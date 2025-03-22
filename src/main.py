import cv2
import numpy as np
import time
import os
from datetime import datetime
from picamera2 import Picamera2
import threading
import schedule

class PlaneDetectionSystem:
    def __init__(self, config):
        """Initialize the plane detection system with configuration parameters."""
        self.config = config
        self.camera = None
        self.running = False
        self.output_dir = config.get('output_dir', 'captured_images')
        self.raw_dir = os.path.join(self.output_dir, 'raw')
        self.processed_dir = os.path.join(self.output_dir, 'processed')
        
        # Create output directories if they don't exist
        for directory in [self.output_dir, self.raw_dir, self.processed_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Initialize background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.get('bg_history', 500),
            varThreshold=config.get('bg_threshold', 16),
            detectShadows=False
        )
        
        # Load the plane detection model if using object detection
        if config.get('use_object_detection', False):
            self._load_detection_model()
    
    def _load_detection_model(self):
        """Load a pre-trained model for plane detection."""
        # For simplicity, we're using a basic Haar cascade
        # In a real implementation, you might want to use a more sophisticated model
        # like YOLO or SSD trained specifically for aircraft
        try:
            self.detector = cv2.CascadeClassifier('models/aircraft_cascade.xml')
            print("Loaded aircraft detection model")
        except Exception as e:
            print(f"Warning: Could not load detection model. Using motion detection only. Error: {e}")
            self.config['use_object_detection'] = False
    
    def initialize_camera(self):
        """Set up and initialize the camera with appropriate settings."""
        try:
            self.camera = Picamera2()
            
            # Configure camera
            config = self.camera.create_still_configuration(
                main={"size": (self.config.get('image_width', 1920), 
                               self.config.get('image_height', 1080))},
                lores={"size": (640, 480)},
                display="lores"
            )
            self.camera.configure(config)
            
            # Set any camera parameters
            if 'exposure_time' in self.config:
                self.camera.set_controls({"ExposureTime": self.config['exposure_time']})
            if 'iso' in self.config:
                self.camera.set_controls({"AnalogueGain": self.config['iso']})
            
            self.camera.start()
            time.sleep(2)  # Allow camera to warm up
            print("Camera initialized successfully")
            return True
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            return False
    
    def capture_image(self):
        """Capture an image from the camera."""
        if not self.camera:
            print("Camera not initialized")
            return None
        
        try:
            raw_image = self.camera.capture_array()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_path = os.path.join(self.raw_dir, f"plane_{timestamp}.jpg")
            cv2.imwrite(raw_path, raw_image)
            print(f"Captured image saved to {raw_path}")
            return raw_image, raw_path
        except Exception as e:
            print(f"Error capturing image: {e}")
            return None, None
    
    def detect_plane(self, image):
        """Detect if a plane is present in the image."""
        if image is None:
            return None, False
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply background subtraction to detect motion
        fg_mask = self.bg_subtractor.apply(gray)
        fg_mask = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)[1]
        
        # Find contours in the foreground mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size to find potential planes
        min_area = self.config.get('min_contour_area', 500)
        potential_planes = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        if not potential_planes:
            return None, False
        
        if self.config.get('use_object_detection', False) and self.detector:
            # Use the trained model for more accurate detection
            planes = self.detector.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            if len(planes) > 0:
                # Return the largest detection
                x, y, w, h = max(planes, key=lambda rect: rect[2] * rect[3])
                return (x, y, w, h), True
        
        # If no specific plane detection or nothing detected,
        # use the largest motion contour
        largest_contour = max(potential_planes, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Expand the bounding box slightly
        padding = int(max(w, h) * 0.2)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        return (x, y, w, h), True
    
    def crop_and_save(self, image, bbox, original_path):
        """Crop the image to center on the detected plane and save it."""
        if image is None or bbox is None:
            return
        
        x, y, w, h = bbox
        cropped = image[y:y+h, x:x+w]
        
        # Generate the output filename based on the original
        filename = os.path.basename(original_path)
        output_path = os.path.join(self.processed_dir, f"cropped_{filename}")
        
        cv2.imwrite(output_path, cropped)
        print(f"Cropped image saved to {output_path}")
        return output_path
    
    def scheduled_capture(self):
        """Perform a scheduled image capture and processing."""
        print(f"Scheduled capture at {datetime.now().strftime('%H:%M:%S')}")
        image, path = self.capture_image()
        if image is not None:
            bbox, detected = self.detect_plane(image)
            if detected:
                self.crop_and_save(image, bbox, path)
                print("Plane detected and image cropped")
            else:
                print("No plane detected in this image")
    
    def start_scheduled_captures(self):
        """Set up a schedule for regular captures."""
        interval_minutes = self.config.get('capture_interval_minutes', 5)
        schedule.every(interval_minutes).minutes.do(self.scheduled_capture)
        
        self.running = True
        while self.running:
            schedule.run_pending()
            time.sleep(1)
    
    def start_continuous_monitoring(self):
        """Continuously monitor for planes."""
        if not self.initialize_camera():
            print("Failed to initialize camera. Exiting.")
            return
        
        print("Starting continuous monitoring...")
        self.running = True
        
        # Set up a background reference
        background = None
        frame_count = 0
        
        while self.running:
            try:
                # Capture a frame
                frame = self.camera.capture_array()
                frame_count += 1
                
                # Only process every few frames to reduce CPU usage
                if frame_count % self.config.get('process_every_n_frames', 5) != 0:
                    continue
                
                bbox, detected = self.detect_plane(frame)
                
                if detected:
                    print("Plane detected!")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    raw_path = os.path.join(self.raw_dir, f"plane_{timestamp}.jpg")
                    cv2.imwrite(raw_path, frame)
                    
                    self.crop_and_save(frame, bbox, raw_path)
                    
                    # After capturing and saving, pause briefly to avoid
                    # capturing too many images of the same plane
                    time.sleep(self.config.get('detection_cooldown', 10))
                
                # Small delay between frames
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                self.running = False
                print("Monitoring stopped by user")
            except Exception as e:
                print(f"Error during monitoring: {e}")
                time.sleep(1)  # Pause briefly on error
    
    def stop(self):
        """Stop the monitoring process."""
        self.running = False
        if self.camera:
            self.camera.stop()
            self.camera = None
        print("System stopped")


# TODO: break up this monster of a file into multiple files: i.e. camera.py, detector.py, util.py
 
def main():
    """Main function to set up and run the plane detection system."""
    # Default configuration
    config = {
        'output_dir': 'plane_images',
        'image_width': 1920,
        'image_height': 1080,
        'capture_interval_minutes': 5,
        'process_every_n_frames': 3,
        'detection_cooldown': 15,
        'min_contour_area': 1000,
        'bg_history': 500,
        'bg_threshold': 16,
        'use_object_detection': False,
        'mode': 'continuous'  # or 'scheduled'
    }

    # TODO: Day/Dusk/Night configs and auto switch based on time of day
    
    # Create the plane detection system
    detector = PlaneDetectionSystem(config)
    
    try:
        if config['mode'] == 'scheduled':
            detector.start_scheduled_captures()
        else:
            detector.start_continuous_monitoring()
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        detector.stop()


if __name__ == "__main__":
    main()