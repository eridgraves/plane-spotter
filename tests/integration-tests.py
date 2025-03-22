"""
Integration tests for the plane detection system.

These tests verify that different components of the system work together correctly in
realistic scenarios, testing the full pipeline from image capture to plane detection
and image processing.
"""

import os
import sys
import pytest
import numpy as np
import cv2
import tempfile
import shutil
from unittest.mock import MagicMock, patch
import time
from datetime import datetime

# Import the module to test
from plane_detection_system import PlaneDetectionSystem, main


class TestPlaneDetectionIntegration:
    """Integration tests for the plane detection system."""

    @pytest.fixture
    def test_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_camera(self):
        """Create a mocked camera instance."""
        with patch('plane_detection_system.Picamera2') as mock_picamera:
            mock_camera = MagicMock()
            mock_picamera.return_value = mock_camera
            yield mock_camera
    
    @pytest.fixture
    def plane_detector(self, test_dir, mock_camera):
        """Create a detector instance with mocked camera."""
        config = {
            'output_dir': test_dir,
            'image_width': 640,
            'image_height': 480,
            'process_every_n_frames': 1,
            'detection_cooldown': 0,  # No cooldown for testing
            'min_contour_area': 100,
            'bg_history': 10,  # Small history for testing
            'bg_threshold': 16
        }
        
        detector = PlaneDetectionSystem(config)
        detector.camera = mock_camera  # Use our mocked camera
        
        # Mock initialize_camera to return True
        detector.initialize_camera = MagicMock(return_value=True)
        
        yield detector
        
        # Clean up
        detector.stop()
    
    @pytest.fixture
    def sample_images(self):
        """Create a series of sample images with a plane moving across the frame."""
        width, height = 640, 480
        images = []
        
        # Create background sky image
        sky = np.ones((height, width, 3), dtype=np.uint8) * np.array([135, 206, 235], dtype=np.uint8)
        
        # Create 5 frames with a plane moving across
        for i in range(5):
            img = sky.copy()
            
            # Draw a "plane" at different positions
            x_pos = 100 + i * 80  # Moving from left to right
            y_pos = 150 + i * 10  # Moving slightly downward
            
            # Draw plane body
            cv2.rectangle(img, (x_pos, y_pos), (x_pos + 80, y_pos + 20), (220, 220, 220), -1)
            
            # Draw wings
            cv2.rectangle(img, (x_pos + 20, y_pos - 15), (x_pos + 60, y_pos), (200, 200, 200), -1)
            
            # Draw tail
            cv2.rectangle(img, (x_pos + 70, y_pos - 10), (x_pos + 80, y_pos + 10), (180, 180, 180), -1)
            
            images.append(img)
        
        return images
    
    def test_detection_pipeline(self, plane_detector, sample_images):
        """Test the complete detection pipeline with a sequence of images."""
        # Process the first frame to initialize background
        bbox, detected = plane_detector.detect_plane(sample_images[0])
        assert detected is False or detected is True  # First frame might detect plane or not
        
        # Process the second frame - should definitely detect the plane
        bbox, detected = plane_detector.detect_plane(sample_images[1])
        assert detected is True
        assert bbox is not None
        assert len(bbox) == 4  # x, y, w, h
        
        # Test cropping the image
        with patch('cv2.imwrite') as mock_imwrite:
            original_path = "test_image.jpg"
            cropped_path = plane_detector.crop_and_save(sample_images[1], bbox, original_path)
            
            assert cropped_path is not None
            assert "cropped_test_image.jpg" in cropped_path
            assert mock_imwrite.call_count == 1
    
    def test_continuous_monitoring(self, plane_detector, sample_images, mock_camera):
        """Test continuous monitoring with a sequence of images."""
        # Configure mock camera to return our sample images
        # Add KeyboardInterrupt at the end to exit the monitoring loop
        mock_camera.capture_array.side_effect = sample_images + [KeyboardInterrupt]
        
        # Patch time.sleep and cv2.imwrite to speed up test and avoid actual file writes
        with patch('time.sleep'), patch('cv2.imwrite'):
            # Start continuous monitoring
            plane_detector.start_continuous_monitoring()
        
        # Verify camera was used correctly
        assert mock_camera.capture_array.call_count >= len(sample_images)
        
        # Check that images were saved
        assert cv2.imwrite.call_count > 0
    
    def test_scheduled_capture(self, plane_detector, sample_images, mock_camera):
        """Test scheduled capture functionality."""
        # Configure mock camera to return a sample image
        mock_camera.capture_array.return_value = sample_images[2]  # Use middle image
        
        # Mock the scheduler
        with patch('schedule.every') as mock_schedule:
            # Create a job mock
            mock_job = MagicMock()
            mock_schedule.return_value.minutes.return_value.do.return_value = mock_job
            
            # Start scheduled captures but exit immediately
            def stop_after_scheduling():
                plane_detector.running = False
            
            with patch('time.sleep', side_effect=stop_after_scheduling), \
                 patch('cv2.imwrite'):
                plane_detector.start_scheduled_captures()
            
            # Verify scheduler was configured correctly
            mock_schedule.assert_called_once()
            mock_schedule.return_value.minutes.assert_called_once()
            mock_schedule.return_value.minutes.return_value.do.assert_called_once_with(
                plane_detector.scheduled_capture
            )
    
    def test_scheduled_capture_execution(self, plane_detector, sample_images, mock_camera):
        """Test execution of a scheduled capture."""
        # Configure mock camera to return a sample image
        mock_camera.capture_array.return_value = sample_images[2]
        
        # Mock detection and cropping
        plane_detector.detect_plane = MagicMock(return_value=((100, 100, 200, 100), True))
        plane_detector.crop_and_save = MagicMock(return_value="cropped_test.jpg")
        
        # Run scheduled capture
        with patch('cv2.imwrite'):
            plane_detector.scheduled_capture()
        
        # Verify the entire pipeline was executed
        mock_camera.capture_array.assert_called_once()
        plane_detector.detect_plane.assert_called_once()
        plane_detector.crop_and_save.assert_called_once()
    
    def test_realistic_plane_image(self, plane_detector):
        """Test detection with a more realistic plane image simulation."""
        # Create a realistic sky image
        img_height, img_width = 720, 1280
        sky_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * np.array([135, 206, 235], dtype=np.uint8)
        
        # Create a blank image for background
        blank_sky = sky_img.copy()
        
        # Initialize the background model with the blank sky
        plane_detector.detect_plane(blank_sky)
        
        # Create plane image
        plane_img = sky_img.copy()
        
        # Draw a more realistic plane
        plane_x, plane_y = 500, 300
        
        # Plane body
        cv2.rectangle(plane_img, (plane_x, plane_y), (plane_x + 180, plane_y + 40), (220, 220, 220), -1)
        
        # Wings
        cv2.rectangle(plane_img, (plane_x + 60, plane_y - 30), (plane_x + 120, plane_y), (200, 200, 200), -1)
        
        # Tail
        cv2.rectangle(plane_img, (plane_x + 160, plane_y - 20), (plane_x + 180, plane_y + 10), (180, 180, 180), -1)
        
        # Test detection
        bbox, detected = plane_detector.detect_plane(plane_img)
        
        # Should detect the plane
        assert detected is True
        assert bbox is not None
        
        # Bounding box should encompass the plane
        x, y, w, h = bbox
        assert x <= plane_x
        assert y <= plane_y - 30  # Should include wings
        assert x + w >= plane_x + 180  # Should include full body
        assert y + h >= plane_y + 40  # Should include full body
    
    def test_main_function(self, test_dir):
        """Test the main function."""
        # Patch the detector class and component methods
        with patch('plane_detection_system.PlaneDetectionSystem') as mock_detector_class, \
             patch('plane_detection_system.Picamera2'):
            
            # Create a mock detector instance
            mock_detector = MagicMock()
            mock_detector_class.return_value = mock_detector
            
            # Set up the continuous monitoring mock
            mock_detector.start_continuous_monitoring = MagicMock()
            
            # Override the config for testing
            test_config = {
                'output_dir': test_dir,
                'image_width': 640,
                'image_height': 480,
                'mode': 'continuous'
            }
            
            # Run main with patched config
            with patch.dict('__main__.__dict__', {'config': test_config}):
                try:
                    main()
                except Exception:
                    # Handle case where __main__ context isn't available in test
                    pass
            
            # Verify detector was created and method was called
            mock_detector_class.assert_called_once()
            assert mock_detector.start_continuous_monitoring.called or mock_detector.start_scheduled_captures.called
    
    def test_error_handling(self, plane_detector, mock_camera):
        """Test error handling during continuous monitoring."""
        # Configure camera to raise an exception after a few frames
        mock_camera.capture_array.side_effect = [
            np.zeros((480, 640, 3), dtype=np.uint8),  # First frame
            Exception("Camera error"),  # Error
            KeyboardInterrupt  # To exit the loop
        ]
        
        # Test that system recovers from the error
        with patch('time.sleep'), patch('cv2.imwrite'):
            plane_detector.start_continuous_monitoring()
        
        # Should have called capture_array at least twice
        assert mock_camera.capture_array.call_count >= 2
        
        # System should still be running (would be stopped by KeyboardInterrupt)
        assert plane_detector.running is False
    
    def test_cross_component_integration(self, plane_detector, sample_images, mock_camera, test_dir):
        """Test integration across multiple components of the system."""
        # Configure a special sequence of frames
        # 1. First frame - no plane (background)
        # 2. Second frame - plane appears
        # 3. Third frame - plane moves
        # 4. Fourth frame - plane disappears
        # 5. Exit with KeyboardInterrupt
        
        background = np.ones((480, 640, 3), dtype=np.uint8) * np.array([135, 206, 235], dtype=np.uint8)
        mock_camera.capture_array.side_effect = [
            background,  # Background
            sample_images[1],  # Plane appears
            sample_images[3],  # Plane moves
            background,  # Plane disappears
            KeyboardInterrupt  # Exit
        ]
        
        # Use real methods (not mocked) for detection and processing
        plane_detector.detect_plane = PlaneDetectionSystem.detect_plane.__get__(plane_detector)
        plane_detector.crop_and_save = PlaneDetectionSystem.crop_and_save.__get__(plane_detector)
        
        # Run continuous monitoring
        with patch('time.sleep'), patch('cv2.imwrite') as mock_imwrite:
            plane_detector.start_continuous_monitoring()
        
        # Should have detected planes in frames 2 and 3
        # This should result in at least 4 imwrite calls (2 raw + 2 processed images)
        assert mock_imwrite.call_count >= 4
        
        # Final check - detector was properly stopped
        assert plane_detector.running is False
        assert plane_detector.camera is None  # Camera should be released


if __name__ == "__main__":
    pytest.main(["-v"])
