"""
Tests for the plane detection system.

This test suite covers the functionality of the plane detection system,
including initialization, camera operations, plane detection, and image processing.
"""

import os
import sys
import pytest
import numpy as np
import cv2
from datetime import datetime
import tempfile
import shutil
from unittest.mock import MagicMock, patch

# Add the parent directory to the path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module to test
from plane_detection_system import PlaneDetectionSystem


class TestPlaneDetectionSystem:
    """Test cases for the plane detection system class."""

    @pytest.fixture
    def test_config(self):
        """Create a test configuration."""
        # Use a temporary directory for test outputs
        temp_dir = tempfile.mkdtemp()
        
        config = {
            'output_dir': temp_dir,
            'image_width': 640,
            'image_height': 480,
            'capture_interval_minutes': 1,
            'process_every_n_frames': 1,
            'detection_cooldown': 1,
            'min_contour_area': 100,
            'bg_history': 50,
            'bg_threshold': 16,
            'use_object_detection': False,
            'mode': 'continuous'
        }
        
        yield config
        
        # Clean up the temporary directory after tests
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def detector(self, test_config):
        """Create a detector instance with mocked camera."""
        with patch('plane_detection_system.Picamera2') as mock_picamera:
            # Setup mock camera
            mock_camera = MagicMock()
            mock_picamera.return_value = mock_camera
            
            # Create detector with the test config
            detector = PlaneDetectionSystem(test_config)
            
            # Provide the detector with the mock camera
            detector.camera = mock_camera
            
            yield detector
            
            # Clean up
            if detector.camera:
                detector.stop()
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        # Create a blank image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw a "plane" (rectangle) on it
        cv2.rectangle(img, (100, 100), (200, 150), (255, 255, 255), -1)
        
        return img
    
    def test_initialization(self, test_config):
        """Test that the system initializes correctly."""
        detector = PlaneDetectionSystem(test_config)
        
        assert detector.output_dir == test_config['output_dir']
        assert detector.raw_dir == os.path.join(test_config['output_dir'], 'raw')
        assert detector.processed_dir == os.path.join(test_config['output_dir'], 'processed')
        
        # Check that the directories were created
        assert os.path.exists(detector.output_dir)
        assert os.path.exists(detector.raw_dir)
        assert os.path.exists(detector.processed_dir)
        
        # Check that the background subtractor was initialized
        assert detector.bg_subtractor is not None
    
    def test_initialize_camera(self, detector):
        """Test camera initialization."""
        with patch.object(detector.camera, 'configure'), \
             patch.object(detector.camera, 'start'), \
             patch('time.sleep'):
            
            result = detector.initialize_camera()
            assert result is True
            detector.camera.configure.assert_called_once()
            detector.camera.start.assert_called_once()
    
    def test_initialize_camera_exception(self, detector):
        """Test camera initialization with exception."""
        detector.camera.configure.side_effect = Exception("Camera error")
        
        result = detector.initialize_camera()
        assert result is False
    
    def test_capture_image(self, detector, test_config, sample_image):
        """Test image capture functionality."""
        # Mock the camera capture_array method to return our sample image
        detector.camera.capture_array.return_value = sample_image
        
        # Call capture_image
        with patch('cv2.imwrite') as mock_imwrite:
            mock_imwrite.return_value = True
            image, path = detector.capture_image()
        
        # Check the results
        assert image is not None
        assert np.array_equal(image, sample_image)
        assert path is not None
        assert path.startswith(detector.raw_dir)
        assert 'plane_' in path
        assert path.endswith('.jpg')
    
    def test_capture_image_no_camera(self, detector):
        """Test image capture with no camera initialized."""
        detector.camera = None
        
        image, path = detector.capture_image()
        
        assert image is None
        assert path is None
    
    def test_capture_image_exception(self, detector):
        """Test image capture with exception."""
        detector.camera.capture_array.side_effect = Exception("Capture error")
        
        image, path = detector.capture_image()
        
        assert image is None
        assert path is None
    
    def test_detect_plane(self, detector, sample_image):
        """Test plane detection in an image."""
        # Process the image through the detector
        bbox, detected = detector.detect_plane(sample_image)
        
        # Check the results
        assert detected is True
        assert bbox is not None
        assert len(bbox) == 4  # x, y, w, h
        
        # The bounding box should roughly correspond to our "plane"
        x, y, w, h = bbox
        # Allow for some padding
        assert 0 <= x <= 100
        assert 0 <= y <= 100
        assert w >= 100  # The rectangle was 100px wide
        assert h >= 50   # The rectangle was 50px tall
    
    def test_detect_plane_none_image(self, detector):
        """Test plane detection with None image."""
        bbox, detected = detector.detect_plane(None)
        
        assert detected is False
        assert bbox is None
    
    def test_detect_plane_no_plane(self, detector):
        """Test plane detection with no plane in the image."""
        # Create a completely black image
        blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # First call will initialize the background model
        detector.detect_plane(blank_img)
        
        # Second call should detect no movement
        bbox, detected = detector.detect_plane(blank_img)
        
        assert detected is False
        assert bbox is None
    
    def test_crop_and_save(self, detector, sample_image):
        """Test image cropping and saving."""
        # Define a bounding box
        bbox = (100, 100, 100, 50)  # x, y, w, h
        original_path = os.path.join(detector.raw_dir, "test_image.jpg")
        
        # Call crop_and_save
        with patch('cv2.imwrite') as mock_imwrite:
            mock_imwrite.return_value = True
            output_path = detector.crop_and_save(sample_image, bbox, original_path)
        
        # Verify the results
        assert output_path is not None
        assert output_path.startswith(detector.processed_dir)
        assert "cropped_test_image.jpg" in output_path
        
        # Check that imwrite was called with the correct arguments
        mock_imwrite.assert_called_once()
        # The first arg is the file path
        assert mock_imwrite.call_args[0][0] == output_path
        # The second arg is the cropped image - we don't check its contents here
    
    def test_scheduled_capture(self, detector, sample_image):
        """Test scheduled capture functionality."""
        # Setup mocks
        detector.capture_image = MagicMock(return_value=(sample_image, "test_path.jpg"))
        detector.detect_plane = MagicMock(return_value=((100, 100, 100, 50), True))
        detector.crop_and_save = MagicMock(return_value="cropped_path.jpg")
        
        # Call scheduled_capture
        detector.scheduled_capture()
        
        # Verify the correct methods were called
        detector.capture_image.assert_called_once()
        detector.detect_plane.assert_called_once_with(sample_image)
        detector.crop_and_save.assert_called_once()
    
    def test_start_continuous_monitoring(self, detector, sample_image):
        """Test continuous monitoring functionality."""
        # Setup mocks
        detector.initialize_camera = MagicMock(return_value=True)
        detector.camera.capture_array = MagicMock(return_value=sample_image)
        detector.detect_plane = MagicMock(side_effect=[
            ((100, 100, 100, 50), True),  # First call - plane detected
            ((100, 100, 100, 50), False), # Second call - no plane
            KeyboardInterrupt                # Third call - interrupt
        ])
        detector.crop_and_save = MagicMock()
        
        # Mock time.sleep to avoid delays
        with patch('time.sleep'), patch('cv2.imwrite'):
            # Start monitoring - it should exit when detect_plane raises KeyboardInterrupt
            detector.start_continuous_monitoring()
        
        # Verify the correct methods were called
        detector.initialize_camera.assert_called_once()
        assert detector.camera.capture_array.call_count > 0
        assert detector.detect_plane.call_count > 0
        detector.crop_and_save.assert_called_once()
    
    def test_stop(self, detector):
        """Test stopping the detection system."""
        # Setup
        detector.running = True
        
        # Call stop
        detector.stop()
        
        # Verify the system was stopped
        assert detector.running is False
        detector.camera.stop.assert_called_once()
        assert detector.camera is None


class TestObjectDetection:
    """Test cases specifically for object detection functionality."""
    
    @pytest.fixture
    def test_config_with_object_detection(self):
        """Create a test configuration with object detection enabled."""
        temp_dir = tempfile.mkdtemp()
        
        config = {
            'output_dir': temp_dir,
            'image_width': 640,
            'image_height': 480,
            'use_object_detection': True,
            'mode': 'continuous'
        }
        
        yield config
        
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_load_detection_model(self, test_config_with_object_detection):
        """Test loading of the detection model."""
        with patch('cv2.CascadeClassifier') as mock_classifier, \
             patch('plane_detection_system.Picamera2'):
            
            # Setup the mock classifier
            mock_detector = MagicMock()
            mock_classifier.return_value = mock_detector
            
            # Create detector with object detection enabled
            detector = PlaneDetectionSystem(test_config_with_object_detection)
            
            # Check that the classifier was created
            mock_classifier.assert_called_once()
            assert detector.detector is mock_detector
    
    def test_load_detection_model_fails(self, test_config_with_object_detection):
        """Test graceful failure when model loading fails."""
        with patch('cv2.CascadeClassifier') as mock_classifier, \
             patch('plane_detection_system.Picamera2'):
            
            # Setup the mock classifier to raise an exception
            mock_classifier.side_effect = Exception("Model not found")
            
            # Create detector with object detection enabled
            detector = PlaneDetectionSystem(test_config_with_object_detection)
            
            # Check that use_object_detection was set to False
            assert detector.config['use_object_detection'] is False
    
    def test_detect_plane_with_model(self, test_config_with_object_detection, sample_image):
        """Test plane detection using the object detection model."""
        with patch('cv2.CascadeClassifier') as mock_classifier, \
             patch('plane_detection_system.Picamera2'):
            
            # Setup the mock classifier
            mock_detector = MagicMock()
            # Make detectMultiScale return a single detection
            mock_detector.detectMultiScale.return_value = np.array([[50, 50, 150, 100]])
            mock_classifier.return_value = mock_detector
            
            # Create detector with object detection enabled
            detector = PlaneDetectionSystem(test_config_with_object_detection)
            
            # Test the detect_plane method
            bbox, detected = detector.detect_plane(sample_image)
            
            # Check that the model was used
            mock_detector.detectMultiScale.assert_called_once()
            
            # Check the results
            assert detected is True
            assert bbox == (50, 50, 150, 100)


if __name__ == "__main__":
    pytest.main(["-v"])
