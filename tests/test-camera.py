"""
Tests for camera functionality in the plane detection system.

These tests focus on the camera initialization and capture functionality,
ensuring proper error handling and configuration.
"""

import os
import sys
import pytest
import numpy as np
import cv2
import tempfile
from unittest.mock import MagicMock, patch, call

# Add the parent directory to the path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module to test
from plane_detection_system import PlaneDetectionSystem


class TestCameraFunctionality:
    """Test cases for camera-specific functionality."""

    @pytest.fixture
    def mock_picamera(self):
        """Create a mock for the Picamera2 class."""
        with patch('plane_detection_system.Picamera2') as mock:
            # Create a mock camera instance
            mock_camera = MagicMock()
            mock.return_value = mock_camera
            
            # Mock the create_still_configuration method
            mock_camera.create_still_configuration.return_value = {"test": "config"}
            
            yield mock, mock_camera

    @pytest.fixture
    def basic_config(self):
        """Create a basic configuration for testing."""
        return {
            'output_dir': tempfile.mkdtemp(),
            'image_width': 800,
            'image_height': 600,
            'exposure_time': 10000,  # 10ms
            'iso': 800
        }

    def test_camera_initialization_with_params(self, mock_picamera, basic_config):
        """Test camera initialization with specific parameters."""
        mock_picamera_class, mock_camera = mock_picamera
        
        # Create detector
        detector = PlaneDetectionSystem(basic_config)
        
        # Initialize camera
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = detector.initialize_camera()
        
        # Verify successful initialization
        assert result is True
        
        # Check that the camera was configured correctly
        mock_camera.create_still_configuration.assert_called_once_with(
            main={"size": (800, 600)},
            lores={"size": (640, 480)},
            display="lores"
        )
        
        # Check that camera parameters were set
        mock_camera.set_controls.assert_any_call({"ExposureTime": 10000})
        mock_camera.set_controls.assert_any_call({"AnalogueGain": 800})
        
        # Check that camera was started
        mock_camera.configure.assert_called_once()
        mock_camera.start.assert_called_once()

    def test_camera_initialization_default_params(self, mock_picamera):
        """Test camera initialization with default parameters."""
        mock_picamera_class, mock_camera = mock_picamera
        
        # Create detector with minimal config
        detector = PlaneDetectionSystem({'output_dir': tempfile.mkdtemp()})
        
        # Initialize camera
        with patch('time.sleep'):
            result = detector.initialize_camera()
        
        # Verify successful initialization
        assert result is True
        
        # Check default resolution
        mock_camera.create_still_configuration.assert_called_once_with(
            main={"size": (1920, 1080)},
            lores={"size": (640, 480)},
            display="lores"
        )
        
        # Check that no additional controls were set (no exposure_time or iso in config)
        assert mock_camera.set_controls.call_count == 0

    def test_camera_initialization_failure(self, mock_picamera):
        """Test handling of camera initialization failure."""
        mock_picamera_class, mock_camera = mock_picamera
        
        # Make the camera.configure method raise an exception
        mock_camera.configure.side_effect = Exception("Camera initialization failed")
        
        # Create detector
        detector = PlaneDetectionSystem({'output_dir': tempfile.mkdtemp()})
        
        # Try to initialize camera
        result = detector.initialize_camera()
        
        # Verify initialization failed
        assert result is False
        
        # Camera start should not have been called
        mock_camera.start.assert_not_called()

    def test_capture_image_successful(self, mock_picamera, basic_config):
        """Test successful image capture."""
        mock_picamera_class, mock_camera = mock_picamera
        
        # Create a sample image as return value
        sample_image = np.zeros((600, 800, 3), dtype=np.uint8)
        mock_camera.capture_array.return_value = sample_image
        
        # Create detector and set the mock camera
        detector = PlaneDetectionSystem(basic_config)
        detector.camera = mock_camera
        
        # Mock cv2.imwrite
        with patch('cv2.imwrite') as mock_imwrite:
            mock_imwrite.return_value = True
            image, path = detector.capture_image()
        
        # Verify image capture
        assert image is sample_image
        assert path is not None
        assert path.startswith(detector.raw_dir)
        assert path.endswith('.jpg')
        
        # Check that image was saved
        mock_imwrite.assert_called_once()
        # First arg is file path, second is the image
        assert mock_imwrite.call_args[0][1] is sample_image

    def test_capture_image_exception(self, mock_picamera, basic_config):
        """Test handling of exception during image capture."""
        mock_picamera_class, mock_camera = mock_picamera
        
        # Make capture_array raise an exception
        mock_camera.capture_array.side_effect = Exception("Capture failed")
        
        # Create detector and set the mock camera
        detector = PlaneDetectionSystem(basic_config)
        detector.camera = mock_camera
        
        # Try to capture image
        image, path = detector.capture_image()
        
        # Verify capture failed
        assert image is None
        assert path is None

    def test_stop_camera(self, mock_picamera, basic_config):
        """Test stopping the camera."""
        mock_picamera_class, mock_camera = mock_picamera
        
        # Create detector and set the mock camera
        detector = PlaneDetectionSystem(basic_config)
        detector.camera = mock_camera
        detector.running = True
        
        # Stop the detector
        detector.stop()
        
        # Verify camera was stopped
        assert detector.running is False
        mock_camera.stop.assert_called_once()
        assert detector.camera is None

    def test_continuous_monitoring_integration(self, mock_picamera, basic_config):
        """Test the integration of camera with continuous monitoring."""
        mock_picamera_class, mock_camera = mock_picamera
        
        # Create sample images for the mock camera to return
        blank_image = np.zeros((600, 800, 3), dtype=np.uint8)
        plane_image = blank_image.copy()
        # Draw a "plane" on the second image
        cv2.rectangle(plane_image, (300, 200), (500, 300), (255, 255, 255), -1)
        
        # Configure mock camera to return different images on subsequent calls
        mock_camera.capture_array.side_effect = [blank_image, plane_image, KeyboardInterrupt]
        
        # Create detector with the mock camera
        detector = PlaneDetectionSystem(basic_config)
        detector.initialize_camera = MagicMock(return_value=True)
        detector.camera = mock_camera
        
        # Mock the detect_plane and crop_and_save methods to check their calls
        detector.detect_plane = MagicMock(side_effect=[
            (None, False),  # No plane in first image
            ((300, 200, 200, 100), True),  # Plane in second image
        ])
        detector.crop_and_save = MagicMock()
        
        # Run continuous monitoring with mocked time.sleep and cv2.imwrite
        with patch('time.sleep'), patch('cv2.imwrite'):
            detector.start_continuous_monitoring()
        
        # Verify the functionality
        detector.initialize_camera.assert_called_once()
        assert mock_camera.capture_array.call_count >= 2
        assert detector.detect_plane.call_count >= 2
        
        # Should save the raw image when a plane is detected
        assert cv2.imwrite.called
        
        # Should call crop_and_save when a plane is detected
        detector.crop_and_save.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-v"])
