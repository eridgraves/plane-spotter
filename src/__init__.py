"""
Plane Spotter - A Raspberry Pi-based system for detecting and capturing images of aircraft.

This package contains modules for camera control, plane detection, and image processing.
"""

__version__ = '0.1.0'
__author__ = 'Eric Graves'

# Make key classes available at the package level for easier imports
from .detector import PlaneDetectionSystem
from .camera import CameraManager
from .utils import setup_logging, load_config

# Set up package-level logger
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())