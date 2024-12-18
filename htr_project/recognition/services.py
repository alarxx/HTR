# recognition/services.py
import os
from django.conf import settings
from .ml_models.recognizer import KazakhTextRecognizer
import cv2
import numpy as np


class TextRecognitionService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TextRecognitionService, cls).__new__(cls)
            # Initialize the instance
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        """Initialize the recognizer with the correct model paths"""
        try:
            model_dir = os.path.join(settings.BASE_DIR, 'recognition', 'ml_models')

            detector_path = os.path.join(model_dir, "frozen_east_text_detection.pb")
            classifier_path = os.path.join(model_dir, "FCNN_CTC_main.pth")
            font_path = os.path.join(model_dir, "arial.ttf")

            # Verify that all required files exist
            if not all(os.path.exists(path) for path in [detector_path, classifier_path, font_path]):
                raise FileNotFoundError("One or more required model files are missing")

            # Initialize the recognizer
            self.recognizer = KazakhTextRecognizer(
                detector_model_path=detector_path,
                classifier_model_path=classifier_path,
                font_path=font_path
            )
        except Exception as e:
            print(f"Error initializing TextRecognitionService: {str(e)}")
            raise

    def process_image(self, image_path):
        """
        Process an image and return the recognized text
        """
        try:
            # Read image using OpenCV
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("Could not read image")

            # Perform text recognition
            text_regions, text_lines, steps = self.recognizer.detect_text_regions(image)
            word_images = self.recognizer.crop_words(image, text_regions)
            recognized_texts = self.recognizer.recognize_text(word_images)

            # Create visualization
            painted_image = self.recognizer.draw_boxes(image, text_lines, text_regions)
            result_image = self.recognizer.draw_recognized_text(painted_image, text_regions, recognized_texts)

            # Convert result image to bytes for saving
            _, buffer = cv2.imencode('.png', result_image)
            image_bytes = buffer.tobytes()

            return {
                'text': ' '.join(recognized_texts),
                'visualization': image_bytes
            }
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            raise