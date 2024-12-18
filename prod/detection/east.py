import cv2
import numpy as np
import math


class EAST:
    """
    EAST TEXT DETECTION EXAMPLE
    https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.py

    Text detection model: https://github.com/argman/EAST
    Download link: https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1
    """

    def __init__(self, model_path, CONF_THRESHOLD=0.5, NMS_THRESHOLD=0.4):
        self.__CONF_THRESHOLD  = CONF_THRESHOLD  # Confidence threshold
        self.__NMS_THRESHOLD   = NMS_THRESHOLD  # Non-Maximum Suppression threshold

        # Load the EAST model
        self.detector = cv2.dnn.readNet(model_path)


    def __decodeBoundingBoxes(self, scores, geometry, scoreThresh):
        # https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.py
        detections = []
        confidences = []

        height, width = scores.shape[2:4]
        for y in range(height):
            scoresData = scores[0, 0, y]
            x0_data = geometry[0, 0, y]
            x1_data = geometry[0, 1, y]
            x2_data = geometry[0, 2, y]
            x3_data = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            for x in range(width):
                score = scoresData[x]

                if score < scoreThresh:
                    continue

                offsetX = x * 4.0
                offsetY = y * 4.0

                angle = anglesData[x]
                cosA = math.cos(angle)
                sinA = math.sin(angle)
                h = x0_data[x] + x2_data[x]
                w = x1_data[x] + x3_data[x]

                offset = (
                    offsetX + cosA * x1_data[x] + sinA * x2_data[x],
                    offsetY - sinA * x1_data[x] + cosA * x2_data[x]
                )

                p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
                p3 = (-cosA * w + offset[0], sinA * w + offset[1])
                center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))

                detections.append((center, (w, h), -angle * 180.0 / math.pi))
                confidences.append(float(score))

        return [detections, confidences]


    def __resize_with_aspect_ratio(self, width, height, max_width=960, max_height=1280):
        # default 4x3=1280x960,
        # но с этой функцией может быть например и 480x960

        if width < height and max_width > max_height:
            # vertical
            max_height, max_width = max_width, max_height

        # Calculate the aspect ratio
        aspect_ratio = width / height

        # Checking the restrictions
        if max_width / aspect_ratio <= max_height:
        # if height become bigger than max_height
            # Limit the width
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:
        # otherwise
            # Limit the height
            new_height = max_height
            new_width = int(max_height * aspect_ratio)

        return new_width, new_height


    def detect(self, image):
        """Detect text regions in the input image."""
        origH, origW = image.shape[:2]

        # Input dimensions for processing
        # Ширина и высота для предобработки должна быть кратна 32
        # Round dimensions to the nearest smaller multiple of 32
        WIDTH, HEIGHT = origW, origH
        WIDTH, HEIGHT = self.__resize_with_aspect_ratio(WIDTH, HEIGHT) # Max values
        WIDTH, HEIGHT = (WIDTH // 32) * 32, (HEIGHT // 32) * 32

        print("HEIGHT: ", HEIGHT, "WIDTH: ", WIDTH)

        rW = origW / float(WIDTH)
        rH = origH / float(HEIGHT)

        # Preprocess the image
            # (123.68, 116.78, 103.94): Normalize with mean values for RGB channels
            # True: Swap red and blue channels (BGR to RGB).
            # False: No cropping applied to the image.
        blob = cv2.dnn.blobFromImage(image, 1.0, (WIDTH, HEIGHT), (123.68, 116.78, 103.94), True, False)
        print('Blob datatype:', blob.dtype, ", dimensions:", blob.shape)

        self.detector.setInput(blob)

        # Run the model
        layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
        scores, geometry = self.detector.forward(layerNames)

        # Decode text regions
        boxes, confidences = self.__decodeBoundingBoxes(scores, geometry, self.__CONF_THRESHOLD)

        # Apply NMS
        indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, self.__CONF_THRESHOLD, self.__NMS_THRESHOLD)
        print('indices datatype:', indices.dtype, ", dimensions:", indices.shape)

        text_regions = []
        for i in indices:
            # boxes[i] ((center_x, center_y), (width, height), angle)
            center, size, angle = boxes[i]
            # vertices = cv2.boxPoints(boxes[i])
            vertices = cv2.boxPoints(((center[0] * rW, center[1] * rH), (size[0] * rW, size[1] * rH), angle))
            # vertices = np.int0(vertices * [rW, rH])
            vertices = np.int0(vertices)
            text_regions.append(vertices)

        return text_regions


    def avg_height(self, image):
        """Detect text regions in the input image."""
        origH, origW = image.shape[:2]

         # Input dimensions for processing
        # Ширина и высота для предобработки должна быть кратна 32
        # Round dimensions to the nearest smaller multiple of 32
        WIDTH, HEIGHT = origW, origH
        WIDTH, HEIGHT = self.__resize_with_aspect_ratio(WIDTH, HEIGHT) # Max values
        WIDTH, HEIGHT = (WIDTH // 32) * 32, (HEIGHT // 32) * 32

        rW = origW / float(WIDTH)
        rH = origH / float(HEIGHT)

        # Preprocess the image
            # (123.68, 116.78, 103.94): Normalize with mean values for RGB channels
            # True: Swap red and blue channels (BGR to RGB).
            # False: No cropping applied to the image.
        blob = cv2.dnn.blobFromImage(image, 1.0, (WIDTH, HEIGHT), (123.68, 116.78, 103.94), True, False)
        print('Blob datatype:', blob.dtype, ", dimensions:", blob.shape)

        self.detector.setInput(blob)

        # Run the model
        layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
        scores, geometry = self.detector.forward(layerNames)

        # Decode text regions
        boxes, confidences = self.__decodeBoundingBoxes(scores, geometry, self.__CONF_THRESHOLD)

        # Apply NMS
        indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, self.__CONF_THRESHOLD, self.__NMS_THRESHOLD)
        print('indices datatype:', indices.dtype, ", dimensions:", indices.shape)

        text_regions = []
        heights = []
        for i in indices:
            # boxes[i] ((center_x, center_y), (width, height), angle)
            center, size, angle = boxes[i]

            # vertices = cv2.boxPoints(boxes[i])
            vertices = cv2.boxPoints(((center[0] * rW, center[1] * rH), (size[0] * rW, size[1] * rH), angle))
            # vertices = np.int0(vertices * [rW, rH])
            vertices = np.int0(vertices)
            text_regions.append(vertices)

            heights.append(size[1] * rH)  # Scale height back to original image dimensions

        # Calculate average height
        avg_height = sum(heights) // len(heights) if heights else 0

        words = self.draw_boxes(image, text_regions)

        # Returns the actual value and steps, but EAST has only one
        return avg_height, [{"image": words, "title": f"EAST: {round(avg_height, 2)}"}]


    def draw_boxes(self, image, text_regions):
        # Draw bounding boxes
        image = image.copy()
        for vertices in text_regions:
            cv2.polylines(image, [vertices], isClosed=True, color=(0, 255, 0), thickness=2)
        return image
