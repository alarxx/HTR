

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


    def __resize_with_aspect_ratio(self, width, height, max_width=960, max_height=1280): # 4x3
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

        heights = []
        for i in indices:
            _, size, _ = boxes[i]
            heights.append(size[1] * rH)  # Scale height back to original image dimensions

        # Calculate average height
        avg_height = sum(heights) // len(heights) if heights else 0
        if avg_height % 2 == 0:
            avg_height -= 1
        return int(avg_height)


    def draw_boxes(self, image, text_regions):
        # Draw bounding boxes
        for vertices in text_regions:
            cv2.polylines(frame, [vertices], isClosed=True, color=(0, 255, 0), thickness=2)
        return image



class Morpological:
    """
    Morphological Text Detection
    https://github.com/alarxx/Handwriting-Recognition/blob/master/android_project/app/src/main/java/com/rat6/utils/WordRec.java
    """
    def __init__(self):
        pass

    def detect(self, image, kernel_size=(15, 7)):
        """
        Detects words in an image by finding bounding boxes around contours.
        """
        # Convert to grayscale if necessary
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = imageCOLOR_BGR2GRAY

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 80, 200)
        # cv2.imshow("Canny Text", edges)


        # Apply morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size) # КАКОГО РАЗМЕРА СДЕЛАТЬ ЯДРО????????????? Среднее от высоты EAST?
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # cv2.imshow("Closed Text", closed)

        # Find contours: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0
        #   RetrievalModes: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71
        #   ContourApproximationModes: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga4303f45752694956374734a03c54d5ff
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter and process bounding boxes
        bound_rects = []
        for contour in contours:
            approx = cv2.approxPolyDP(contour, epsilon=2, closed=True)
            bounding_box = cv2.boundingRect(approx)  # (x, y, w, h)

            x, y, w, h = bounding_box

            # Apply filtering criteria
            # mbWord.width()>img.cols()/33 Ширина должна быть больше чем 33 части ширины изображения, мы убираем слишком маленькие box-ы
            # mbWord.width()<img.cols()/2 Слово не может быть шире, чем половина ширины самого изображения
            # mbWord.height()<img.rows()/2 - Слово не может быть выше, чем половина высоты самого изображения
            # mbWord.width()>mbWord.height() - У слова ширина должна быть больше, чем высота, НО ЕСЛИ ЭТО ОДНОБУКВЕННОЕ СЛОВО??!!
            # if w > image.shape[1] // 33 and w < image.shape[1] // 2 and h < image.shape[0] // 2: # and w > h:
            bound_rects.append(bounding_box)

        # Sort rectangles by x-coordinate
        sorted_rects = sorted(bound_rects, key=lambda rect: rect[0])
        return sorted_rects


    def draw_boxes(self, image, bounding_boxes):
        """
        Draws bounding boxes on the image.
        """
        for (x, y, w, h) in bounding_boxes:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return image


class MyDetector:
    def __init__(self, east_model_path):
        self.east = EAST(model_path=east_model_path)
        self.morpho = Morpological()

    def detect(self, image):
        height, width = image.shape[:2] # row x col x ch
        # print('image datatype:', image.dtype, ", dimensions:", image.shape)

        avg_height = self.east.avg_height(image)
        # print("avg_height: ", avg_height)
        text_regions = self.morpho.detect(image, kernel_size=(max(15, avg_height), 7))
        # text_regions = east.detect(frame)

        # self.morpho.draw_boxes(image, text_regions)
        # east.draw_boxes(image, text_regions)

        return text_regions

    def draw_boxes(self, image, text_regions):
        return self.morpho.draw_boxes(image, text_regions)



if __name__ == "__main__":

    INPUT_IMAGE     = "../assets/gnhk_019.jpg" # Path to the input image

    # Load the input image
    frame = cv2.imread(INPUT_IMAGE) # BGR
    print('Frame datatype:', frame.dtype, ", dimensions:", frame.shape)

    # frame = cv2.resize(frame, (width, height))

    mydetector = MyDetector(east_model_path="../assets/frozen_east_text_detection.pb")

    text_regions = mydetector.detect(frame)
    mydetector.draw_boxes(frame, text_regions)

    # Display the result
    cv2.imshow("Detected Text", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
Нужно сделать полный pipline для text detection-а.
КАКОГО РАЗМЕРА СДЕЛАТЬ ЯДРО????????????? Среднее от высоты EAST?
Сравнить с tesseract text detection,
tesseract не создан для распознавание печатного текста.

Morphological Text Detection
Tesseract Detection

Tesseract

FCNN
CTC
CRNN


Мы же в CTC Loss последовательно идем с определенным шагом. А что если поступил квадратный region слова. Как будут распознавать слова nxn? Вдруг там буква? Можем ли мы просто расстанять слово до nx2n? Или nx1.5n?


қанағаттандырылмағандықтарыңыздан

А что если взять тупо самое длинное слово в словаре и расстянуть относительно чтобы самое длинное слово могло распознавать?

32x32
"""
