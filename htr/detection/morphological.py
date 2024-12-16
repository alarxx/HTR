import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


class Morphological:
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
            gray = image

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 80, 200)
        # cv2.imshow("Canny Text", edges)

        # Apply morphological closing to find lines of text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6*kernel_size[0], kernel_size[1])) # actually it is 3 avg word height
        closed_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        text_lines = self.__get_text_lines(image, closed_lines)

        if len(text_lines["bounding_rects"])>0:
            # Sorting by the y-coordinate of bounding_rects
            sorted_data = sorted(zip(text_lines["bounding_rects"], text_lines["coefficients"]), key=lambda x: x[0][1])  # x[0][1] is the y-coordinate
            text_lines["bounding_rects"], text_lines["coefficients"] = map(list, zip(*sorted_data))

        # self.draw_boxes(image, text_lines["bounding_rects"], box_color=(255, 0, 0))
        # self.draw_lines(image, text_lines["line_coefficients"])


        # Apply morphological closing to find words
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size) # КАКОГО РАЗМЕРА СДЕЛАТЬ ЯДРО????????????? Среднее от высоты EAST?
        closed_words = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow("Closed Text", closed_words)
        bound_word_rects = self.__get_word_bound_rects(image, closed_words)

        # Sort rectangles by y and then x coordinates
        # sorted_word_rects = sorted(bound_word_rects, key=lambda rect: (rect[1], rect[0]))
        # Слова на одной линии могут находиться ниже, либо выше
        # Поэтому метод с сортировкой сначала по y, потом по x не работает
        # Вместо этого мы сортируем внутри линии
        sorted_word_rects = self.__sort_word_rects(bound_word_rects, text_lines)

        steps = [
            {"image": gray, "title": "Grayscale Image"},
            {"image": edges, "title": "Canny Edges"},
            {"image": closed_lines, "title": "Morphological Line Detection"},
            {"image": closed_words, "title": "Morphological Word Detection"}
        ]

        return sorted_word_rects, text_lines, steps


    def draw_boxes(self, image, bounding_boxes, box_color=(0, 255, 0), index_color=((255, 0, 0))):
        """
        Draws bounding boxes on the image.

        By Functional Programming we must not change the argument.
        """
        image = image.copy()
        for i, (x, y, w, h) in enumerate(bounding_boxes):
            # rectangle(image, topLeft, bottomRight, color, thickness, lineType, shift)
            cv2.rectangle(img=image, pt1=(x, y), pt2=(x + w, y + h), color=box_color, thickness=2) # BGR
            # Put the index number
            cv2.putText(img=image, text=str(i), org=(x, y+h), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=index_color, thickness=2)
        return image


    def draw_lines(self, image, coefficients, color=(255, 0, 0)):
        image = image.copy()
        for i, coeffs in enumerate(coefficients):
            [vx, vy, x0, y0] = coeffs
            # Вычисляем начальную и конечную точки линии для рисования
            left_y = int((-x0 * vy / vx) + y0)  # y при x = 0
            right_y = int(((image.shape[1] - x0) * vy / vx) + y0)  # y при x = ширина изображения
            # Рисуем линию на изображении
            cv2.line(image, (0, left_y), (image.shape[1], right_y), color, 2)
        return image


    def __get_text_lines(self, image, closed_lines):
        contours, _ = cv2.findContours(closed_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow("Closed Text Lines", closed_lines)
        text_lines = {
            "bounding_rects": [],
            "coefficients": [] # fit line coefficients
        }
        print("contour length:", len(contours))
        for contour in contours:
            approx = cv2.approxPolyDP(contour, epsilon=2, closed=True)

            # Smallest rectangle that covers the entire contour.
            bounding_box = cv2.boundingRect(approx)
            # (topLeft.x, topLeft.y, width, height)
            x, y, w, h = bounding_box
            if w < image.shape[1] // 33:
                continue

            points = approx.reshape(-1, 2)  # Преобразуем в массив Nx2 (x, y)
            # print("approx.shape:", approx.shape)
            # reshaped = arr.reshape(-1, 2)  # 2 столбца, количество строк определяется автоматически

            # print("contour.shape:", contour.shape)
            # print("points.shape:", points.shape)
            # Находим параметры линии методом наименьших квадратов
            [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 1, 1)
            coefficients = [vx, vy, x0, y0]

            text_lines["bounding_rects"].append(bounding_box)
            text_lines["coefficients"].append(coefficients)

        return text_lines


    def __get_word_bound_rects(self, image, closed_words):
        # Find contours: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0
        #   RetrievalModes: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71
        #   ContourApproximationModes: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga4303f45752694956374734a03c54d5ff
        contours, _ = cv2.findContours(closed_words, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter and process bounding boxes
        bound_rects = []
        for contour in contours:
            # simplifies the contour by removing unnecessary points, preserving its shape
            approx = cv2.approxPolyDP(contour, epsilon=2, closed=True)


            # cv2.polylines(image, [approx], isClosed=True, color=(255, 0, 0), thickness=2)

            # Smallest rectangle that covers the entire contour.
            bounding_box = cv2.boundingRect(approx)
            # (topLeft.x, topLeft.y, width, height)
            x, y, w, h = bounding_box

            # Apply filtering criteria
            # mbWord.width()>img.cols()/33 Ширина должна быть больше чем 33 части ширины изображения, мы убираем слишком маленькие box-ы
            # mbWord.width()<img.cols()/2 Слово не может быть шире, чем половина ширины самого изображения
            # mbWord.height()<img.rows()/2 - Слово не может быть выше, чем половина высоты самого изображения
            # mbWord.width()>mbWord.height() - У слова ширина должна быть больше, чем высота, НО ЕСЛИ ЭТО ОДНОБУКВЕННОЕ СЛОВО??!!
            if w > image.shape[1] // 33 and w < image.shape[1] // 2 and h < image.shape[0] // 2: # and w > h:
                bound_rects.append(bounding_box)

        return bound_rects


    def __rectangles_overlap_area(self, rect1, rect2):
        # Old idea from book https://www.amazon.co.uk/Beginning-Android-Games-Mario-Zechner/dp/1430230428

        # (x, y, width, height)
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        # Rectangle boundaries
        left1, right1, top1, bottom1 = x1, x1 + w1, y1, y1 + h1
        left2, right2, top2, bottom2 = x2, x2 + w2, y2, y2 + h2

        # Check
        if left1 < right2 and right1 > left2 and top1 < bottom2 and bottom1 > top2:
            inter_width = min(x1 + w1, x2 + w2) - max(x1, x2) # в одномерном рассматривай
            inter_height = min(y1 + h1, y2 + h2) - max(y1, y2)
            return inter_width * inter_height

        return 0


    def __sort_word_rects(self, word_rects, text_lines):

        sorted_word_rects = []
        for i in range(0, len(text_lines["bounding_rects"])):
            sorted_word_rects.append([])

        # print("sorted_word_rects:", sorted_word_rects)

        for word_rect in word_rects:

            max_area, max_i = -1, -1
            for i, line_rect in enumerate(text_lines["bounding_rects"]):
                area = self.__rectangles_overlap_area(word_rect, line_rect)
                if area > max_area:
                    max_area = area
                    max_i = i

            sorted_word_rects[max_i].append(word_rect)

        # print("sorted_word_rects:", sorted_word_rects)

        flatten = []
        for rects_in_line in sorted_word_rects:
            flatten += sorted(rects_in_line, key = lambda rect: rect[0])
        # print("flatten:", flatten)

        return flatten
