import cv2
import matplotlib.pyplot as plt

from detection.mydetector import MyDetector


if __name__ == "__main__":

    # INPUT_IMAGE     = "./detection/assets/gnhk_019.jpg" # Path to the input image
    INPUT_IMAGE     = "./detection/assets/handwritten_kaz.jpg" # Path to the input image

    # Load the input image
    frame = cv2.imread(INPUT_IMAGE) # BGR
    print("Frame")
    print("frame.shape:", frame.shape)
    print('frame datatype:', frame.dtype, ", dimensions:", frame.shape)

    # frame = cv2.resize(frame, (width, height))

    mydetector = MyDetector(east_model_path="./detection/assets/frozen_east_text_detection.pb")

    # text_regions = [(topLeft.x, topLeft.y, width, height)]
    # text_lines = { "bounding_rects": [], "coefficients": [fit line coefficients] }
    text_regions, text_lines, steps = mydetector.detect(frame)

    mydetector.visualize_steps(steps)

    painted = mydetector.draw_boxes(frame, text_lines["bounding_rects"], box_color=(255, 0, 0), index_color=(0, 255, 0))
    # painted = mydetector.draw_lines(painted, text_lines["coefficients"], color=(0, 255, 0))
    painted = mydetector.draw_boxes(painted, text_regions, index_color=(0, 0, 255))
    mydetector.imshow(painted, "Detected Text")

    word_images = mydetector.crop_words(frame, text_regions)
    # Display the first cropped region
    if len(word_images) > 0:
        plt.imshow(cv2.cvtColor(word_images[0], cv2.COLOR_BGR2RGB))
        plt.title("Cropped Text Region")
        plt.show()


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
