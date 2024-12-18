
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def draw_kazakh_text(image, text, position, font_path, font_scale=1, color=(0, 255, 0), thickness=2):
    """
    Draws text on an image using Pillow for proper Unicode support, mimicking cv2.putText.

    Args:
        image (numpy.ndarray): Input image in BGR format.
        text (str): Text to draw.
        position (tuple): Position to draw the text (x, y).
        font_path (str): Path to the .ttf font file.
        font_scale (float): Scale of the font (relative to default size).
        color (tuple): Color of the text in BGR format.
        thickness (int): Thickness of the text outline (not supported in Pillow, included for compatibility).

    Returns:
        numpy.ndarray: Image with text drawn.
    """
    # Convert the image to RGB for Pillow
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    # Create a drawing object
    draw = ImageDraw.Draw(pil_image)

    # Load a font and adjust size using font_scale
    base_font_size = 32  # Default font size
    font_size = int(base_font_size * font_scale)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        raise ValueError(f"Font file not found at {font_path}")

    # Draw text
    draw.text((position[0], position[1]-font_size), text, font=font, fill=color[::-1])  # Convert BGR to RGB for Pillow

    # Convert the image back to BGR for OpenCV
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def detect_text_with_tesseract(image, lang="eng", font_path="./assets/fonts/arial.ttf", isHandwritten=False): # lang="kaz"
    """
    Detects text regions in an image using Tesseract OCR and visualizes the results.

    Args:
        image (numpy.ndarray): Input image in BGR format.
        visualize (bool): If True, displays the image with detected text regions.

    Returns:
        list: Detected text regions as a list of dictionaries with text and bounding box coordinates.
    """
    # Convert image to RGB (Tesseract works better with RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use Tesseract to detect text
    data = pytesseract.image_to_data(image_rgb, lang=lang, output_type=Output.DICT)

    # Extract text and bounding box information
    results = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 0 and data["level"][i] > 2:  # Filter by confidence level
            x, y, w, h = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            text = data['text'][i]
            results.append({'text': text, 'bbox': (x, y, w, h)})

    # Visualize results
    for result in results:
        x, y, w, h = result['bbox']
        # Top left, bottom right points
        cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        print(result['text'])
        if lang == "kaz":
            image = draw_kazakh_text(image, result['text'], (x, y), font_path, 1, (0, 0, 255), 2)
        else:
            cv2.putText(image, result['text'], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the image with matplotlib
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Tesseract {'Handwritten' if isHandwritten else ''} Text Detection")
    plt.axis("off")
    plt.show()

    return results


# Example usage:
if __name__ == "__main__":
    INPUT_IMAGE = "./assets/handwritten_kaz.jpg"  # Path to the input image
    # LANG = "eng"
    LANG = "kaz"
    IS_HANDWRITTEN=True

    # Load the input image
    frame = cv2.imread(INPUT_IMAGE)
    print('Frame datatype:', frame.dtype, ", dimensions:", frame.shape)

    # Perform Tesseract text detection and visualize
    detected_text = detect_text_with_tesseract(frame, lang=LANG, isHandwritten=IS_HANDWRITTEN)

    # Print detected text
    for item in detected_text:
        print(f"Detected text: '{item['text']}' at {item['bbox']}")

# Написать от руки текст
