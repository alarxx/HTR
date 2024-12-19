import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.transforms import transforms

from .data_transforms.trans import MinMaxWidth
from .detection.mydetector import MyDetector
from .classificator.cnns import FCNN


class KazakhTextRecognizer:
    def __init__(self,
                 detector_model_path,
                 classifier_model_path,
                 font_path="./arial.ttf"):
        self.alphabet = [
            '<blank>', 'А', 'Ә', 'Б', 'В', 'Г', 'Ғ', 'Д', 'Е', 'Ё', 'Ж', 'З',
            'И', 'Й', 'К', 'Қ', 'Л', 'М', 'Н', 'Ң', 'О', 'Ө', 'П', 'Р', 'С',
            'Т', 'У', 'Ұ', 'Ү', 'Ф', 'Х', 'Һ', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы',
            'І', 'Ь', 'Э', 'Ю', 'Я'
        ]
        self.char_to_idx = {ch: i for i, ch in enumerate(self.alphabet)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}

        self.font_path = font_path

        # Load detector
        self.detector = MyDetector(east_model_path=detector_model_path)

        # Load classifier
        self.model = FCNN(num_classes=len(self.alphabet))

        # Load the model on CPU
        checkpoint = torch.load(classifier_model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Transformation pipeline
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            MinMaxWidth(),
        ])

    def detect_text_regions(self, image):
        text_regions, text_lines, steps = self.detector.detect(image)
        return text_regions, text_lines, steps

    def visualize_detection_steps(self, steps):
        self.detector.visualize_steps(steps)

    def draw_boxes(self, image, text_lines, text_regions):
        painted = self.detector.draw_boxes(
            image, text_lines["bounding_rects"], box_color=(255, 0, 0), index_color=(0, 255, 0)
        )
        painted = self.detector.draw_boxes(painted, text_regions, index_color=(0, 0, 255))
        return painted

    def crop_words(self, image, text_regions):
        return self.detector.crop_words(image, text_regions)

    def recognize_text(self, word_images):
        recognized_texts = []
        with torch.no_grad():
            for img_bgr in word_images:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                input_img = self.transform(pil_img).unsqueeze(0)
                logits = self.model(input_img)
                preds = self._greedy_decode(logits)
                recognized_texts.extend(preds)
        return recognized_texts

    def _greedy_decode(self, logits):
        argmaxes = torch.argmax(logits, dim=2)
        results = []
        for b in range(argmaxes.size(1)):
            seq = argmaxes[:, b].cpu().numpy()
            decoded = []
            prev = None
            for s in seq:
                if s != 0 and s != prev:
                    decoded.append(s)
                prev = s
            text = ''.join(self.idx_to_char[idx] for idx in decoded)
            results.append(text)
        return results

    def draw_kazakh_text(self, image, text, position, font_scale=0.5, color=(0, 0, 255)):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        base_font_size = 32
        font_size = int(base_font_size * font_scale)
        try:
            font = ImageFont.truetype(self.font_path, font_size)
        except IOError:
            raise ValueError(f"Font file not found at {self.font_path}")
        draw.text((position[0], position[1] - font_size), text, font=font, fill=color[::-1])
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def display_image(self, image, title="Image"):
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(title)
        plt.show()

    def draw_recognized_text(self, image, text_regions, recognized_texts):
        for (x, y, w, h), text in zip(text_regions, recognized_texts):
            image = self.draw_kazakh_text(image, text, (x, y))

        return image

    def end2end(self, input_image_path):
        image = cv2.imread(input_image_path)

        text_regions, text_lines, steps = recognizer.detect_text_regions(image)

        recognizer.visualize_detection_steps(steps)

        painted_image = recognizer.draw_boxes(image, text_lines, text_regions)
        recognizer.display_image(painted_image, "Detected Text")

        word_images = recognizer.crop_words(image, text_regions)
        recognized_texts = recognizer.recognize_text(word_images)


        painted_with_text = recognizer.draw_recognized_text(painted_image, text_regions, recognized_texts)
        recognizer.display_image(painted_with_text, "Recognized Text Overlaid")

        return ' '.join(recognized_texts)


if __name__ == "__main__":

    recognizer = KazakhTextRecognizer(
        detector_model_path="./frozen_east_text_detection.pb",
        classifier_model_path="FCNN_CTC_main.pth",
    )

    recognized_texts = recognizer.end2end(input_image_path="./handwritten_kaz.jpg")

    print("Recognized Texts:", recognized_texts)


