import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.transforms import transforms
from ctc.data_transforms.trans import MinMaxWidth
from detection.mydetector import MyDetector
from classificator.cnns import FCNN

def draw_kazakh_text(image, text, position, font_path="./arial.ttf", font_scale=0.5, color=(0, 0, 255), thickness=2):
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


# Алфавит (должен быть тем же, что и в train.py)
alphabet = [
    '<blank>', 'А', 'Ә', 'Б', 'В', 'Г', 'Ғ', 'Д', 'Е', 'Ё', 'Ж', 'З',
    'И', 'Й', 'К', 'Қ', 'Л', 'М', 'Н', 'Ң', 'О', 'Ө', 'П', 'Р', 'С',
    'Т', 'У', 'Ұ', 'Ү', 'Ф', 'Х', 'Һ', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы',
    'І', 'Ь', 'Э', 'Ю', 'Я'
]
char_to_idx = {ch: i for i, ch in enumerate(alphabet)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}

def greedy_decode(logits):
    # logits: (T, B, C)
    argmaxes = torch.argmax(logits, dim=2)  # (T, B)
    results = []
    for b in range(argmaxes.size(1)):
        seq = argmaxes[:, b].cpu().numpy()
        decoded = []
        prev = None
        for s in seq:
            if s != 0 and s != prev:
                decoded.append(s)
            prev = s
        text = ''.join(idx_to_char[idx] for idx in decoded)
        results.append(text)
    return results


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



    # Создаём и загружаем модель (предположим, что у вас есть сохранённый чекпойнт)
    model = FCNN(num_classes=len(alphabet))
    checkpoint = torch.load("FCNN_CTC_e99.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Трансформации для входных данных модели.
    # Важно: Они должны соответствовать тем, что использовались при обучении.
    # Здесь мы используем упрощённый вариант, вы можете добавить Resize, Normalization и т.п.
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Можно добавить трансформации типа Resize высоты в 32px, т.к. в train.py часто высота фиксируется.
        MinMaxWidth(),
    ])

    recognized_texts = []
    with torch.no_grad():
        for img_bgr in word_images:
            # Преобразуем BGR в RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            # Применяем трансформацию
            input_img = transform(pil_img).unsqueeze(0)  # (1, C, H, W)

            # Прогон через модель
            # Модель выдает: (T, B, C), где T - размер по ширине после свертки
            logits = model(input_img)  # (T, B, C)

            # Расшифровываем выходы модели
            preds = greedy_decode(logits)
            recognized_texts.append(preds)  # preds - список строк длиной B (тут B=1)

    # Выводим распознанный текст для каждого слова
    # for i, text in enumerate(recognized_texts):
    #     print(f"Word {i + 1}: {text}")

    print(recognized_texts)

    # Можно при желании отобразить поверх изображения
    # как пример: предположим, что text_regions соответствует координатам [x,y,w,h]
    # painted_recognized = painted.copy()
    # for (x, y, w, h), txt in zip(text_regions, recognized_texts):
    #     painted_recognized = draw_kazakh_text(painted_recognized, txt, (x, y))
    # mydetector.imshow(painted_recognized, "Recognized Words")
