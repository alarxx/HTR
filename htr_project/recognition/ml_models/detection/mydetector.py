import cv2
import matplotlib.pyplot as plt

from . import east  # Changed to relative import
from . import morphological  # Changed to relative import


class MyDetector(morphological.Morphological):

    def __init__(self, east_model_path):
        self.east = east.EAST(model_path=east_model_path)
        self.morpho = morphological.Morphological()


    def detect(self, image):
        height, width = image.shape[:2] # row x col x ch
        # print('image datatype:', image.dtype, ", dimensions:", image.shape)

        avg_height, east_steps = self.east.avg_height(image)

        kernel_width = int(avg_height/2)
        # The size of the kernel should conventionally be odd
        if kernel_width % 2 == 0:
            kernel_width -= 1
        # print("kernel_width: ", kernel_width)

        # text_regions = [(topLeft.x, topLeft.y, w, h)]
        text_regions, text_lines, morpho_steps = self.morpho.detect(image, kernel_size=(max(15, kernel_width), 7))
        # text_regions = east.detect(frame)

        image = self.morpho.draw_boxes(image, text_regions)
        # east.draw_boxes(image, text_regions)

        steps = east_steps + morpho_steps + [{"image": image, "title": "Result"}]

        return text_regions, text_lines, steps


    def imshow(self, image, title=""):
        # cv2.imshow(title, image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        plt.figure(figsize=(16, 9))  # Adjust the figure size as needed
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # rgb
        plt.axis('off')  # Turn off axis for better visualization
        plt.title(title)
        plt.show()


    def visualize_steps(self, steps):
        """
        Visualizes the processing steps with their corresponding titles.

        steps (list of dict):
            - 'image'
            - 'title'
        """
        num_steps = len(steps)
        cols = 3  # Number of columns
        rows = 1 + (num_steps - 1) // cols  # Calculate required rows

        # Create a figure with subplots for each step
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

        # Flatten axes for easier iteration, in case it's a 2D array
        axes = axes.flatten()

        for ax, step in zip(axes, steps):
            image = step['image']
            title = step['title']

            # Convert BGR to RGB for displaying images with matplotlib
            if len(image.shape) == 3 and image.shape[2] == 3:  # Check if the image is in color
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Display the image
            ax.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
            ax.set_title(title)
            ax.axis('off')

        # Hide any unused subplots
        for ax in axes[num_steps:]:
            ax.axis('off')

        plt.tight_layout()
        plt.show()


    def crop_words(self, image, regions):
        # rectangle(image, topLeft, bottomRight, color, thickness, lineType, shift)
        # cv2.rectangle(image, (0, 0), (100, 100), (0, 255, 0), 2)

        # regions = [(topLeft.x, topLeft.y, width, height)]
        # print(regions)
        word_images = []
        for region in regions:
            x, y, w, h = region
            # By Functional Programming we must not change the argument.
            # Crop the region actually references the same values, so we need to copy it
            cropped_image = image[y:y+h, x:x+w, 0:3]
            word_images.append(cropped_image.copy())

        return word_images
