import numpy as np
import cv2
import torch
import torchvision
import matplotlib.pyplot as plt

from utils import convert



if __name__ == "__main__":

    # my_list = [[1, 2, 3], [4, 5, 6]]
    # print(my_list)
    # print()
    #
    # nparray = np.array(my_list, dtype=np.float32)
    # print(type(nparray), nparray.dtype)
    # print(nparray)
    # print()
    #
    # # tensor = torch.tensor(my_list) # torch.FloatTensor(list)
    # tensor = torch.from_numpy(nparray)
    # print(type(tensor), tensor.dtype)
    # print(tensor)
    # print()
    #
    #
    # nparray = tensor.numpy()
    # print(type(nparray), nparray.dtype)
    # print(nparray)
    # print()


    # img is numpy.ndarray
    bgr_img = cv2.imread('./printed_text.jpg') # BGR
    # Converting to RGB
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    print(type(rgb_img), rgb_img.dtype)
    print(rgb_img.shape) # [H, W, C]
    print()

    # Convert to PyTorch tensor
    # [H, W, C] -> [C, H, W]
    # tensor_img = convert.cvmat2ntensor(rgb_img)

    # torchvision.io.read_image(path: str, mode: torchvision.io.image.ImageReadMode = <ImageReadMode.UNCHANGED: 0>) → torch.Tensor
    tensor_img = torchvision.io.read_image('./printed_text.jpg')
    # [H, W, C] uint8 between 0 and 255
    print(type(tensor_img), tensor_img.dtype)
    print(tensor_img.shape) # [H, W, C]
    print()

    # Convert tensor back to OpenCV format
    # [C, H, W] -> [H, W, C]
    numpy_image = convert.tensor2cvmat(tensor_img)
    print()

    # Display the image
    # Matplotlib expects RGB format
    plt.imshow(numpy_image)
    # plt.imshow(numpy_image)
    plt.axis("off")
    plt.show()


    # https://pytorch.org/docs/stable/generated/torch.permute.html
    # nparray = np.array([
    #     [[1, 2, 3],
    #      [4, 5, 6],
    #      [4, 5, 6]],
    #
    #     [[4, 5, 6],
    #      [7, 8, 9],
    #      [4, 5, 6]]
    #     ])
    #
    # print(nparray.shape) # (C, H, W)
    # print(nparray)
    #
    # # x = torch.randn(2, 3, 5)
    # x = torch.from_numpy(nparray)
    # x = torch.permute(x, (2, 0, 1))
    # print(x.size())
    # print(x)

