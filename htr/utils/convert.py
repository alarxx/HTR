import numpy as np
import cv2
import torch
import torchvision
import matplotlib.pyplot as plt


# Do we assume by default that tensor values in [0, 1] range?
# No, read_image gives range of [0, 255] in uint8

def cvmat2ntensor(hwc_mat):
    # Convert OpenCV Matrix format (NumPy Arrat) to PyTorch tensor
    # [H, W, C] -> [C, H, W]
    tensor = torch.from_numpy(hwc_mat).permute(2, 0, 1).float() / 255.0 # [0, 1]
    print(type(tensor), tensor.dtype)
    print(tensor.shape) # [C, H, W]
    return tensor


def cvmat2tensor(hwc_mat):
    # Convert OpenCV Matrix format (NumPy Arrat) to PyTorch tensor
    # [H, W, C] -> [C, H, W]
    tensor = torch.from_numpy(hwc_mat).permute(2, 0, 1).byte() # [0, 255]
    print(type(tensor), tensor.dtype)
    print(tensor.shape) # [C, H, W]
    return tensor


def ntensor2cvmat(chw_tensor):
    # Convert tensor back to OpenCV format
    # [C, H, W] -> [H, W, C]
    numpy_img = chw_tensor.permute(1, 2, 0).mul(255.0).byte().numpy()  # [0, 255]
    print(type(numpy_img), numpy_img.dtype)
    print(numpy_img.shape) # [H, W, C]
    return numpy_img


def tensor2cvmat(chw_tensor):
    # Convert tensor back to OpenCV format
    # [C, H, W] -> [H, W, C]
    numpy_img = chw_tensor.permute(1, 2, 0).byte().numpy()
    print(type(numpy_img), numpy_img.dtype)
    print(numpy_img.shape) # [H, W, C]
    return numpy_img





# DON'T USE THOSE FUNCTIONS BELOW, THEY'RE ONLY EXAMPLES

def __how_rgb_cvmat(numpy_image):
    # Display the image
    # Matplotlib expects RGB format
    plt.imshow(numpy_image)
    # plt.imshow(numpy_image)
    plt.axis("off")
    plt.show()


def __show_rgb_ntensor(ntensor):
    numpy_image = ntensor2cvmat(ntensor)
    return show_rgb_cvmat(numpy_image)


def __cv_read(path):
    # img is numpy.ndarray
    bgr_img = cv2.imread(path) # BGR
    # Converting to RGB
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    # print(type(rgb_img), rgb_img.dtype)
    # print(rgb_img.shape) # [H, W, C]
    # print()
    ntensor = cvmat2ntensor(rgb_img)

    return ntensor


def __torch_read(path):
    # torchvision.io.read_image(path: str, mode: torchvision.io.image.ImageReadMode = <ImageReadMode.UNCHANGED: 0>) → torch.Tensor
    tensor_img = torchvision.io.read_image(path) # RGB [C, H, W] uint8 between 0 and 255
    print(type(tensor_img), tensor_img.dtype)
    print(tensor_img.shape) # [C, H, W]
    print()

    ntensor = tensor_img.float() / 255.0 # [0, 1]

    return ntensor
