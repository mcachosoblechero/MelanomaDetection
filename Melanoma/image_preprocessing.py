import pandas as pd
import numpy as np
import random
import cv2
import os
import albumentations as A
import matplotlib.pyplot as plt


def ResizeImage(img, targetRes, verbose=False, plot=False):
    #####################################
    # Resize Image to Target Resolution #
    # Input:
    # img - Input image
    # targetRes - Output resolution
    # Output: Resized Image
    #####################################
    dim = (targetRes[0], targetRes[1])
    img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    if verbose:
        print("Initial Image Size", img.shape)
        print("Resized Image", img_resized.shape)

    if plot:
        # Analysis of results of resized
        plt.subplots(1, 2, figsize=(8, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(img_resized)
        plt.title("Resized Image")
        plt.axis('off')
        plt.show()

    return img_resized


def MicroscopeAug(image):
    #####################################
    # Augment Image with a microscope effect
    # Input:
    # img - Input image
    # Output: Augmented Image in a dictionary
    #####################################
    center = (image.shape[1]//2, image.shape[0]//2)
    radious = random.randint(image.shape[0]//2 - 3, image.shape[0]//2 + 15)
    circle = cv2.circle((np.ones(image.shape) * 255).astype(np.uint8),
                        center,
                        radious,
                        (0, 0, 0),
                        -1)

    mask = circle - 255
    aug_img = np.multiply(image, mask)
    return {'image': aug_img}


def HairAug(image):
    #####################################
    # Augment Image with hair effect
    # Input:
    # image - Input image
    # Output: Augmented Image in a dictionary
    #####################################
    # Default parameters
    n_hairs = 5
    folder_hair = "Datasets/resources/Hair/"
    # Target image width and height
    height, width, _ = image.shape
    # Obtain list of hair images
    hair_images = [im for im in os.listdir(folder_hair) if 'png' in im]
    # Copy the original image for comparison
    aug_img = image.copy()

    for _ in range(n_hairs):
        # Select a random hair image
        hair = cv2.imread(os.path.join(
            folder_hair, random.choice(hair_images)))
        # Select random V/H rotation
        hair = cv2.flip(hair, random.choice([-1, 0, 1]))
        # Select random rotation
        hair = cv2.rotate(hair, random.choice([0, 1, 2]))

        h_height, h_width, _ = hair.shape
        # Select random position for the image - Height
        roi_ho = random.randint(0, aug_img.shape[0] - hair.shape[0])
        # Select random position for the image - Width
        roi_wo = random.randint(0, aug_img.shape[1] - hair.shape[1])
        roi = aug_img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width]

        img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        hair_fg = cv2.bitwise_and(hair, hair, mask=mask)

        dst = cv2.add(img_bg, hair_fg)
        aug_img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width] = dst

    return {'image': aug_img}


def ImgAug(img, aug, plot=False):
    #####################################
    # Augment Image with various filters#
    # Input:
    # img - Input image
    # aug - Target Filter
    # Output: Augmented Image
    #####################################
    Aug_Dic = {
        "Gauss": A.GaussNoise(p=1),
        # Missing - Rotation
        "HFlip": A.HorizontalFlip(p=1),
        "VFlip": A.VerticalFlip(p=1),
        "Micro": MicroscopeAug,
        "Hair":  HairAug
    }
    Filter = Aug_Dic[aug]
    img_augmented = Filter(image=img)['image']

    if plot:
        # Analysis of results of resized
        plt.subplots(1, 2, figsize=(8, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(img_augmented)
        plt.title("Augmented Image")
        plt.axis('off')
        plt.show()

    return img_augmented
