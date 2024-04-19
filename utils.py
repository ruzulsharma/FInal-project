# -*- coding: utf-8 -*-
from PIL import Image
import os
import numpy as np
from skimage.io import imread
from tqdm import tqdm

OGIMG_SIZE = 1500
IMG_SIZE = 224
OVERLAP = 14

def imgstitch(img_path):
    """
    This function overlays the predicted image patches with each other to produce the final output image.
    It should be noted that the overlap regions are INTENDED to be averaged to minimize error. This is not
    done in this function. They are just overwritten by the next image patch. This will be implemented in the future 
    for a more robust approach
    
    Parameters
    ----------
    img_path : STRING
        Path to directory with test image patches

    Returns
    -------
    None. The final stitched image will be automatically saved in the same directory as the image patches with the 
    name 'output.png''

    """
    _, _, img_files = next(os.walk(img_path))
    
    img_files = sorted(img_files, key=lambda x: int(os.path.splitext(x)[0]))
    IMG_WIDTH, IMG_HEIGHT = (Image.open(os.path.join(img_path, '11.png'))).size
    
    full_img = Image.new('RGB', (IMG_WIDTH * 7, IMG_HEIGHT * 7))
    x, y = (0, 0)
    
    for n, id_ in enumerate(img_files):
        img = Image.open(os.path.join(img_path, id_))
        if x < IMG_WIDTH * 6:
            full_img.paste(img, (x, y))
            x += IMG_WIDTH - OVERLAP
        if x >= IMG_WIDTH * 6:
            x = 0
            y += IMG_HEIGHT - OVERLAP
            full_img.paste(img, (x, y))
    
    full_img.save(os.path.join(img_path, 'output.png'), 'PNG')
    
def DatasetLoad(train_dataset, test_dataset, val_dataset):
    """
    Load and preprocess the dataset for training, testing, and validation.

    Parameters
    ----------
    train_dataset : str
        Path to the directory containing training images and masks.
    test_dataset : str
        Path to the directory containing test images and masks.
    val_dataset : str
        Path to the directory containing validation images and masks.

    Returns
    -------
    X_train : numpy.ndarray
        Training dataset features. Shape: [num_samples, IMG_SIZE, IMG_SIZE, 3]
    Y_train : numpy.ndarray
        Training dataset labels. Shape: [num_samples, IMG_SIZE, IMG_SIZE, 1]
    X_test : dict
        Test dataset features for different classes. Keys: class names, Values: numpy.ndarray
        Shape: [num_samples, IMG_SIZE, IMG_SIZE, 3]
    Y_test : dict
        Test dataset labels for different classes. Keys: class names, Values: numpy.ndarray
        Shape: [num_samples, IMG_SIZE, IMG_SIZE, 1]
    X_val : numpy.ndarray
        Validation dataset features. Shape: [num_samples, IMG_SIZE, IMG_SIZE, 3]
    Y_val : numpy.ndarray
        Validation dataset labels. Shape: [num_samples, IMG_SIZE, IMG_SIZE, 1]
    """

    def load_images_and_masks(dataset_path, is_test=False):
        image_files = sorted(os.listdir(os.path.join(dataset_path, 'image')))
        num_samples = len(image_files)

        X = np.zeros((num_samples, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        Y = np.zeros((num_samples, IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)  # Assuming categorical labels

        for i, img_filename in tqdm(enumerate(image_files), total=num_samples, desc=f"Loading {os.path.basename(dataset_path)} images"):
            img_path = os.path.join(dataset_path, 'image', img_filename)
            mask_path = os.path.join(dataset_path, 'mask', img_filename)

            img = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE))
            mask = Image.open(mask_path).resize((IMG_SIZE, IMG_SIZE))

            X[i] = np.array(img)
            Y[i] = np.array(mask)

        return X, Y

    X_train, Y_train = load_images_and_masks(train_dataset)
    X_val, Y_val = load_images_and_masks(val_dataset)

    X_test, Y_test = {}, {}
    test_folders = sorted(os.listdir(test_dataset))
    for folder in test_folders:
        test_folder_path = os.path.join(test_dataset, folder)
        X_test[folder], Y_test[folder] = load_images_and_masks(test_folder_path, is_test=True)

    return X_train, Y_train, X_test, Y_test, X_val, Y_val