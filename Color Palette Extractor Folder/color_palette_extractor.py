"""
COLOR_PALETTE_EXTRACTOR

This script extracts a color palette from an input image using the KMeans clustering algorithm. 
It returns a list of hex values for the dominant colors in the image.

To use it, call the get_palette function with the path to the input image. 
The function also has an optional 'show' parameter that can be set to False to suppress 
the display of the image and its color palette.

The script requires the following libraries to be installed: 
PIL (Python Imaging Library), 
scikit-learn, 
numpy, 
pandas, 
requests,
matplotlib
"""


#imports
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from io import BytesIO



#rgb to hex function
def rgb_to_hex(colors_list):
    hex_values = [] #empty list to store hex values
    for rgbs in colors_list:
        hex_values.append("#{0:02x}{1:02x}{2:02x}".format(*rgbs))
    return hex_values


def preprocess(img):

    # check image size and resize if too big (>1000x1000)
    if img.size[0] > 1000 and img.size[1] > 1000:
        newsize_img = img.resize((900, 600))
    else:
        newsize_img = img.copy()

    #make sure the image is in rgb mode (not grayscale, cmyk, etc.)
    newsize_img = newsize_img.convert('RGB')

    #get rgb values of pixels as numpy array
    pixels = np.array(newsize_img)

    #reshape into 2D array with rgb tuples
    pixels = pixels.reshape((-1, 3))

    return pixels


#Fit Kmeans and return list of hex values
def get_palette(image_path, show = True):
    if image_path.startswith('http'):
        response = requests.get(image_path)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(image_path)

    pixels = preprocess(img) 
    clf = KMeans(n_clusters = 5)
    clf.fit(pixels)
    colors_list = clf.cluster_centers_.astype(int)
    hex_values = rgb_to_hex(colors_list)

    if show:
        display_image_with_palette(img, hex_values)

    return hex_values


def display_image_with_palette(image, colors_list):
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax[0].imshow(image)
    ax[0].axis('off')
    ax[1].bar(range(len(colors_list)), [1]*len(colors_list), color=colors_list)
    ax[1].set_xticks(range(len(colors_list)))
    ax[1].set_xticklabels(['']*len(colors_list))
    ax[1].set_yticks([])
    ax[1].set_title('Color Palette')
    plt.show()
