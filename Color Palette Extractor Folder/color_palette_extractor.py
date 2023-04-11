
"""
COLOR PALETTE EXTRACTOR V3

This script extracts a color palette from an input image using a KMeans clustering algorithm. 
It uses KMeans clustering algorithm twice to improve the accuracy of the color palette extraction. 
First a KMeans algorithm extracts the main color palette of the image. The pixels are then each re-examined and compared to the dominant colors. 
After then extracting the pixels with color labels most dissimilar to palette colors, the remaining pixels are clustered with a KMeans algo, 
and a representative color is chosen from among the cluster centers. 

This process makes the color palette extraction more robust and accurate for a wider range of images, 
especially with small sections of highly contrastive colors not captured by the original KMeans clustering.


To use the script, call the `get_palette` function with the path to the input image. The function returns a list of hex values for the dominant colors in the image. 
The function also has an optional `show` parameter that can be set to `False` to suppress the display of the image and its color palette.

The updated script also allows for customization of several hard-coded values, including the number of clusters used in the KMeans algorithm 
and the threshold for determining whether a pixel is significantly different from the color palette. 
These values can be adjusted to optimize the performance of the color palette extractor for different types of images.


The resulting list of hex values includes the main color palette as well as any additional unique colors found in the image. 
The script also includes a display function that shows the image and its color palette, making it easier to visualize the results.


The script requires the following libraries to be installed: 
PIL (Python Imaging Library), 
scikit-learn, 
numpy, 
pandas, 
requests,
matplotlib.

Note: This code has been updated from an earlier version.
"""


#imports
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from io import BytesIO
import math





#Conversion Functions

#RGB to HEX Convrsion:
def rgb_to_hex(rgb_pixels):
    hex_values = []
    for rgbs in rgb_pixels:
        rgbs = rgbs.astype(int) # convert to integers
        hex_values.append("#{0:02x}{1:02x}{2:02x}".format(*rgbs))
    return hex_values


def rgb_to_hsl(rgb):
    r, g, b = rgb[0], rgb[1], rgb[2]
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    chroma = max_val - min_val
    if chroma == 0:
        hue = 0
    elif max_val == r:
        hue = ((g - b) / chroma) % 6
    elif max_val == g:
        hue = (b - r) / chroma + 2
    else:
        hue = (r - g) / chroma + 4
    hue = int(hue * 60)
    lightness = (max_val + min_val) / 2
    if chroma == 0:
        saturation = 0
    else:
        saturation = chroma / (1 - abs(2 * lightness - 1))
    saturation = int(saturation * 100)
    lightness = int(lightness * 100)
    return hue, saturation, lightness



def hsl_to_hex(hsl):
    """Convert an HSL color value to a hexadecimal color code."""
    if hsl is None:
        return None
    hue, saturation, lightness = hsl
    saturation /= 100
    lightness /= 100
    chroma = (1 - abs(2 * lightness - 1)) * saturation
    hue_ = hue / 60
    x = chroma * (1 - abs(hue_ % 2 - 1))
    if hue_ < 1:
        r_, g_, b_ = chroma, x, 0
    elif hue_ < 2:
        r_, g_, b_ = x, chroma, 0
    elif hue_ < 3:
        r_, g_, b_ = 0, chroma, x
    elif hue_ < 4:
        r_, g_, b_ = 0, x, chroma
    elif hue_ < 5:
        r_, g_, b_ = x, 0, chroma
    else:
        r_, g_, b_ = chroma, 0, x
    m = lightness - chroma / 2
    r, g, b = (r_ + m) * 255, (g_ + m) * 255, (b_ + m) * 255
    hex_code = "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))
    return hex_code



#extracts HSL from RGB, and returns both as well as a combined pixel array
def combine_pixels(rgb_pixels):

    # Get HSL values for pixels from RGB values
    hsl_pixels = np.array([rgb_to_hsl(pixel) for pixel in rgb_pixels])
    
    # Concatenate RGB and HSL arrays so we can use both to get more accurate results
    combined_pixels = np.hstack((rgb_pixels, hsl_pixels))
    
    return combined_pixels , rgb_pixels , hsl_pixels


#extract rgb values from combined pixel tuples
def combined_pixels_to_rgb(combined_pixels):
    new_rgb_pixels = []
    for values in combined_pixels:
        new_rgb_pixels.append(values[0:3])
    return new_rgb_pixels



def color_distance_hsl(color1, color2):
    """
    Calculates the distance between two colors in the HSL color space.
    """
    h1, s1, l1 = color1
    h2, s2, l2 = color2
    dh = min(abs(h1 - h2), 360 - abs(h1 - h2)) / 360
    ds = abs(s1 - s2) / 100
    dl = abs(l1 - l2) / 100
    return math.sqrt(dh ** 2 + ds ** 2 + dl ** 2)



# Preprocess Image


#resize function
def resize_image(img, quick):
    # check image size and resize if too big (>1000x1000)
    if img.size[0] > 199 or img.size[1] > 299:
        if quick:
            newsize_img = img.resize((100, 150))
        else:
            newsize_img = img.resize((200, 300))
    else:
        newsize_img = img.copy()

    return newsize_img


# Resize image, convert to rgb mode, and return as a 2d array of rgb pixels
def preprocess(img, quick):
    # resize image
    newsize_img = resize_image(img, quick)

    # make sure the image is in RGB mode (not grayscale, CMYK, etc.)
    newsize_img = newsize_img.convert('RGB')

    # get RGB values of pixels as numpy array
    rgb_pixels = np.array(newsize_img)

    # reshape into 2D array with RGB tuples
    rgb_pixels = rgb_pixels.reshape((-1, 3))

    return rgb_pixels







def compare_color_labels(unique_color_labels, unique_color_pixels, cluster_color_labels): #SIMILARITY FUNCTION
    # This function looks at the pixel color labels and the main color palette color labels and determines whether they are similar enough to ignore. 

    black_similar = ["Blacks", "Dark-Greys", "Dark-Browns"]
    blue_similar = ["Blue-Cyans", "Violet-Blues, Purples"]
    red_orange_similar = ["Red-Oranges", "Reds", "Oranges"]
    brown_similar = ["Dark-Browns", "Light-Browns / Tans"]
    grey_similar = ["Dark-Greys", "Light-Greys / Off-Whites"]
    green_similar = ["Greens", "Cyans", "Yellow-Greens"]
    white_similar = ["Whites", "Light-Greys / Off-Whites"]
    yellow_similar = ["Yellows", "Oranges", "Yellow-Greens"]
    pink_similar = ["Pinks", "Reds", "Magenta-Pinks"]

    # Make copies of the input lists to avoid modifying them during iteration
    unique_color_labels_copy = unique_color_labels.copy()
    unique_color_pixels_list = list(unique_color_pixels)

     # Loop through each color label in unique colors in reverse order
    indices_to_remove = []
    for i, unique_color_label in enumerate(reversed(unique_color_labels_copy)):
        for cluster_color_label in cluster_color_labels:
            if (cluster_color_label in black_similar and unique_color_label in black_similar) or \
               (cluster_color_label in blue_similar and unique_color_label in blue_similar) or \
               (cluster_color_label in red_orange_similar and unique_color_label in red_orange_similar) or \
               (cluster_color_label in grey_similar and unique_color_label in grey_similar) or \
               (cluster_color_label in green_similar and unique_color_label in green_similar) or \
               (cluster_color_label in brown_similar and unique_color_label in brown_similar) or \
               (cluster_color_label in white_similar and unique_color_label in white_similar) or \
               (cluster_color_label in yellow_similar and unique_color_label in yellow_similar) or \
               (cluster_color_label in pink_similar and unique_color_label in pink_similar):
                indices_to_remove.append(len(unique_color_labels_copy) - 1 - i)
                break  # Move on to the next unique color label

    # Remove the similar color labels and pixels from the original lists.
    for i in sorted(indices_to_remove, reverse=True):
        unique_color_labels.pop(i)
        unique_color_pixels_list.pop(i)

    unique_color_labels_dissimilar = unique_color_labels

    # Convert the pixels list back to a numpy array.
    unique_color_pixels_dissimilar = np.array(unique_color_pixels_list)

    print("unique_color_labels_dissimilar length", len(unique_color_labels))
    return unique_color_labels_dissimilar, unique_color_pixels_dissimilar






def get_h_range_color_name(hsl_pixel):
    """Return the color label based on the hue value."""
    h, s, l = hsl_pixel
        
    color_dict = {
        (0.0, 11): "Reds",
        (11, 21): "Red-Oranges",
        (21, 45): "Oranges",
        (45, 68): "Yellows",
        (68, 75): "Yellow-Greens",
        (75, 140): "Greens",
        (140, 180): "Cyans",
        (180, 220): "Blues",
        (220, 299): "Purples",
        (299, 310): "Magenta-Pinks",
        (310, 360): "Pinks"
    }
    
    for range_, color_name in color_dict.items():
        lower, upper = range_
        if lower <= h < upper:
            return color_name

    else:
        print(f"The hue value {h} is not within any specified range.")



def get_pixel_color_label(hsl_pixel):
    """Return the color label for a pixel in HSL format."""

    color_label = get_h_range_color_name(hsl_pixel)
    h, s, l = (hsl_pixel)
    
    #Check for Black and White
    if l < 15:
        color_label = "Blacks"
    elif l > 98:
        color_label = "Whites" 
    
    #Check s and l ranges for Browns and Greys
    elif l >= 13 and l < 15:
        if color_label not in ["Oranges", "Red-Oranges", "Yellows", "Reds", "Pinks"]:
            if s < 74:
                color_label = "Dark-Greys"
        else:
            if 74 < s < 99:
                 color_label = "Dark-Browns"

        
    elif l >= 15 and l < 20:
        if color_label not in ["Oranges", "Red-Oranges", "Yellows", "Reds", "Pinks"]:
            if s < 25:
                color_label = "Dark-Greys"
        elif 30 < s < 98:
            color_label = "Dark-Browns"
        
    elif l >= 20 and l < 25:
        if color_label not in ["Oranges", "Red-Oranges", "Yellows", "Reds", "Pinks"]:
            if s < 20:
                color_label = "Dark-Greys"
        elif color_label in ["Reds", "Pinks"]:
            if 15 < s < 50:
                color_label = "Dark-Browns"
        else:
            if 20 < s < 97:             
                color_label = "Dark-Browns"
                

    elif l >= 25 and l < 30:
        if color_label not in ["Oranges", "Red-Oranges", "Yellows", "Reds", "Pinks"]:
            if s < 17:
                color_label = "Dark-Greys"
        elif color_label in ["Reds", "Pinks"]:
            if 17 < s < 30:
                color_label = "Dark-Browns"
        else: 
            if 19 < s < 90:
                color_label = "Dark-Browns"


    elif l >= 30 and l < 35:
        if color_label not in ["Oranges", "Red-Oranges", "Yellows"]:
            if s < 16:
                color_label = "Dark-Greys"  
        elif color_label in ["Reds", "Pinks"]:
            if 17 < s < 30:
                color_label = "Dark-Browns"         
        elif 17 < s < 85:
            color_label = "Dark-Browns"
            


    elif l >= 35 and l < 45:
        if color_label not in ["Oranges", "Red-Oranges"]:
            if s < 14:
                color_label = "Dark-Greys"
        elif color_label == "Reds":
            if 14 < s < 45:
                color_label = "Dark-Browns"
        elif 14 < s < 65:
                color_label = "Dark-Browns"
        
        


    elif l >= 45 and l < 50:
        if color_label not in ["Oranges", "Red-Oranges"]:
            if s < 12:
                color_label = "Dark-Greys"
        elif 12 < s < 60:
            color_label = "Dark-Browns"
        


    elif l >= 50 and l < 60:
        if color_label not in ["Oranges", "Red-Oranges"]:   
            if s < 12:
                color_label = "Light-Greys / Off-Whites" 
        elif color_label == "Reds":
            if 12< s < 23:
                color_label = "Light-Browns / Tans"
        elif 12 < s < 65:
            color_label = "Light-Browns / Tans"
       


    elif l >= 60 and l < 70:
        if color_label not in ["Oranges", "Red-Oranges"]:
            if s < 14:
                color_label = "Light-Greys / Off-Whites"
        elif 14 < s < 75:
            color_label = "Light-Browns / Tans"



    elif l >= 70 and l < 80:
        if color_label not in ["Oranges", "Red-Oranges"]:
            if s < 20:
                color_label = "Light-Greys / Off-Whites"
        elif 20 < s < 85:
            color_label = "Light-Browns / Tans"
        


    elif l >= 80 and l < 85:
        if color_label not in ["Oranges", "Red-Oranges"]:
            if s < 30:
                color_label = "Light-Greys / Off-Whites"
        elif 30 < s < 90:
            color_label = "Light-Browns / Tans"
        


    elif l >= 85 and l < 90:
        if color_label not in ["Oranges", "Red-Oranges"]:
            if s < 43:
                color_label = "Light-Greys / Off-Whites"
        elif 43 < s < 91:
            color_label = "Light-Browns / Tans"


    elif l >= 90 and l < 91:
        if color_label not in ["Oranges", "Red-Oranges"]:
            if s < 44:
                color_label = "Light-Greys / Off-Whites"            
        elif 44 < s < 93:
            color_label = "Light-Browns / Tans"
        


    elif l >= 91 and l < 93:
        if color_label not in ["Oranges", "Red-Oranges"]:
            if s < 50:
                color_label = "Light-Greys / Off-Whites"  
        elif 50 < s < 96:
            color_label = "Light-Browns / Tans"
        


    elif l >= 93 and l < 96:
        if color_label not in ["Oranges", "Red-Oranges"]:
            if s < 55:
                color_label = "Light-Greys / Off-Whites"
        elif 55 < s < 97:
            color_label = "Light-Browns / Tans"
        


    elif l >= 96 and l < 97:
        if color_label not in ["Oranges", "Red-Oranges"]:
            if s < 65:
                color_label = "Light-Greys / Off-Whites"
        elif 65 < s < 98:
              color_label = "Light-Browns / Tans"
        


    elif l >= 97 and l < 98:
        if color_label not in ["Oranges", "Red-Oranges"]:
            if s < 75:
                color_label = "Light-Greys / Off-Whites"
        elif 75 < s < 98:
            color_label = "Light-Browns / Tans"


    elif l >= 98 and l < 99:
        if color_label not in ["Oranges", "Red-Oranges"]:
            if s < 90:
                color_label = "Light-Greys / Off-Whites"
        elif 90 < s < 99:
            color_label = "Light-Browns / Tans"

    return color_label


def get_color_labels_list(hsl_pixels):
    """Return a list of color labels for a list of pixels in HSL format."""

    pixel_color_labels_list = []

    for hsl_pixel in hsl_pixels:

        pixel_color_labels_list.append(get_pixel_color_label(hsl_pixel))
    return pixel_color_labels_list




# Function to Identify Unique Colors (if any) That is, colors that standout but are not captured in the Kmeans Palette
def identify_unique_colors(pixel_color_labels_list, hsl_pixels, cluster_center_palette, cluster_color_labels, min_count=50, max_count=1000):

    # Create an empty dictionary to store counts of each unique color
    color_counts = {}

    # Go through each color label and count how many times it appears
    for label in pixel_color_labels_list:
        if label not in cluster_color_labels:
            if label not in color_counts:
                color_counts[label] = 1
            else:
                color_counts[label] += 1

    # Keep only colors with counts between the minimum and maximum thresholds
    unique_color_labels = [label for label, count in color_counts.items() if min_count < count]


    # If there are no unique colors, return an empty list
    if not unique_color_labels:        
        print("No Unique Color Pixel Values")
        print( "No Unique Color Labels")
        return [], []


    # Get the pixel values for the unique colors
    unique_color_pixels = np.array([hsl_pixels[i] for i, label in enumerate(pixel_color_labels_list) if label in unique_color_labels])

    # Create a list of labels that correspond to the unique color pixels
    unique_color_labels_final = [label for i, label in enumerate(pixel_color_labels_list) if (label in unique_color_labels) and (hsl_pixels[i] in unique_color_pixels)]

    unique_color_labels = unique_color_labels_final

    print("Unique color Pixel length: ", len(unique_color_pixels))
    print("Unique Color Labels Length: ", len(unique_color_labels))

    return unique_color_pixels, unique_color_labels


#This function sets color priorities for each category and returns the first color label with the highest priority that appears in the unique colors. 
def select_most_contrasting_label(image_category, unique_color_labels_dissimilar):
    priorities = {
        "Brownish": ["Cyans", "Purples", "Greens", "Reds", "Red-Oranges", "Pinks"],
        "Yellowish": ["Reds", "Red-Oranges",  "Magenta-Pinks", "Pinks" , "Blues", "Cyans", "Purples", "Greens"],
        "Greenish": ["Reds", "Red-Oranges", "Oranges", "Magenta-Pinks", "Pinks"],
        "Redish": ["Cyans", "Blues", "Greens", "Violet-Blues"],
        "Blueish": ["Reds", "Yellows", "Red-Oranges", "Oranges", "Magenta-Pinks", "Pinks"],
        "White-ish": ["Blacks", "Dark-Greys", "Reds", "Red-Oranges", "Oranges", "Magenta-Pinks", "Pinks", "Greens", "Cyans", "Blues", "Purples"],
        "Blackish" : ["Whites", "Reds", "Red-Oranges", "Oranges", "Magenta-Pinks", "Pinks", "Greens", "Blue-Cyans", "Blues", "Violet-Blues", "Purples"],
        "Varied" : ["Reds", "Blues", "Greens"],
        "Greyish" : ["Reds", "Red-Oranges", "Oranges", "Magenta-Pinks", "Pinks", "Greens", "Cyans", "Blues"]
    }
    
    if image_category in priorities:
        for label in priorities[image_category]:
            if label in unique_color_labels_dissimilar:
                return label
    return None



#This function removes all of the color's with labels that are too similar, as determined by the similarity function
def pop_unique_similars(most_contrasting_label, unique_color_labels_dissimilar, unique_color_pixels_dissimilar):

    new_unique_color_pixels = []
    for i, label in enumerate(unique_color_labels_dissimilar):
        if label == most_contrasting_label:
             new_unique_color_pixels.append(unique_color_pixels_dissimilar[i])
              
    return   new_unique_color_pixels




#This function tries to extract the most visually interesting color from among the unique color pixels. It tries to balance visual interest with what is appropriate for the image.   
def get_most_interesting_color(new_unique_color_pixels):
    if not new_unique_color_pixels:
        return None

    # Fits new_unique_color_pixels to KMeans classifier so we can select a more representative color than just an average.
    new_unique_color_pixels = np.array(new_unique_color_pixels)
    unique_clf = KMeans(n_clusters=10) 
    unique_clf.fit(new_unique_color_pixels)  

    # Get the cluster centers and their HSL values
    unique_cluster_centers = unique_clf.cluster_centers_
    print("UNIQUE CLUSTER CENTERS:", unique_cluster_centers)
    

    # Find the cluster center with the highest saturation value
    s_values = [hsl[2] for hsl in unique_cluster_centers]
    highest_s_value = np.argmax(s_values)
    s = s_values[highest_s_value]

    # Find the cluster center with the lightness value closest to 50. 
    l_values = [hsl[1] for hsl in unique_cluster_centers]
    closest_to_50_index = np.argmin(np.abs(np.array(l_values) - 50))
    l = l_values[closest_to_50_index]

    # Get the average hue value
    h_values = [hsl[0] for hsl in unique_cluster_centers]
    h = sum(h_values) / len(h_values)


    print( "MOST INTERESTING HSL BEFORE MODIFICATION", h, s, l )

        #If l is below 35, and s is below 40, raise them
    if l < 35:
        l = 35
    if l < 36 and s < 40:
        s = 40


    return (h, s, l)




def select_most_unique_color(unique_color_labels_dissimilar, unique_color_pixels, cluster_color_labels):

    #First categorize image using the cluster color labels:
    image_category = categorize_image(cluster_color_labels)
    most_contrasting_label = select_most_contrasting_label(image_category, unique_color_labels_dissimilar)
    print("MOST CONTRASTING LABEL: ", most_contrasting_label)
    new_unique_color_pixels = pop_unique_similars(most_contrasting_label, unique_color_labels_dissimilar, unique_color_pixels)
    most_interesting_color = get_most_interesting_color(new_unique_color_pixels)

    return most_interesting_color, image_category




def categorize_image(color_labels):
    categories = {
        "Brownish": ["Dark-Browns", "Light-Browns / Tans"],
        "Yellowish": ['Yellows', "Yellow-Greens" "Light-Browns / Tans"],
        "Greenish": ["Greens", "Cyans", "Yellow-Greens"],
        "Redish": ["Reds", "Red-Oranges", "Oranges", "Magenta-Pinks", "Pinks"],
        "Blueish": [ "Cyans", "Blues", "Purples"],
        "White-ish": ["Whites", "Light-Greys / Off-Whites"],
        "Blackish" : ["Blacks", "Dark-Greys", "Dark-Browns"],
        "Varied" : ["Yellows", "Greens", "Reds", "Blues"],
        "Greyish" : ["Blacks", "Whites", "Dark-Greys", "Light-Greys / Off-Whites"]
    }

    # Categorize the image based on the cluster_center_palette
    category_counts = {category: 0 for category in categories.keys()}
    for label in color_labels:
        for category, colors in categories.items():
            if label in colors:
                category_counts[category] += 1
                break

    # Find the category with the highest count
    image_category = max(category_counts, key=category_counts.get)

    return image_category






# Display the Image and Color Palette
def display_image_with_palette(img, cluster_center_palette, unique_color_hex, image_category):
    if unique_color_hex is None:
        print('This image does not have contrasting Unique Colors.')
        fig, ax = plt.subplots(1, 2, figsize=(10, 6))
        ax[0].imshow(img)
        ax[0].axis('off')
        ax[1].bar(range(len(cluster_center_palette)), [1]*len(cluster_center_palette), color=cluster_center_palette)
        ax[1].set_xticks(range(len(cluster_center_palette)))
        ax[1].set_xticklabels(['']*len(cluster_center_palette))
        ax[1].set_yticks([])
        ax[1].set_title('Color Palette')
    else:
        fig, ax = plt.subplots(1, 3, figsize=(10, 6))
        ax[0].imshow(img)
        ax[0].axis('off')
        ax[1].bar(range(len(cluster_center_palette)), [1]*len(cluster_center_palette), color=cluster_center_palette)
        ax[1].set_xticks(range(len(cluster_center_palette)))
        ax[1].set_xticklabels(['']*len(cluster_center_palette))
        ax[1].set_yticks([])
        ax[1].set_title('Color Palette')
        ax[2].bar(1, 1, color=unique_color_hex)
        ax[2].set_xticklabels(['']*len(unique_color_hex))
        ax[2].set_yticks([])
        ax[2].set_title('Unique Colors')
        print(unique_color_hex)
        print(image_category)
    plt.show()

    




# Main Function Call to Get Palette

def get_palette(image_path, quick=True, show=True):
    
    #Access image
    if image_path.startswith('http'):
        response = requests.get(image_path)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(image_path)

    print(f"Image opened successfully: {img.size}")


    #Preprocess Image and get combined pixels
    rgb_pixels = preprocess(img, quick)
    combined_pixels, rgb_pixels, hsl_pixels = combine_pixels(rgb_pixels)
    print(f"Preprocessing complete: {len(combined_pixels)} pixels processed")



    #Get Color Labels for Pixels
    print('Labeling Pixels')
    print( "HSL PIXELS SAMPLE: ", hsl_pixels[:3]) 
    pixel_color_labels_list = get_color_labels_list(hsl_pixels)
    print("Pixels Labeled. Sample: ", pixel_color_labels_list[:10])

        
    
    
    print("Beginning KMeans clustering")
    #KMean clustering to get Main Cluster-Center Color Palette
    clf = KMeans(n_clusters=6)
    clf.fit(combined_pixels)  # Fit KMeans using HSL values
    cluster_center_palette = clf.cluster_centers_ #Assigns the 6 color centers as 6 colors in color palette
    np.set_printoptions(precision=2, suppress=True)
    cluster_center_palette = np.round(cluster_center_palette, decimals=0).astype(int)
    print(f"KMeans clustering complete. Palette pixel values: {cluster_center_palette[:, 3:]}")


    #Label Main Cluster Palette colors according to basic hue categories using HSL ([:, 3:])
    print("Getting Cluster Label Names")
    cluster_color_labels= get_color_labels_list(cluster_center_palette[:, 3:]) # The get_color_labels_list function takes HSL values array
    print("Cluster Label Names: ", cluster_color_labels)
        
    #Get HEX values of Main Cluster Palette
    print("Getting Hex Values for Palette: ")
    color_palette_hex_values = rgb_to_hex(cluster_center_palette[:, :3]) #rgb_to_hex takes rgb values array
    print(f"Main cluster palette colors in HEX: {color_palette_hex_values}")
    

    #Identify Unique Colors (if any) and get hex value(s)

    print("Identifying Unique Colors not in Palette (if any).")
    
    unique_color_pixels, unique_color_labels = identify_unique_colors(pixel_color_labels_list, hsl_pixels, cluster_center_palette, cluster_color_labels)

    
    #print(f"Unique color pixels before dissimilar function: {unique_color_pixels}")
    print(f"Unique color labels unique labels before dissimilar function: {np.unique(unique_color_labels)}")

    print("Removing too similar Unique Color Labels...")
    unique_color_labels_dissimilar, unique_color_pixels_dissimilar = compare_color_labels(unique_color_labels, unique_color_pixels, cluster_color_labels)
    unique_labels, counts = np.unique(unique_color_labels_dissimilar, return_counts=True)
    print("Dissimilar Unique Color Labels: ", unique_labels)
    print("Counts: ", counts)




    #Get most unique unique color
    print("Getting most Unique Color")

    most_interesting_color, image_category = select_most_unique_color(unique_color_labels_dissimilar, unique_color_pixels_dissimilar, cluster_color_labels)
    print(" most_interesting_color: ", most_interesting_color)
    print( "image_category", image_category)


    unique_color_hex = hsl_to_hex(most_interesting_color)


    
    if show:
        display_image_with_palette(img, color_palette_hex_values, unique_color_hex, image_category)

    return color_palette_hex_values, image_category, unique_color_hex
                         
