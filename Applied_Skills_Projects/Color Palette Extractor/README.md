# Color Palette Extractor
### Overview:
Color Palette Extractor is a tool I developed to automatically create a color palette for a given image, using K Means Clustering. However, color palettes functions that utilize KMeans in this way often struggle to capture small splashes of contrasting colors. To address this issue, I added a feature that reexamines the image after KMeans clustering is performed. Unique/Contrasting colors are identified using a labeling function that assigns a broad color name to each pixel, based on the hue value. These labels are compared to the color palette labels to eliminate non-unique colors. Pixels with the most dissimilar labels are further filtered and finally clustered using KMeans. From the resulting cluster centers, a representative color is automatically chosen. Additionally, the program categorizes the image overall. The output includes the color palette hex values, the unique color hex value (if present), and the image category.





### Usage
To use Color Palette Extractor, you must have Python 3.x installed on your computer along with the following dependencies:  
- numpy
- pandas
- scikit-learn
- matplotlib
- Pillow

To use this script, download or clone the repository to your local machine.
You can install these packages using the following command:
	**pip install -r requirements.txt**
	
Once you have the dependencies installed, you can run the program using the **get_palette()** function. This function takes one argument, which is the path to the image you want to extract a color palette from.
The **get_palette()** function returns three values:
- **palette**: a list of hex color codes representing the dominant colors in the image
- **category**: a string indicating the overall category of the image
- **unique_colors**: Hex color code of unique color (if present)

Here's an example of how to use the **get_palette()**  function:  
_______________________________________________________________________________
from **color_palette_extractor** import **get_palette**  
  
palette, category, unique_color = **get_palette('path/to/image.jpg')**
_______________________________________________________________________________

### Notes
Please note that the program assumes the image is in RGB mode (not grayscale, CMYK, etc.).

ALso note that, while Color Palette Extractor includes a feature to display the extracted colors, this is primarily intended for testing and debugging purposes. If you want to use the program to display the hex values in a visually appealing way, I recommend that you create your own function or use an existing library. However, the built-in display function can still be useful for quickly verifying that the program is working correctly and extracting the expected colors from an image, or if a visually appealing display is not required.


**Thank you for using Color Palette Extractor! Feel free to use this code in your projects! If you find it helpful, please cite me in your work by including a link to this repository.**

