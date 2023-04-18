# Color Palette Extractor
### Overview:
Color Palette Extractor Color Palette Extractor is a Python program that automatically generates a color palette for a given image. The program uses K Means Clustering to get the dominant colors of the image. However, since previous versions of the program were not able to identify small splashes of contrasting colors which should be in the color palette, it reexamines the image to find instances of color which contrast with the color palette, i.e., colors which are “unique”. After extracting the pixels with color labels most dissimilar to palette colors, the remaining pixels are clustered with a KMeans algorithm and a representative color is chosen from among the cluster centers.


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


**Thank you for using Color Palette Extractor! Feel free to use this code in your projects! If you find it helpful, please cite me in your work by including a link to this repository.**

