Color Palette Extractor

This script extracts a color palette from an input image using the KMeans clustering algorithm. It returns a list of hex values for the dominant colors in the image.

Installation

To use this script, you'll need to set up a virtual environment and install the required packages. Follow these steps:

Clone this repository to your local machine.
Navigate to the directory where you cloned the repository in your terminal or command prompt.
Create a new virtual environment by running the following command:
pip install -r requirements.txt

Usage

To use the script, call the get_palette function with the path to the input image. The function also has an optional show parameter that can be set to False to suppress the display of the image and its color palette.

Example:

palette = get_palette("path/to/image.jpg", show=True)

License

This project is licensed under the MIT License. See the LICENSE file for more information.