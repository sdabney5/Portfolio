# Famous Paintings Color Palette Extraction
I scraped Painting Data and used a custom python script to produce a Famous Paitings Color Palette Database.

Highlights:
* Web Scraping
* Data Cleaning
* Data Visualization

--- --- --- 

This project used Beautiful Soup to scrape data from https://en.most-famous-paintings.com, a website that lists and displays famous paintings. 
The data includes the image URL, artist name, and painting title. The data was then cleaned and exported to a csv.

Then, a custom script called color_palette_extractor (see my Projects repo) was used to extract the color palettes of each painting, and categorize the image.
The information was then stored as a CSV file.

The resulting CSV file contains color palette information for each of the paintings, including hex values for the color palette, the image category, and
unique contrasting color hex value (if present)

This Jupyter Notebook provides a step-by-step guide to how the data was scraped, processed, and analyzed. It also includes visualizations of the color palettes of a selection of the paintings.

Note that this project was created as an example of using Beautiful Soup and the color_palette_extractor script to extract color palette information. It is not intended for actual use, but the resulting CSV file may be useful for anyone who needs color palette information for famous paintings.
