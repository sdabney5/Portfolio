# K-Nearest Neighbors (KNN) Implementation from Scratch

This is a simple K-Nearest Neighbors (KNN) implementation written from scratch in Python.  
I did this project to better understand the inner workings of the KNN algorithm.

## Features
- Written from scratch using only NumPy for vector operations.
- Customizable K value.
- Uses Euclidean distance for calculating the distances between data points.
- Handles ties by choosing the label that appears first among the k-nearest neighbors.
- Easy-to-understand code with comments for better readability.

## Usage
1. Import the KNN class from the script.
2. Instantiate the KNN class with the desired value of K.
3. Fit the model using the `fit` method with training data.
4. Predict target values using the `predict` method with new data points.

## Credits
- [YouTube video by AssemblyAI](https://www.youtube.com/watch?v=rTEtEy5o3X0) for the idea and inspiration for this KNN implementation.
- [GeeksforGeeks article on numpy.argsort](https://www.geeksforgeeks.org/numpy-argsort-in-python/) for information on using numpy's argsort function.
- [NIST Dictionary of Algorithms and Data Structures article on Euclidean distance](https://xlinux.nist.gov/dads/HTML/euclidndstnc.html) for the Euclidean distance formula.

