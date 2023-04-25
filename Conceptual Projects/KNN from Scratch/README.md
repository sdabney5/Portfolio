

# KNN Classifier
## Overview
K-Nearest Neighbors (KNN) algorithm is a type of supervised learning that can be used for classification or regression. In KNN, we predict the label of a data point by finding the k-nearest data points to it in the training set, and then assigning the label that is most common among these neighbors. The distance between the data points can be calculated using different metrics, such as the Euclidean distance. The value of k is a hyperparameter that needs to be set before training the model, and it determines how many neighbors will be considered for classification.  
  
## Usage
To use the KNN classifier, create an instance of the KNN class and provide the value of k (number of neighbors to consider) as an argument (default is 15). Then use the fit() method to train the model on a dataset, and predict() method to predict target values for new data. Note, this algorithm is less efficient than others. This was a learning/educational project more than an attempt to offer a robust and well-tuned algorithm for public use. Even so, feel free to use it if you want.
  
##Requirements
•	NumPy
•	Collections
  
  
## Installation  
•	Clone this repository  
•	Navigate to the project directory  
•	Create a new virtual environment and activate it
•	Install dependencies from requirements.txt: pip install -r requirements.txt


___
Thanks for checking out my KNN Algo.

