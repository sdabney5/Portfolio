
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=15):
        self.k = k
        
    # Calculates the Euclidean distance between two points
    def euclidean_distance(self, x1, x2):
        distance = np.sqrt(np.sum((x1-x2)**2))
        return distance
    
    # Trains the model on the provided data
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    # #Predict target values for each x value dataset X using the fitted model
    def predict(self, X):
        predictions = []
        for x in X:
            prediction = self.dist_calc_sort_classify(x)
            predictions.append(prediction)
        return predictions
    
    
    # Computes distances, finds the nearest neighbors, and classifies based on the most common label
    def dist_calc_sort_classify(self, x):

    	#Calculate the distance between new datapoint x and each
		#datapoint x_train in X_train
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        

        #Sort distances in ascending order (argsort's default)
		# Note that argsort(distances) returns the indices, not the values at those indices.
		# And [:self.k] says we want only the k lowest distances (from 0 to k of the sorted distances)
        k_indices = np.argsort(distances)[:self.k]



        k_nearest_labels = [self.y_train[i] for i in k_indices]
        

		#Note: most_common returns ordered list of tuples (label, count) for all elements:
		# we want only the first value (label) of first tuple (the lowest distance)
		# and so we need most_common[0][0]

        #We need to also add a statement to handle the case of ties

        most_common = Counter(k_nearest_labels).most_common()


        # Handle ties by choosing the label that appears first among the k-nearest neighbors
        if len(most_common) == 1:
            return most_common[0][0]
        else:
            for label, count in most_common:
                if label in k_nearest_labels:
                    return label
            return most_common[0][0]





# Credits to the following sources:
# - https://www.youtube.com/watch?v=rTEtEy5o3X0 for the idea and inspiration for this KNN implementation.
# - https://www.geeksforgeeks.org/numpy-argsort-in-python/ for information on using numpy's argsort function.
# - https://xlinux.nist.gov/dads/HTML/euclidndstnc.html#:~:text=Note%3A%20In%20N%20dimensions%2C%20the,or%20q)%20in%20dimension%20i.
# 		-"In N dimensions, the Euclidean distance between two points p and q is √(∑i=1N (pi-qi)²) 
#		  where pi (or qi) is the coordinate of p (or q) in dimension i."