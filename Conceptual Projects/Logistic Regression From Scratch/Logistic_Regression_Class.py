# Set the value of Euler's number. 
# I am doing this instead of math.exp simply because this project does not use imported modules or libraries
e = 2.71828

class LogisticRegression:
    
    #initialize hyperparameters
    def __init__(self, alpha=0.001, num_iterations=1000, threshold = 0.5):
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.theta = None
        self.threshold = threshold
        
    def fit(self, X, y):
        # Add a column of 1's to X to represent the intercept
        X = X.insert(0, "ONES", 1) 
        
        # Rows (m) and columns (n)
        m , n = X.shape
        
        # Initialize a list of 0's length "n" for first theta values
        self.theta = [0]*n
        
        # Hypothesis function
        for _ in range(self.num_iterations):
            # Create a 0's list of predicted values for y_hat
            y_hat = [0]*m 
            for i in range(m):
                for j in range(n):
                    y_hat[i] += (X[i][j] * self.theta[j])  #this is just the hypothesis function from linear regression.
                
                # For Logistic Regression, we need to apply the sigmoid function to each predicted value                
                y_hat[i] = 1 / (1 + e**(-y_hat[i])) 
                        
            # Gradient calculation of cross-entropy (J(theta)) w.r.t. theta[j]
            dJ_dtheta = [0]*n
            
            for j in range(n):
                for i in range(m):
                    dJ_dtheta[j] += 1/m * ((y_hat[i] - y[i]) * X[i][j]) # This gives the partial derivative of the cross-entropy w.r.t theta[j]
             
            # Update theta values. Previous theta value minus the corresponding gradient value times the learning rate
            for j in range(n):
                self.theta[j] -=  self.alpha * dJ_dtheta[j]
            
    def predict_prob(self, X):
        # Add a column of 1's to X to represent the intercept
        X = X.insert(0, "ONES", 1) 
        
        # Rows (m) and columns (n)
        m , n = X.shape 
        
        # Create a list of zeros for probability predictions
        y_predicted_prob = [0]*m
        
        # Compute predictions
        for i in range(m):
            for j in range(n):
                y_predicted_prob[i] += X[i][j] * self.theta[j]
                
                # Apply sigmoid function to the predicted values
                y_predicted_prob[i] = 1 / (1 + e**(-y_predicted_prob[i]))

        return y_predicted_prob
    
    def predict(self, X):
        # Add a column of 1's to X to represent the intercept
        X = X.insert(0, "ONES", 1) 
        
        # Rows (m) and columns (n)
        m , n = X.shape 
        
        # Create a list of zeros for class predictions and probability predictions
        y_prob_pred = [0]*m
        y_pred  = [0]*m
        
        # Compute class predictions according to the threshold
        for i in range(m):
            for j in range(n):
                y_prob_pred[i] += X[i][j] * self.theta[j]
                
                # Apply sigmoid function to the predicted values
                y_prob_pred[i] = 1 / (1 + e**(-y_prob_pred[i]))
                
            if y_prob_pred[i] >= self.threshold:
                y_pred[i] = 1
        

        return y_pred
        
        
