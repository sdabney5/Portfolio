class LinearRegression:
    
    #initialize hyperparameters
    def __init__(self, alpha = 0.001, num_iterations = 1000):
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.theta = None
        
    def fit(self, X, y):
        # Convert X to a list of lists
        X = X.tolist()

        # Add a column of 1's to X to represent the intercept
        for row in X:
            row.insert(0, 1)
        
        # Rows (m) and columns (n)
        m, n = len(X), len(X[0])
        
        # Initialize a list of 0's of length "n" for first theta values
        self.theta = [0]*n         
        
        #Hypothesis function
        for _ in range(self.num_iterations):
            # Create a 0's list of predicted values for y_hat
            y_hat = [0]*m 
            for i in range(m):
                 for j in range(n):
                        y_hat[i] += (X[i][j] * self.theta[j])  #this is the hypothesis function.
                        
                
            #Gradient calculation of MSE (J(theta)) w.r.t. theta[j]
            dJ_dtheta = [0]*n 
            
            for j in range(n):
                for i in range(m):
                    dJ_dtheta[j] += 1/m * ((y_hat[i] - y[i]) * X[i][j]) # The partial derivative of the MSE w.r.t theta[j]
                    
             
            #update theta values
            for j in range(n):
                self.theta[j] -=  self.alpha * dJ_dtheta[j]
            
    def predict(self, X):
        # Convert X to a list of lists
        X = X.tolist()

        # Add a column of 1's to X to represent the intercept
        for row in X:
            row.insert(0, 1)
        
        # Rows (m) and columns (n)
        m , n = len(X), len(X[0])
        
        #Create a list of zeros for predictions corresponding to each row
        y_prediction = [0]*m 
        
        #Compute predictions
        for i in range(m):
            for j in range(n):
                y_prediction[i] += X[i][j] * self.theta[j]
        return y_prediction
