import numpy as np

class WeightedLogisticRegression:
    '''
    A class to represent a generalized logistic regression model and
    binary classifier with the option of having weighted inputs for use
    in the AdaBoost Algorithm 

    Attributes
    --------

    alpha: float
        The learning rate for this model
    tolerance: float
        The minimum difference between consecutive weights to determine convergence
    max_iter: int
        The maximum number of training epochs
        
    theta_: 
        The weights for the model
    converged_: bool
        Whether the gradient descent has converged
    
    '''

    def __init__(self, alpha = 0.1, tolerance = 0.01, max_iter = 1000):

        self.alpha = alpha
        self.tolerance = tolerance
        self.max_iter = max_iter

        self.theta_ = None
        self.converged_ = False

    def compute_gradient(self, theta, X, y, sample_weight):
        '''
        Calculates the gradient of the Cross Entropy Loss of the model,
        to be used for gradient descent, 

        Parameters
        ----------
        theta:
            The weights for loss calculation
        X:
            The training features matrix
        y:
            The training data labels, binary classes 1 and 0
        sample_weghts:
            The weights for each data point in X and y

        Returns:
        grad:
            The calculated gradient
        '''
        sigmoid = lambda x: 1 / (1 + np.exp(-x))

        N, Dplus1 = X.shape
        grad = [0.0]*Dplus1
        thetaT = theta.transpose()
        for i in range(N):
            err = sigmoid(np.matmul(thetaT, X[i])) - y[i]
            grad += (sample_weight[i]*err*X[i])
        return grad

    def fit(self, X, y, sample_weight):
        '''
        Fits the model to the training data using gradient descent

        Parameters
        ----------
        X:
            The training features matrix
        y:
            The training class labels, binary 1 or 0
        sample_weghts:
            The weights for each data point in X and y
        '''
        N, D = X.shape

        # ones column for bias
        ones_col = np.ones((N, 1))
        X = np.hstack((ones_col, X))

        theta_prev = np.zeros((D + 1,))
        theta_new = theta_prev.copy()

        #gradient descent loop
        for i in range(self.max_iter):
            theta_new = theta_prev - self.alpha * self.compute_gradient(theta_prev, X, y, sample_weight)

            # convergence check
            if np.linalg.norm(theta_new - theta_prev) / (np.linalg.norm(theta_prev) + self.tolerance) <= self.tolerance:
                self.converged_ = True
                break
            
            theta_prev = theta_new.copy()

        #saving weights
        self.theta_ = theta_new

    def predict_probability(self, X):
        '''
        Predicts the probability that the label is 1 for every item in X

        Parameters
        ----------
        X:
            The features matrix on which predictions are to be made

        Returns
        ---------
        y_est:
            The predicted probabilities
        
        '''
        N = X.shape[0]

        ones_col = np.ones((N, 1))
        X = np.hstack((ones_col, X))

        # sigmoid to convert regression output into a probability
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        y_est = sigmoid(X.dot(self.theta_))
        return y_est

    def predict(self, X):
        '''
        Converts predicted probabilities into predicted class labels

        Parameters
        ----------
        X:
            The features matrix on which predictions are to be made

        Returns
        ---------
        y_pred:
            The predicted class labels
        
        '''
        y_est = self.predict_probability(X)
        y_pred = y_est.copy()
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        return y_pred


