import numpy as np


class LinearRegression:
    '''
    A class to represent a generalized linear regression model
    with regularization

    Attributes
    --------

    alpha: float
        The learning rate for this model
    tolerance: float
        The minimum difference between consecutive weights to determine convergence
    max_iter: int
        The maximum number of training epochs
    reg_type: str
        The type of regularization: None, l1(lasso) or l2(ridge)
    reg_rate: float
        The regularization rate, lambda
    theta_: 
        The weights for the model
    '''

    def __init__(self, alpha = 0.01, tolerance = 0.0001, max_iter = 100, reg_type = None, reg_rate = 0.0):
        self.alpha = alpha
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.reg_type = reg_type
        self.reg_rate = reg_rate

        self.theta_ = None

    def compute_cost(self, theta, X, y):
        '''
        Calculates the Mean Squared Error of the model

        Parameters
        ----------
        theta:
            The weights for loss calculation
        X:
            The training features matrix
        y:
            The training data labels

        Returns
        --------
        loss:
            The MSE loss given the current theta
        '''
        N, D = X.shape
        loss = 0.0
        for i in range(N):
            thetaT = np.transpose(theta)
            y_est = np.matmul(thetaT, X[i])
            err = y_est - y[i]
            loss += err**2
        loss = loss/N

        # handling regularization
        if self.reg_type == None:
            return loss
        elif self.reg_type == 'l1':
            sum = 0.0
            for i in range(1, theta.size):
                sum += abs(theta[i])
            return (loss + (self.reg_rate)*sum)
        elif self.reg_type == 'l2':
          sum = 0.0
          for i in range(1, theta.size):
            sum += (theta[i])*(theta[i])
          return (loss + (self.reg_rate)*sum)

    def compute_gradient(self, theta, X, y):
        '''
        Calculates the gradient of the Mean Squared Error of the model,
        to be used for gradient descent

        Parameters
        ----------
        theta:
            The weights for loss calculation
        X:
            The training features matrix
        y:
            The training data labels

        Returns:
        grad:
            The calculated gradient
        '''
        N = X.shape[0]
        XT = np.transpose(X)
        XTX = np.matmul(XT, X)
        grad = (-2.0/N)*np.matmul(XT, y) +(2.0/N)*np.matmul(XTX, theta)

        # handling regularization
        if self.reg_type == None:
            return grad
        elif self.reg_type == 'l1':
            sign = lambda n: 1 if n>0 else -1
            for i in range(1, grad.size):
                grad[i] += (self.reg_rate*sign(theta[i]))
            return grad
        elif self.reg_type == 'l2':
            for i in range(1, grad.size):
                grad[i] += (2.0*self.reg_rate*theta[i])
            return grad
        

    def fit(self, X, y):
        '''
        Fits the model to the training data using gradient descent

        Parameters
        ----------
        X:
            The training features matrix
        y:
            The training data labels
        '''

        N, D = X.shape

        # adding col of ones for bias 
        ones_col = np.ones((N, 1))
        X = np.hstack((ones_col, X))

        theta_prev = np.zeros((D + 1,))
        theta_new = theta_prev.copy()

        # gradient descent loop
        for i in range(self.max_iter):
            theta_new = theta_prev - self.alpha * self.compute_gradient(theta_prev, X, y)

            # check convergence
            if np.linalg.norm(theta_new - theta_prev) / (np.linalg.norm(theta_prev) + self.tolerance) <= self.tolerance:
                self.converged_ = True
                break
                
            theta_prev = theta_new.copy()
        
        # save the weights
        self.theta_ = theta_new

    def predict(self, X):
        '''
        Predicts the label for every item in X

        Parameters
        ----------
        X:
            The features matrix on which predictions are to be made

        Returns
        ---------
        y_pred:
            The predictions using the model given the features in X
        
        '''
        N = X.shape[0]
        
        ones_col = np.ones((N, 1))
        X = np.hstack((ones_col, X))
        y_pred = X.dot(self.theta_)
        return y_pred

        
