import numpy as np

class LinearRegression:
    def __int__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape

        #initialize weights and bias to zero
        self.weight = np.zeros(num_features)
        self.bias = 0

        #Gradient descent
        for _ in range(self.iterations):
            #predict the output using current weigths and bias
            y_pred = self.predict(X)

            #calculate the gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            #update the weights and bias
            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db


    def predict(self, X):
        return np.dot(X, self.weight) + self.bias





