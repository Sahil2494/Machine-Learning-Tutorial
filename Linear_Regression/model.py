import numpy as np

class Linear_Regression:
  #Initiating the hyperparameters
  def __init__(self, learning_rate, no_of_iterations):

    self.learning_rate = learning_rate
    self.no_of_iterations = no_of_iterations

  #Fit the data to model
  def fit(self,X,Y):
    #number of records and number of features
    self.m, self.n = X.shape

    #initiating weight and bias value
    self.w = np.zeros(self.n) #weight for each feature
    self.b = 0

    # Flattening Y to prevent broadcasting issues (ensures shape is (m,))
    Y = np.ravel(Y)

    #implementing gradient descent
    for i in range(self.no_of_iterations):
      self.update_weights(X,Y)

  #Updating the weights and bias using gradient descent
  def update_weights(self,X,Y):
    y_prediction = self.predict(X)

    #calculate gradients
    dw = - (2 * (X.T).dot(Y - y_prediction)) / self.m
    db = - 2 * np.sum(Y - y_prediction) / self.m

    #updating the weights and bias
    self.w = self.w - self.learning_rate * dw
    self.b = self.b - self.learning_rate * db

  def predict(self,X):
    return X.dot(self.w) + self.b