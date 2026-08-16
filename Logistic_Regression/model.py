import numpy as np

class LogisticRegression:
  #initiating hyperparameters
  def __init__(self, learning_rate, number_of_iterations):
    self.learning_rate = learning_rate
    self.no_of_iterations = number_of_iterations

  #fit() to train the model using dataset
  def fit(self,X,Y):
    #m=number of record, n=features
    self.m, self.n = X.shape

    #initiating weights and bias
    self.W = np.zeros(self.n)
    self.b = 0

    for i in range(self.no_of_iterations):
      self.update_weights(X, Y)


  def update_weights(self, X, Y):
    #calculating z
    #X -> (m,n), w -> (n,), for correct ordering: X.dot(w)
    #z -> (m,)
    z = X.dot(self.W) + self.b

    #Y -> (m,), m = number of records
    y_cap = 1/(1+np.exp(-z))

    #gradients
    #dw should be for each feature, so its dimensions should be (n,)
    dw = (1/self.m) * X.T.dot(y_cap - Y.squeeze())

    #db should be a scalar value
    db = np.sum(y_cap - Y.squeeze()) * (1/self.m)

    self.W = self.W - self.learning_rate * dw
    self.b = self.b - self.learning_rate * db

  #sigmoid equation and decision boundary
  def predict(self, X):
    z = X.dot(self.W) + self.b
    y_pred = 1/(1+np.exp(-z))
    y_pred = np.where(y_pred > 0.5, 1, 0)
    return y_pred