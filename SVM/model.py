import numpy as np

class SVM_classifier:

  #initialize the hyper-parameters:
  #learning rate, epochs, regularization parameter
  def __init__(self, learning_rate, epochs, lambda_parameter):
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.lambda_parameter = lambda_parameter


  #fit the dataset to the model
  def fit(self, X, Y):
    self.no_of_records, self.no_of_features = X.shape

    self.w = np.zeros(self.no_of_features) #weights for each feature

    self.b = 0 #bias

    #label encoding
    y_label = np.where(Y <= 0, -1, 1)

    for i in range(self.epochs):
      self.updateWeights(X, y_label)

  #Update the weights and bias value
  def updateWeights(self, X, y_label):

    for index, x_i in enumerate(X):

      # Calculate the scaled regularization term ONCE
      reg_term = (2 * self.lambda_parameter * self.w) / self.no_of_records

      condition = y_label[index] * (np.dot(x_i, self.w) - self.b) >= 1

      if condition:
        dw = reg_term
        db = 0
      else:
        dw = reg_term - (x_i * y_label[index])
        db = y_label[index]

      self.w = self.w - self.learning_rate * dw
      self.b = self.b - self.learning_rate * db

  #Predict the label for given input value
  def predict(self, X):

    #y = w.x - b
    output = np.dot(X, self.w) - self.b

    predicted_labels = np.sign(output) #either +1 or -1

    y_hat = np.where(predicted_labels <= -1, 0, 1)

    return y_hat