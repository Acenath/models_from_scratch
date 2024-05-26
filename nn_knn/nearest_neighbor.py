import numpy as np

class NearestNeighbor:

    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, metric = "euclidean"):
        metric = metric.lower() # Prevent typing error

        num_test = X.shape[0]
        y_pred = np.zeros(num_test, dtype = self.y_train.dtype)
        
        #Manhattan distance
        if metric == "manhattan":

            for i in range(num_test):
                distances = np.sum(np.abs(self.X_train - X[i, :]), axis = 1)
                min_index = np.argmin(distances)
                y_pred[i] = self.y_train[min_index]

        #Euclidean distance
        elif metric == "euclidean":
            for i in range(num_test):
                distances = np.sqrt(np.sum(np.square(self.X_train - X[i, :]), axis = 1))
                min_index = np.argmin(distances)
                y_pred[i] = self.y_train[min_index]


        return y_pred
    
    def accuracy_score(self, y_test, prediction):
        try:
            if y_test.shape[0] != prediction.shape[0]:
                raise Exception("Length of the y_test and prediction arrays are not match!!")
        except:
            print("Entered parameters should be in type of numpy array (np.array())!!")


        return np.mean(y_test == prediction) * 100


