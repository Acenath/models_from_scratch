import numpy as np

class KNearestNeighbor:
    
    def __init__(self, k):
        self.k = k

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        test_row = X_test.shape[0]
        y_predict = np.zeros(test_row, dtype=self.y_train.dtype)

        for i in range(test_row):
            distances = np.sqrt(np.sum(np.square(self.X_train - X_test[i, :]), axis=1))  # Euclidean distance
            labels = self.__find_labels(distances)
            counter = self.__count_occurences(labels)
            label = self.__max_count_index(counter)
            y_predict[i] = label
            
        return y_predict
            
    def __find_labels(self, distances):
        labels = []
        for _ in range(self.k):
            min_index = np.argmin(distances)
            labels.append(self.y_train[min_index])
            distances[min_index] = np.inf 
        return labels
    
    def __count_occurences(self, labels):
        counter = {}
        for label in labels:
            counter[label] = counter.get(label, 0) + 1
        return counter

    def __max_count_index(self, counter):
        max_val = -1
        label = 0
        for key, value in counter.items():
            if value >= max_val:
                max_val = value
                label = key

        return label
    
    def accuracy_score(self, y_test, prediction):
        try:
            if y_test.shape[0] != prediction.shape[0]:
                raise Exception("Length of the y_test and prediction arrays are not match!!")
        except:
            print("Entered parameters should be in type of numpy array (np.array())!!")


        return np.mean(y_test == prediction) * 100
        



            


    