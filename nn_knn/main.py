from nearest_neighbor import NearestNeighbor
from k_nearest_neighbor import KNearestNeighbor
import cifar10
import numpy as np

def get_result_model(model, X_train, y_train, X_test, y_test):
    model.train(X_train, y_train)
    prediction = model.predict(X_test)
    score = model.accuracy_score(y_test, prediction)
    print(f"ACCURACY SCORE FOR {model}: {score}")

def menu():
    X_train = np.random.rand(1000, 8)  # 50 samples, each with 10 features
    y_train = np.random.randint(0, 2, size=1000)  # Random binary labels for training data

    X_test = np.random.rand(1000, 8)  # 10 samples for testing
    y_test = np.random.randint(0, 2, size=1000)  # Random binary labels for testing data

    nn_model = NearestNeighbor()
    knn_model = KNearestNeighbor(3)

    models = [nn_model, knn_model]

    for model in models:
        get_result_model(model, X_train, y_train, X_test, y_test)
        print(" ")


if __name__ == "__main__":
    menu()









