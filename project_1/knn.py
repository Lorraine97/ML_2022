import matplotlib.pyplot as plt
import shared

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# the following code refers sklearn documentation site and a tutorial on plotting learning curves:
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
class KNNAnalysis:
    
    def __init__(self, x_train, x_test, y_train, y_test, best_k=1):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.common_attributes = {
            'weights': 'distance',
            'p': 2  # euclidean_distance
        }
        self.knn_best = KNeighborsClassifier(n_neighbors=best_k, **self.common_attributes)
    
    def k_performance(self):
        train_scores, test_scores = [], []
        for k in range(1, 10):
            knn = KNeighborsClassifier(n_neighbors=k, **self.common_attributes)
            knn.fit(self.x_train, self.y_train)
            train_scores.append(knn.score(self.x_train, self.y_train))
            test_scores.append(knn.score(self.x_test, self.y_test))
        return train_scores, test_scores
    
    def plot_pruning(self):
        train_scores, test_scores = self.k_performance()
        fig, ax = plt.subplots()
        ax.set_xlabel("K")
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy vs K for training and testing sets")
        ax.plot(range(1, 10), train_scores, marker="o", label="train", drawstyle="steps-post")
        ax.plot(range(1, 10), test_scores, marker="o", label="test", drawstyle="steps-post")
        ax.legend()
        plt.show()


if __name__ == "__main__":
    tester1 = KNNAnalysis(x_train=shared.loan_x_train,
                          x_test=shared.loan_x_test,
                          y_train=shared.loan_y_train,
                          y_test=shared.loan_y_test,
                          best_k=5)
    
    tester1.plot_pruning()
    shared.learning_analysis(shared.loan_attr_data, shared.loan_status_data, tester1.knn_best)
    
    tester2 = KNNAnalysis(shared.math_x_train,
                          shared.math_x_test,
                          shared.math_y_train,
                          shared.math_y_test)
    
    tester2.plot_pruning()
    shared.learning_analysis(shared.math_attr_data, shared.math_status_data, tester2.knn_best)
