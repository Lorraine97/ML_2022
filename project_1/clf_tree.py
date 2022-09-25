import matplotlib.pyplot as plt
import shared

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# the following code refers sklearn documentation site and a tutorial on pruning:
# https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
# https://www.kaggle.com/code/arunmohan003/pruning-decision-trees-tutorial
class DecisionTreeAnalysis:
    def __init__(self, x_train, x_test, y_train, y_test, random_state=0):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.best_alpha = 0
        self.random_state = random_state
        self.init_clf = DecisionTreeClassifier(random_state=random_state)
        self.best_clf = self.init_clf
        init_path = self.init_clf.cost_complexity_pruning_path(self.x_train, self.y_train)
        self.alphas = init_path.ccp_alphas
    
    def pruning_performance(self):
        train_scores, test_scores = [], []
        for alpha in self.alphas:
            clf = DecisionTreeClassifier(random_state=self.random_state, ccp_alpha=alpha)
            clf.fit(self.x_train, self.y_train)
            train_scores.append(clf.score(self.x_train, self.y_train))
            train_scores.append(clf.score(self.x_test, self.y_test))
        return train_scores, test_scores
    
    def plot_pruning(self):
        train_scores, test_scores = self.pruning_performance()
        fig, ax = plt.subplots()
        ax.set_xlabel("alpha")
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy vs alpha for training and testing sets")
        ax.plot(self.alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
        ax.plot(self.alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
        ax.legend()
        plt.show()

    def evaluate_predict(self, model):
        y_train_pred = model.predict(self.x_train)
        y_test_pred = model.predict(self.x_test)
        print(f'Train score {accuracy_score(y_train_pred, self.y_train)}')
        print(f'Test score {accuracy_score(y_test_pred, self.y_test)}')
        return y_train_pred, y_test_pred

    def analysis_for_best(self, best_alpha=None):
        if best_alpha:
            self.best_alpha = best_alpha
            self.best_clf = DecisionTreeClassifier(random_state=self.random_state, ccp_alpha=self.best_alpha)
        self.best_clf.fit(self.x_train, self.y_train)
        print(f"Current tree has {self.best_clf.tree_.node_count} nodes, with alpha: {self.best_alpha}")
        y_train_pred, y_test_pred = self.evaluate_predict(self.best_clf)
        shared.plot_confusionmatrix(y_train_pred, self.y_train, title='Train')
        shared.plot_confusionmatrix(y_test_pred, self.y_test, title='Test')


if __name__ == "__main__":
    tester1 = DecisionTreeAnalysis(x_train=shared.loan_x_train,
                                   x_test=shared.loan_x_test,
                                   y_train=shared.loan_y_train,
                                   y_test=shared.loan_y_test)
    tester1.plot_pruning()
    tester1.analysis_for_best()
    tester1.analysis_for_best(best_alpha=0.005)
    shared.learning_analysis(shared.loan_attr_data, shared.loan_status_data, tester1.best_clf)

    tester2 = DecisionTreeAnalysis(shared.math_x_train,
                                   shared.math_x_test,
                                   shared.math_y_train,
                                   shared.math_y_test)
    tester2.plot_pruning()
    tester2.analysis_for_best()
    tester2.analysis_for_best(best_alpha=0.012)
    shared.learning_analysis(shared.math_attr_data, shared.math_status_data, tester2.best_clf)
