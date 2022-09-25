import shared

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from clf_tree import DecisionTreeAnalysis


# the following code refers sklearn documentation site and a tutorial on pruning:
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier

class BaggedTreeAnalysis(DecisionTreeAnalysis):
    def __init__(self, x_train, x_test, y_train, y_test, random_state=0, alpha=0.0):
        super().__init__(x_train, x_test, y_train, y_test, random_state)
        self.bagged_tree = None
        self.best_alpha = alpha
    
    def set_bagged_tree(self):
        tree = DecisionTreeClassifier(random_state=self.random_state, ccp_alpha=self.best_alpha)
        self.bagged_tree = BaggingClassifier(tree, max_samples=0.3, max_features=0.8)
        self.bagged_tree.fit(self.x_train, self.y_train)
        self.evaluate_predict(model=self.bagged_tree)


if __name__ == "__main__":
    tester1 = BaggedTreeAnalysis(x_train=shared.loan_x_train,
                                 x_test=shared.loan_x_test,
                                 y_train=shared.loan_y_train,
                                 y_test=shared.loan_y_test,
                                 alpha=0.007)
    tester1.set_bagged_tree()
    shared.learning_analysis(shared.loan_attr_data, shared.loan_status_data, tester1.bagged_tree)
    
    tester2 = BaggedTreeAnalysis(x_train=shared.math_x_train,
                                 x_test=shared.math_x_test,
                                 y_train=shared.math_y_train,
                                 y_test=shared.math_y_test,
                                 alpha=0.014)
    tester2.set_bagged_tree()
    shared.learning_analysis(shared.math_attr_data, shared.math_status_data, tester2.bagged_tree)
