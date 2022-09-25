import shared

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# the following code refers sklearn documentation site and a tutorial on plotting learning curves:
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html
# https://www.dataquest.io/blog/learning-curves-machine-learning/
class SVMAnalysis:
    # C-Support Vector Classification
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.rbf = SVC(gamma='scale', kernel='rbf')
        self.sig = SVC(gamma='scale', kernel='sigmoid')
    
    def result_analysis(self):
        self.rbf.fit(self.x_train, self.y_train)
        y_train_pred = self.rbf.predict(self.x_train)
        y_test_pred = self.rbf.predict(self.x_test)
        print(f'(RBG) Train score {accuracy_score(y_train_pred, self.y_train)}')
        print(f'(RBG) Test score {accuracy_score(y_test_pred, self.y_test)}')
        shared.plot_confusionmatrix(y_train_pred, self.y_train, title='Train')
        shared.plot_confusionmatrix(y_test_pred, self.y_test, title='Test')
        
        self.sig.fit(self.x_train, self.y_train)
        y_train_pred = self.sig.predict(self.x_train)
        y_test_pred = self.sig.predict(self.x_test)
        print(f'(SIG) Train score {accuracy_score(y_train_pred, self.y_train)}')
        print(f'(SIG) Test score {accuracy_score(y_test_pred, self.y_test)}')
        shared.plot_confusionmatrix(y_train_pred, self.y_train, title='Train')
        shared.plot_confusionmatrix(y_test_pred, self.y_test, title='Test')


if __name__ == "__main__":
    tester1 = SVMAnalysis(x_train=shared.loan_x_train,
                          x_test=shared.loan_x_test,
                          y_train=shared.loan_y_train,
                          y_test=shared.loan_y_test)
    
    tester1.result_analysis()
    shared.learning_analysis(shared.loan_attr_data, shared.loan_status_data, tester1.rbf)
    shared.learning_analysis(shared.loan_attr_data, shared.loan_status_data, tester1.sig)
    
    tester2 = SVMAnalysis(shared.math_x_train,
                          shared.math_x_test,
                          shared.math_y_train,
                          shared.math_y_test)
    
    tester2.result_analysis()
    shared.learning_analysis(shared.math_attr_data, shared.math_status_data, tester2.rbf)
    shared.learning_analysis(shared.math_attr_data, shared.math_status_data, tester2.sig)
