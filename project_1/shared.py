import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import learning_curve, ShuffleSplit


# Start to parse loan eligibility data
loan_data = pd.read_csv('data/loan_eligibility.csv')

loan_attr_data = loan_data.loc[:, loan_data.columns != 'Loan_Status']
loan_status_data = loan_data['Loan_Status']
loan_x_train, loan_x_test, loan_y_train, loan_y_test = train_test_split(loan_attr_data, loan_status_data, random_state=0)


# Start to parse Math performance data
math_data = pd.read_csv('data/Maths.csv')

school_dict = {"GP": 1, "MS": 2}
gender_dict = {"F": 0, "M": 1}
famsize_dict = {"GT3": 1, "LE3": 2}
addr_dict = {"U": 1, "R": 2}
Pstatus_dict = {"A": 1, "T": 2}
job_dict = {"at_home": 1, "health": 2, 'other': 3, 'services': 4, 'teacher': 5}
reason_dict = {"home": 1, "reputation": 2, 'course': 3, 'other': 4}
guardian_dict = {"mother": 0, "father": 1, 'other': 2}
bool_dict = {"yes": 1, "no": 0}
math_data['school'] = math_data['school'].apply(lambda x: school_dict[x])
math_data['sex'] = math_data['sex'].apply(lambda x: gender_dict[x])
math_data['famsize'] = math_data['famsize'].apply(lambda x: famsize_dict[x])
math_data['address'] = math_data['address'].apply(lambda x: addr_dict[x])
math_data['Pstatus'] = math_data['Pstatus'].apply(lambda x: Pstatus_dict[x])
math_data['Mjob'] = math_data['Mjob'].apply(lambda x: job_dict[x])
math_data['Fjob'] = math_data['Fjob'].apply(lambda x: job_dict[x])
math_data['reason'] = math_data['reason'].apply(lambda x: reason_dict[x])
math_data['guardian'] = math_data['guardian'].apply(lambda x: guardian_dict[x])
math_data['schoolsup'] = math_data['schoolsup'].apply(lambda x: bool_dict[x])
math_data['famsup'] = math_data['famsup'].apply(lambda x: bool_dict[x])
math_data['paid'] = math_data['paid'].apply(lambda x: bool_dict[x])
math_data['activities'] = math_data['activities'].apply(lambda x: bool_dict[x])
math_data['nursery'] = math_data['nursery'].apply(lambda x: bool_dict[x])
math_data['higher'] = math_data['higher'].apply(lambda x: bool_dict[x])
math_data['internet'] = math_data['internet'].apply(lambda x: bool_dict[x])
math_data['romantic'] = math_data['romantic'].apply(lambda x: bool_dict[x])
math_data['G3'] = math_data['G3'].apply(lambda x: 0 if x < 16 else 1)


math_attr_data = math_data.loc[:, math_data.columns != 'G3']
math_status_data = math_data['G3']

math_x_train, math_x_test, math_y_train, math_y_test = train_test_split(math_attr_data, math_status_data, random_state=0)


def plot_confusionmatrix(y_train_pred, y_train, title):
    print(f'{title} Confusion matrix')
    cf = confusion_matrix(y_train_pred, y_train)
    sns.heatmap(cf, annot=True, yticklabels=['No', 'Yes']
                , xticklabels=['No', 'Yes'], cmap='Blues', fmt='g')
    plt.tight_layout()
    plt.show()


def learning_analysis(x_data, y_data, mmodel=None):
    """
    The following code is adopted from sklearn:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    """
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    _, axes = plt.subplots(1, 3, figsize=(20, 5))
    total_sample = x_data.shape[0]
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator=mmodel,
        X=x_data,
        y=y_data,
        cv=cv,
        train_sizes=[1, int(total_sample/4), int(total_sample/2), int(total_sample/1.5)],
        scoring='accuracy',
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")
    plt.show()
    return plt

