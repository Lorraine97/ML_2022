# Running the project 2 code files

## Set up environment

For this project, you will need to run pip install for the following packages:

- mlrose
- pandas
- numpy
- scikit-learn
- spicy
- seaborn
- matplotlib
- jupyter

And python version of 3.8 or higher.

## Running scripts

Before running the script, need to update mlrose package scripts where it uses `from sklearn.externals import six` to use `import six` instead. Only in this way it can properly runs the algorithms. 

The repository is accessible at: https://github.com/Lorraine97/ML_2022/tree/main/project_2

The `tutorial_examples.ipynb` is where you would run the examples. It is adopted from [mlrose tutorial](https://github.com/gkhayes/mlrose/blob/master/tutorial_examples.ipynb)
Note: all of the scripts here include both model training and plotting of the performance.

## References
Data files used are stored in the `data` folder, and images generated from running the scripts are stored under
each model's subdirectory in the `images` folder.