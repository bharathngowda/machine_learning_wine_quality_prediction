
# Red Wine Quality Prediction

![App Screenshot](https://github.com/bharathngowda/machine_learning_wine_quality_prediction/blob/main/1_oXOppV_n4hv-tf8Y96QHxQ.png)

### Table of Contents

1. [Problem Statement](#Problem-Statement)
2. [Data Pre-Processing](#Data-Pre-Processing)
3. [Exploratory Data Analysis](#Exploratory-Data-Analysis)
4. [Model Training](#Model-Building)
5. [Model Selection](#Model-Selection)
6. [Model Evaluation](#Model-Evaluation)
7. [Dependencies](#Dependencies)
8. [Installation](#Installation)

### Problem Statement

The goal of the project is to predict the quality of the wine as good or bad based on the features in the dataset.

**Quick Start:** [View](https://github.com/bharathngowda/machine_learning_wine_quality_prediction/blob/main/Wine%20Quality%20Prediction.ipynb) a static version of the notebook in the comfort of your own web browser

### Data Pre-Processing

- Loaded the train and test data
- Splitting the dataset to test and train set as there is no separate test set.
- Checkig for null values


### Exploratory Data Analysis

- **Correlation Plot** to determine if there is a linear relationship between the 2 variables and the strength of the relationship
- Checking how each feature affects the quality of the wine
- checking if the data set is balanced or not

### Model Training

Models used for the training the dataset are - 

- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Linear Discriminant Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
- [Linear Support Vector Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC)
- [Bagging](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)
- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)


### Model Selection

Since the dataset is imbalanced, I have used [f1 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) as my scorer and used [k-fold cross-validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)
to select the model with highest **'f1 score'**.

### Model Evaluation

I fit the final model on the train data and predicted the wine quality for the test data and obtained the below result-

| Metric        | Score    |
| :--------     | :------- |
| F1 Score	    |0.809035  |
| Precision	    |0.831224  |
| Recall	    |0.788000  |
| Accuracy	    |0.806250  |


Confusion Matrix and ROC Curve

![App Screenshot](https://github.com/bharathngowda/machine_learning_wine_quality_prediction/blob/main/ROC%20and%20CM.PNG)

### Dependencies
* [NumPy](http://www.numpy.org/)
* [IPython](http://ipython.org/)
* [Pandas](http://pandas.pydata.org/)
* [SciKit-Learn](http://scikit-learn.org/stable/)
* [SciPy](http://www.scipy.org/)
* [Matplotlib](http://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)

### Installation

To run this notebook interactively:

1. Download this repository in a zip file by clicking on this [link](https://github.com/bharathngowda/machine_learning_wine_quality_prediction/archive/refs/heads/main.zip) or execute this from the terminal:
`git clone https://github.com/bharathngowda/machine_learning_wine_quality_prediction.git`

2. Install [virtualenv](http://virtualenv.readthedocs.org/en/latest/installation.html).
3. Navigate to the directory where you unzipped or cloned the repo and create a virtual environment with `virtualenv env`.
4. Activate the environment with `source env/bin/activate`
5. Install the required dependencies with `pip install -r requirements.txt`.
6. Execute `ipython notebook` from the command line or terminal.
7. Click on `Wine Quality Prediction.ipynb` on the IPython Notebook dasboard and enjoy!
8. When you're done deactivate the virtual environment with `deactivate`.
