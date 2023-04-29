
# Loan Approval Project

This Python code is a machine learning-based system for predicting loan approval. It uses the following libraries:

- pandas
- numpy
- scikit-learn

## Installation

To run this code, you will need to install the following dependencies:

- Python 3.5+
- pandas
- numpy
- scikit-learn

You can install these dependencies using pip. Here's how:

```
pip install pandas
pip install numpy
pip install scikit-learn
```

## Usage

To use this system, you will need to provide it with a dataset of loan applications and their labels (Y i.e. approved or N i.e. rejected). You can then use the system to train a machine learning model on the data, and use the model to predict the labels of new loan applications.

Here's how to use the code:

1. Load the dataset into a Pandas dataframe using the `pd.read_excel()` function.
2. Preprocess the data as required, including any missing value imputation, encoding categorical variables, scaling numerical features and splitting the data into training and testing sets using the `train_test_split()` function.
3. Train a Random Forest Classifier model on the training data using the `fit()` function.
4. Use the model to predict the labels of the test data using the `predict()` function.
5. Evaluate the accuracy of the model using the `accuracy_score()` and `classification_report()` functions.

## Data

To train and test the machine learning model, you will need a dataset of loan applications labeled as Y or N. The dataset should be in XLSX or CSV format, with one row per application and several columns containing information about the applicant, such as gender, Education, and credit history.

## Models

This code uses the scikit-learn library to implement the machine learning model. The model used in this code is a Random Forest Classifier, which is an ensemble learning method that uses multiple decision trees to make predictions.

## Evaluation

To evaluate the performance of the machine learning model, this code uses the accuracy score and classification report. The accuracy score measures the proportion of correct predictions, while the classification report provides a detailed breakdown of the model's performance on each class.

## Contributing

If you would like to contribute to this project, feel free to fork the repository and submit a pull request. We welcome contributions from anyone!


## Contact

If you have any questions or comments about this code, please contact yamsanikavya15@gmail.com 
