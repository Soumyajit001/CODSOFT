# Credit Card Fraud Detection

## Problem Statement
The problem statement chosen for this project is to predict fraudulent credit card transactions with the help of machine learning models.

Build a model to detect fraudulent credit card transactions. Use a dataset containing information about credit card transactions, and experiment with algorithms like Logistic Regression, Decision Trees, or Random Forests to classify transactions as fraudulent or legitimate.

## About Dataset

This is a simulated credit card transaction dataset containing legitimate and fraud transactions from the duration 1st Jan 2019 - 31st Dec 2020. It covers credit cards of 1000 customers doing transactions with a pool of 800 merchants.

https://www.kaggle.com/datasets/kartik2112/fraud-detection/data

## Source of Simulation

This was generated using Sparkov Data Generation | Github tool created by Brandon Harris. This simulation was run for the duration - 1 Jan 2019 to 31 Dec 2020. The files were combined and converted into a standard format.

## Project Pipeline

The project pipeline can be briefly summarized in the following four steps:

Data Understanding: Here, we need to load the data and understand the features present in it. This would help us choose the features that we will need for your final model.

Exploratory data analytics (EDA): Normally, in this step, we need to perform univariate and bivariate analyses of the data, followed by feature transformations, if necessary. However, we can check if there is any skewness in the data and try to mitigate it, as it might cause problems during the model-building phase.

Train/Test Split: Now we are familiar with the train/test split, which we can perform in order to check the performance of our models with unseen data.

Model-Building/Hyperparameter Tuning: This is the final step at which we can try different models and fine-tune their hyperparameters until we get the desired level of performance on the given dataset. We should try and see if we get a better model by the various sampling techniques.

Model Evaluation: We need to evaluate the models using appropriate evaluation metrics. Note that since the data is imbalanced it is is more important to identify which are fraudulent transactions accurately than the non-fraudulent. We need to choose an appropriate evaluation metric which reflects this business goal.
