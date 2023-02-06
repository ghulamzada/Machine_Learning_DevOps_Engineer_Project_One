# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity
– Student: Taj Mohammad Ghulam Zada
– Date of submission: 05. Feb. 2023

## Project Description
The purpose of this project is to predict the customer churn using machine learning model.
For that purpose, two models were used: Random forest and logistic regression classification algorithyms. Following libraries are used for this project:
– os
– sklearn
– joblib
– pandas
– numpy
– matplotlib
– seaborn


## Files and data description
Following files are used for this project:
**churn_library.py**: This files include all the required functions for this project. These functions are shortly described as follow:

– import_data(): Imports CSV-File
– perform_eda(): Saved different statiscall plots from the dataframe
– categorical_list(): Outputs only the desired names of categorical columns as a list
– encoder_helper(): Converts all categorical columns into a numerical values.
– perform_feature_engineering(): Applies train-test-split on the dataset
– classification_report_image(): Saves the result of trained models as an image
– feature_importance_plot(): saves the feature importance plots
– train_models(): Trains the classification models.


**test_and_logging_churn_script.py**: This files include all tests and log files including the infos and erros for this project. Each function mentioned above will be test here.

**churn_notebook.ipynb**: This files include all codes during the development of this project as well as completed additional funtions that might an alternative solution for some functions. The file result of this file is in **churn_library.py**.

**Guide.ipynb**: This files shows the introduction as well the guidance on how to complete this project and pass it successfully.


**requirements_py3.6.txt and requirements_py3.8**: This text file includes all the required libraries including their respective version that are needed for execution of this project.

**data**: This folder includes the raw CSV-File.
**images**: This folder all the saved plots and model classification report as an image.
**logs**: This folder includes the result of unit-test with pytest.
**models**: This folder include the saved model with random foret and logistic regression algorithym.


## Running Files
The files can be run in terminal with "python nameOfFile.py" or seperately. It should be mentioned that the duration of file execution (specially model training) takes around 20 minutes or more depending our your PC.

– To test the functions in "churn_library.py" and check the result of unit-test (using pytest), just type "pytest test_and_logging_churn_script.py" on terminal. Make sure your are alread in project folder.



