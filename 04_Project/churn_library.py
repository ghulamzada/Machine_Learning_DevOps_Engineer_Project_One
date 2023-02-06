"""
This file includes all functions for customer prediction churn
with random forest and logistic regression model.

Author: Taj Mohammad Ghulam Zada
Date: 05. Feb. 2023
"""

# import libraries
import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth
    input:
            pth: a path to the csv
    output:
            data_raw: pandas dataframe
    '''
    # Loading CSV-Data
    data_raw = pd.read_csv(pth)
    return data_raw


def perform_eda(data_df):
    '''
    perform eda on data and save figures to images folder
    input:
            data_df: pandas dataframe

    output:
            None
    '''
    # Creating y-variable depending on cloum"attribution_flag"
    data_df['Churn'] = data_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Histogram of y-variable
    plt.figure(figsize=(10, 5))
    data_df['Churn'].hist().figure.savefig('./images/eda/01_df_histogram.png')

    # Histogram of "Customer_Age" column
    plt.figure(figsize=(10, 5))
    data_df['Customer_Age'].hist().figure.savefig(
        './images/eda/02_df_customer_age.png')

    # Plot of "Marital_Status" column
    plt.figure(figsize=(10, 5))
    data_df.Marital_Status.value_counts('normalize').plot(
        kind='bar').figure.savefig('./images/eda/03_df_marital_status.png')

    # Histogram of "Total_Trans_Ct" column
    plt.figure(figsize=(10, 5))
    sns.histplot(
        data_df['Total_Trans_Ct'],
        stat='density',
        kde=True).figure.savefig('./images/eda/04_df_density.png')

    # Correlation plot
    plt.figure(figsize=(10, 5))
    sns.heatmap(data_df.corr(), annot=False, cmap='Dark2_r',
                linewidths=2).figure.savefig('./images/eda/05_df_heatmap.png')

    # Check the number of files including folder
    len(os.listdir('images/eda'))
    # Check only number of files
    num_files_saved = len(next(os.walk('./images/eda/'))[2])
    return data_df, num_files_saved


def categorical_list(data_cat):
    '''
    This function will create the categorical list to assist in the encoder_help funtion.

    Input:
    data_cat: pandas dataframe

    Output:
    categorical_list: names of columns that include the desired categorical variables
    '''
    # All categorical variables in DataFrame
    cat_columns_general = list(
        data_cat.select_dtypes(
            exclude='number').columns)
    # Few specified categorical variables in Dataframe
    cat_columns = [
        col for col in cat_columns_general if col != "Attrition_Flag"]

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    return cat_columns, keep_cols


def encoder_helper(data_encode, category_lst, response="Churn"):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_encode: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could
            be used for naming variables or index y column]

    output:
            data_encode: pandas dataframe with new columns for
    '''

    # looping through all columns with categorical variables
    for colname, colval in data_encode.iteritems():
        if colname in category_lst:
            # emtpy list
            empty_lst = []
            # grouping columns depending on each categorical variables and get
            # mean values
            cat_groups = data_encode.groupby(colname).mean()[response]
            # appending the "cat_groups" to "empty_list"
            for val in data_encode[colname]:
                empty_lst.append(cat_groups.loc[val])
            # adding the "empty_list" as a new column in dataframe
            data_encode[str(colname) + "_" + str(response)] = empty_lst
    return data_encode


def perform_feature_engineering(data_feat_eng, response):
    '''
    input:
              data_feat_eng: pandas dataframe
              response: string of response name [optional argument that could
              be used for naming variables or index y column]
    output:
              xtrain: x training data
              xtest: X testing data
              ytrain: y training data
              ytest: y testing data
    '''
    # y-variables
    target_variable = data_feat_eng['Churn']
    # specifying columns depending on new column-names
    dependent_variables = data_feat_eng[response]
    # Train-test-split
    xtrain, xtest, ytrain, ytest = train_test_split(
        dependent_variables,  # Training data
        target_variable,  # target values
        test_size=0.3,  # size of test dataset
        random_state=42)  # seed number for rondom mixture of data-points

    return xtrain, xtest, ytrain, ytest


def classification_report_image(ytrain,
                                ytest,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and
    stores report as image in images folder
    input:
            ytrain: training response values
            ytest: test response values
            ytrain_preds_lr: training predictions from logistic regression
            ytrain_preds_rf: training predictions from random forest
            ytest_preds_lr: test predictions from logistic regression
            ytest_preds_rf: test predictions from random forest
    output:
             None
    '''

    # scores
    # random forest results: test and train results
    plt.rc('figure', figsize=(10, 5))
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(ytrain, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')    # approach improved by OP -> monospace!

    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach

    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(ytest, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    # save the random forest plot
    plt.savefig('images/results/01_random_forest_test_and_train_result.png')

    # logistic regression: test results
    plt.figure(figsize=(5, 5))
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(ytest, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!

    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(ytrain, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    # save the logistic regression plot
    plt.savefig(
        'images/results/02_logistic_regression_test_and_train_result.png')


def feature_importance_plot(model, dependent_variables, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            dependent_variables: pandas dataframe of dependent_variables values
            output_pth: path to store the figure
    output:
             None
    '''

    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [dependent_variables.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(dependent_variables.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(dependent_variables.shape[1]), names, rotation=90)
    plt.savefig(output_pth + "feature_importance.png")


def train_models(xtrain, xtest, ytrain, ytest, dependent_variables):
    '''
    train, store model results: images + scores, and store models
    input:
              xtrain: x training data
              xtest: x testing data
              ytrain: y training data
              ytest: y testing data
    output:
              None
    '''
    # Random forest classifier
    rfc = RandomForestClassifier(random_state=42)
    # Logistic regression classifier
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    # parameter for model training
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # grid search
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    # fitting random forest using grid search
    cv_rfc.fit(xtrain, ytrain)

    lrc.fit(xtrain, ytrain)
    #rfc.fit(xtrain, ytrain)

    # Prediction of training data from with forest model
    y_train_preds_rf = cv_rfc.best_estimator_.predict(xtrain)
    # Prediction of test data from with forest model
    y_test_preds_rf = cv_rfc.best_estimator_.predict(xtest)

    # Prediction of training data from with forest model
    y_train_preds_lr = lrc.predict(xtrain)
    # Prediction of test data from with forest model
    y_test_preds_lr = lrc.predict(xtest)
    
    # create ROC plot
    plt.figure(figsize=(15, 8))
    lrc_plot = plot_roc_curve(lrc, xtest, ytest)
    plt.savefig("images/results/ROC_logistic_regression.png")
    plt.figure(figsize=(15, 8))
    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, xtest, ytest)
    plt.savefig("images/results/ROC_random_forest.png")

    # Create and save classification report
    classification_report_image(ytrain,
                                ytest,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    feature_importance_plot(cv_rfc,
                            dependent_variables,
                            "images/results/")
    # save best model combination
    # save best random forest model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    # save best logistic regression model
    joblib.dump(lrc, './models/logistic_model.pkl')


if __name__ == "__main__":
    DATA, NUM_FILES_SAVED = perform_eda(import_data("./data/bank_data.csv"))
    CAT_LIST, KEEP_COL = categorical_list(DATA)
    ENCODED_DATA = encoder_helper(DATA, CAT_LIST, "Churn")
    DEP_VAR = DATA[KEEP_COL]
    XTRAIN, XTEST, YTRAIN, YTEST = perform_feature_engineering(
        ENCODED_DATA,
        KEEP_COL)
