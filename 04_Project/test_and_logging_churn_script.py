'''
This file is for testing and logging of customer prediction to churn (churn_library.py)

Author: Taj Mohammad Ghulam Zada
Date: 05. Feb. 2023
'''

# Import libraries
import logging
import churn_library as cls

logging.basicConfig(
    filename='logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import

    Input: CSV Path

    Output: CSV-data loaded as dataframe
    '''

    try:
        data = cls.import_data("data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function

    Input: Dataframe
    Ouptut: EDA and saved plots
    '''
    try:
        data = cls.import_data("data/bank_data.csv")
        data, numFile = cls.perform_eda(data)
        assert numFile > 4
        logging.info("Testing EDA plots: SUCCESS")

    except NameError as err:
        logging.error("Testing EDA plots: No dataframe found")
        raise err

    except AssertionError as err:
        logging.error("Testing EDA plots: Plots not saved")
        raise err


def test_encoder_helper():
    '''
    test encoder helper
    Input: Dataframe with categorical variables

    Output: Dataframe with encoded categorical variabes
    '''
    try:
        data = cls.import_data("data/bank_data.csv")
        data, numFile = cls.perform_eda(data)
        catColumns, keepCol = cls.categorical_list(data)
        data = cls.encoder_helper(data, catColumns, response="Churn")
        logging.info("Testing loading data for encoding: SUCCESS")

    except AssertionError as err:
        logging.error("Testing loading data for encoding: ERROR")
        raise err

    try:
        assert data.shape[1] == 28
        logging.info("Testing dataframe shape after encoding: SUCCESS")

    except AssertionError as err:
        logging.error(
            "Testing dataframe shape after encoding: Categorical columns aren't encoded properly")
        raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    Input: Dataframe cleaned
    Output: Train-test-split variables
    '''
    try:
        data = cls.import_data("data/bank_data.csv")
        data, numFile = cls.perform_eda(data)
        catColumns, keepCol = cls.categorical_list(data)
        data = cls.encoder_helper(data, catColumns, response="Churn")
        logging.info("Loading data feature engineering: SUCCESS")

    except NameError as err:
        logging.error("Loading data feature engineering: ERROR")
        raise err
    try:
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
            data, "Churn")
        assert x_train.shape[0] == 7088
        assert x_test.shape[0] == 3039
        logging.info(
            "Testing dataframe shape after feature engineering: SUCCESS")

    except AssertionError as err:
        logging.error(
            "Testing dataframe shape after feature engineering: Dataset is not splitted corretly")
        raise err


def test_train_models():
    '''
    test train_models

    Input: Four Train-test-splitted variables
    Ouput: Saved models + feature importance plote + model summary
    '''

    try:
        data = cls.import_data("data/bank_data.csv")
        data, numFile = cls.perform_eda(data)
        catColumns, keepCol = cls.categorical_list(data)
        data = cls.encoder_helper(data, catColumns, response="Churn")
        dependent_variables = data[keepCol]
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
            data, keepCol)
        cls.train_models(
            x_train,
            x_test,
            y_train,
            y_test,
            dependent_variables)
        logging.info("Testing model training: SUCCESS")

    except AssertionError as err:
        logging.error("Testing model training: Error in model training")
        raise err


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
