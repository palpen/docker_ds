import logging
import sys
import pathlib

import numpy as np
import pandas as pd
import feather

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger(__name__)

id_col = 'id'
target_col = 'target'
raw_num_cols = ['total_acc', 'annual_inc', 'installment', 'open_acc',
                'mort_acc', 'emp_title', 'fico_range_high', 'pub_rec',
                'fico_range_low', 'revol_util', 'issue_d', 'title',
                'earliest_cr_line', 'loan_amnt', 'pub_rec_bankruptcies',
                'dti', 'int_rate', 'revol_bal']
nom_cats_cols = ['addr_state', 'application_type',
                 'home_ownership', 'initial_list_status', 'purpose',
                 'term', 'verification_status', 'zip_code']
ord_cats_cols = ['emp_length', 'grade', 'sub_grade']
loan_years_training = [2013, 2014, 2015]
loan_years_oos = [2016]

def get_high_missing_cols(df, thresh=40):
    count = 0
    missing_cols = []
    num_records = len(df)
    for c in df.columns:
        perc_missing = (df[c].isna().sum() / num_records)*100
        if perc_missing > thresh:
            count += 1
            missing_cols.append(c)
    log.info(f"Num cols with more than {thresh}% missing: {count}")
    return missing_cols

if __name__ == '__main__':

    log.info("Loading data")
    df = pd.read_csv("/data/sample_data.csv")

    log.info("Cleaning id column")
    # The id column is a mixed type array (it is a pandas object, which allows
    # numerical values and strings to be mixed together)
    # We will first convert the entire array into a string,
    # then clean out the non-numeric elements from the column
    df['id'] = df['id'].astype(str)
    non_digit_mask = ~df['id'].str.contains("^[0-9]*$", na=False, regex=True)
    non_num_records = (df.
                       loc[non_digit_mask, 'id'].
                       str.
                       extract('\d: (\d+)').
                       values)
    df.loc[non_digit_mask, 'id'] = non_num_records
    df['id'] = df['id'].astype(float)

    log.info("Restrict to complete loans")
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]

    log.info("Create target variable")
    df['target'] = 1
    df['target'] = (df['target'].
                    where(df['loan_status'] == 'Charged Off', 0))
    df = df.drop('loan_status', axis=1)

    log.info("Exclude columns with a high amount of missing records")
    missing_cols = get_high_missing_cols(df, thresh=40)
    valid_cols = [c for c in df.columns if c not in missing_cols]
    df = df.loc[:,valid_cols]

    log.info("Use only variables available at loan application time"
             " during training")
    df = df[raw_num_cols + nom_cats_cols + ord_cats_cols + [target_col, id_col]]

    log.info("Convert dates to datetime object")
    dates_cols = ['earliest_cr_line', 'issue_d']
    for c in dates_cols:
            df[c] = pd.to_datetime(df[c], format='%b-%Y')

    log.info("Number of days between the issue date of a "
             "loan and the earliest reported credit")
    df['days_from_issue_to_earliest_cr'] = (df['issue_d'] -
            df['earliest_cr_line']).apply(lambda x: x.days)

    log.info("Convert object to categorical dtype")
    df[nom_cats_cols + ord_cats_cols] = \
            df[nom_cats_cols + ord_cats_cols].astype('category')
    # TODO !!! Use set_categories, ordered==True for ordinal categorical vars

    log.info("Seperate training set from testing by year")
    log.info(f"Training years: {loan_years_training}")
    log.info(f"Testing years: {loan_years_oos}")
    df_train = df[df['issue_d'].dt.year.isin(loan_years_training)]
    df_oos = df[df['issue_d'].dt.year.isin(loan_years_oos)]

    final_num_cols = df_train.select_dtypes(np.number).columns.tolist()
    final_num_cols.remove('id')
    final_num_cols.remove('target')

    log.info("Writing training and test data for modelling")
    feather.write_dataframe(df_train, '/data/train')
    feather.write_dataframe(df_oos, '/data/test')

    with open('/data/final_num_cols.txt', 'w') as f:
        for item in final_num_cols:
            f.write(f"{item}\n")

