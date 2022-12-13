import numpy as np
import pandas as pd
from sklearn import model_selection


def create_column_filter(dataframe):
    """
    Input:
        df - a dataframe (where the column names "CLASS" and "ID" have special meaning)
    Output:
        df            - a new dataframe, where columns, except "CLASS" and "ID", containing only missing values 
                        or only one unique value (apart from the missing values) have been dropped
        column_filter - a list of the names of the remaining columns, including "CLASS" and "ID"

    """
    df = dataframe.copy()
    for column in df.columns:
        if column not in ["SMILES", "ACTIVE"]:
            if len(pd.unique(df[column].dropna())) <= 1:
                df.drop(column, axis = 1, inplace = True)
    return df, list(df.columns)

def preprocess_data():
    df = pd.read_csv('../data/features.csv')
    df.drop(['ACIVE'], axis = 1, inplace = True)

    print('Before Column filter, number of columns: ', len(df.columns))
    cols = set(df.columns)

    df, column_filter = create_column_filter(df)

    cols_after = set(df.columns)
    print('After Column filter, number of columns: ', len(df.columns))

    print('Columns dropped: ', cols - cols_after)

    df_train = df[df['ACTIVE'].notnull()].copy()
    df_test = df[df['ACTIVE'].isnull()].copy()


    # we create a new column called kfold and fill it with -1
    df_train['kfold'] = -1

    # the next step is to randomize the rows of the data
    df_train = df_train.sample(frac=1).reset_index(drop=True)

    # fetch labels
    y = df_train['ACTIVE'].values

    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill the new kfold column
    for fold, (trn_, val_) in enumerate(kf.split(X=df_train, y=y)):
        df_train.loc[val_, 'kfold'] = fold

    print('Fold Stats: ', df_train['kfold'].value_counts())
    print()

    print('Train shape: ', df_train.shape)
    print('Test shape: ', df_test.shape)

    print(df_train['ACTIVE'].value_counts())

    df_train.to_csv('../data/train_folds.csv', index = False)
    df_test.to_csv('../data/test_folds.csv', index = False)


if __name__ =='__main__':
    preprocess_data()