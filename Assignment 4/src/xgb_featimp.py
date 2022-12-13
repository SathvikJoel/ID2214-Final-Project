import pandas as pd
import numpy as np

import xgboost as xgb

from sklearn import metrics
from tqdm import tqdm

from sklearn import ensemble
from sklearn import feature_selection

def create_imputation(dataframe):
    df = dataframe.copy()
    imputation = {}
    for column in df.columns:
        # Numerical Columns replace with mean
        if column not in ["CLASS", "ID"] and (df[column].dtype == 'float64' or df[column].dtype == 'int64'):
            # in rare case of all values being NaN, replace with 0
            if len(pd.unique(df[column].dropna())) == 0:
                df.loc[:, column] = df[column].fillna(0) 
            

            imputation[column] = df[column].mean()
            df.loc[:, column] = df[column].fillna(df[column].mean())

        
        # Categorical Columns replace with mode
        elif column not in ["SMILES", "ACTIVE", "ID"] and (df[column].dtype == 'object' or df[column].dtype == 'category'):
            # in rare case of all values being NaN, replace with 
            if len(pd.unique(df[column].dropna())) == 0:
                if df[column].dtype == 'object':
                    df.loc[:, column] = df[column].fillna("")
                elif df[column].dtype == 'category':
                    df.loc[:, column] = df[column].fillna(df[column].cat.categories[0])
            
            imputation[column] = df[column].mode()[0]
            df.loc[:, column] = df[column].fillna(df[column].mode()[0])
            
    return df, imputation
# Input to apply_imputation:
# df         - a dataframe
# imputation - a mapping (dictionary) from column name to value that should replace missing values
#
# Output from apply_imputation:
# df - a new dataframe, where each missing value has been replaced according to the mapping
#
# Hint 1: First copy the input dataframe and modify the copy (the input dataframe should be kept unchanged)
#
# Hint 2: Consider using fillna
def apply_imputation(dataframe, imputation):
    df = dataframe.copy()
    for column in df.columns:
        if column not in ["CLASS", "ID"] and imputation.get(column) is not None:
                df.loc[:, column] = df[column].fillna(imputation[column])
    return df

def run(fold):
    df = pd.read_csv('../data/train_folds.csv')

    df['ACTIVE'] = df['ACTIVE'].astype('category')

    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)

    df_train, imputation = create_imputation(df_train)
    df_valid = apply_imputation(df_valid, imputation)

    
    y_train = df_train.ACTIVE.values
    x_train = df_train.drop(['ACTIVE'], axis = 1).values

    model = ensemble.RandomForestClassifier(n_estimators = 100, n_jobs=-1, verbose= 0)
    model.fit(x_train, y_train)

    # get the feature importances and take the first 200 features from the x_train
    selection = feature_selection.SelectFromModel(model, max_features = 200)
    selection.fit(x_train, y_train)

    x_train = selection.transform(x_train)


    y_valid = df_valid.ACTIVE.values
    x_valid = df_valid.drop(['ACTIVE'], axis = 1).values
    
    x_valid = selection.transform(x_valid)

    model = xgb.XGBC
    lassifier(n_jobs=-1)

    model.fit(x_train, y_train)

    valid_preds = model.predict_proba(x_valid)[:, 1]

    auc = metrics.roc_auc_score(y_valid, valid_preds)

    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    for fold_ in tqdm(range(5)):
        run(fold_)
