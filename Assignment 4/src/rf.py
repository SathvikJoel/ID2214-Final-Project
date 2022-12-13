import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn import ensemble
from sklearn import metrics
from tqdm import tqdm

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

 

if __name__ == '__main__':
    df = pd.read_csv('../data/train_folds.csv')

    df, imputation = create_imputation(df)

    X = df[df['kfold'] != 0].drop(['kfold', 'ACTIVE'], axis=1).values
    y = df[df['kfold'] != 0].ACTIVE.values

    X_test = pd.read_csv('../data/test_folds.csv')
    X_test = X_test.drop(['ACTIVE'], axis=1).values


    clf = ensemble.RandomForestClassifier(n_jobs=-1)
    param_grid = {
            "n_estimators": [100, 200, 400, 300],
            "max_depth": [5, 25, 30, None],
            "max_features": ["sqrt", "log2", None],
            "criterion" : ['gini', 'entropy']
        }
    model = model_selection.RandomizedSearchCV(
        estimator=clf,
        param_distributions = param_grid,
        n_iter = 50,
        scoring = 'roc_auc',
        verbose = 10,
        n_jobs = 1,
        cv = 2
    )
    model.fit(X, y)
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    best_model = model.best_estimator_

    best_model.fit(X, y)
    preds = best_model.predict_proba(X_test)[:, 1]
    fold = 0
    x_valid = df[df.kfold == fold].drop(['kfold', 'ACTIVE'], axis=1).values
    y_valid = df[df.kfold == fold].ACTIVE.values
    valid_preds = best_model.predict_proba(x_valid)[:, 1]

    auc = metrics.roc_auc_score(y_valid, valid_preds)

    print(f"Fold = {fold}, AUC = {auc}")

