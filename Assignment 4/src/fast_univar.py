from fastai.tabular.all import *
import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm import tqdm


from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest


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
        if column not in ["SMILES", "ACTIVE", "ID"] and imputation.get(column) is not None:
                df.loc[:, column] = df[column].fillna(imputation[column])
    return df


def load_train(fold):
    df = pd.read_csv('../data/train_folds.csv', index_col = 0)

    selection = SelectKBest(f_classif, k=200)

    # Train data

    X_train = df[df.kfold != 0].reset_index().copy()
    X_train = X_train.drop(['kfold'], axis=1)
    X_train['ACTIVE'] = X_train['ACTIVE'].astype('category')
    X_train, imputation = create_imputation(X_train)

    y_train = X_train['ACTIVE']
    x_train = X_train.drop(['ACTIVE'], axis = 1)

    selection.fit(x_train, y_train)
    x_train = selection.transform(x_train)
    X_train = pd.DataFrame(x_train, columns = [str(i) for i in range(x_train.shape[1])])
    X_train['ACTIVE'] = y_train

    # valid data



    X_valid = df[df.kfold == 0].reset_index().copy()
    X_valid = X_valid.drop(['kfold'], axis=1)
    X_valid = apply_imputation(X_valid, imputation)
    X_valid['ACTIVE'] = X_valid['ACTIVE'].astype('category')


    y_valid = X_valid['ACTIVE']
    x_valid = X_valid.drop(['ACTIVE'], axis = 1)


    x_valid = selection.transform(x_valid)
    X_valid = pd.DataFrame(x_valid, columns = [str(i) for i in range(x_valid.shape[1])])
    X_valid['ACTIVE'] = y_valid

    
    return X_train, X_valid



def train(fold):
    print('\nLoading data...')
    X_train, X_valid = load_train(fold)
    print('Data loaded...')



    splits = RandomSplitter(valid_pct=0.2)(range_of(X_train))

    to = TabularPandas(X_train, procs=[Categorify ,FillMissing, Normalize],
                    cont_names = [f for f in X_train.columns if f not in ['ACTIVE']], 
                    y_names='ACTIVE',
                    splits=splits,
                    y_block=CategoryBlock())

    dls = to.dataloaders(bs=64)

    roc_auc = RocAucBinary()
    learn = tabular_learner(dls, metrics=[roc_auc])

    learn.fit_one_cycle(20)

   
    dl = learn.dls.test_dl(X_valid)

    predictions = learn.get_preds(dl=dl)[0][:, 1]

    auc = metrics.roc_auc_score(X_valid.ACTIVE.values, predictions)

    print(f'Fold = {fold}, AUC = {auc}')

if __name__ == '__main__':
    for fold in tqdm(range(5)):
        train(fold)