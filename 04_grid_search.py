from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from xgboost import XGBRegressor
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import json


DATADIR = Path('../Data/')
USE_CACHED_RFE = False

class NaiveRegressor(BaseEstimator, RegressorMixin):
    """ Naive estimator that predicts mean/median of y_train for all values of y_test.
    """
    
    def __init__(self, use_mean=False):
        self.use_mean = use_mean
        
    def fit(self, X, y):
        # Make sure X, y have correct shapes
        X, y = check_X_y(X, y)
        
        self.X_ = X
        self.y_ = y
        
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        avg = np.median(self.y_) if not self.use_mean else np.mean(self.y_)
        preds = np.full(X.shape[0], avg)
        
        return preds


def markov_stacking(df, length=7):
    """ Creates `length` new features for each column, corresponding to time lag.
    Excludes columns with `target` and `is_pure` in their name.
    """
    _df = df.copy()
    numeric_cols = _df.select_dtypes(np.number).columns
    for c in [col for col in numeric_cols if ('target' not in col and 'is_pure' not in col)]:
        for i in range(1, length+1):
            _df[f'{c}_-{i}day'] = _df[c].shift(i)
    return _df


def custom_split(df: pd.DataFrame, n_test=40, replacement=False):
    """ Train-test split with special properties
    TODO: stratify on hemisphere?
    """

    # Select only the 'pure' observations (minimal imputation)
    valid_for_test = df[df['is_pure'] == 1].dropna(axis=1)

    # Sample some specified number of values from 'pure' observations as test set
    test = valid_for_test.sample(n_test, replace=replacement)
    # Get the indices of the test set observations
    test_ind = test.index
    # Subset the observations that are NOT in the test set
    remaining = df.copy().drop(index=test_ind, axis=0)
    # Ensure the same columns
    remaining = remaining[test.columns]
    train_full = remaining
    train = remaining.dropna(axis=0)

    # Drop the "pure" marker
    train = train.drop('is_pure', axis=1)
    test = test.drop('is_pure', axis=1)
    train_full = train_full.drop('is_pure', axis=1)

    return train, train_full, test

if __name__ == '__main__':
    # Load datasets into a dictionary of dataframes
    datasets = {}
    for path in (DATADIR/'Trainable/').glob('pre_markov*'):
        _df = pd.read_csv(path)
        datasets[path.stem] = _df.copy()


    # Apply "Markov stacking" and saving with updated names
    current_names = list(datasets.keys())
    current_datasets = list(datasets.values())
    for dataset_name, _df in zip(current_names, current_datasets):
        _df_stacked = markov_stacking(_df)
        print("Stacking", dataset_name, _df.shape, "->", _df_stacked.shape)
        new_name = '_'.join(dataset_name.split('_')[2:])
        datasets[new_name] = _df_stacked

    # Configure grid search parameters
    results = defaultdict(list)
    n_features = [*[i for i in range(2, 20)], *[i for i in range(20, 200, 10)]]
    n_repeats = 10
    algorithms = [DecisionTreeRegressor,
                  RandomForestRegressor, 
                  XGBRegressor, 
                  Lasso, 
                  Ridge, 
                  NaiveRegressor,
                 ]

    if not USE_CACHED_RFE:
        # Do an initial RFE (for each dataset) on a random split and save the ordering
        feature_ranks = {}
        for dataset_name, df in tqdm(datasets.items(), desc='RFE'):
            try:
                train, train_full, test = custom_split(
                    df, n_test=100, replacement=True)
                X_train, y_train = train.drop('target', axis=1), train['target']
                X_test, y_test = test.drop('target', axis=1), test['target']

                # Scale train and test separately (retaining column names)
                X_train = pd.DataFrame(StandardScaler().fit_transform(
                    X_train), columns=X_train.columns)
                X_test = pd.DataFrame(StandardScaler().fit_transform(
                    X_test), columns=X_test.columns)

                rfe = RFE(DecisionTreeRegressor(),
                          n_features_to_select=1).fit(X_train, y_train)

                df_feats = pd.DataFrame.from_dict({X_train.columns[i]: v for i, v in enumerate(
                    rfe.ranking_)}, orient='index', columns=['ranking'])
                ranked_feats = list(df_feats.sort_values(by='ranking').index.values)
                feature_ranks[dataset_name] = ranked_feats
            except Exception as e:
                print(dataset_name, e)
                import ipdb; ipdb.set_trace()
        try:
            # Persist best features per dataset to JSON file
            with open(DATADIR/'rfe_features.json', 'w') as outfile:
                json.dump(feature_ranks, outfile)
        except Exception as e:
            print(e)
    else:
        with open(DATADIR/'rfe_features.json', 'r') as infile:
            feature_ranks = json.load(infile)


    for repeat in tqdm(range(n_repeats), desc='Repeats'):

        for dataset_name, df in datasets.items():
            # Use custom train-test split function
            train, train_full, test = custom_split(df)
            X_train, y_train = train.drop('target', axis=1), train['target']
            X_train_full, y_train_full = train_full.drop(
                'target', axis=1), train_full['target']
            X_test, y_test = test.drop('target', axis=1), test['target']

            # Scale train and test separately (retaining column names)
            X_train = pd.DataFrame(StandardScaler().fit_transform(
                X_train), columns=X_train.columns)
            X_train_full = pd.DataFrame(StandardScaler().fit_transform(
                X_train_full), columns=X_train_full.columns)
            X_test = pd.DataFrame(StandardScaler().fit_transform(
                X_test), columns=X_test.columns)

            if X_train.isna().sum().sum() > 0 or X_test.isna().sum().sum() > 0:
                import ipdb
                ipdb.set_trace()

            for n_feats in n_features:
                _feats = feature_ranks[dataset_name][:n_feats]
                _X_train_full = X_train_full[_feats]
                _X_train = X_train[_feats]
                _X_test = X_test[_feats]

                for alg in algorithms:
                    mod = alg()
                    try:
                        if 'nan' in dataset_name and alg == XGBRegressor:
                            preds = mod.fit(_X_train_full.values,
                                            y_train_full.values).predict(_X_test.values)
                            train_shape = _X_train_full.shape
                        else:
                            preds = mod.fit(_X_train.values, y_train.values).predict(
                                _X_test.values)
                            train_shape = _X_train.shape

                        score = mean_squared_error(y_test, preds, squared=False)

                        # Save results
                        results['dataset'].append(dataset_name)
                        results['algorithm'].append(alg)
                        results['n_features'].append(n_feats)
                        results['train_examples'].append(train_shape)
                        results['RMSE'].append(score)

                        # Save intermediate results to csv
                        df_res = pd.DataFrame(results)
                        df_res.to_csv(f'../checkpoints/latest_results.csv')
                    except Exception as e:
                        print(e)
                        import ipdb
                        ipdb.set_trace()


    now = datetime.now().strftime("%m-%d-%H_%M_%S")
    df_res = pd.DataFrame(results)
    df_res.to_csv(f'../results/{now}.csv')
    print(df_res)
