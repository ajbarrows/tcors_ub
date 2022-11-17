### TCORS UB Analysis Functions

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

def load_data(filename = "*.csv", path = '../../data/clean/' ):
    ''' Load cleaned data files '''
    filepath = path + filename
    df = pd.read_csv(filepath)

    return df

def make_training_split(df: pd.DataFrame, y: list, test_size = 0.2, random_state=42):
    ''' Takes dataframe for analysis. Specify dependent outcomes
    as list of column names. All others are included as predictors.'''

    y = df[y]
    X = df.drop(y, axis=1)

    return train_test_split(X, y)


