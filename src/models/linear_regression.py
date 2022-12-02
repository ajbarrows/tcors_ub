import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from collections import namedtuple

def bin_prp_change(df, var_name, step):
    upper_limit = 2
    bins = np.arange(-1, upper_limit, step=step)
    bins = np.append(bins, np.inf)

     # construct variable based on prop_change
    df[var_name + "_bin"] = pd.cut(df[var_name], bins, include_lowest=True).astype("category")
    df[var_name + "_bin_label"] = df[var_name + "_bin"].cat.codes

    return df

def assess_clf_perf(preds, y_test, step):
    pred_df = pd.DataFrame(preds).reset_index(drop=True)
    pred_df.columns = ['prp_change_pred']
    pred_df = bin_prp_change(pred_df, 'prp_change_pred', step=step)

    actual_df = pd.DataFrame(y_test).reset_index(drop=True)
    actual_df = bin_prp_change(actual_df, 'prp_change', step=step)


    merged = pd.concat([actual_df, pred_df], axis=1)
    merged['success'] = merged['prp_change_bin_label'] == merged['prp_change_pred_bin_label']

    return merged


def extract_coefficients(mod, X_train):
    # prep = mod.named_steps['preprocessor']
    colnames = X_train.columns
    coef = mod.named_steps['regression'].coef_ 

    coef_df =  pd.concat([
        pd.DataFrame(colnames), 
        pd.DataFrame(coef.T)
        ], 
        axis=1
        )

    return coef_df

def gather_scores(cv_scores, test_score):
    d = {
        'mean_cv_r2': np.mean(cv_scores),
        'sd_cv_r2' : np.std(cv_scores),
        'test_r2': test_score
    }
    df = pd.DataFrame(d, index = [1])
    return df

def clf_metrics(clf_augment_df):

    vars = ['prp_change_bin_label', 'prp_change_pred_bin_label']
    df = clf_augment_df[vars]

    clf_rep = f1_score(
        y_true=df['prp_change_bin_label'], 
        y_pred=df['prp_change_pred_bin_label'],
        average = "micro"
    )
    return clf_rep

def fit_linear(X_train, X_test, y_train, y_test,  bin_step = 0.1, random_state=42):

    reg = Pipeline(
        steps = [("regression", LinearRegression())]
    )
    # estimate training performance
    cv_scores = cross_val_score(reg, X_train, y_train, cv = 5)

    # assess test set performance fit to all training data
    reg.fit(X_train, y_train)
    test_score = reg.score(X_test, y_test)

    # make prediction with test set
    preds = reg.predict(X_test)

    # gather regression scores
    avg_scores = gather_scores(cv_scores, test_score)
    
    # extract coefficients
    coefs = extract_coefficients(reg, X_train)

    # assess performance as though it were a classifier
    clf_augment_df = assess_clf_perf(preds, y_test, step = bin_step)
    clf_mets = clf_metrics(clf_augment_df)


    summary = namedtuple("summary", [
        "cv_scores", "test_score", "avg_scores",
        "mod", "preds", "coefs", "clf_augment_df", "clf_mets",
        "X_train", "X_test", "y_train", "y_test"
        ])

    return summary(
        cv_scores, 
        test_score,
        avg_scores, 
        reg, 
        preds,
        coefs,
        clf_augment_df,
        clf_mets,
        X_train,
        X_test,
        y_train,
        y_test)
