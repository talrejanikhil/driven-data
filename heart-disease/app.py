import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate

from feature_selector import FeatureSelector


def eda(x):
    print(x.info())
    print(x.var())


def categorize_label(x):
    return x.astype('category')


def clean_data(df, use_fs=True):
    # convert object to categorical data
    string_labels = ['thal']
    df[string_labels] = df[string_labels].apply(categorize_label, axis=0)
    df = pd.get_dummies(df, drop_first=True)
    # drop some columns
    to_drop = ['fasting_blood_sugar_gt_120_mg_per_dl','slope_of_peak_exercise_st_segment']
    df.drop(to_drop, axis=1, inplace=True)
    # normalize high variance columns
    # high_variance_cols = ['resting_blood_pressure']
    # df[high_variance_cols] = np.log(df[high_variance_cols])
    # convert int to float
    # df = df.apply(lambda c : c.astype(float), axis=1)
    if use_fs:
        fs = FeatureSelector(data=df, labels=y)
        fs.identify_zero_importance(task='classification', eval_metric='auc',
                                    n_iterations=10, early_stopping=False)
        fs.plot_feature_importances(threshold=0.99, plot_n=14)
    # print(train_removed_all_once)
    # standard scaling
    # scaler = RobustScaler()
    # df[df.columns] = scaler.fit_transform(df[df.columns])
    # print(df.info())
    # print('\nFeature Selector analysis')
    # feature selector
    return df


def train_predict(algo_name, x, y, clf, params, create_submission=False):
    # grid search CV
    grid = GridSearchCV(clf, params, cv=5, iid=True)

    if create_submission:
        # fit and predict on actual test values
        grid.fit(x, y)
        test_df = pd.read_csv('test_values.csv')
        patient_ids = test_df.patient_id.values
        x_sub = clean_data(test_df.drop('patient_id', axis=1), use_fs=False).values
        y_prob = grid.predict_proba(x_sub)[:, 1]
        df_out = pd.DataFrame(data={'patient_id': patient_ids, 'heart_disease_present': y_prob})
        print(df_out.head())
        df_out.to_csv('submission.csv', index=False)
    else:
        # train and score on test data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
        grid.fit(x_train, y_train)
        log_loss_score = log_loss(y_test, grid.predict_proba(x_test)[:, 1])
        y_score = grid.score(x_test, y_test)
        return algo_name, grid.best_score_, grid.best_params_, y_score, log_loss_score


df_train = pd.read_csv('train_values.csv').drop(['patient_id'], axis=1)
eda(df_train)

y = pd.read_csv('train_labels.csv').heart_disease_present.values
df_train = clean_data(df_train)

X = df_train.values

classifiers = {
    'KNeighborsClassifier': {'clf': KNeighborsClassifier(), 'params': {'n_neighbors': np.arange(1, 10)}},
    'GaussianNB': {'clf': GaussianNB(), 'params': {}},
    'LogisticRegression': {'clf': LogisticRegression(solver='liblinear'), 'params': {'C': [0.0001, 0.001, 0.01, 1, 10]},
                           'penalty': ['l1', 'l2']},
    'RandomForestClassifier': {'clf': RandomForestClassifier(), 'params': {'n_estimators': np.arange(1, 100, 10)}},
    'DecisionTreeClassifier': {'clf': DecisionTreeClassifier(criterion='entropy', max_depth=3), 'params': {}}
}

table = []
scores = []
algos = []

for k, v in classifiers.items():
    algos.append(k)
    a, best_score_, best_params_, y_pred, log_loss_score = train_predict(k, X, y, v['clf'], v['params'])
    scores.append(log_loss_score)
    table.append([a, best_score_, best_params_, y_pred, log_loss_score])

plt.scatter(algos, scores)
plt.xlabel('Algorithm')
plt.xticks(rotation=60)
plt.ylabel('Log Loss Score')
plt.show()

print(tabulate(table, headers=['Algorithm', 'Best Score', 'Best Params', 'Test Score', 'Log Loss Score']))

# choose best algorithm
train_predict('LogisticRegression', X, y, LogisticRegression(solver='liblinear'), {'C': [0.0001, 0.001, 0.01, 1, 10]},
              create_submission=True)
