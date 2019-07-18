import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate


def eda(x):
    print(x.info())
    print(x.var())


def categorize_label(x):
    return x.astype('category')


def clean_data(df):
    # convert object to categorical data
    string_labels = ['thal']
    df[string_labels] = df[string_labels].apply(categorize_label, axis=0)
    df = pd.get_dummies(df, drop_first=True)
    # normalize high variance columns
    high_variance_cols = ['serum_cholesterol_mg_per_dl', 'max_heart_rate_achieved']
    df[high_variance_cols] = np.log(df[high_variance_cols])
    print(df.columns)
    return df


def train_predict(algo_name, x, y, clf, params, create_submission=False):
    # grid search CV
    grid = GridSearchCV(clf, params, cv=5, iid=True)

    # train and score on test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
    grid.fit(x_train, y_train)
    y_score = grid.score(x_test, y_test)
    log_loss_score = log_loss(y_test, grid.predict_proba(x_test)[:, 1])

    if create_submission:
        # fit and predict on actual test values
        grid.fit(x, y)
        test_df = pd.read_csv('test_values.csv')
        patient_ids = test_df.patient_id.values
        x_sub = clean_data(test_df.drop('patient_id', axis=1)).values
        y_prob = grid.predict_proba(x_sub)[:, 1]
        df_out = pd.DataFrame(data={'patient_id': patient_ids, 'heart_disease_present': y_prob})
        df_out.to_csv('submission.csv', index=False)

    return algo_name, grid.best_score_, grid.best_params_, y_score, log_loss_score


df_train = pd.read_csv('train_values.csv').drop(['patient_id'], axis=1)
eda(df_train)
df_train = clean_data(df_train)

X = df_train.values
y = pd.read_csv('train_labels.csv').heart_disease_present.values

classifiers = {
    'KNeighborsClassifier': {'clf': KNeighborsClassifier(), 'params': {'n_neighbors': np.arange(1, 10)}},
    'GaussianNB': {'clf': GaussianNB(), 'params': {}},
    'LogisticRegression': {'clf': LogisticRegression(solver='liblinear'), 'params': {}},
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
train_predict('LogisticRegression', X, y, LogisticRegression(solver='liblinear'), {}, create_submission=True)
