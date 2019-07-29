import pandas as pd
# the model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
# for optimizing parameters of the pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
# for combining the preprocess with model training
from sklearn.pipeline import Pipeline
# for preprocessing the data
from sklearn.preprocessing import StandardScaler


def categorize_label(x):
    return x.astype('category')


# one hot encode
def one_hot_encode(df):
    if 'thal' in df.columns:
        df[['thal']] = df[['thal']].apply(categorize_label, axis=0)
        return pd.get_dummies(df, drop_first=True)
    return df


# train values
train_values = pd.read_csv('train_values.csv', index_col='patient_id')
train_labels = pd.read_csv('train_labels.csv', index_col='patient_id')

# drop less important columns
to_drop = ['fasting_blood_sugar_gt_120_mg_per_dl', 'slope_of_peak_exercise_st_segment']
train_values = train_values.drop(to_drop, axis=1)
train_values = one_hot_encode(train_values)

# pipeline
pipe = Pipeline(steps=[('scale', StandardScaler()),
                       ('logistic', LogisticRegression(solver='liblinear'))])

param_grid = {'logistic__C': [0.0001, 0.001, 0.01, 1, 10],
              'logistic__penalty': ['l2'],
              'logistic__solver': ['liblinear', 'lbfgs']}
cv = GridSearchCV(pipe, param_grid, scoring='neg_log_loss', cv=10, iid=False)
X = train_values
y = train_labels.heart_disease_present
x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3, random_state=42, stratify=y)
cv.fit(x_train, y_train)

in_sample_preds = cv.predict_proba(x_test)
print(log_loss(y_test, in_sample_preds))

# create submission
test_values = pd.read_csv('test_values.csv', index_col='patient_id')
test_values_subset = test_values.drop(to_drop, axis=1)
test_values_subset = one_hot_encode(test_values_subset)
cv.fit(X, y)
predictions = cv.predict_proba(test_values_subset)[:, 1]
submission_format = pd.read_csv('submission_format.csv', index_col='patient_id')
my_submission = pd.DataFrame(data=predictions,
                             columns=submission_format.columns,
                             index=submission_format.index)
my_submission.to_csv('submission.csv')
