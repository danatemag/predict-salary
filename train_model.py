# Script to train machine learning model.
import os
import json
import logging
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (train_model, compute_model_metrics,
                      compute_model_metrics_by_cat)


logging.basicConfig(level=logging.DEBUG)

# Add code to load in the data.
logging.debug('Loading data ...')
cur_dir = os.path.dirname(os.path.realpath(__file__))
data = pd.read_csv(cur_dir + '/data/census.csv')

logging.debug('Splitting data ...')
# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

logging.debug('Processing data ...')

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# # Train and save a model.
logging.debug('Training model ...')
model = train_model(X_train, y_train)

logging.debug('Predicting test data ...')
y_preds = model.predict(X_test)

logging.debug('Generating metrics ...')
metrics = compute_model_metrics(y_test, y_preds)

logging.debug('General metrics:')
print(metrics)

cat_metrics = compute_model_metrics_by_cat(
    test, y_test, y_preds, cat_features[0])

with open(f'{cur_dir}/slice_output.txt', 'w') as f:
    f.write(json.dumps(cat_metrics, indent=4))

save_files = {
    'model': model,
    'encoder': encoder,
    'lb': lb
}

logging.debug('Saving model files ...')
for key in save_files.keys():
    with open(f'{cur_dir}/model/{key}.pkl', 'wb') as f:
        pickle.dump(save_files[key], f)
