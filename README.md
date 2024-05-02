# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model is a sklearn.linear_model.LogisticRegression with parameter max_iter of 500 trained on the following dataset https://archive.ics.uci.edu/dataset/20/census+income

## Intended Use

The goal of the model is to predict if the salary of a person is above or below 50k in the context of a ML Pipiline exercise

## Training Data

To obtain the training data we randomly select 80% of the rows from the dataset

## Evaluation Data

To obtaine the evaluation data we use the remaining 20% of the rows after we substracted the training data

## Metrics

Precision 0.69
Recall 0.50
F-beta 0.58


## Ethical Considerations

The dataset is using sensitive data like marital status and relationship status

## Caveats and Recommendations

The dataset is biased in some categories.
There are 66% Males and 33% Females
The White race is overly represented with 85% of the total data

We recommend not to take any decisions based solely on this model
