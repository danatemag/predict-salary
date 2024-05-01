from starter.ml.model import train_model, compute_model_metrics, inference
from sklearn.linear_model import LogisticRegression


def test_train_model():
    X = [[0], [1]]
    y = [0, 1]

    model = train_model(X, y)
    assert type(model).__name__ == 'LogisticRegression'


def test_compute_model_metrics():
    y = [1, 1, 1, 1]
    preds = [1, 1, 1, 1]
    metrics = compute_model_metrics(y, preds)

    assert (1.0, 1.0, 1.0) == metrics

    preds = [0, 0, 0, 0]
    metrics = compute_model_metrics(y, preds)

    assert (1.0, 0.0, 0.0) == metrics


def test_inference():
    model = LogisticRegression()
    model.fit([[1], [0]], [1, 0])

    preds = inference(model, [[1]])
    assert [1] == preds
