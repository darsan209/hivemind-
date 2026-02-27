import time
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(model, X_train, y_train, X_test, y_test):

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    y_pred = model.predict(X_test)

    results = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1": f1_score(y_test, y_pred, average="weighted"),
        "TrainingTime": training_time
    }

    return results


def cross_validation(model, X, y):
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

    scores = cross_validate(
        model,
        X,
        y,
        cv=5,
        scoring=scoring
    )

    return {
        "CV_Accuracy_Mean": np.mean(scores['test_accuracy']),
        "CV_Accuracy_Std": np.std(scores['test_accuracy']),
        "CV_F1_Mean": np.mean(scores['test_f1_weighted']),
        "CV_F1_Std": np.std(scores['test_f1_weighted'])
    }