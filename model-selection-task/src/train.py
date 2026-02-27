import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Allow imports from src folder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.preprocessing import build_preprocessor
from src.models import get_models
from src.evaluation import evaluate_model, cross_validation


def main():

    # Load dataset
    data_path = os.path.join(project_root, "data", "dataset.csv")
    df = pd.read_csv(data_path)

    target_column = "target"  # CHANGE to your actual target

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Build preprocessor
    preprocessor = build_preprocessor(X_train)

    models = get_models()

    final_results = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", model)
        ])

        eval_results = evaluate_model(
            pipeline,
            X_train,
            y_train,
            X_test,
            y_test
        )

        cv_results = cross_validation(
            pipeline,
            X_train,
            y_train
        )

        final_results[name] = {**eval_results, **cv_results}

    results_df = pd.DataFrame(final_results).T
    print("\nFinal Comparison:\n")
    print(results_df)

    results_path = os.path.join(project_root, "reports", "model_comparison.csv")
    results_df.to_csv(results_path)


if __name__ == "__main__":
    main()