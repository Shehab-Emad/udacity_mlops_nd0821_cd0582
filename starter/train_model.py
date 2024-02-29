# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

def run_train(dataset):
    # Add code to load in the data.
    data = pd.read_csv(dataset)

    print(data.columns)
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

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

    # Proces the test data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )


    # Train and save a model.
    model = train_model(X_train, y_train)
    pd.to_pickle(model, "starter/model/model.pkl")

    #Saving the encoder and the LabelBinarizer for being used in the API later
    pd.to_pickle(encoder, "starter/model/encoder.pkl")
    pd.to_pickle(lb, "starter/model/lb.pkl")

    preds = inference(model, X_test)
    p, r, fb = compute_model_metrics(y_test, preds)
    print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

if __name__ == '__main__':
    dataset = r"starter/data/census_cleaned.csv"
    run_train(dataset)