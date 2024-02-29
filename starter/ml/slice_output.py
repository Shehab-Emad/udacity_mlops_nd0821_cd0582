# Script to Compute the performance metrics on slices of categorical features

# Add the necessary imports
import pandas as pd
from data import process_data
from model import inference, compute_model_metrics



def run_slice(feature):
    """ Computes the performance metrics of feature is held fixed.

    Inputs
    ------
    feature : (str) The name of the column that we want to slice data based on

    Returns
    -------
    None

    This function generates a txt file with the Performance metrics
    """

    # Add code to load in the data, model and encoder
    data = pd.read_csv(r"../starter/data/census_cleaned.csv")
    model = pd.read_pickle(r"../starter/model/model.pkl")
    encoder = pd.read_pickle(r"../starter/model/encoder.pkl") 
    lb = pd.read_pickle(r"../starter/model/lb.pkl")
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

    # Get unique slices and their corresponding data
    unique_slices = data.groupby(feature)
    
    # Create a string to accumulate results
    result_string = ''
    
    # Iterate over unique slices and calculate performance metrics
    for slice_value, slice_data in unique_slices:
        result_string += f"{slice_value}\n"
        X_slice, y_slice, _, _ = process_data(
            slice_data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)
        slice_preds = inference(model, X_slice)
        slice_precision, slice_recall, slice_fbeta = compute_model_metrics(y_slice, slice_preds)
        result_string += f'Precision: {slice_precision}\n'
        result_string += f'Recall: {slice_recall}\n'
        result_string += f'Fbeta: {slice_fbeta}\n'
        result_string += "-------------\n"

    # Write results to file
    with open(f'slice_output.txt', 'w') as f:
        f.write(result_string)


run_slice("education")