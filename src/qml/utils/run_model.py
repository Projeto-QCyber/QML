import joblib
import numpy as np
import pandas as pd

def predict_model(input_data:pd.DataFrame) -> np.ndarray:
    """
    Args:
        input_data (str): The input data for the model.
    return:
        str: The prediction result from the model.
    """

    # Prepare data
    y = input_data["Attack_label"].astype(int)
    X = input_data.drop(["Attack_label"], axis=1)
    #data_test = X.reset_index(drop=True)
    #dictionary = data_test.to_dict(orient='list')

    # Load the model
    model_path = "D:/Area de trabalho/Faculdade/BioData/Q-Cyber/QML/IA/models/traditional/random_forest_model.joblib"
    model = joblib.load(model_path)

    # Make a prediction
    prediction = model.predict(X)

    return prediction, y, X