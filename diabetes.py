import numpy as np
import joblib


def getPrediction(data):
    """Get the prediction from the SVM model.

    Args:
        data (dict): Contains the data from the POST request.
        It's keys are:
            pregnancies
            glucose
            blood-pressure
            skin-thickness
            insulin
            bmi
            diabetes-pedigree-function
            age

    Returns:
        int: The prediction from the SVM model.
    """

    # Get the values in a list
    input_data = [data[x] for x in data]

    # convert to numpy array & reshape
    input_data_reshaped = np.asarray(input_data).reshape(1, -1)

    # Load scaler & transform data
    scaler = joblib.load("./static/model/diabetes-prediction-scaler.joblib")
    std_data = scaler.transform(input_data_reshaped)
    print(f"std_data: {std_data}")

    # Load model & predict
    model = joblib.load("./static/model/diabetes-prediction-model.joblib")
    prediction = model.predict(std_data)
    print(f"\033[94mPredction\033[0m: {prediction}")  # For debugging

    return prediction[0]


if __name__ == "__main__":

    data = {
        "pregnancies": 5,
        "glucose": 166,
        "blood-pressure": 72,
        "skin-thickness": 19,
        "insulin": 175,
        "bmi": 25.8,
        "diabetes-pedigree-function": 0.587,
        "age": 51
    }

    print(f"\033[94mData\033[0m: {data}")  # For debugging
    print(f"\033[94mPrediction\033[0m: {getPrediction(data)}")  # For debugging
