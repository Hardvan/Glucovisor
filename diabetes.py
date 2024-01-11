import numpy as np
import joblib

data_cache = {}
scaler = joblib.load("./static/model/diabetes-prediction-scaler.joblib")
model = joblib.load("./static/model/diabetes-prediction-model.joblib")


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
    input_data = tuple([data[x] for x in data])

    # Check if the data is in the cache
    if input_data in data_cache:
        return data_cache[input_data]

    # convert to numpy array & reshape
    input_data_reshaped = np.asarray(input_data).reshape(1, -1)

    # Transform data using scaler
    std_data = scaler.transform(input_data_reshaped)
    print(f"std_data: {std_data}")

    # Get prediction from model
    prediction = model.predict(std_data)
    print(f"\033[94mPredction\033[0m: {prediction}")  # For debugging

    # Add to cache
    data_cache[input_data] = prediction[0]

    return prediction[0]


if __name__ == "__main__":

    import time

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

    # Test cache
    start = time.time()
    print(f"\033[94mPrediction\033[0m: {getPrediction(data)}")
    end = time.time()
    time_taken = round(end - start, 2) * 1000
    print(f"\033[94mTime taken\033[0m: {time_taken} ms")
    print()

    # Same data again
    start = time.time()
    print(f"\033[94mPrediction\033[0m: {getPrediction(data)}")
    end = time.time()
    time_taken = round(end - start, 2) * 1000
    print(f"\033[94mTime taken\033[0m: {time_taken} ms")

    # Different input
    data = {
        "pregnancies": 1,
        "glucose": 85,
        "blood-pressure": 66,
        "skin-thickness": 29,
        "insulin": 0,
        "bmi": 26.6,
        "diabetes-pedigree-function": 0.351,
        "age": 31
    }

    start = time.time()
    print(f"\033[94mPrediction\033[0m: {getPrediction(data)}")
    end = time.time()
    time_taken = round(end - start, 2) * 1000
    print(f"\033[94mTime taken\033[0m: {time_taken} ms")
    print()

    # Same data again
    start = time.time()
    print(f"\033[94mPrediction\033[0m: {getPrediction(data)}")
    end = time.time()
    time_taken = round(end - start, 2) * 1000
    print(f"\033[94mTime taken\033[0m: {time_taken} ms")
    print()

    print(f"\033[94mData cache\033[0m: {data_cache}")
