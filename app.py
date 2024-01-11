from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    # Get the data from the POST request
    data = {"pregnancies": request.form['pregnancies'],
            "glucose": request.form['glucose'],
            "blood-pressure": request.form['blood-pressure'],
            "skin-thickness": request.form['skin-thickness'],
            "insulin": request.form['insulin'],
            "bmi": request.form['bmi'],
            "diabetes-pedigree-function": request.form['diabetes-pedigree-function'],
            "age": request.form['age']}

    print(f"\033[94mData\033[0m: {data}")  # For debugging

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
