# Glucovisor: Diabetes Prediction Flask App

## [Link to the Website](https://glucovisor.onrender.com)

## Introduction

Glucovisor is a Flask web application designed for predicting diabetes based on user-provided health data. It utilizes a SVM machine learning model to make predictions and provides users with an interactive interface for obtaining predictions and exploring the results.

## Tech Stack

- **Flask**: A lightweight web framework for building web applications in Python.
- **Tailwind CSS**: A utility-first CSS framework for rapidly building custom user interfaces.
- **Plotly**: A library for creating interactive and dynamic visualizations.
- **Joblib**: Used for saving and loading machine learning models.
- **scikit-learn**: A machine learning library for building and evaluating models.

## Features

1. **User-friendly Interface**: An intuitive web interface for users to input their health data.

2. **Predictive System**: Utilizes a machine learning model to predict whether the user has diabetes.

3. **Interactive Visualizations**: Employs Plotly for creating dynamic visualizations during exploratory data analysis.

## Installation

1. Clone the repo

   ```bash
   git clone https://github.com/Hardvan/Glucovisor.git
   ```

2. Navigate to the folder

   ```bash
   cd Glucovisor
   ```

3. Create a virtual python environment by typing the following in the terminal

   ```bash
   python -m venv .venv
   ```

4. Activate the virtual environment

   Windows:

   ```bash
   .\.venv\Scripts\activate
   ```

   Linux:

   ```bash
   source .venv/bin/activate
   ```

5. Install dependencies by typing the following in the terminal

   ```bash
   pip install -r requirements.txt
   ```

6. Run the app

   ```bash
   python app.py
   ```

7. Click on the link in the terminal to open the website

   It will look something like this:

   ```bash
   Running on http://127.0.0.1:5000
   ```
