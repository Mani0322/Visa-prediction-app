from flask import Flask, render_template, request
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))

preprocessor = pickle.load(open("preprocessor.pkl","rb"))

 
        



@app.route("/", methods=["GET", "POST"])
def form():
    if request.method == "POST":
        # Collect form data
        form_data = {
            'continent': request.form['continent'],
            'education_of_employee': request.form['education_of_employee'],
            'has_job_experience': request.form['has_job_experience'],
            'requires_job_training': request.form['requires_job_training'],
            'no_of_employees': request.form['no_of_employees'],
            'company_age': request.form['company_age'],
            'region_of_employment': request.form['region_of_employment'],
            'prevailing_wage': request.form['prevailing_wage'],
            'unit_of_wage': request.form['unit_of_wage'],
            'full_time_position': request.form['full_time_position']
        }

        data = pd.DataFrame([form_data])

        processed_data = preprocessor.transform(data)

        print(processed_data.shape)

        prediction = model.predict(processed_data)[0]
        
        status = None
        if prediction == 1:
            status = "Visa-approved"
        else:
            status = "Visa Not-Approved"
    
    return render_template("index.html", context=status)

if __name__ == "__main__":
    app.run(debug=True)