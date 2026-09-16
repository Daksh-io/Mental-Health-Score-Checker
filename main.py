import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware

model = joblib.load('Mental_Health_Model.pkl')
top_countries = ['Other','India','USA','Canada','Australia','UK','Germany','Mexico','Turkey','France']

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


#A first Pydantic Model
class StudentData(BaseModel):
    age                     : int = Field(..., ge=10, le=100)
    gender                  : Literal['Male', 'Female']
    country                 : str
    academic_level          : Literal['Undergraduate', 'Graduate', 'High School']
    most_used_platform      : Literal['Facebook', 'LinkedIn', 'Instagram', 'Snapchat','Twitter','YouTube', 'TikTok', 'LINE', 'KakaoTalk', 'VKontakte', 'WhatsApp','WeChat']
    purpose_of_use          : Literal['Networking', 'Education', 'Entertainment', 'News']
    avg_daily_usage_hours   : float = Field(..., ge=0, le=24)
    daily_unlocks           : int   = Field(..., ge=0)
    study_hours             : float = Field(..., ge=0, le=24)
    physical_activity_hours : float = Field(..., ge=0, le=24)
    sleep_hours_per_night   : float = Field(..., ge=0, le=24)
    stress_level            : Literal['Medium', 'Low', 'Very High', 'High']




# Describe what we send back
class PredictionResponse(BaseModel):
    predicted_mental_health_score:float
    #6.777777 -> float




@app.get('/')
def greet():
    return {'Welcome to Mental Health Checkup For Students, Say It True, Because we are here to help you!!!'}


@app.post('/predict', response_model=PredictionResponse) #6.77777
def predict(data: StudentData):

   country_group = data.country if data.country in top_countries else "Other"

   input_row = pd.DataFrame([{
        'Age'                       :data.age,
        'Gender'                    :data.gender,
        'Country'                   :data.country,
        'Academic_Level'            :data.academic_level,
        'Most_Used_Platform'        :data.most_used_platform,
        'Purpose_Of_Use'            :data.purpose_of_use,
        'Avg_Daily_Usage_Hours'     :data.avg_daily_usage_hours,
        'Daily_Unlocks'             :data.daily_unlocks,
        'Study_Hours'               :data.study_hours,
        'Physical_Activity_Hours'   :data.physical_activity_hours,
        'Sleep_Hours_Per_Night'     :data.sleep_hours_per_night,
        'Stress_Level'              :data.stress_level,
        'Grouped_country'           :country_group
   }])

   prediction = model.predict(input_row)[0] #6.77
   return PredictionResponse(predicted_mental_health_score=round(float(prediction),2))

## Quick Note!!!
'''
Simple Project Explanation

This FastAPI application creates an API for predicting a student's mental health score using a trained machine learning model.

Loads the ML model
The application loads the previously trained Mental_Health_Model.pkl file using joblib.
Accepts student information
The /predict API receives information such as:
Age and gender
Country and academic level
Most-used social media platform
Purpose of social media usage
Daily usage and unlocks
Study, physical activity, and sleep hours
Stress level
Validates the input
Pydantic checks that the submitted information has the correct format and reasonable values. For example,
age must be between 10 and 100, and daily usage hours must be between 0 and 24.
Groups countries
The code checks whether the student's country is in the predefined top_countries list. If it isn't, the country is grouped as "Other".
Prepares the data
The student's information is converted into a Pandas DataFrame with the same column structure expected by the trained ML model.
Makes the prediction
The trained model receives the student's information and predicts a mental health score.
Returns the result
The predicted score is converted to a number and rounded to 2 decimal places, then returned as a JSON response.
API flow

Student information → Input validation → Data preparation → ML model → Mental health score

For example, the API could receive student details and return:
    {
      "predicted_mental_health_score": 6.78
    }
'''

"""In one sentence....
This FastAPI application provides a REST API that accepts student lifestyle and academic information,
sends the validated data to a trained machine learning model,
and returns the student's predicted mental health score.
"""
