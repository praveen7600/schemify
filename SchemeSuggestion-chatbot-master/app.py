"""
Disclaimer: Ensure that you have reviewed and comply with the licensing terms and usage policies associated with any third-party libraries or data sources used in this script.
Author: Chiristo Selva Nimal
"""

import json
from flask import Flask, render_template, request, jsonify
import random
import joblib
import numpy as np
import warnings

# Load the model
model = joblib.load("./model/model.joblib")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

app = Flask(__name__)

# Load intents.json for response generation
with open('./data/intents.json', 'r') as file:
    intents = json.load(file)

# Helper methods
def dict_to_feature(survey_responses):
    default_values = {
        'age': 0,
        'income': 0,
        'religion_christian': 0,
        'religion_hindu': 0,
        'religion_muslim': 0,
        'religion_others': 0,
        'community_bc': 0,
        'community_mbc': 0,
        'community_oc': 0,
        'community_others': 0,
        'community_sc': 0,
        'gender_female': 0,
        'gender_male': 0,
        'gender_others': 0,
        'segment_farmer': 0,
        'segment_governmentemployee': 0,
        'segment_sportsperson': 0,
        'segment_student': 0,
        'segment_unemployed': 0,
        'segment_widow': 0,
    }

    for key, value in survey_responses.items():
        if key == 'religion':
            default_values[f'religion_{value.lower()}'] = 1
        elif key == 'community':
            community = 'sc' if value.lower() in ['sc', 'st'] else value.lower()
            default_values[f'community_{community}'] = 1
        elif key == 'gender':
            default_values[f'gender_{value.lower()}'] = 1
        elif key == 'segment':
            segment = 'widow' if value.lower() in ['widow', 'destitute women'] else value.lower().replace(' ', '')
            default_values[f'segment_{segment}'] = 1
        else:
            default_values[key] = value

    features = [default_values[key] for key in default_values]
    return features

def make_prediction(survey_responses):
    try:
        features = dict_to_feature(survey_responses)
        prediction = model.predict([features])
        return prediction.tolist()
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None

def determine_intent(user_input):
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            if user_input.lower() in pattern:
                return intent['tag']
    return 'fallback'

def generate_response(intent):
    for i in intents['intents']:
        if i['tag'] == intent:
            return random.choice(i['responses'])

# Routes
@app.route('/get_survey')
def generate_survey():
    global survey_responses, options
    message = request.args.get('msg')
    try:
        if message == '/start':
            survey_responses = {}
            return jsonify({
                'process': True,
                'type': "complex",
                'response': {
                    "reply": "Please answer the following survey for personalised scheme recommendations.",
                    "options": ["What is your age? eg. 1, 18, 33, etc."]
                }
            })
        
        elif 'age' not in survey_responses:
            if message is not None:
                try:
                    age = int(message)
                    if age < 0 or age > 120:
                        raise ValueError("Age must be a valid number.")
                    survey_responses['age'] = age
                    return jsonify({
                        'process': True,
                        'type': "complex",
                        'response': {
                            "reply": "Thank you for your coordination! What is your Religion? enter the number alone",
                            "options": ["hindu", "muslim", "christian", "others"]
                        }
                    })
                except ValueError:
                    return jsonify({
                        'process': True,
                        'type': "simple",
                        'response': "Please provide a valid positive integer for the age question."
                    })
        
        elif 'religion' not in survey_responses:
            if message is not None:
                try:
                    option = int(message)
                    if option <= 0 or option > 4:
                        raise ValueError("Enter a valid number")
                    
                    religion_mapping = {1: "hindu", 2: "muslim", 3: "christian", 4: "others"}
                    survey_responses['religion'] = religion_mapping.get(option, "others")
                    return jsonify({
                        'process': True,
                        'type': "complex",
                        'response': {
                            "reply": "Things going well :) What is your community? enter the number alone",
                            "options": ["bc", "mbc", "oc", "sc/st", "others"]
                        }
                    })
                except ValueError:
                    return jsonify({
                        'process': True,
                        'type': "simple",
                        'response': "Please provide a valid positive number"
                    })   
            else:
                return jsonify({
                    'process': True,
                    'type': "simple",
                    'response': "Please provide a response for the religion question."
                })
            
        elif 'community' not in survey_responses:
            if message is not None:
                try:
                    option = int(message)
                    if option <= 0 or option > 5:
                        raise ValueError("Enter a valid number")
                    
                    community_mapping = {1: "bc", 2: "mbc", 3: "oc", 4: "sc", 5: "others"}
                    survey_responses['community'] = community_mapping.get(option, "others")
                    return jsonify({
                        'process': True,
                        'type': "simple",
                        'response': "Ok, Next What is your family annual income?"
                    })
                except ValueError:
                    return jsonify({
                        'process': True,
                        'type': "simple",
                        'response': "Please provide a valid positive number"
                    })   
            else:
                return jsonify({
                    'process': True,
                    'type': "simple",
                    'response': "Please provide a response for the community question."
                })
            
        elif 'income' not in survey_responses:
            if message is not None:
                try:
                    income = float(message)
                    if income < 0:
                        raise ValueError("Income must be a positive number.")
                    survey_responses['income'] = income
                    return jsonify({
                        'process': True,
                        'type': "complex",
                        'response': {
                            "reply": "Thanks! What is your gender? enter the number",
                            "options": ["male", "female", "others"]
                        }
                    })
                except ValueError:
                    return jsonify({
                        'process': True,
                        'type': "simple",
                        'response': "Enter a valid annual income"
                    })
            else:
                return jsonify({
                    'process': True,
                    'type': "simple",
                    'response': "Please provide a response for the income question."
                })
        
        elif 'gender' not in survey_responses:
            if message is not None:
                try:
                    option = int(message)
                    if option <= 0 or option > 3:
                        raise ValueError("Enter a valid number")
                    
                    gender_mapping = {1: "male", 2: "female", 3: "others"}
                    survey_responses['gender'] = gender_mapping.get(option, "others")

                    gender_option = survey_responses.get('gender', '').lower()
                    age_option = survey_responses.get('age', 0)
                    options = ["Student", "Farmer", "Government Employee", "Sports person", "Unemployed", "Others"]

                    if gender_option == 'male':
                        if age_option <= 18:
                            options = [opt for opt in options if opt not in ["Farmer", "Government Employee"]]
                        elif age_option > 24:
                            options = [opt for opt in options if opt != "Student"]
                        options = [opt for opt in options if opt != "Widow/Destitute women"]
                    elif gender_option == 'female':
                        options = ["Student", "Farmer", "Government Employee", "Sports person", "Unemployed", "Widow/Destitute women", "Others"]
                        if age_option <= 18:
                            options = [opt for opt in options if opt not in ["Farmer", "Widow/Destitute women", "Government Employee"]]
                        elif age_option > 24:
                            options = [opt for opt in options if opt != "Student"]
                    else:  
                        if age_option <= 18:
                            options = [opt for opt in options if opt not in ["Farmer", "Government Employee", "Widow/Destitute women"]]
                        elif age_option > 24:
                            options = [opt for opt in options if opt != "Student"]

                    return jsonify({
                        'process': True,
                        'type': "complex",
                        'response': {
                            "reply": "Got it! What is your beneficiary segment? enter the number",
                            "options": options
                        }
                    })
                except ValueError:
                    return jsonify({
                        'process': True,
                        'type': "simple",
                        'response': "Please provide a valid positive number"
                    })   
            else:
                return jsonify({
                    'process': True,
                    'type': "simple",
                    'response': "Please provide a response for the religion question."
                })
        
        elif 'segment' not in survey_responses:
            if message is not None:
                try:
                    option = int(message)
                    if option <= 0 or option > len(options):
                        raise ValueError("Enter a valid number")
                    
                    survey_responses['segment'] = options[option-1].lower().replace(' ', '')
                    prediction = make_prediction(survey_responses)
                    return jsonify({
                        'process': False,
                        'type': "result",
                        'response': {
                            'reply': "According to our database, the scheme(s) you might be eligible for is/are",
                            'schemes': prediction[0]
                        }
                    })
                except ValueError:
                    return jsonify({
                        'process': True,
                        'type': "simple",
                        'response': "Please provide a valid positive number"
                    })   
            else:
                return jsonify({
                    'process': True,
                    'type': "simple",
                    'response': "Please provide a response for the beneficiary segment question."
                })  
    except Exception as e:
        return jsonify({
            'process': False,
            'type': "simple",
            'response': "Error occurred while conducting the survey. Please enter /start to start the survey again."
        })

@app.route('/get_response')
def get_response():
    message = request.args.get('msg')
    intent = determine_intent(message)
    response = generate_response(intent)
    return jsonify({'process': False, 'response': response})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
