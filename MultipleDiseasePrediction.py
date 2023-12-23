# import necessary libraries

import streamlit as st
import pickle
from streamlit_option_menu import option_menu
import numpy as np

# trained models to predict

diab_svm_model = pickle.load(open("trained_SVM_model_diabetes.sav", "rb"))
heart_LR_model=pickle.load(open("trained_LR_model_for_heart.sav", "rb"))



def predict_diabetes(input_data):
    input_array = np.asarray(input_data)
    input_stand_array_reshaped = input_array.reshape(1, -1)
    #svm_model = pickle.load(open("trained_SVM_model_diabetes.sav", "rb"))
    prediction = diab_svm_model.predict(input_stand_array_reshaped)
    if prediction[0]==1:
        return 'The person is diabetic'
    else:
        return 'The person is not diabetic'

# create a sidebar with multiple diseases
with st.sidebar:
    selected=option_menu('Multiple Disease Predictor Web App',
                ['Diabetes Prediction',
                'Heart disease Prediction',
                'BP level checker'],
                icons=['activity','heart-fill','heart-pulse-fill'],
                default_index=0)




# Diabetes prediction
if selected=='Diabetes Prediction':
    st.title('Diabetes prediction web app')
    col1,col2,col3=st.columns(3)
    with col1:
        Pregnancies = st.slider('Number of pregnancies', 0, 10,value=2)
    with col2:
        Glucose = st.slider('Glucose level', 0, 200,value=120)
    with col3:
        BloodPressure = st.slider('BP level', 0, 150,value=90)
    with col1:
        SkinThickness = st.slider('Skin thickness value', 0, 100,value=40)
    with col2:
        Insulin = st.slider('Insulin level', 0, 900,value=50)
    with col3:
        BMI = st.number_input('BMI value', 0.0)
    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes pedigree function value', 0.000)
    with col2:
        Age = st.slider('Age of the person', 0, 110)
    result = ""

    if st.button('Diabetes Test Result'):
        input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        result = predict_diabetes(input_data)
        st.success(result)

    if st.button('Suggestions to prevent diabetis'):
        st.write('*  📊 Track Carbs: Be mindful of carbohydrate consumption.')
        st.write('*  🍬 Limit Sugars: Cut back on added sugars and sugary treats.')
        st.write('*  🗓️ Consistent Habits: Maintain a consistent routine for health.')
        st.write('*  🏥 Regular Check-ups: Visit your doctor for diabetes screenings.')
        st.write('*  📈 Monitor Blood Sugar: Keep an eye on your blood glucose levels')
        st.write('*  🌾 Choose Whole Grains: Opt for whole grain foods over refined grains.')
        st.write('*  🧘 Reduce stress: taking steps to reduce stress ')






# heart disease prediction

# Function to predict heart disease
def predict_heart_disease(model, input_data):
    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if selected=='Heart disease Prediction':
    st.title('Heart disease prediction web app')
    # Input widgets
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider('Age of the patient', 1, 90, 35)
    with col2:
        sex = st.radio('Gender', ['Male', 'Female'])
        sex = 1 if sex == 'Male' else 0
    with col3:
        chest_pain = st.selectbox('Type of chest pain',
                              ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
        chest_pain_encoded = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'].index(chest_pain)
    with col1:
        bp = st.number_input('Resting blood pressure (mm Hg)', 70)
    with col2:
        chol = st.number_input('Cholesterol (mg/dl)', 126)
    with col3:
        fbs = st.radio('Fasting blood sugar > 120 mg/dl', ['True', 'False'])
        fbs = 1 if fbs == 'True' else 0
    with col1:
        ecg_result = st.selectbox('Resting electrocardiographic results',
                              ['Normal', 'ST-T wave normality', 'Left ventricular hypertrophy'])
        ecg_result_encoded = ['Normal', 'ST-T wave normality', 'Left ventricular hypertrophy'].index(ecg_result)
    with col2:
        max_heart_rate = st.number_input('Maximum heart rate achieved', 70)
    with col3:
        exercise = st.radio('Exercise', ['Yes', 'No'])
        exercise = 1 if exercise == 'Yes' else 0
    # adding some additional features

    with col1:
        peak_value = st.slider('ST Depression Peak Value', 0.0, 7.0)
    with col2:
        slope = st.slider('Slope of ST Segment', 0, 2)
    with col3:
        num_vessels = st.number_input('Number of major vessels', min_value=0, max_value=5, value=2)
    with col1:
        thalium_stress = st.number_input('Thalium stress result', 0)

    result = ""

    if st.button('Predict Heart Disease'):
        input_data = [age, sex, chest_pain_encoded, bp, chol, fbs, ecg_result_encoded, max_heart_rate, exercise,
                      peak_value, slope, num_vessels, thalium_stress]
        model = heart_LR_model
        prediction = predict_heart_disease(model, input_data)

        if prediction == 0:
            result = "The person does not have heart disease"
        else:
            result = "The person has heart disease"

    st.success(result)
    if st.button('Suggestions to prevent heart disease'):
        st.write('*  Medication Adherence: 💊 Follow prescribed medications as directed by doctor.')
        st.write('*  Stay Hydrated: 💧 - Drink plenty of water to maintain good circulation.')
        st.write('*  Choose Healthy Fats: 🥑 -Opt for unsaturated fats found in avocados, nuts 🌰, and olive oil.')
        st.write('*  Practice Yoga: 🧘‍️ - Yoga can help reduce stress and improve flexibility.')
        st.write('*  Manage Blood Sugar: 🩸📉 - Keep blood sugar levels in a healthy range.')
        st.write('*  Eat More Fish: 🐟 - Fatty fish like salmon is rich in Omega-3 fatty acids.')
        st.write('*  Engage in Cardio Workouts: 🏃  - Incorporate aerobic exercises like running, swimming, or cycling.')
        st.write('*  Balanced Lifestyle: 🌞 Maintain work-life balance for better heart health.')

# Bp level checker
if selected=='BP level checker':
    st.title('BP level checker web app')

    # bp level checking function
    def bpcheck(sys, dia):
        if sys < 90 and dia < 60:
            return "Low Blood Pressure 🩸 ( 'Hypotension' ) "
        elif sys <= 120 and dia <= 80:
            return 'Normal Blood Pressure 🩸 ✅'
        elif sys < 130 and dia < 80:
            return 'Elevated Blood Pressure 🩸'
        elif sys < 140 or dia < 90:
            return 'High Blood Pressure Stage-1 🩸'
        elif sys < 180 or dia < 120:
            return 'Hypertensive Stage-2 🩸'
        else:
            return 'Hypertensive Stage-3 take immediate actions 🩸❗'

    sys = st.number_input('Enter the Systolic value in (mm Hg) ', min_value=60, max_value=200, value=120)
    dia = st.number_input('Enter the Diastolic value in (mm Hg) ', min_value=30, max_value=140, value=80)
    result = ''

    if st.button('Test Blood Pressure level'):
        result = bpcheck(sys, dia)
    st.success(result)
    sug = ''
    if st.button('Suggestions to control BP'):
        st.write(" *  eating a healthful diet that includes 🍴 fruits 🍅, vegetables, lean proteins, and carbohydrates ")
        st.write(" *  exercising regularly, particularly cardio workouts, such as walking, cycling, or running 🏃‍")
        st.write(" *  stop smoking and limiting alcohol 🚭🍷 ")
        st.write(" *  consumption restricting consumption of processed foods 🍔🚫")
        st.write(' *  limiting sodium intake to less than 2 grams daily 🧂⬇️')
        st.write(' *  treating sleep apnea 😴💤')
        st.write(' *  managing and regulating diabetes 🩸💉')
        st.write(' *  reducing weight if overweight 🏋️‍🥗')







