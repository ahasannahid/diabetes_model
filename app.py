import gradio as gr
import pandas as pd
import pickle

# Load trained model
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

INSULIN_MEDIAN = 125.0

def predict_outcome(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    dpf,
    age
):
    pregnancies = int(pregnancies)
    glucose = float(glucose)
    blood_pressure = float(blood_pressure)
    skin_thickness = float(skin_thickness)
    insulin = float(insulin)
    bmi = float(bmi)
    dpf = float(dpf)
    age = float(age)

    glucose_bmi = glucose * bmi
    high_insulin = int(insulin > INSULIN_MEDIAN)

    input_df = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
        "Glucose_BMI": glucose_bmi,
        "High_Insulin": high_insulin
    }])

    pred = model.predict(input_df)[0]
    return int(pred)   # only 0 or 1

inputs = [
    gr.Radio([0, 1], label="Pregnancies (0/1)", value=0),
    gr.Number(label="Glucose", value=120),
    gr.Number(label="BloodPressure", value=70),
    gr.Number(label="SkinThickness", value=20),
    gr.Number(label="Insulin", value=80),
    gr.Number(label="BMI", value=30.0),
    gr.Number(label="DiabetesPedigreeFunction", value=0.5),
    gr.Number(label="Age", value=30),
]

app = gr.Interface(
    fn=predict_outcome,
    inputs=inputs,
    outputs=gr.Number(label="Prediction (0/1)"),
    title="Diabetes Outcome Predictor"
)

app.launch(share=True)
