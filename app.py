import pandas as pd
import pickle
import gradio as gr
from prometheus_client import Gauge, start_http_server, generate_latest
from flask import Flask, Response
from sklearn.metrics import r2_score, f1_score, precision_score, recall_score
from fastapi import FastAPI
import uvicorn

CUSTOM_PATH = "/"


def load_model(model_path):
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

model = load_model('xgboost-model.pkl')
test_data = pd.read_csv('heart_failure_clinical_records_dataset.csv')


R2_METRIC = Gauge('model_r2_score', 'R2 score for model predictions')
F1_METRIC = Gauge('model_f1_score', 'F1 score for model predictions')
PRECISION_METRIC = Gauge('model_precision_score', 'Precision for model predictions')
RECALL_METRIC = Gauge('model_recall_score', 'Recall for model predictions')

app = FastAPI()



@app.get("/metrics")
async def metrics():
    sample = test_data.sample(100)
    features = sample.drop('DEATH_EVENT', axis=1)
    true_values = sample['DEATH_EVENT'].values
    
    predictions = model.predict(features)
    r2 = round(r2_score(true_values, predictions), 3)
    f1 = round(f1_score(true_values, predictions, average='macro'),3)
    precision = round(precision_score(true_values, predictions, average='macro'),3)
    recall = round(recall_score(true_values, predictions, average='macro'),3)

    R2_METRIC.set(r2)
    F1_METRIC.set(f1)
    PRECISION_METRIC.set(precision)
    RECALL_METRIC.set(recall)
    
    return Response(generate_latest(), mimetype="text/plain")


def predict_death_event(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                        high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex,
                        smoking, time):
    input_data = pd.DataFrame([{
        'age': age,
        'anaemia': anaemia,
        'creatinine_phosphokinase': creatinine_phosphokinase,
        'diabetes': diabetes,
        'ejection_fraction': ejection_fraction,
        'high_blood_pressure': high_blood_pressure,
        'platelets': platelets,
        'serum_creatinine': serum_creatinine,
        'serum_sodium': serum_sodium,
        'sex': sex,
        'smoking': smoking,
        'time': time
    }])
    
    prediction = model.predict(input_data)
    return "High risk of death" if prediction[0] == 1 else "Low risk of death"

iface = gr.Interface(
    fn=predict_death_event,
    inputs=[
        gr.Slider(minimum=0, maximum=100, step=1, label="Age"),
        gr.Radio(choices=[0, 1], label="Anaemia"),
        gr.Slider(minimum=0, maximum=10000, step=1, label="CPK Level"),
        gr.Radio(choices=[0, 1], label="Diabetes"),
        gr.Slider(minimum=10, maximum=80, step=1, label="Ejection Fraction"),
        gr.Radio(choices=[0, 1], label="High Blood Pressure"),
        gr.Slider(minimum=10000, maximum=500000, step=1000, label="Platelets"),
        gr.Slider(minimum=0.5, maximum=10.0, step=0.1, label="Serum Creatinine"),
        gr.Slider(minimum=100, maximum=150, step=1, label="Serum Sodium"),
        gr.Radio(choices=[0, 1], label="Sex"),
        gr.Radio(choices=[0, 1], label="Smoking"),
        gr.Slider(minimum=0, maximum=300, step=1, label="Follow-up Time")
    ],
    outputs="text",
    title="Patient Survival Prediction",
    description="Predict survival of patient with heart failure, given their clinical record"
)

#iface.launch(server_name="0.0.0.0", server_port=8002)

app = gr.mount_gradio_app(app, iface, path=CUSTOM_PATH)

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8002)
