import pandas as pd
import pickle
import gradio as gr
from prometheus_client import Gauge, start_http_server, generate_latest
from flask import Flask, Response
from sklearn.metrics import r2_score, f1_score, precision_score, recall_score

# Load model
def load_model(model_path):
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

model = load_model('xgboost-model.pkl')
test_data = pd.read_csv('path_to_your_test_dataset.csv')  # Ensure this path is correct and accessible

# Prometheus Metrics
R2_METRIC = Gauge('model_r2_score', 'R2 score for model predictions')
F1_METRIC = Gauge('model_f1_score', 'F1 score for model predictions')
PRECISION_METRIC = Gauge('model_precision_score', 'Precision for model predictions')
RECALL_METRIC = Gauge('model_recall_score', 'Recall for model predictions')

app = Flask(__name__)

@app.route("/metrics")
def metrics():
    sample = test_data.sample(100)  # Randomly sample 100 data points for metrics calculation
    features = sample.drop('DEATH_EVENT', axis=1)  # Assuming 'DEATH_EVENT' is the label column
    true_values = sample['DEATH_EVENT'].values
    
    predictions = model.predict(features)
    r2 = r2_score(true_values, predictions).round(3)
    f1 = f1_score(true_values, predictions, average='macro').round(3)
    precision = precision_score(true_values, predictions, average='macro').round(3)
    recall = recall_score(true_values, predictions, average='macro').round(3)

    R2_METRIC.set(r2)
    F1_METRIC.set(f1)
    PRECISION_METRIC.set(precision)
    RECALL_METRIC.set(recall)
    
    return Response(generate_latest(), mimetype="text/plain")

# Gradio app setup
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

iface = gr.Interface(fn=predict_death_event, inputs="text", outputs="text", title="Patient Survival Prediction")
iface.launch(server_name="0.0.0.0", server_port=8002)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8002)
