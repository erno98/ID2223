import gradio as gr
import numpy as np
from PIL import Image
import requests

import hopsworks
import joblib

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("titanic_surv_modal", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_model.pkl")


def titanic(age, embarked, fare, parch, pclass, sex, sibsp):
    input_list = []
    input_list.append(age)
    input_list.append(embarked)
    input_list.append(fare)
    input_list.append(parch)
    input_list.append(pclass)
    input_list.append(sex)
    input_list.append(sibsp)

    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(input_list).reshape(1, -1))            
    return res
        
demo = gr.Interface(
    fn=titanic,
    title="Titanic Survival Predictive Analytics",
    description="Experiment with passanger data to predict whether they would survive in titanic.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=1.0, label="Age (years)"),
        gr.inputs.Number(default=1.0, label="Embarked (0 (S), 1 (C), or 2 (Q))"),
        gr.inputs.Number(default=1.0, label="Fare (USD)"),
        gr.inputs.Number(default=1.0, label="Parch (from 0 to 6, integer values)"),
        gr.inputs.Number(default=1.0, label="Class (0, 1, or 2)"),
        gr.inputs.Number(default=1.0, label="Sex (0-male, 1-female)"),
        gr.inputs.Number(default=1.0, label="sibsp (from 0 to 5, integer values)")
        ],
    outputs=gr.Text(value="none")
    )

demo.launch()

