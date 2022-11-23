import gradio as gr
from PIL import Image
import hopsworks

project = hopsworks.login(project="ID2223_Anton")
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()

dataset_api.download("Resources/images/latest_passenger.png")
dataset_api.download("Resources/images/actual_passenger.png")
dataset_api.download("Resources/images/df_recent_titanic.png")
dataset_api.download("Resources/images/confusion_matrix_titanic.png")

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Label("Today's Predicted survival of the passenger")
            input_img = gr.Image("latest_passenger.png",
                                 elem_id="predicted-img")
        with gr.Column():
            gr.Label("Today's Actual survival of the passenger")
            input_img = gr.Image("actual_passenger.png", elem_id="actual-img")
    with gr.Row():
        with gr.Column():
            gr.Label("Recent Prediction History")
            input_img = gr.Image("df_recent_titanic.png",
                                 elem_id="recent-predictions")
        with gr.Column():
            gr.Label("Confusion Maxtrix with Historical Prediction Performance")
            input_img = gr.Image(
                "confusion_matrix_titanic.png", elem_id="confusion-matrix")


demo.launch()
