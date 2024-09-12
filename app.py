import json
import os
import sys

import gradio as gr
from loguru import logger
from planqk.service.client import PlanqkServiceClient

logging_level = os.environ.get("LOG_LEVEL", "DEBUG")
logger.configure(handlers=[{"sink": sys.stdout, "level": logging_level}])
logger.info("Starting Gradio Demo")

consumer_key = os.getenv("CONSUMER_KEY", None)
consumer_secret = os.getenv("CONSUMER_SECRET", None)
service_endpoint = os.getenv(
    "SERVICE_ENDPOINT",
    "https://gateway.platform.planqk.de/anaqor/quantum-random-number-generator/1.0.0",
)

title = "A PlanQK Demo using Gradio!"
description = '<div align="center"> <h1>A descriptive description!</h1> </div>'


def upload_json_file(file):
    file_path = file.name
    parts = file_path.split("/")
    filename = parts[-1]
    parts = filename.split(".")
    if len(parts) < 2:
        raise Exception("File name is not long enough.")

    file_extension = parts[-1]
    if file_extension not in ["json"]:
        raise Exception("File name is not a JSON file.")

    with open(file_path) as f:
        data = json.load(f)

    keys = ["X_train", "y_train"]
    for key in keys:
        if key not in data.keys():
            raise Exception("Mandatory key: {key} not in JSON file.")

    return file_path, data


def train(
    data_file,
    regression_choice,
    rescaling_choice,
    encoding_choice,
    dim_reduction,
    problem_type,
    mode,
    time_budget,
):
    file_path = data_file.name
    with open(file_path) as f:
        data = json.load(f)

    keys = ["X_train", "y_train"]
    for key in keys:
        if key not in data.keys():
            raise Exception("Mandatory key: {key} not in JSON file.")

    custom_config = {
        "autoqml_lib.search_space.regression.RegressionChoice__choice": regression_choice,
        "autoqml_lib.search_space.preprocessing.rescaling.RescalingChoice__choice": rescaling_choice,
        "autoqml_lib.search_space.preprocessing.encoding.EncoderChoice__choice": encoding_choice,
        "autoqml_lib.search_space.preprocessing.dim_reduction.DimReductionChoice__choice": dim_reduction,
        "autoqml_lib.search_space.preprocessing.dim_reduction.autoencoder.Autoencoder__latent_dim": 10,
    }

    params = dict()
    params["custom_config"] = custom_config
    params["mode"] = mode
    params["time_budget_for_this_task"] = time_budget
    params["problem_type"] = problem_type

    client = PlanqkServiceClient(service_endpoint, consumer_key, consumer_secret)
    job = client.start_execution(data=data, params=params)
    result = client.get_result(job.id)

    return result, data, params


with gr.Blocks(title=title) as demo:
    gr.Markdown(description)
    with gr.Tab("Training"):
        with gr.Row():
            with gr.Column():
                regression_choice = gr.Dropdown(
                    label="Regression", choices=["classic:svr", "qc:qsvr"]
                )
                rescaling_choice = gr.Dropdown(
                    label="Rescaling",
                    choices=[
                        "standard_scaling",
                        "normalization",
                        "min_max_scaling",
                        "no-op",
                    ],
                )
                encoding_choice = gr.Dropdown(
                    label="Encoding", choices=["categorical", "one-hot", "no-op"]
                )
                dim_reduction = gr.Dropdown(
                    label="Dimension reduction", choices=["pca", "autoencoder"]
                )
            with gr.Column():
                time_budget = gr.Number(label="Time budget(seconds)", value=60)
                problem_type = gr.Dropdown(label="Problem type", choices=["classification", "regression"])

                data_file = gr.File()
                upload_button = gr.UploadButton(
                    "Click to Upload a File",
                    file_types=["text"],
                    file_count="single",
                )
                with gr.Accordion("Inspect Data", open=False):
                    data_json_box = gr.JSON()

                upload_button.upload(
                    upload_json_file, upload_button, [data_file, data_json_box]
                )

        with gr.Row():
            with gr.Column():
                train_button = gr.Button(value="Train")
            with gr.Column():
                ...
            with gr.Column():
                ...
            with gr.Column():
                ...

        with gr.Accordion("Inspect Data", open=False):
            with gr.Row():
                send_data_json_box = gr.JSON()
                send_params_json_box = gr.JSON()

        with gr.Column():
            result_json_box = gr.JSON()

            mode = gr.Text("train", visible=False)
            train_button.click(
                train,
                inputs=[
                    data_file,
                    regression_choice,
                    rescaling_choice,
                    encoding_choice,
                    dim_reduction,
                    problem_type,
                    mode,
                    time_budget,
                ],
                outputs=[result_json_box, send_data_json_box, send_params_json_box],
                api_name="train",
            )
    with gr.Tab("Prediction"):
        ...

demo.launch()
