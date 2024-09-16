import json
import os
import sys
import time

import gradio as gr
from loguru import logger
from planqk.service.client import PlanqkServiceClient
from planqk.service.sdk import JobStatus

logging_level = os.environ.get("LOG_LEVEL", "DEBUG")
logger.configure(handlers=[{"sink": sys.stdout, "level": logging_level}])
logger.info("Starting Gradio Demo")

consumer_key = os.getenv("CONSUMER_KEY", None)
consumer_secret = os.getenv("CONSUMER_SECRET", None)
service_endpoint = os.getenv("SERVICE_ENDPOINT", None)

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
        n_reduction_dims,
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
        'autoqml_lib.search_space.preprocessing.encoding.one_hot.OneHotEncoder__max_categories': 17,
        'autoqml_lib.search_space.preprocessing.encoding.one_hot.OneHotEncoder__min_frequency': 1,
        'autoqml_lib.search_space.data_cleaning.imputation.ImputationChoice__choice': 'no-op',
        "autoqml_lib.search_space.preprocessing.dim_reduction.DimReductionChoice__choice": dim_reduction,
        "autoqml_lib.search_space.preprocessing.dim_reduction.autoencoder.Autoencoder__latent_dim": n_reduction_dims,
        'autoqml_lib.search_space.preprocessing.downsampling.DownsamplingChoice__choice': 'no-op',
    }

    params = dict()
    params["custom_config"] = custom_config
    params["mode"] = mode
    params["time_budget_for_this_task"] = time_budget
    params["problem_type"] = problem_type

    client = PlanqkServiceClient(service_endpoint, consumer_key, consumer_secret)
    logger.info("Starting execution of the service...")
    job = client.start_execution(data=data, params=params)

    timeout = 25
    sleep = 5
    count = 0
    while True:
        try:
            count += 1
            client.wait_for_final_state(job.id, timeout=timeout, wait=sleep)
            logger.info(f"{count:03d} | ...Found result!")
            result = client.get_result(job.id)
            break
        except Exception as e:
            logger.info(f"{e}")
            if count >= int(600 / timeout):
                logger.info(f"{count:03d} | ...Found no result...stop.")
                result = {"result": None}
                break

    return result, data, params


with gr.Blocks(title=title) as demo:
    gr.Markdown(description)
    with gr.Tab("Training"):
        with gr.Row():
            with gr.Column():
                regression_choice = gr.Dropdown(
                    label="Regression", choices=["svr", "qsvr"], value="qsvr",
                )
                rescaling_choice = gr.Dropdown(
                    label="Rescaling",
                    choices=[
                        "standard_scaling",
                        "normalization",
                        "min_max_scaling",
                        "no-op",
                    ],
                    value="standard_scaling",
                )
                encoding_choice = gr.Dropdown(
                    label="Encoding", choices=["categorical", "one-hot", "no-op"], value="one-hot"
                )
                dim_reduction = gr.Dropdown(
                    label="Dimension reduction", choices=["pca", "autoencoder"], value="autoencoder"
                )
                n_reduction_dims = gr.Number(
                    label="Reduction dims", value=5
                )
            with gr.Column():
                time_budget = gr.Number(label="Time budget(seconds)", value=60)
                problem_type = gr.Dropdown(label="Problem type", choices=["classification", "regression"],
                                           value="regression")

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
                    n_reduction_dims,
                    problem_type,
                    mode,
                    time_budget,
                ],
                outputs=[result_json_box, send_data_json_box, send_params_json_box],
                api_name="train",
            )
    with gr.Tab("Prediction"):
        ...

demo.queue()
demo.launch()
