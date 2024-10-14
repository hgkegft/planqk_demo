import os
import sys

import gradio as gr

from loguru import logger
from lib import (
    upload_json_file,
    predict,
    create_predict_data_and_params,
    ref_identifier
)
from train import train_trigger, create_train_data_and_params


logging_level = os.environ.get("LOG_LEVEL", "DEBUG")
logger.configure(handlers=[{"sink": sys.stdout, "level": logging_level}])
logger.info("Starting Gradio Demo")


title = "A PlanQK Demo using Gradio!"
description = '<div align="center"> <h1>A descriptive description!</h1> </div>'


with gr.Blocks(title=title, theme=gr.themes.Soft()) as demo:
    gr.Markdown(description)
    with gr.Tab("Training"):
        with gr.Row():
            with gr.Column():
                with gr.Accordion("Regression", open=False):
                    regression_choice = gr.Dropdown(
                        label="Regression method",
                        choices=["svr", "qsvr"],
                        value="qsvr",
                    )
                with gr.Accordion("Rescaling", open=False):
                    rescaling_choice = gr.Dropdown(
                        label="Rescaling method",
                        choices=[
                            "standard_scaling",
                            "normalization",
                            "min_max_scaling",
                            "no-op",
                        ],
                        value="standard_scaling",
                    )
                with gr.Accordion("Encoding", open=False):
                    encoding_choice = gr.Dropdown(
                        label="Encoding method",
                        choices=["categorical", "one-hot", "no-op"],
                        value="one-hot",
                    )
                with gr.Accordion("Dimension reduction", open=False):
                    dim_reduction = gr.Dropdown(
                        label="Dimension reduction method",
                        choices=["pca", "autoencoder"],
                        value="autoencoder",
                    )
                    n_reduction_dims = gr.Number(label="Reduction dims", value=5)
                with gr.Accordion("General", open=False):
                    time_budget = gr.Number(label="Time budget(seconds)", value=60)
                    problem_type = gr.Dropdown(
                        label="Problem type",
                        choices=["classification", "regression"],
                        value="regression",
                    )
            with gr.Column():
                with gr.Tab("Reference"):
                    dataset_reference = gr.Dropdown(
                        label="Datasets",
                        choices=ref_identifier,
                        value=None
                    )
                    ...
                with gr.Tab("Upload"):
                    is_reference_data = gr.Text("No", visible=False)
                    train_data_file = gr.File()
                    upload_button = gr.UploadButton(
                        "Click to upload a file",
                        file_types=["text"],
                        file_count="single",
                    )
                    with gr.Accordion("Inspect Data", open=False):
                        data_json_box = gr.JSON()

                    upload_button.upload(
                        upload_json_file, upload_button, [train_data_file, data_json_box]
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

        with gr.Accordion("Inspect data", open=False):
            with gr.Row():
                gr.Markdown("Data", scale=3)
                gr.Markdown("Params", scale=4)
            with gr.Row():
                send_data_json_box = gr.JSON(scale=3)
                send_params_json_box = gr.JSON(scale=4)

        with gr.Column():
            with gr.Accordion("Result", open=False):
                result_json_box_train = gr.JSON()

        mode = gr.Text("train", visible=False)
        train_button.click(
            train_trigger,
            inputs=[
                regression_choice,
                rescaling_choice,
                encoding_choice,
                dim_reduction,
                n_reduction_dims,
                problem_type,
                mode,
                time_budget,
                train_data_file,
                dataset_reference,
            ],
            outputs=[result_json_box_train],
            api_name="train",
        )
        gr.on(
            [train_button.click],
            create_train_data_and_params,
            inputs=[
                regression_choice,
                rescaling_choice,
                encoding_choice,
                dim_reduction,
                n_reduction_dims,
                problem_type,
                mode,
                time_budget,
                train_data_file,
                dataset_reference,
            ],
            outputs=[
                send_params_json_box,
                send_data_json_box,
            ],
        )
    with gr.Tab("Prediction"):
        with gr.Accordion("Result data", open=False):
            result_json_box_train_output = gr.JSON()
            gr.on(
                [result_json_box_train.change],
                lambda value: value,
                inputs=[result_json_box_train],
                outputs=[result_json_box_train_output],
            )

        gr.Markdown(value="Specify data")

        predict_data_file = gr.File()
        upload_button = gr.UploadButton(
            "Click to upload a file",
            file_types=["text"],
            file_count="single",
        )
        with gr.Accordion("Inspect Data", open=False):
            data_json_box = gr.JSON()

        upload_button.upload(
            upload_json_file, upload_button, [predict_data_file, data_json_box]
        )

        with gr.Row():
            with gr.Column():
                predict_button = gr.Button(value="Predict")
            with gr.Column():
                ...
            with gr.Column():
                ...
            with gr.Column():
                ...

        with gr.Accordion("Inspect data", open=False):
            with gr.Row():
                gr.Markdown("Data", scale=3)
                gr.Markdown("Params", scale=4)
            with gr.Row():
                send_data_json_box = gr.JSON(scale=3)
                send_params_json_box = gr.JSON(scale=4)

        with gr.Column():
            with gr.Accordion("Result", open=False):
                result_json_box_predict = gr.JSON()

            mode = gr.Text("predict", visible=False)
            predict_button.click(
                predict,
                inputs=[
                    predict_data_file,
                    is_reference_data,
                    result_json_box_train_output,
                    mode,
                ],
                outputs=[result_json_box_predict],
                api_name="predict",
            )
            gr.on(
                [predict_button.click],
                create_predict_data_and_params,
                inputs=[
                    predict_data_file,
                    result_json_box_train_output,
                    mode,
                ],
                outputs=[
                    send_params_json_box,
                    send_data_json_box,
                ],
            )

demo.queue()
demo.launch()
