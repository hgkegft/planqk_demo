import gradio as gr
import numpy as np

from lib import (
    upload_json_file,
    ref_identifier
)
from predict import create_predict_data_and_params, predict_trigger
from train import train_trigger, create_train_data_and_params


def get_config_elements():
    with gr.Accordion("Regression", open=False):
        regression_choice = gr.Dropdown(
            label="Regression method",
            choices=["svr", "qsvr"],
            value="qsvr",
        )
    with gr.Accordion("Classification", open=False):
        classification_choice = gr.Dropdown(
            label="Classification method",
            choices=["qgpc", "qnn", "qsvc", "random_forest", "svc"],
            value="qsvc",
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
        rescaling_min_max_feature_range = gr.Number(label="MinMax Feature Range", minimum=0.0, value=0.5, maximum=1.0)
        rescaling_normalization_norm = gr.Dropdown(
            label="Normalization norm",
            choices=[
                "l1",
                "l2",
                "max",
            ],
            value="l1",
        )
    with gr.Accordion("Encoding", open=False):
        encoding_choice = gr.Dropdown(
            label="Encoding method",
            choices=["categorical", "one-hot", "no-op"],
            value="one-hot",
        )
        one_hot_min_frequency = gr.Number(label="One-Hot Min Frequency", minimum=0.0, value=0.5, maximum=1.0)
        one_hot_max_categories = gr.Number(label="One-Hot Max categories", minimum=0.0, value=0.5, maximum=1.0)
    with gr.Accordion("Rescaling", open=False):
        ...
    with gr.Accordion("Down sampling", open=False):
        ...
    with gr.Accordion("Dimension reduction", open=False):
        dim_reduction = gr.Dropdown(
            label="Dimension reduction method",
            choices=["pca", "autoencoder"],
            value="autoencoder",
        )
        n_reduction_dims = gr.Number(label="Reduction dims", value=5)
    with gr.Accordion("General", open=False):
        time_budget = gr.Number(label="Time budget(seconds)", value=5)
        problem_type = gr.Dropdown(
            label="Problem type",
            choices=["classification", "regression"],
            value="regression",
        )

    return (regression_choice,
            classification_choice,
            rescaling_choice,
            rescaling_min_max_feature_range,
            rescaling_normalization_norm,
            encoding_choice,
            one_hot_min_frequency,
            one_hot_max_categories,
            dim_reduction,
            n_reduction_dims,
            time_budget,
            problem_type)


def handle_dataset_reference():
    with gr.Tab("Reference data"):
        dataset_reference = gr.Dropdown(
            label="Reference Datasets",
            choices=ref_identifier,
            value=None
        )
        data_pool_reference = gr.Textbox(label="Reference data pool")

    return dataset_reference, data_pool_reference


def handle_data_upload():
    with gr.Tab("Upload data"):
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
    return train_data_file, upload_button


def get_inspect_block():
    with gr.Accordion("Inspect data", open=False):
        with gr.Row():
            gr.Markdown("Data", scale=3)
            gr.Markdown("Params", scale=4)
        with gr.Row():
            send_data_json_box = gr.JSON(scale=3)
            send_params_json_box = gr.JSON(scale=4)
    return send_data_json_box, send_params_json_box


def training_ui():
    with gr.Tab("Training"):
        with gr.Row():
            with gr.Column():
                config_elements = get_config_elements()
            with gr.Column():
                dataset_reference, data_pool_reference = handle_dataset_reference()
                train_data_file, upload_button = handle_data_upload()

        with gr.Row():
            with gr.Column():
                train_button = gr.Button(value="Train")
            with gr.Column():
                ...
            with gr.Column():
                ...
            with gr.Column():
                ...

        send_data_json_box, send_params_json_box = get_inspect_block()

        with gr.Column():
            with gr.Accordion("Result", open=False):
                result_json_box_train = gr.JSON()

        mode = gr.Text("train", visible=False)
        train_button.click(
            train_trigger,
            inputs=[
                *config_elements,
                mode,
                train_data_file,
                dataset_reference,
                data_pool_reference,
            ],
            outputs=[result_json_box_train],
            api_name="train",
        )
        gr.on(
            [train_button.click],
            create_train_data_and_params,
            inputs=[
                *config_elements,
                mode,
                train_data_file,
                dataset_reference,
                data_pool_reference,
            ],
            outputs=[
                send_params_json_box,
                send_data_json_box,
            ],
        )
    return result_json_box_train


def prediction_ui(result_json_box_train):
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

        with gr.Row():
            with gr.Column():
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
            with gr.Column():
                ...

        with gr.Row():
            with gr.Column():
                predict_button = gr.Button(value="Predict")
            with gr.Column():
                ...
            with gr.Column():
                ...
            with gr.Column():
                ...

        send_data_json_box, send_params_json_box = get_inspect_block()

        with gr.Column():
            with gr.Accordion("Result", open=False):
                result_json_box_predict = gr.JSON()

            mode = gr.Text("predict", visible=False)
            predict_button.click(
                predict_trigger,
                inputs=[
                    result_json_box_train_output,
                    mode,
                    predict_data_file,
                ],
                outputs=[result_json_box_predict],
                api_name="predict",
            )
            gr.on(
                [predict_button.click],
                create_predict_data_and_params,
                inputs=[
                    result_json_box_train_output,
                    mode,
                    predict_data_file,
                ],
                outputs=[
                    send_params_json_box,
                    send_data_json_box,
                ],
            )
