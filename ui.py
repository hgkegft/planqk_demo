import gradio as gr

from lib import upload_json_file
from predict import create_predict_data, predict_trigger
from train import train_trigger, create_train_data
from constants import *


def get_config_elements():
    with gr.Accordion("General", open=False):
        time_budget = gr.Number(label="Time budget(seconds)", minimum=5, value=5)
        problem_type = gr.Dropdown(
            label="Problem type",
            choices=["classification", "regression"],
            value="regression",

        )
    with gr.Accordion("Regression", open=False):
        regression_choice = gr.Dropdown(
            label="Regression method",
            choices=list(regression_dict.keys()),
            value=["SVR"],
            multiselect=True,
        )
    with gr.Accordion("Classification", open=False):
        classification_choice = gr.Dropdown(
            label="Classification method",
            choices=list(classification_dict.keys()),
            value=["Gaussian Process Classifier", "QGPC: Quantum Gaussian Process Classifier"],
            multiselect=True,
        )
    with gr.Accordion("Rescaling", open=False):
        rescaling_choice = gr.Dropdown(
            label="Rescaling method",
            choices=list(rescaling_dict.keys()),
            value="Standard Scaling",
            multiselect=True,
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
            multiselect=True,
        )
    with gr.Accordion("Encoding", open=False):
        encoding_choice = gr.Dropdown(
            label="Encoding method",
            choices=list(encoding_dict.keys()),
            value="One-Hot Encoding",
            multiselect=True,
        )
        one_hot_min_frequency = gr.Number(label="One-Hot Min Frequency", minimum=0.5, value=0.5, maximum=0.5)
        one_hot_max_categories = gr.Number(label="One-Hot Max categories", minimum=0.5, value=0.5, maximum=0.5)
    with gr.Accordion("Imputation", open=False):
        imputation_choice = gr.Dropdown(
            label="Imputation method",
            choices=list(imputation_dict.keys()),
            value="no Imputation",
            multiselect=True,
        )
    with gr.Accordion("Downsampling", open=False):
        downsampling_choice = gr.Dropdown(
            label="Downsampling method",
            choices=list(downsampling_dict.keys()),
            value="no Downsampling",
            multiselect=True,
        )
    with gr.Accordion("Dimension reduction", open=False):
        dim_reduction_choice = gr.Dropdown(
            label="Dimension reduction method",
            choices=list(dim_reduction_dict.keys()),
            value="PCA",
            multiselect=True,
        )
        n_reduction_dims = gr.Number(label="Reduction dims", minimum=2, value=2)

    return (regression_choice,
            classification_choice,
            rescaling_choice,
            rescaling_min_max_feature_range,
            rescaling_normalization_norm,
            encoding_choice,
            imputation_choice,
            downsampling_choice,
            one_hot_min_frequency,
            one_hot_max_categories,
            dim_reduction_choice,
            n_reduction_dims,
            time_budget,
            problem_type)


def handle_data_upload():
    upload_button = gr.UploadButton(
        "Click to upload a file",
        file_types=["text"],
        file_count="single",
    )
    with gr.Accordion("Upload info", open=True):
        data_file = gr.File()
        with gr.Accordion("Inspect uploaded data", open=False):
            data_json_box = gr.JSON()

    upload_button.upload(
        upload_json_file, upload_button, [data_file, data_json_box]
    )
    return data_file, data_json_box


def get_inspect_block():
    with gr.Accordion("Inspect data", open=False):
        with gr.Row():
            gr.Markdown("Data", scale=3)
        with gr.Row():
            send_data_json_box = gr.JSON(scale=3)
    return send_data_json_box


def training_ui():
    with gr.Tab("Training"):
        with gr.Row():
            with gr.Column():
                config_elements = get_config_elements()
            with gr.Column():
                with gr.Accordion("Data", open=True):
                    data_file, data_json_box = handle_data_upload()

        with gr.Row():
            with gr.Column():
                train_button = gr.Button(value="Train")
            with gr.Column():
                ...
            with gr.Column():
                ...
            with gr.Column():
                ...

        send_data_json_box = get_inspect_block()

        with gr.Column():
            with gr.Accordion("Result", open=False):
                result_json_box_train = gr.JSON()

        train_button.click(
            train_trigger,
            inputs=[
                *config_elements,
                data_file,
                data_json_box,
            ],
            outputs=[result_json_box_train],
            api_name="train",
        )
        gr.on(
            [train_button.click],
            create_train_data,
            inputs=[
                *config_elements,
                data_json_box,
            ],
            outputs=[
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
                ...
            with gr.Column():
                ...

        with gr.Row():
            with gr.Column():
                data_file, data_json_box = handle_data_upload()
                predict_button = gr.Button(value="Predict")
            with gr.Column():
                ...
            with gr.Column():
                ...
            with gr.Column():
                ...

        send_data_json_box = get_inspect_block()

        with gr.Column():
            with gr.Accordion("Result", open=False):
                result_json_box_predict = gr.JSON()

            predict_button.click(
                predict_trigger,
                inputs=[
                    data_file,
                    data_json_box,
                    result_json_box_train_output
                ],
                outputs=[result_json_box_predict],
                api_name="predict",
            )
            gr.on(
                [predict_button.click],
                create_predict_data,
                inputs=[
                    data_json_box,
                ],
                outputs=[
                    send_data_json_box,
                ],
            )
