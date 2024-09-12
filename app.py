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


def run() -> str:
    # client = PlanqkServiceClient(service_endpoint, consumer_key, consumer_secret)
    # data = {"n_numbers": n_numbers}
    # params = {"n_bits": 4, "backend": "qasm_simulator"}
    # job = client.start_execution(data=data, params=params)
    # result = client.get_result(job.id)
    return "txt"


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
                time_budget_for_this_task = gr.Number(
                    label="Time budget(seconds)", value=60
                )
                problem_type = gr.Dropdown(choices=["classification", "regression"])
    with gr.Tab("Prediction"):
        ...

demo.launch()
