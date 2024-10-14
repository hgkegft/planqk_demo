import os
import sys

import gradio as gr

from loguru import logger

from ui import training_ui, prediction_ui

logging_level = os.environ.get("LOG_LEVEL", "DEBUG")
logger.configure(handlers=[{"sink": sys.stdout, "level": logging_level}])
logger.info("Starting Gradio Demo")

title = "A PlanQK Demo using Gradio!"
description = '<div align="center"> <h1>A descriptive description!</h1> </div>'


with gr.Blocks(title=title, theme=gr.themes.Soft()) as demo:
    gr.Markdown(description)
    result_json_box_train = training_ui()
    prediction_ui(result_json_box_train)

demo.queue()
demo.launch()
