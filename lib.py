import os
import json

import gradio as gr

from loguru import logger
from planqk.service.client import PlanqkServiceClient

consumer_key = os.getenv("CONSUMER_KEY", None)
consumer_secret = os.getenv("CONSUMER_SECRET", None)
service_endpoint = os.getenv("SERVICE_ENDPOINT", None)


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

    return file_path, data


def execute_on_planqk(data=None, params=None, data_ref=None):
    logger.info(params)

    client = PlanqkServiceClient(service_endpoint, consumer_key, consumer_secret)
    logger.info("Starting execution of the service...")

    job = client.start_execution(data=data, params=params, data_ref=data_ref)

    MAX_TIME = 600
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
            if count >= int(MAX_TIME / timeout):
                logger.info(f"{count:03d} | ...Found no result...stop.")
                result = {"result": None}
                break
    return result


def train(
        data_file,
        is_reference_data,
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

    custom_config = {
        "autoqml_lib.search_space.regression.RegressionChoice__choice": regression_choice,
        "autoqml_lib.search_space.preprocessing.rescaling.RescalingChoice__choice": rescaling_choice,
        "autoqml_lib.search_space.preprocessing.encoding.EncoderChoice__choice": encoding_choice,
        "autoqml_lib.search_space.preprocessing.encoding.one_hot.OneHotEncoder__max_categories": 17,
        "autoqml_lib.search_space.preprocessing.encoding.one_hot.OneHotEncoder__min_frequency": 1,
        "autoqml_lib.search_space.data_cleaning.imputation.ImputationChoice__choice": "no-op",
        "autoqml_lib.search_space.preprocessing.dim_reduction.DimReductionChoice__choice": dim_reduction,
        "autoqml_lib.search_space.preprocessing.dim_reduction.autoencoder.Autoencoder__latent_dim": int(
            n_reduction_dims
        ),
        "autoqml_lib.search_space.preprocessing.downsampling.DownsamplingChoice__choice": "no-op",
    }

    params = dict()
    params["custom_config"] = custom_config
    params["mode"] = mode
    params["time_budget_for_this_task"] = int(time_budget)
    params["problem_type"] = problem_type

    if is_reference_data == "Yes":
        data = None
        data_ref = data
    else:
        data_ref = None

    result = execute_on_planqk(data, params, data_ref)

    return result


def predict(data_file, is_reference_data, result_json_box_train_output, mode):
    file_path = data_file.name
    with open(file_path) as f:
        data = json.load(f)

    params = dict()
    params["mode"] = mode
    params["model_as_string_base64"] = result_json_box_train_output["result"]

    if is_reference_data == "Yes":
        data = None
        data_ref = data
    else:
        data_ref = None

    result = execute_on_planqk(data, params, data_ref)

    return result


def create_train_data_and_params(
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

    custom_config = {
        "autoqml_lib.search_space.regression.RegressionChoice__choice": regression_choice,
        "autoqml_lib.search_space.preprocessing.rescaling.RescalingChoice__choice": rescaling_choice,
        "autoqml_lib.search_space.preprocessing.encoding.EncoderChoice__choice": encoding_choice,
        "autoqml_lib.search_space.preprocessing.encoding.one_hot.OneHotEncoder__max_categories": 17,
        "autoqml_lib.search_space.preprocessing.encoding.one_hot.OneHotEncoder__min_frequency": 1,
        "autoqml_lib.search_space.data_cleaning.imputation.ImputationChoice__choice": "no-op",
        "autoqml_lib.search_space.preprocessing.dim_reduction.DimReductionChoice__choice": dim_reduction,
        "autoqml_lib.search_space.preprocessing.dim_reduction.autoencoder.Autoencoder__latent_dim": int(
            n_reduction_dims
        ),
        "autoqml_lib.search_space.preprocessing.downsampling.DownsamplingChoice__choice": "no-op",
    }

    params = dict()
    params["custom_config"] = custom_config
    params["mode"] = mode
    params["time_budget_for_this_task"] = int(time_budget)
    params["problem_type"] = problem_type

    return params, data, params


def create_predict_data_and_params(data_file, result_json_box_train_output, mode):
    file_path = data_file.name
    with open(file_path) as f:
        data = json.load(f)

    params = dict()
    params["mode"] = mode
    params["model_as_string_base64"] = result_json_box_train_output["result"]

    return params
