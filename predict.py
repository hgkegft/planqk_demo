import json

from lib import execute_with_upload_data


def predict_trigger(
        result_json_box_train_output,
        mode,
        data_file=None,
):

    params = dict()
    params["mode"] = mode
    params["model_as_string_base64"] = result_json_box_train_output["result"]

    file_path = data_file.name
    with open(file_path) as f:
        data = json.load(f)

    return execute_with_upload_data(data, params)


def create_predict_data_and_params(
        result_json_box_train_output,
        mode,
        data_file=None,
):
    params = dict()
    params["mode"] = mode
    params["model_as_string_base64"] = result_json_box_train_output["result"]

    file_path = data_file.name
    with open(file_path) as f:
        data = json.load(f)

    return params, data
