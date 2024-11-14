import json

from lib import execute_with_upload_data, execute_with_reference_data


def predict_trigger(
        result_json_box_train_output,
        mode,
        data_reference,
):
    params = dict()
    params["mode"] = mode
    params["model_as_string_base64"] = result_json_box_train_output["result"]

    return execute_with_reference_data(data_ref, params)


def create_predict_data_and_params(
        result_json_box_train_output,
        mode,
        data_reference,
):
    params = dict()
    params["mode"] = mode
    params["model_as_string_base64"] = result_json_box_train_output["result"]

    data = data_reference

    return params, data
