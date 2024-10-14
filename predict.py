import json

from lib import execute_on_planqk, execute_with_reference_data, ref_datasets


def predict_trigger(
        is_reference_data,
        result_json_box_train_output,
        mode,
        data_file=None,
        data_ref_identifier=None
):
    file_path = data_file.name
    with open(file_path) as f:
        data = json.load(f)

    params = dict()
    params["mode"] = mode
    params["model_as_string_base64"] = result_json_box_train_output["result"]

    if is_reference_data == "Yes":
        data_ref = data
        data = None
    else:
        data_ref = None

    result = execute_on_planqk(data, params, data_ref)

    if data_file is None:
        data_ref = ref_datasets[data_ref_identifier]
        return execute_with_reference_data(data_ref, params)
    elif data_ref_identifier is None:
        return execute_with_reference_data(data_file, params)

    return result


def create_predict_data_and_params(
        result_json_box_train_output,
        mode,
        data_file=None,
        data_ref_identifier=None
):
    params = dict()
    params["mode"] = mode
    params["model_as_string_base64"] = result_json_box_train_output["result"]

    if data_file is not None:
        file_path = data_file.name
        with open(file_path) as f:
            data = json.load(f)
    elif data_ref_identifier is not None:
        data = ref_datasets[data_ref_identifier]
    else:
        data = dict()

    return params, data
