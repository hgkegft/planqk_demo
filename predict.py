import json

from lib import execute_with_upload_data, execute_with_reference_data


def predict_trigger(
        result_json_box_train_output,
        mode,
        data_file=None,
):
    params = dict()
    params["mode"] = mode
    params["model_as_string_base64"] = result_json_box_train_output["result"]

    # return execute_with_upload_data(data_file, params)

    data_ref = {"dataPoolId": "95b5dd46-8188-4e3b-8fa3-cc6e2289d596",
                "dataSourceDescriptorId": "e34d112b-5e8f-4494-969f-d7f215c24259",
                "fileId": "c52abef3-49dd-4e03-a9d3-417c533a714f"}
    return execute_with_reference_data(data_ref, params)


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
