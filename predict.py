from lib import execute_on_planqk


def predict_trigger(
        file_obj,
        data,
        result,
):

    data_tmp = dict()
    data_tmp["mode"] = "predict"
    data_tmp["X"] = data["X"]
    data_tmp["y"] = data["y"]
    data_tmp["model_as_string_base64"] = result["result"]

    return execute_on_planqk(data_tmp, file_obj.name)


def create_predict_data(
        data_ref,
):
    data = data_ref

    return data
