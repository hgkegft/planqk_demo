import json

from lib import execute_with_upload_data, execute_with_reference_data


def predict_trigger(
        data_ref,
        params_ref,
):

    return execute_with_reference_data(data_ref, params_ref)


def create_predict_data_and_params(
        data_ref,
        params_ref,
):

    data = data_ref
    params = params_ref

    return params, data
