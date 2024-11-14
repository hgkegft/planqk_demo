from lib import execute_on_planqk


def predict_trigger(
        data,
        params,
):

    return execute_on_planqk(data, params)


def create_predict_data_and_params(
        data_ref,
        params_ref,
):

    data = data_ref
    params = params_ref

    return params, data
