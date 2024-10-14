import json

from lib import ref_datasets, execute_on_planqk, execute_with_reference_data


def train_trigger(
        regression_choice,
        rescaling_choice,
        encoding_choice,
        dim_reduction,
        n_reduction_dims,
        problem_type,
        mode,
        time_budget,
        data_file=None,
        data_ref_identifier=None,
):
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

    if data_file is None:
        data_ref = ref_datasets[data_ref_identifier]
        return execute_with_reference_data(data_ref, params)
    elif data_ref_identifier is None:
        return execute_with_reference_data(data_file, params)


def train_with_upload_data(
        data_file,
        params
):
    file_path = data_file.name
    with open(file_path) as f:
        data = json.load(f)

    result = execute_on_planqk(data, params, data_ref=None)

    return result


def train_with_reference_data(
        data_ref,
        params
):
    return execute_on_planqk(data=None, params=params, data_ref=data_ref)


def create_train_data_and_params(
        regression_choice,
        rescaling_choice,
        encoding_choice,
        dim_reduction,
        n_reduction_dims,
        problem_type,
        mode,
        time_budget,
        data_file=None,
        data_ref_identifier=None,
):
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

    if data_file is not None:
        file_path = data_file.name
        with open(file_path) as f:
            data = json.load(f)
    elif data_ref_identifier is not None:
        data = ref_datasets[data_ref_identifier]
    else:
        data = dict()

    return params, data
