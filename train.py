import json

from lib import ref_datasets, execute_on_planqk, execute_with_reference_data, execute_with_upload_data


def train_trigger(
        regression_choice,
        classification_choice,
        rescaling_choice,
        rescaling_min_max_feature_range,
        rescaling_normalization_norm,
        encoding_choice,
        one_hot_min_frequency,
        one_hot_max_categories,
        dim_reduction,
        n_reduction_dims,
        time_budget,
        problem_type,
        mode,
        data_ref=None,
):
    custom_config = {
        "autoqml_lib.search_space.regression.RegressionChoice__choice": regression_choice,
        "autoqml_lib.search_space.classification.ClassificationChoice__choice": classification_choice,
        "autoqml_lib.search_space.preprocessing.rescaling.RescalingChoice__choice": rescaling_choice,
        "autoqml_lib.search_space.preprocessing.rescaling.min_max_scaling.MinMaxScaling.__feature_range": rescaling_min_max_feature_range,
        "autoqml_lib.search_space.preprocessing.rescaling.normalization.Normalization.__norm": rescaling_normalization_norm,
        "autoqml_lib.search_space.preprocessing.encoding.EncoderChoice__choice": encoding_choice,
        "autoqml_lib.search_space.preprocessing.encoding.one_hot.OneHotEncoder__max_categories": one_hot_max_categories,
        "autoqml_lib.search_space.preprocessing.encoding.one_hot.OneHotEncoder__min_frequency": one_hot_min_frequency,
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

    return execute_with_reference_data(data_ref, params)


def create_train_data_and_params(
        regression_choice,
        classification_choice,
        rescaling_choice,
        rescaling_min_max_feature_range,
        rescaling_normalization_norm,
        encoding_choice,
        one_hot_min_frequency,
        one_hot_max_categories,
        dim_reduction,
        n_reduction_dims,
        time_budget,
        problem_type,
        mode,
        data_ref=None,
):
    custom_config = {
        "autoqml_lib.search_space.regression.RegressionChoice__choice": regression_choice,
        "autoqml_lib.search_space.classification.ClassificationChoice__choice": classification_choice,
        "autoqml_lib.search_space.preprocessing.rescaling.RescalingChoice__choice": rescaling_choice,
        "autoqml_lib.search_space.preprocessing.rescaling.min_max_scaling.MinMaxScaling.__feature_range": rescaling_min_max_feature_range,
        "autoqml_lib.search_space.preprocessing.rescaling.normalization.Normalization.__norm": rescaling_normalization_norm,
        "autoqml_lib.search_space.preprocessing.encoding.EncoderChoice__choice": encoding_choice,
        "autoqml_lib.search_space.preprocessing.encoding.one_hot.OneHotEncoder__max_categories": one_hot_max_categories,
        "autoqml_lib.search_space.preprocessing.encoding.one_hot.OneHotEncoder__min_frequency": one_hot_min_frequency,
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

    data = data_ref

    return params, data
