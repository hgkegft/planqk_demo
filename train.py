import json

from lib import execute_on_planqk
from constants import *


def train_trigger(
        regression_choice,
        classification_choice,
        rescaling_choice,
        rescaling_min_max_feature_range,
        rescaling_normalization_norm,
        encoding_choice,
        imputation_choice,
        downsampling_choice,
        one_hot_min_frequency,
        one_hot_max_categories,
        dim_reduction_choice,
        n_reduction_dims,
        time_budget,
        problem_type,
        file_obj,
        data,
):
    regression_list = [regression_dict[key] for key in regression_choice]
    if len(regression_list) > 1:
        regression_list = [entry for entry in regression_list if entry != "no-op"]

    classification_list = [classification_dict[key] for key in classification_choice]
    if len(classification_list) > 1:
        classification_list = [entry for entry in classification_list if entry != "no-op"]

    rescaling_list = [rescaling_dict[key] for key in rescaling_choice]
    if len(rescaling_list) > 1:
        rescaling_list = [entry for entry in rescaling_list if entry != "no-op"]

    encoding_list = [encoding_dict[key] for key in encoding_choice]
    if len(encoding_list) > 1:
        encoding_list = [entry for entry in encoding_list if entry != "no-op"]

    dim_reduction_list = [dim_reduction_dict[key] for key in dim_reduction_choice]
    if len(dim_reduction_list) > 1:
        dim_reduction_list = [entry for entry in dim_reduction_list if entry != "no-op"]

    custom_config = {
        "autoqml_lib.search_space.regression.RegressionChoice__choice": regression_list,
        "autoqml_lib.search_space.classification.ClassificationChoice__choice": classification_list,
        "autoqml_lib.search_space.preprocessing.rescaling.RescalingChoice__choice": rescaling_list,
        "autoqml_lib.search_space.preprocessing.rescaling.min_max_scaling.MinMaxScaling.__feature_range": rescaling_min_max_feature_range,
        "autoqml_lib.search_space.preprocessing.rescaling.normalization.Normalization.__norm": rescaling_normalization_norm,
        "autoqml_lib.search_space.preprocessing.encoding.EncoderChoice__choice": encoding_list,
        "autoqml_lib.search_space.preprocessing.encoding.one_hot.OneHotEncoder__max_categories": one_hot_max_categories,
        "autoqml_lib.search_space.preprocessing.encoding.one_hot.OneHotEncoder__min_frequency": one_hot_min_frequency,
        "autoqml_lib.search_space.data_cleaning.imputation.ImputationChoice__choice": "no-op",
        "autoqml_lib.search_space.preprocessing.dim_reduction.DimReductionChoice__choice": dim_reduction_list,
        "autoqml_lib.search_space.preprocessing.dim_reduction.autoencoder.Autoencoder__latent_dim": int(
            n_reduction_dims
        ),
        "autoqml_lib.search_space.preprocessing.downsampling.DownsamplingChoice__choice": "no-op",
    }

    data["custom_config"] = custom_config
    data["mode"] = "train"
    data["data_mode"] = "by_reference"
    data["time_budget_for_this_task"] = int(time_budget)
    data["problem_type"] = problem_type

    return execute_on_planqk(data, file_obj.name)


def create_train_data(
        regression_choice,
        classification_choice,
        rescaling_choice,
        rescaling_min_max_feature_range,
        rescaling_normalization_norm,
        encoding_choice,
        imputation_choice,
        downsampling_choice,
        one_hot_min_frequency,
        one_hot_max_categories,
        dim_reduction_choice,
        n_reduction_dims,
        time_budget,
        problem_type,
        data,
):
    regression_list = [regression_dict[key] for key in regression_choice]
    if len(regression_list) > 1:
        regression_list = [entry for entry in regression_list if entry != "no-op"]

    classification_list = [classification_dict[key] for key in classification_choice]
    if len(classification_list) > 1:
        classification_list = [entry for entry in classification_list if entry != "no-op"]

    rescaling_list = [rescaling_dict[key] for key in rescaling_choice]
    if len(rescaling_list) > 1:
        rescaling_list = [entry for entry in rescaling_list if entry != "no-op"]

    encoding_list = [encoding_dict[key] for key in encoding_choice]
    if len(encoding_list) > 1:
        encoding_list = [entry for entry in encoding_list if entry != "no-op"]

    dim_reduction_list = [dim_reduction_dict[key] for key in dim_reduction_choice]
    if len(dim_reduction_list) > 1:
        dim_reduction_list = [entry for entry in dim_reduction_list if entry != "no-op"]

    custom_config = {
        "autoqml_lib.search_space.regression.RegressionChoice__choice": regression_list,
        "autoqml_lib.search_space.classification.ClassificationChoice__choice": classification_list,
        "autoqml_lib.search_space.preprocessing.rescaling.RescalingChoice__choice": rescaling_list,
        "autoqml_lib.search_space.preprocessing.rescaling.min_max_scaling.MinMaxScaling.__feature_range": rescaling_min_max_feature_range,
        "autoqml_lib.search_space.preprocessing.rescaling.normalization.Normalization.__norm": rescaling_normalization_norm,
        "autoqml_lib.search_space.preprocessing.encoding.EncoderChoice__choice": encoding_list,
        "autoqml_lib.search_space.preprocessing.encoding.one_hot.OneHotEncoder__max_categories": one_hot_max_categories,
        "autoqml_lib.search_space.preprocessing.encoding.one_hot.OneHotEncoder__min_frequency": one_hot_min_frequency,
        "autoqml_lib.search_space.data_cleaning.imputation.ImputationChoice__choice": "no-op",
        "autoqml_lib.search_space.preprocessing.dim_reduction.DimReductionChoice__choice": dim_reduction_list,
        "autoqml_lib.search_space.preprocessing.dim_reduction.autoencoder.Autoencoder__latent_dim": int(
            n_reduction_dims
        ),
        "autoqml_lib.search_space.preprocessing.downsampling.DownsamplingChoice__choice": "no-op",
    }

    data["custom_config"] = custom_config
    data["mode"] = "mode"
    data["data_mode"] = "by_reference"
    data["time_budget_for_this_task"] = int(time_budget)
    data["problem_type"] = problem_type

    return data
