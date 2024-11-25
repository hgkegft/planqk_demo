import json

import numpy as np

from autoqml_lib.automl import AutoQMLTabularRegression, AutoQMLTimeSeriesClassification
from datetime import timedelta
from autoqml_lib.messages import AutoQMLFitCommand

with open("data/train_data.json") as f:
    data = json.load(f)

with open("data/train_params.json") as f:
    params = json.load(f)

X = np.asarray(data["X_train"])
y = np.asarray(data["y_train"])

custom_config = params["custom_config"]
custom_config["autoqml_lib.search_space.regression.RegressionChoice__choice"] = ["qsvr", "svr"]
custom_config["autoqml_lib.search_space.preprocessing.encoding.EncoderChoice__choice"] = ["one-hot", "categorical"]
custom_config["autoqml_lib.search_space.preprocessing.dim_reduction.DimReductionChoice__choice"] = ["pca"]

time_budget_for_this_task = 60 # params["time_budget_for_this_task"]
problem_type = params["problem_type"]

if problem_type == "classification":
    auto_qml = AutoQMLTimeSeriesClassification()
elif problem_type == "regression":
    auto_qml = AutoQMLTabularRegression()
else:
    raise Exception("No valid problem_type.")

cmd = AutoQMLFitCommand(
    X,
    y,
    timedelta(seconds=time_budget_for_this_task),
    configuration=custom_config
)

auto_qml = auto_qml.fit(cmd)

print(auto_qml.pipeline_)
