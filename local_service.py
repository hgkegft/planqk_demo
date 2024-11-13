import os

from planqk.service.client import PlanqkServiceClient

consumer_key = os.getenv("CONSUMER_KEY", "QuAIhJULrDtOu62HMsjCih8QL0oa")
consumer_secret = os.getenv("CONSUMER_SECRET", "sBdY25szBMZ6IevOIq1J3FH39w0a")
service_endpoint = os.getenv("SERVICE_ENDPOINT", "https://gateway.platform.planqk.de/418f4736-0ed9-46cf-8e09-68a16dada3bc/planqk-autoqml-8et4w/1.0.0")

data = None
data_ref = {
    "dataPoolId": "b9db3d59-164f-4ae4-a42e-e4f4f331cf33",
    "dataSourceDescriptorId": "0772df64-ecac-462d-9967-62a4816de2ae",
    "fileId": "e540e51c-561c-4628-af37-45bb88ab2abe"
}

params = {
    "custom_config": {
        "autoqml_lib.search_space.regression.RegressionChoice__choice": "qsvr",
        "autoqml_lib.search_space.classification.ClassificationChoice__choice": "qsvc",
        "autoqml_lib.search_space.preprocessing.rescaling.RescalingChoice__choice": "standard_scaling",
        "autoqml_lib.search_space.preprocessing.rescaling.min_max_scaling.MinMaxScaling.__feature_range": 0.5,
        "autoqml_lib.search_space.preprocessing.rescaling.normalization.Normalization.__norm": "l1",
        "autoqml_lib.search_space.preprocessing.encoding.EncoderChoice__choice": "one-hot",
        "autoqml_lib.search_space.preprocessing.encoding.one_hot.OneHotEncoder__max_categories": 0.5,
        "autoqml_lib.search_space.preprocessing.encoding.one_hot.OneHotEncoder__min_frequency": 0.5,
        "autoqml_lib.search_space.data_cleaning.imputation.ImputationChoice__choice": "no-op",
        "autoqml_lib.search_space.preprocessing.dim_reduction.DimReductionChoice__choice": "autoencoder",
        "autoqml_lib.search_space.preprocessing.dim_reduction.autoencoder.Autoencoder__latent_dim": 5,
        "autoqml_lib.search_space.preprocessing.downsampling.DownsamplingChoice__choice": "no-op"
    },
    "mode": "train",
    "time_budget_for_this_task": 60,
    "problem_type": "regression"
}

client = PlanqkServiceClient(service_endpoint, consumer_key, consumer_secret)
job = client.start_execution(data=data, params=params, data_ref=data_ref)
