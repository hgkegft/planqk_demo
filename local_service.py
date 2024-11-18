import json
import os

from planqk.service.client import PlanqkServiceClient

from data_pools import create_data_pool, add_file_to_data_pool

consumer_key = os.getenv("CONSUMER_KEY", None)
consumer_secret = os.getenv("CONSUMER_SECRET", None)
service_endpoint = os.getenv("SERVICE_ENDPOINT", None)

with open("data/train_data.json") as f:
    data_raw = json.load(f)

with open("data/train_params.json") as f:
    params_raw = json.load(f)

data = dict()
data["X"] = data_raw["X_train"]
data["y"] = data_raw["y_train"]
data["custom_config"] = params_raw["custom_config"]
data["mode"] = params_raw["mode"]
data["time_budget_for_this_task"] = params_raw["time_budget_for_this_task"]
data["problem_type"] = params_raw["problem_type"]


api_key = "PLANQK_API_KEY"
data_pool_name = "autoqml_demo_data"
filename = "train_data.json"
data_pool_id = create_data_pool(api_key, data_pool_name)
file_reference = add_file_to_data_pool(data_pool_id, api_key, filename, data)

client = PlanqkServiceClient(service_endpoint, consumer_key, consumer_secret)
job = client.start_execution(data_ref=file_reference)

timeout = 600
sleep = 5
client.wait_for_final_state(job.id, timeout=timeout, wait=sleep)
result = client.get_result(job.id)

if "result" in result.keys():
    print("Predict")
    with open("data/test_data.json") as f:
        data_raw = json.load(f)

    with open("data/test_params.json") as f:
        params_raw = json.load(f)

    data = dict()
    data["X"] = data_raw["X_test"]
    data["y"] = data_raw["y_test"]
    data["mode"] = params_raw["mode"]
    data["model_as_string_base64"] = result["result"]

    api_key = "PLANQK_API_KEY"
    data_pool_name = "autoqml_demo_data"
    filename = "test_data.json"
    data_pool_id = create_data_pool(api_key, data_pool_name)
    file_reference = add_file_to_data_pool(data_pool_id, api_key, filename, data)

    client = PlanqkServiceClient(service_endpoint, consumer_key, consumer_secret)
    job = client.start_execution(data_ref=file_reference)

    timeout = 600
    sleep = 5
    client.wait_for_final_state(job.id, timeout=timeout, wait=sleep)
    result = client.get_result(job.id)
    if "result" in result.keys():
        print(result["result"])
    elif "code" in result.keys():
        print(result["result"])

elif "code" in result.keys():
    print(result["detail"])
