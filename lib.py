import os
import json

from loguru import logger
from planqk.service.client import PlanqkServiceClient

ref_datasets = dict()
ref_datasets["Zeppelin"] = {"dataPoolId": "95b5dd46-8188-4e3b-8fa3-cc6e2289d596",
                            "dataSourceDescriptorId": "4dbffd52-519c-47cf-a4ed-330ef17f301a",
                            "fileId": "9ebc4d35-c9ab-4741-9a7d-b56894d6d099"}
ref_datasets["KEB"] = None
ref_datasets["IAV"] = None
ref_datasets["Trumpf"] = None

ref_identifier = []
for key, value in ref_datasets.items():
    if value is not None:
        ref_identifier.append(key)
    else:
        ref_identifier.append(f"{key} - Not available.")

consumer_key = os.getenv("CONSUMER_KEY", None)
consumer_secret = os.getenv("CONSUMER_SECRET", None)
service_endpoint = os.getenv("SERVICE_ENDPOINT", None)


def upload_json_file(file):
    file_path = file.name
    parts = file_path.split("/")
    filename = parts[-1]
    parts = filename.split(".")
    if len(parts) < 2:
        raise Exception("File name is not long enough.")

    file_extension = parts[-1]
    if file_extension not in ["json"]:
        raise Exception("File name is not a JSON file.")

    with open(file_path) as f:
        data = json.load(f)

    return file_path, data


def execute_on_planqk(data=None, params=None, data_ref=None, params_ref=None):
    logger.info(data)
    logger.info(params)
    logger.info(data_ref)
    logger.info(params_ref)

    data_ref = json.loads(data_ref)
    params_ref = json.loads(params_ref)

    client = PlanqkServiceClient(service_endpoint, consumer_key, consumer_secret)
    logger.info("Starting execution of the service...")

    job = client.start_execution(data=data, params=params, data_ref=data_ref, params_ref=params_ref)

    timeout = 600
    sleep = 5
    try:
        client.wait_for_final_state(job.id, timeout=timeout, wait=sleep)
        logger.info(f"Found result!")
        result = client.get_result(job.id)
    except Exception as e:
        logger.info(f"{e}")
        logger.info(f"Found no result...stop.")
        result = {"result": None}
    return result


def execute_with_upload_data(
        data_file,
        params
):
    file_path = data_file.name
    with open(file_path) as f:
        data = json.load(f)

    result = execute_on_planqk(data, params, data_ref=None, params_ref=None)

    return result


def execute_with_reference_data(
        data_ref,
        params_ref
):
    return execute_on_planqk(data=None, params=None, data_ref=data_ref, params_ref=params_ref)
