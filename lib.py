import os
import json

from loguru import logger
from planqk.service.client import PlanqkServiceClient

from data_pools import create_data_pool, add_file_to_data_pool

consumer_key = os.getenv("CONSUMER_KEY", None)
consumer_secret = os.getenv("CONSUMER_SECRET", None)
service_endpoint = os.getenv("SERVICE_ENDPOINT", None)
planqk_api_key = os.getenv("PLANQK_API_KEY", None)


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


def execute_on_planqk(data, filename):
    logger.info(data)

    data_pool_name = "autoqml_demo_data"
    data_pool_id = create_data_pool(planqk_api_key, data_pool_name)
    file_reference = add_file_to_data_pool(data_pool_id, planqk_api_key, filename, data)

    logger.info(f"File reference: {file_reference}")

    client = PlanqkServiceClient(service_endpoint, consumer_key, consumer_secret)
    logger.info("Starting execution of the service...")

    job = client.start_execution(data_ref=file_reference)

    timeout = None
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
