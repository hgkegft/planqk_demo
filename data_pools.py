import json
import os

import requests


def get_data_pool_id(data_pool_name, api_key):
    url = f"https://platform.planqk.de/qc-catalog/data-pools?search={data_pool_name}&page=0&size=20"
    headers = {
        'accept': '*/*',
        'X-Auth-Token': api_key,
    }

    response = requests.get(url, headers=headers)
    try:
        data_pool_id = response.json()["content"][0]["id"]
    except Exception as e:
        data_pool_id = None
    return data_pool_id


def create_data_pool(api_key, data_pool_name):

    data_pool_id = get_data_pool_id(data_pool_name, api_key)
    if data_pool_id is not None:
        return data_pool_id

    url = 'https://platform.planqk.de/qc-catalog/data-pools'
    headers = {
        'accept': '*/*',
        'X-Auth-Token': api_key,
        'Content-Type': 'application/json'
    }
    data = {
        'name': data_pool_name
    }

    response = requests.post(url, headers=headers, json=data)
    data_pool_id = response.json()["id"]
    return data_pool_id


def get_file_reference_from_data_pool(data_pool_id, filename, api_key):
    url = f"https://platform.planqk.de/qc-catalog/data-pools/{data_pool_id}/data-sources"
    headers = {
        'accept': '*/*',
        'X-Auth-Token': api_key,
    }

    response = requests.get(url, headers=headers)

    found = False
    file_reference = dict()
    file_reference["dataPoolId"] = None
    file_reference["dataSourceDescriptorId"] = None
    file_reference["fileId"] = None

    try:
        content = response.json()[0]
        files = content["files"]
        for file in files:
            if filename == file["name"]:

                found = True
                file_reference = dict()
                file_reference["dataPoolId"] = data_pool_id
                file_reference["dataSourceDescriptorId"] = content["id"]
                file_reference["fileId"] = file["id"]

                return found, file_reference
    except Exception as e:
        pass
    return found, file_reference

def add_file_to_data_pool(data_pool_id, api_key, filename, data):

    found, file_reference = get_file_reference_from_data_pool(data_pool_id, filename, api_key)
    if found:
        return file_reference

    url = f"https://platform.planqk.de/qc-catalog/data-pools/{data_pool_id}/files"
    headers = {
        'accept': '*/*',
        'X-Auth-Token': api_key,
    }

    tmp_path = os.path.join("/tmp", filename)

    with open(tmp_path, "w") as f:
        json.dump(data, f)

    files = {
        'file': (f"{filename}", open(tmp_path, 'rb'), 'application/json')
    }

    response = requests.post(url, headers=headers, files=files)

    content = response.json()

    file_reference = dict()
    file_reference["dataPoolId"] = data_pool_id
    file_reference["dataSourceDescriptorId"] = content["id"]
    file_reference["fileId"] = content["files"][0]["id"]

    return file_reference


def get_file_content(data_pool_id, file_id, api_key):
    url = f"https://platform.planqk.de/qc-catalog/data-pools/{data_pool_id}/data-sources/{file_id}/file"
    headers = {
        'accept': 'application/octet-stream',
        'X-Auth-Token': api_key
    }

    response = requests.get(url, headers=headers)
    return response.json()


if __name__ == "__main__":
    # api_key = os.environ.get("PLANQK_API_KEY")
    api_key = "plqk_exOpA7jse3x3KfVLtcRsoKv94gucB8ExWxGnbCUfSI"
    data_pool_name = "autoqml_demo_data"

    data_pool_id = create_data_pool(api_key, data_pool_name)
    print(data_pool_id)
    found, file_reference = get_file_reference_from_data_pool(data_pool_id, "train_data.json", api_key)
    print(file_reference)
