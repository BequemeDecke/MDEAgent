"""
Script to fetch the latest models from the FAU server and store them in .mdeagent-benchmark/models.json.

You need to set the FAU_API_KEY environment variable before running this script.
"""

import json
import os
from pathlib import Path

import requests

from scripts.json_to_csv import json_to_csv


def main():
    # Define the URL of the FAU server
    url = "https://hub.nhr.fau.de/api/llmgw/v1/models/"  # Replace with the actual URL
    api_key = os.getenv("FAU_API_KEY")
    if not api_key:
        raise ValueError("FAU_API_KEY environment variable is not set")

    models = retrieve_fau_models(url, api_key)

    # Define the path to store the models
    models_path = Path(".mdeagent-benchmark/models.json")
    models_path.parent.mkdir(
        parents=True, exist_ok=True
    )  # Create parent directories if they don't exist

    # Write the models to the JSON file
    with open(models_path, "w", encoding="utf-8") as f:
        json.dump(models, f, ensure_ascii=False, indent=4)

    print(f"Successfully fetched and stored models in {models_path}")

    json_to_csv(models_path, Path(".mdeagent-benchmark/models.csv"))
    print(f"Successfully converted models.json to models.csv in {models_path.parent}")


def retrieve_fau_models(url, api_key):
    # Fetch the models from the FAU server
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise an error for bad responses

    # Parse the JSON response
    models = response.json()["data"]
    return models
