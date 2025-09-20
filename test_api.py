import requests
import pandas as pd
import random
import json
import os
from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

# Get the API port from environment or use 5000 as a default
API_PORT = os.getenv('FLASK_API_PORT', 5000)
API_URL = f"http://localhost:{API_PORT}/analisar"

# Path to the CSV file with test data (as used in your main.py)
DATA_PATH = "data/dados_de_teste.csv"

def run_test():
    """
    Fetches a random sample, sends it to the /analisar endpoint,
    and prints the response.
    """
    print(f"🎯 Attempting to send a request to: {API_URL}")

    # --- 1. Load and Prepare 5 Data Samples ---
    try:
        # Read the test data and grab five random rows (like src/qml/main.py)
        df = pd.read_csv(DATA_PATH)
        samples_df = df.sample(5, random_state=None)
        print("\n1. Selected 5 random samples from the dataset.")

        # Drop labels and convert to list[dict]
        features_df = samples_df.drop(columns=['Attack_label', 'Attack_type'], errors='ignore')
        samples_list = features_df.to_dict(orient='records')

    except FileNotFoundError:
        print(f"\n❌ ERROR: The test data file was not found at '{DATA_PATH}'.")
        print("Please ensure the CSV file exists to run this test.")
        return
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load or process the data file: {e}")
        return

    # --- 2. Construct the API Payload ---
    # The API can accept 'samples' (list[dict]) or a single 'features' dict.
    payload = {
        "device_id": f"test-device-{random.randint(100, 999)}",
        "samples": samples_list,
    }
    
    print("\n2. Constructed the following JSON payload:")
    print(json.dumps(payload, indent=2))

    # --- 3. Send the POST Request ---
    try:
        print("\n3. Sending request to the API...")
        response = requests.post(API_URL, json=payload)
        
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Connection failed. Is the Docker container running?")
        print("Please run 'docker compose up' and try again.")
        return
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ ERROR: The API returned an error (Status Code: {e.response.status_code}).")
        print("Response body:", e.response.text)
        return
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        return

    # --- 4. Print the API Response ---
    print("\n4. ✅ Success! Received response from the API:")
    print(f"   Status Code: {response.status_code}")
    print("   Response JSON:")
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    run_test()