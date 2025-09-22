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
        # Read the test data
        df = pd.read_csv(DATA_PATH)

        # Determine how many samples to send (up to 5 if available)
        n_total = min(5, len(df))

        # Try to ensure the FIRST sample is an ATTACK, based on available columns
        attack_df = pd.DataFrame()
        if 'Attack_label' in df.columns:
            # Consider both numeric and string encodings
            series_num = pd.to_numeric(df['Attack_label'], errors='coerce')
            series_str = df['Attack_label'].astype(str).str.lower().str.strip()
            mask_attack = (series_num == 1) | (series_str.isin(['1', 'attack', 'malicious', 'true', 'yes']))
            attack_df = df[mask_attack]
        elif 'Attack_type' in df.columns:
            series_type = df['Attack_type'].astype(str).str.lower().str.strip()
            # Treat non-benign/normal as attack
            mask_attack = ~series_type.isin(['benign', 'normal', 'none', ''])
            attack_df = df[mask_attack]

        if not attack_df.empty and n_total > 0:
            # Pick one attack sample to be FIRST
            first_attack_row = attack_df.sample(1, random_state=None)
            # Sample the remaining rows from the rest of the dataset (avoid duplicate index)
            remaining_pool = df.drop(first_attack_row.index, errors='ignore')
            remaining_n = max(0, n_total - 1)
            if len(remaining_pool) >= remaining_n:
                remaining_rows = remaining_pool.sample(remaining_n, random_state=None) if remaining_n > 0 else remaining_pool.iloc[0:0]
            else:
                # If dataset is tiny, sample with replacement to reach desired count
                remaining_rows = remaining_pool.sample(remaining_n, replace=True, random_state=None) if remaining_n > 0 else remaining_pool.iloc[0:0]
            samples_df = pd.concat([first_attack_row, remaining_rows], ignore_index=True)
            print("\n1. Selected samples ensuring the FIRST is an ATTACK.")
        else:
            # Fallback: random selection (cannot determine attacks)
            samples_df = df.sample(n_total, random_state=None)
            print("\n1. Selected random samples (could not ensure first is attack).")

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
    # Before dropping label columns, log what the FIRST sample was in terms of labels
    try:
        first_label_val = samples_df['Attack_label'].iloc[0] if 'Attack_label' in samples_df.columns else None
        first_type_val = samples_df['Attack_type'].iloc[0] if 'Attack_type' in samples_df.columns else None
    except Exception:
        first_label_val, first_type_val = None, None

    payload = {
        "device_id": f"test-device-{random.randint(100, 999)}",
        "samples": samples_list,
    }

    print("\n2. Constructed the following JSON payload (features only):")
    print(json.dumps(payload, indent=2))
    print("\n2a. Debug — first sample original labels before drop:")
    print(f"   Attack_label={first_label_val} | Attack_type={first_type_val}")

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