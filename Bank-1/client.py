import os
import time
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import FraudDetectionModel, load_and_prepare_data, train_model, evaluate_model

SERVER_URL = "http://localhost:5000"
CLIENT_ID = "Bank-1"
ROUNDS = 5
EPOCHS_PER_ROUND = 5          # reduce for quicker iterations if needed
SLEEP_BETWEEN_STEPS = 2

def serialize_state(state_dict):
    """Convert torch state_dict to JSON-serializable lists"""
    return {k: v.cpu().numpy().tolist() for k, v in state_dict.items()}

def deserialize_state_to_model(model, state_json):
    """Load JSON state into model (handles missing keys and dtype)"""
    state = {}
    for k, v in state_json.items():
        # guard: strip possible 'module.' prefix
        norm_k = k
        if norm_k.startswith("module."):
            norm_k = norm_k[len("module."):]
        state[norm_k] = torch.tensor(v, dtype=torch.float32)
    # Only load keys that match
    model_state = model.state_dict()
    for k in list(model_state.keys()):
        if k in state and state[k].shape == model_state[k].shape:
            model_state[k] = state[k]
    model.load_state_dict(model_state)
    return model

def send_model_to_server(model):
    payload = {
        'client_id': CLIENT_ID,
        'model_state': serialize_state(model.state_dict())
    }
    try:
        r = requests.post(f"{SERVER_URL}/submit_model", json=payload, timeout=10)
        return r.status_code == 200 and r.json().get('status') == 'received'
    except Exception as e:
        print(f"❌ Error sending model: {e}")
        return False

def fetch_global_model():
    try:
        r = requests.get(f"{SERVER_URL}/get_global", timeout=10)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception as e:
        print(f"❌ Error fetching global model: {e}")
        return None

def wait_for_aggregation(target_round, poll_interval=2, timeout=120):
    """Poll server until server.round >= target_round (or timeout)"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{SERVER_URL}/status", timeout=5)
            if r.status_code == 200:
                jr = r.json()
                if jr.get('round', 0) >= target_round:
                    return True
        except:
            pass
        time.sleep(poll_interval)
    return False

def main():
    # load data
    file_path = os.path.join(os.path.dirname(__file__), "bank1.csv")
    X_train, X_test, y_train, y_test, input_dim = load_and_prepare_data(file_path, apply_smote=True)

    # initialize model
    model = FraudDetectionModel(input_dim)

    # main federated loop
    for rnd in range(ROUNDS):
        print(f"\n🔁 CLIENT {CLIENT_ID} — Local Round {rnd+1}/{ROUNDS}")

        # local training (you can change to train only a few epochs per round)
        model = train_model(model, X_train, y_train, epochs=EPOCHS_PER_ROUND, lr=0.0007)

        # evaluate locally for logging
        evaluate_model(model, X_test, y_test)

        # send model to server
        success = send_model_to_server(model)
        if not success:
            print("❌ Failed to send model to server. Will retry shortly...")
            time.sleep(5)
            success = send_model_to_server(model)
            if not success:
                print("❌ Second attempt failed. Exiting client.")
                return

        # wait until server finishes aggregation for this round (server.round should increment)
        current_server_round = 0
        # we expect server to increase to rnd+1 after aggregation
        target_round = rnd + 1
        print(f"⏳ Waiting for server aggregation to complete for round {target_round}...")
        ok = wait_for_aggregation(target_round, poll_interval=2, timeout=180)
        if not ok:
            print("❌ Timeout waiting for aggregation. Continuing anyway.")
        else:
            print("✅ Detected aggregated global model on server. Fetching...")

            gm = fetch_global_model()
            if gm and gm.get('status') == 'ok':
                model = deserialize_state_to_model(model, gm['model_state'])
                print("✅ Global model loaded into client.")
            else:
                print("⚠️ Could not fetch global model content.")

        time.sleep(SLEEP_BETWEEN_STEPS)

    print("\n🎉 CLIENT finished all rounds. Exiting.")

if __name__ == "__main__":
    main()
