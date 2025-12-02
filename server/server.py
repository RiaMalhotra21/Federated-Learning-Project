import torch
import requests
from flask import Flask, request, jsonify
import json
import sys
import os
import pickle
import threading
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ensure project root is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import FraudDetectionModel, load_and_prepare_data

app = Flask(__name__)

class SimpleServer:
    def __init__(self):
        self.global_model = None
        self.client_models = {}           # client_id -> state_dict (torch tensors)
        self.round = 0
        self.evaluation_history = []
        self.lock = threading.Lock()
        self.expected_clients = 2         # number of clients to wait for before aggregating
        self.input_dim = None
        self.X_test, self.y_test = None, None

        # Load test data and initialize global model
        self.load_test_data()

    def load_test_data(self):
        """Load test data and initialize model"""
        try:
            # Use Bank-1 test file as canonical test set (adjust path if needed)
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            test_csv_path = os.path.join(base_dir, "Bank-1", "bank1.csv")

            _, X_test, _, y_test, input_dim = load_and_prepare_data(
                test_csv_path,
                apply_smote=False)
            self.X_test, self.y_test, self.input_dim = X_test, y_test, input_dim
            self.global_model = FraudDetectionModel(input_dim)
            print(f"✅ Loaded test data: {len(X_test)} samples | Input dim: {input_dim}")
        except Exception as e:
            print(f"❌ Error loading test data: {e}")

    def evaluate_global_model(self):
        """Evaluate global model on server test set"""
        if self.X_test is None or self.y_test is None or self.global_model is None:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}

        self.global_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(self.X_test)
            outputs = self.global_model(X_tensor)
            probs = outputs.cpu().numpy().flatten()
            y_pred = (probs >= 0.5).astype(int)
            y_true = np.array(self.y_test).astype(int)

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            print(f"📊 Eval — Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

            return {
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1_score': float(f1)
            }

    def save_global_model(self):
        """Save global model and metrics to each bank folder"""
        try:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            model_state = self.global_model.state_dict()

            for bank in ["Bank-1", "Bank-2"]:
                model_path = os.path.join(base_dir, bank, "global_model.pth")
                torch.save(model_state, model_path)

            metrics = self.evaluate_global_model()
            self.evaluation_history.append({'round': self.round, 'metrics': metrics})

            for bank in ["Bank-1", "Bank-2"]:
                metrics_path = os.path.join(base_dir, bank, "evaluation_metrics.pkl")
                with open(metrics_path, 'wb') as f:
                    pickle.dump({
                        'current_round': self.round,
                        'current_metrics': metrics,
                        'history': self.evaluation_history
                    }, f)

            print("✅ Saved global model and metrics to both banks.")
        except Exception as e:
            print(f"❌ Error saving global model: {e}")

    def aggregate_models(self):
        """Federated Averaging (robust): average only common keys, keep other params intact."""
        with self.lock:
            if not self.client_models:
                print("⚠️ No client models received yet.")
                return False

            num_models = len(self.client_models)
            print(f"🔄 Aggregating {num_models} client models...")

            try:
                global_state = self.global_model.state_dict()

                # collect client key sets
                client_key_sets = [set(ms.keys()) for ms in self.client_models.values()]

                # intersection of keys across global and clients
                common_keys = set(global_state.keys())
                for s in client_key_sets:
                    common_keys &= s

                if not common_keys:
                    print("❌ No common parameter keys found between global and client models.")
                    for cid, ms in self.client_models.items():
                        print(f"Client {cid} keys (example): {sorted(list(ms.keys()))[:10]} ... total {len(ms)}")
                    print(f"Global keys (example): {sorted(list(global_state.keys()))[:10]} ... total {len(global_state)}")
                    return False

                # initialize accumulators using shape of global params
                averaged_state = {}
                for k in common_keys:
                    averaged_state[k] = torch.zeros_like(global_state[k], dtype=torch.float32)

                # sum client parameters
                for client_id, model_state in self.client_models.items():
                    for k in common_keys:
                        client_tensor = model_state[k].to(dtype=averaged_state[k].dtype)
                        if client_tensor.shape != averaged_state[k].shape:
                            raise ValueError(f"Shape mismatch for key '{k}' from client '{client_id}': "
                                             f"client {client_tensor.shape} vs global {averaged_state[k].shape}")
                        averaged_state[k] += client_tensor

                # divide to get average
                for k in common_keys:
                    averaged_state[k] = averaged_state[k] / float(num_models)

                # update global state (only keys in common_keys)
                new_global_state = global_state.copy()
                for k, v in averaged_state.items():
                    new_global_state[k] = v

                # load into global model
                self.global_model.load_state_dict(new_global_state)
                self.round += 1
                self.save_global_model()
                print(f"✅ Aggregation complete — Global Round {self.round}")
                return True

            except Exception as e:
                print(f"❌ Error during aggregation: {e}")
                return False

server = SimpleServer()

# -------------------------
# Flask endpoints
# -------------------------
@app.route('/submit_model', methods=['POST'])
def submit_model():
    """
    Accepts JSON:
      {
        "client_id": "Bank-1",
        "model_state": { "layer.weight": [[...], ...], ... },
        "round": 0   # optional: client's local round index
      }
    The server converts values -> torch tensors and strips 'module.' prefix if present.
    """
    data = request.json
    client_id = data.get('client_id', 'unknown')
    raw_state = data.get('model_state', {})

    try:
        model_state = {}
        for k, v in raw_state.items():
            # normalize key names
            norm_k = k
            if norm_k.startswith("module."):
                norm_k = norm_k[len("module."):]
            # convert to tensor (ensure float32)
            t = torch.tensor(v, dtype=torch.float32)
            model_state[norm_k] = t
    except Exception as e:
        print(f"❌ Error converting incoming model_state from {client_id}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 400

    with server.lock:
        server.client_models[client_id] = model_state
        print(f"📩 Received model from {client_id} — Total received: {len(server.client_models)}")
    return jsonify({'status': 'received'})

@app.route('/aggregate', methods=['POST'])
def aggregate():
    """Trigger aggregation when coordinator asks for it."""
    with server.lock:
        if len(server.client_models) < server.expected_clients:
            return jsonify({'status': 'waiting', 'clients': len(server.client_models)}), 200

    success = server.aggregate_models()
    if success:
        metrics = server.evaluate_global_model()
        # clear client models to prepare for next round
        with server.lock:
            server.client_models.clear()
        return jsonify({'status': 'success', 'round': server.round, 'metrics': metrics}), 200
    else:
        return jsonify({'status': 'failed'}), 500

@app.route('/status', methods=['GET'])
def status():
    """Return current server status and evaluation metrics"""
    metrics = server.evaluate_global_model()
    with server.lock:
        num_clients = len(server.client_models)
    return jsonify({
        'round': server.round,
        'num_clients': num_clients,
        'metrics': metrics
    }), 200

@app.route('/get_global', methods=['GET'])
def get_global():
    """Return the current global model state as JSON-serializable lists"""
    if server.global_model is None:
        return jsonify({'status': 'no_model'}), 404

    state = server.global_model.state_dict()
    serial = {k: v.cpu().numpy().tolist() for k, v in state.items()}
    return jsonify({'status': 'ok', 'round': server.round, 'model_state': serial}), 200

@app.route('/evaluation_history', methods=['GET'])
def get_evaluation_history():
    """Return evaluation metrics history for dashboard"""
    return jsonify({
        'history': server.evaluation_history,
        'current_round': server.round
    }), 200

if __name__ == '__main__':
    print("🚀 Simple Federated Learning Server Starting...")
    print("Server will listen on http://localhost:5000")
    # Ensure we run from server directory (this file's dir)
    app.run(host='0.0.0.0', port=5000, debug=True)