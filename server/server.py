import torch
import requests
from flask import Flask, request, jsonify
import json
import sys
import os
import pickle
import threading
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Add parent directory to path so we can import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SimpleFraudModel, load_and_prepare_data  # keep evaluate in server

app = Flask(__name__)

# Simple server class
class SimpleServer:
    def __init__(self):
        self.global_model = SimpleFraudModel()
        self.client_models = {}      # dict: client_id -> state_dict
        self.round = 0               # server-side canonical round counter
        self.evaluation_history = []
        self.lock = threading.Lock() # protect client_models & round
        self.expected_clients = 2    # number of banks/clients expected each round
        self.aggregating = False     # guard to prevent concurrent aggregations
        # Load test data for evaluation
        self.load_test_data()
    
    def load_test_data(self):
        """Load test data for evaluation"""
        try:
            # Use Bank-1 data for testing
            self.X_test, self.y_test = load_and_prepare_data("../Bank-1/bank1.csv")
            print(f"✅ Loaded test data: {len(self.X_test)} samples")
        except Exception as e:
            print(f"❌ Error loading test data: {e}")
            self.X_test, self.y_test = None, None

    def evaluate_global_model(self):
        """Evaluate global model and return metrics (robust for binary/multi-class), with debug stats."""
        if self.X_test is None or self.y_test is None:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'confusion_matrix': None, 'label_dist': None}
        
        try:
            self.global_model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(self.X_test)
                outputs = self.global_model(X_tensor)  # shape -> (N,) or (N, C)

                # log the raw outputs shape and stats for debugging
                try:
                    out_np = outputs.cpu().numpy()
                    print(f"🔎 Raw outputs shape: {out_np.shape}; dtype: {out_np.dtype}")
                    print(f"    out stats -> min: {out_np.min():.6f}, max: {out_np.max():.6f}, mean: {out_np.mean():.6f}, std: {out_np.std():.6f}")
                except Exception as e:
                    print(f"⚠️ Could not convert outputs to numpy for stats: {e}")

                # Binary vs multi-class handling
                if outputs.dim() == 1 or (outputs.dim() == 2 and outputs.shape[1] == 1):
                    # Single-logit case (BCEWithLogits style) — apply sigmoid
                    logits = outputs.view(-1)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    y_pred = (probs >= 0.5).astype(int)
                    try:
                        print(f"    probs stats -> min: {probs.min():.6f}, max: {probs.max():.6f}, mean: {probs.mean():.6f}")
                    except Exception:
                        pass
                else:
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    y_pred = np.argmax(probs, axis=1)
                    try:
                        print(f"    softmax probs sample row sums (should be 1): {probs.sum(axis=1)[:5]}")
                    except Exception:
                        pass

                y_true = np.array(self.y_test).astype(int)
                unique, counts = np.unique(y_true, return_counts=True)
                label_dist = dict(zip(unique.tolist(), counts.tolist()))
                cm = confusion_matrix(y_true, y_pred)

                acc = accuracy_score(y_true, y_pred)
                if len(unique) > 2:
                    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
                    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
                    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
                else:
                    prec = precision_score(y_true, y_pred, average='binary', zero_division=0)
                    rec = recall_score(y_true, y_pred, average='binary', zero_division=0)
                    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

                print("🧾 Evaluation summary:")
                print(f"  Round (server-side): {self.round}")
                print(f"  Label distribution (y_true): {label_dist}")
                print(f"  Confusion matrix:\n{cm}")
                print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

                return {
                    'accuracy': float(acc),
                    'precision': float(prec),
                    'recall': float(rec),
                    'f1_score': float(f1),
                    'confusion_matrix': cm.tolist(),
                    'label_dist': label_dist
                }
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'confusion_matrix': None, 'label_dist': None}
    
    def save_global_model(self):
        """Save global model to both bank directories and persist evaluation metrics"""
        try:
            # Save model state dict
            model_state = self.global_model.state_dict()
            
            # Save to Bank-1 directory
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            bank1_path = os.path.join(base_dir, "Bank-1", "global_model.pth")
            torch.save(model_state, bank1_path)
            print(f"✅ Saved global model to Bank-1: {bank1_path}")
            
            # Save to Bank-2 directory
            bank2_path = os.path.join(base_dir, "Bank-2", "global_model.pth")
            torch.save(model_state, bank2_path)
            print(f"✅ Saved global model to Bank-2: {bank2_path}")
            
            # Save evaluation metrics
            metrics = self.evaluate_global_model()
            # Append snapshot with the round AFTER aggregation (server.round currently the new value)
            self.evaluation_history.append({
                'round': self.round,
                'metrics': metrics
            })
            
            # Save metrics to both directories
            metrics_data = {
                'current_round': self.round,
                'current_metrics': metrics,
                'history': self.evaluation_history
            }
            
            # Save to Bank-1
            metrics_path1 = os.path.join(base_dir, "Bank-1", "evaluation_metrics.pkl")
            with open(metrics_path1, 'wb') as f:
                pickle.dump(metrics_data, f)
            
            # Save to Bank-2
            metrics_path2 = os.path.join(base_dir, "Bank-2", "evaluation_metrics.pkl")
            with open(metrics_path2, 'wb') as f:
                pickle.dump(metrics_data, f)
            
            print(f"✅ Saved evaluation metrics to both banks")
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
    
    def _state_dict_param_summary(self, state):
        """Return quick summary: param keys, shapes, and sum total for debugging"""
        summary = {}
        total_sum = 0.0
        for k, v in state.items():
            try:
                arr = v.cpu().numpy()
                total_sum += float(arr.sum())
                summary[k] = {'shape': arr.shape}
            except Exception:
                summary[k] = {'shape': 'unknown'}
        return total_sum, summary

    def aggregate_models(self):
        """Simple FedAvg - average the models (thread-safe)"""
        with self.lock:
            if not self.client_models:
                print("No models to aggregate!")
                return False
            
            num_models = len(self.client_models)
            print(f"Aggregation requested. {num_models} models available for aggregation. Server round={self.round}")
            
            # Get all model states
            model_states = list(self.client_models.values())
            
            # Print per-model debug summary
            for idx, state in enumerate(model_states):
                ssum, ssummary = self._state_dict_param_summary(state)
                print(f"  - Model {idx} param sum (checksum): {ssum:.6e}; keys sample: {list(ssummary.keys())[:5]}")
            
            # Average the parameters into a new averaged_state
            averaged_state = {}
            global_keys = self.global_model.state_dict().keys()
            # initialize zero tensors with same shape/dtype
            for param_name in global_keys:
                averaged_state[param_name] = torch.zeros_like(self.global_model.state_dict()[param_name], dtype=torch.float32)
            
            for model_state in model_states:
                for param_name in global_keys:
                    averaged_state[param_name] += model_state[param_name].float()
            
            # finalize average
            for param_name in global_keys:
                averaged_state[param_name] /= float(num_models)
            
            # Update global model with averaged_state
            try:
                self.global_model.load_state_dict(averaged_state)
            except Exception as e:
                print(f"❌ Error loading averaged state into global model: {e}")
                return False
            
            # Increment round AFTER successful load
            self.round += 1
            
            # Save the global model and evaluation metrics
            self.save_global_model()
            
            print(f"✅ Global model updated! New server round: {self.round}")
            # NOTE: do NOT clear client_models here; clearing is done by the route (so we keep atomicity)
            return True

# Create server instance
server = SimpleServer()

@app.route('/get_model', methods=['GET'])
def get_model():
    """Send global model to clients"""
    model_state = server.global_model.state_dict()
    
    # Convert to lists for JSON (keep small sample logs)
    serializable_state = {}
    for key, value in model_state.items():
        serializable_state[key] = value.cpu().numpy().tolist()
    
    return jsonify({
        'model_state': serializable_state,
        'round': server.round
    })

@app.route('/submit_model', methods=['POST'])
def submit_model():
    """Receive model from client"""
    data = request.json
    client_id = data.get('client_id', 'unknown')
    model_state = data['model_state']
    
    # Convert back to tensors
    state_dict = {}
    for key, value in model_state.items():
        state_dict[key] = torch.tensor(value)
    
    # Store client model in a thread-safe way and log summary
    with server.lock:
        server.client_models[client_id] = state_dict
        num_submitted = len(server.client_models)
    
    # Quick debug summary of this submission
    ssum, ssummary = server._state_dict_param_summary(state_dict)
    print(f"Received model from {client_id}; param checksum: {ssum:.6e}; submitted count: {num_submitted}")
    
    return jsonify({'status': 'success', 'submitted_count': num_submitted})

@app.route('/aggregate', methods=['POST'])
def aggregate():
    """Trigger aggregation but only when expected_clients have submitted."""
    # Quick guard to avoid concurrent aggregator calls
    with server.lock:
        if server.aggregating:
            return jsonify({'status': 'busy', 'message': 'Aggregation already in progress', 'round': server.round}), 200
        # Check how many clients have submitted
        num_submitted = len(server.client_models)
        current_round = server.round
        # If not enough submissions, skip aggregation
        if num_submitted < server.expected_clients:
            print(f"[Aggregate] Not enough models submitted ({num_submitted}/{server.expected_clients}) — skipping aggregation for round {current_round}.")
            return jsonify({
                'status': 'not_ready',
                'message': f'Need {server.expected_clients} submissions but only {num_submitted} present.',
                'round': current_round,
                'submitted': num_submitted
            }), 200
        # mark aggregating
        server.aggregating = True

    # perform aggregation (aggregation function uses its own locking)
    try:
        success = server.aggregate_models()
        if success:
            # Clear client_models for next round (do inside lock)
            with server.lock:
                server.client_models = {}
                server.aggregating = False
            current_metrics = server.evaluate_global_model()
            return jsonify({'status': 'success', 'round': server.round, 'metrics': current_metrics})
        else:
            with server.lock:
                server.aggregating = False
                num_clients = len(server.client_models)
            return jsonify({'error': 'aggregation_failed', 'round': server.round, 'submitted_clients': num_clients}), 500
    except Exception as e:
        with server.lock:
            server.aggregating = False
        print(f"❌ Exception during aggregation route: {e}")
        return jsonify({'error': 'exception', 'message': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    """Get server status"""
    current_metrics = server.evaluate_global_model()
    with server.lock:
        num_clients = len(server.client_models)
    return jsonify({
        'round': server.round,
        'num_clients': num_clients,
        'metrics': current_metrics
    })

@app.route('/evaluation_history', methods=['GET'])
def get_evaluation_history():
    """Get evaluation history"""
    return jsonify({
        'history': server.evaluation_history,
        'current_round': server.round
    })

if __name__ == '__main__':
    print("🚀 Simple Federated Learning Server Starting...")
    print("Server will listen on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
