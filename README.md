# Federated Learning for Bank Fraud Detection

## 🔍 Overview
This project simulates a federated learning setup between two banks (Bank1 and Bank2)
to collaboratively train a model for fraud detection without sharing raw data.

## ⚙️ Architecture
- Each bank trains a local Neural Network on its private transaction data.
- The central server aggregates model parameters (FedAvg approach).
- A dashboard visualizes accuracy, loss, and dataset stats.

## 📂 Repository Structure
├── bank1_client.py  
├── bank2_client.py  
├── server.py  
├── model.py  
├── dataset/  
├── static/ & templates/ (for dashboard)  
├── results/ (plots, metrics)  

## 📈 Results
Current accuracy: ~50%  
Next improvements planned:
- Model tuning (optimizer, dropout)
- Balanced dataset handling
- Increased training rounds

## 🧩 Tech Stack
Python, TensorFlow/Keras, Flask, HTML/CSS/JS

## 🚀 How to Run
1. Run `server.py`
2. Start each bank client (`bank1_client.py`, `bank2_client.py`)
3. Open `localhost:5000` to view dashboard
