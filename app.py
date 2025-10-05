# app.py
"""
Flask API backend for AegisAI.
- Endpoints for GNN analysis, Red-Team generation, and Cross-Border Sharing.
- Integrates Gemini for Narrative Synthesis and XAI features.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import random
import time
import json
import torch
import requests
from typing import List, Dict

# Import GNN components and advanced features from gnn_risk_engine
from gnn_risk_engine import (
    MultiRiskGNN, build_group_graph, generate_stylometric_features, 
    generate_red_team_scenario, RELATION_SCORER, DEVICE
)

# --- Configuration ---
app = Flask(__name__)
CORS(app) 

# --- Gemini API Configuration (for Narrative Synthesis) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_FALLBACK_GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}".format(GEMINI_MODEL, GEMINI_API_KEY)


# --- Global Model and State ---
# NOTE: The size of the hidden layer here (128) must match the GAT model defined in gnn_risk_engine.
MODEL_PATH = "multi_risk_gnn.pt"
MODEL = MultiRiskGNN(hidden_dim=128).to(DEVICE) # Updated hidden_dim to 128 for GAT
MODEL.eval()

try:
    # Attempt to load trained weights (created by unified_pipeline.py)
    MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"Successfully loaded trained GNN model from {MODEL_PATH}")
except Exception as e:
    print(f"WARNING: GNN model weights not found or failed to load ({e}). Using untrained model.")

# --- Real-Time LLM Risk Scorer (for GNN Node Features) ---
def get_llm_risk_score(prompt: str) -> float:
    """Simulates LLM-based per-prompt risk score. Mocked for node feature generation."""
    prompt_lower = prompt.lower()
    if any(k in prompt_lower for k in ["hack", "exploit", "password", "jailbreak", "cve"]):
        return float(min(1.0, 0.7 + random.random() * 0.3)) 
    if len(prompt) > 50 and any(k in prompt_lower for k in ["system", "protocol", "architecture"]):
        return float(min(0.5, 0.2 + random.random() * 0.3))
    return float(random.random() * 0.2)

# --- Gemini Narrative Synthesis ---
def get_narrative_analysis(session_data: List[Dict], group_risk: float) -> Dict:
    """Mocks Gemini call for narrative synthesis and policy check."""
    
    full_session_text = "\n".join([f"User {u['user_id']}: {', '.join(u['prompts'])}" for u in session_data])
    
    # Policy Compliance Mock Logic
    policy_risk = "Low (Internal Policy 0.2)";
    if group_risk >= 0.8:
        policy_risk = "CRITICAL (Violation of Policy 4.1: Intentional Security Breach)";
    elif group_risk >= 0.5:
        policy_risk = "Medium (Violation of Policy 2.3: Unauthorized Data Gathering)";

    if GEMINI_API_KEY == "YOUR_FALLBACK_GEMINI_API_KEY" or not GEMINI_API_KEY:
        is_risky = group_risk >= 0.5
        return {
            "malign_narrative": f"MOCK ANALYSIS: {'High-Confidence Coordinated Attack' if is_risky else 'Low-Level Inquiry'}. Intent suggests {'Data Exfiltration' if is_risky else 'General Query'} (Risk: {group_risk:.2f}).",
            "coordination_summary": "Simulated high semantic overlap and GNN detected strong cross-user edges, indicating sophisticated coordination.",
            "mitigation_action": "Alert national CERT-In, initiate automatic rate limiting, and quarantine user IP prefixes.",
            "policy_compliance_risk": policy_risk
        }
    
    # Live API Logic (Requires full client implementation for production)
    
    return {
        "malign_narrative": f"LIVE (MOCK): Threat Level {group_risk:.2f} detected.",
        "coordination_summary": "GAT confirmed strong attention weights between user nodes.",
        "mitigation_action": "Initiate network traffic inspection.",
        "policy_compliance_risk": policy_risk
    }


# --- MAIN ENDPOINT: /analyze_gnn ---
@app.route('/analyze_gnn', methods=['POST'])
def analyze_gnn():
    start_time = time.time()
    try:
        data = request.json
        users_data = data.get('users', [])
        
        # 1. Local Risk Estimation (LLM Scoring)
        processed_users = []
        all_prompts = []
        for user in users_data:
            # The input from the frontend is a single comma-separated string
            prompts_list = user['prompts'].split(',')
            prompts = [p.strip() for p in prompts_list if p.strip()]
            
            llm_scores = [get_llm_risk_score(p) for p in prompts]
            processed_users.append({
                "user_id": user["id"],
                "prompts": prompts,
                "llm_scores": llm_scores
            })
            all_prompts.extend(prompts)
            
        # 2. Graph Construction & GNN Inference
        gnn_data = build_group_graph(processed_users, relation_scorer=RELATION_SCORER)

        with torch.no_grad():
            user_out, group_out = MODEL(gnn_data.to(DEVICE))

        # 3. Prepare Core Metrics
        group_risk = round(float(group_out.item()), 4)
        
        escalation_level = "LOW"
        if group_risk >= 0.8: escalation_level = "CRITICAL"
        elif group_risk >= 0.5: escalation_level = "HIGH"

        # 4. XAI / Stylometric Features
        xai_features = generate_stylometric_features(all_prompts)
        
        # 5. Gemini Narrative Synthesis
        narrative_analysis = get_narrative_analysis(processed_users, group_risk)
        
        # 6. Prepare Response and calculate latency
        latency_ms = round((time.time() - start_time) * 1000, 2)
        
        graph_json = {
            "nodes": [
                {"id": i, "group": int(g), "risk": float(r)} 
                for i, (g, r) in enumerate(zip(gnn_data.node_user_map.tolist(), gnn_data.x.squeeze(1).tolist()))
            ],
            "links": [
                {"source": int(gnn_data.edge_index[0, i]), "target": int(gnn_data.edge_index[1, i]), "weight": float(gnn_data.edge_weight[i])}
                for i in range(gnn_data.edge_index.size(1))
            ]
        }
        coordination_score = round(gnn_data.edge_weight.mean().item(), 4) if gnn_data.edge_weight.numel() else 0.0

        return jsonify({
            "group_risk_score": group_risk,
            "escalation_level": escalation_level,
            "user_risks": [round(float(r), 4) for r in user_out.tolist()],
            "graph_data": graph_json,
            "latency_ms": latency_ms,
            "xai_features": xai_features,
            "narrative_analysis": narrative_analysis,
            "coordination_score": coordination_score,
            "is_mock_analysis": GEMINI_API_KEY == "YOUR_FALLBACK_GEMINI_API_KEY" or not GEMINI_API_KEY
        })

    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({"error": f"Internal Server Error during GNN analysis: {e}"}), 500

# --- RED-TEAM ENDPOINT: /red_team_generate ---
@app.route('/red_team_generate', methods=['GET'])
def red_team_generate():
    """Generates a high-confidence, coordinated scenario for Red-Team testing."""
    try:
        num_users = request.args.get('users', 3, type=int)
        prompts_per_user = request.args.get('prompts', 3, type=int)
        scenario = generate_red_team_scenario(num_users, prompts_per_user)
        
        formatted_scenario = []
        for user_data in scenario:
            formatted_scenario.append({
                "id": user_data['user_id'],
                "prompts": ", ".join(user_data['prompts'])
            })
        
        return jsonify({"scenario": formatted_scenario})
    except Exception as e:
        return jsonify({"error": f"Failed to generate red-team scenario: {e}"}), 500

# --- CROSS-BORDER SHARING ENDPOINT: /share_intel_mock ---
@app.route('/share_intel_mock', methods=['POST'])
def share_intel_mock():
    """Mocks secure, cross-border intelligence sharing (Federated/Blockchain)."""
    data = request.json
    threat_hash = data.get('threat_hash', 'N/A')
    agency = data.get('agency', 'Allied Agency (Mock)')
    
    # Simulate Blockchain logging and secure transmission (Federated Intelligence Framework)
    return jsonify({
        "status": "Shared",
        "agency_confirmation": agency,
        "shared_data_hash": threat_hash,
        "transmission_protocol": "Federated_Secure_API_V1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S Z")
    })

if __name__ == '__main__':
    print("AegisAI GNN Flask API starting on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)