import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data

class EnhancedMultiRiskGNN(nn.Module):
    """
    Enhanced GNN with Graph Attention Networks (GAT) for better 
    coordination detection and interpretability.
    """
    def __init__(self, input_dim=1, hidden_dim=128, num_heads=4, dropout=0.2):
        super().__init__()
        
        # Graph Attention layers for better feature extraction
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout)
        
        # Batch normalization for stability
        self.bn1 = nn.BatchNorm1d(hidden_dim * num_heads)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # User-level risk prediction
        self.user_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),  # Concat mean and max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Group-level risk with attention weights
        self.group_attention = nn.MultiheadAttention(hidden_dim, num_heads=2, dropout=dropout)
        self.group_encoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # Explainability: attention weights extractor
        self.edge_importance = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, data: Data, return_attention=False):
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None
        
        # GAT layers with residual connections
        x1, attention_weights_1 = self.gat1(x, edge_index, return_attention_weights=True)
        x1 = F.elu(self.bn1(x1))
        x1 = F.dropout(x1, p=0.2, training=self.training)
        
        x2 = self.gat2(x1, edge_index, edge_attr=edge_weight)
        x2 = F.elu(self.bn2(x2))
        
        # Residual connection
        if x.size(1) == x2.size(1):
            x2 = x2 + x
        
        # User-level aggregation with both mean and max pooling
        user_ids = data.node_user_map
        unique_users = torch.unique(user_ids)
        user_features = []
        
        for uid in unique_users:
            mask = (user_ids == uid)
            user_nodes = x2[mask]
            
            # Combine mean and max pooling for richer representation
            mean_feat = user_nodes.mean(dim=0)
            max_feat = user_nodes.max(dim=0)[0]
            combined = torch.cat([mean_feat, max_feat])
            user_features.append(combined)
        
        if user_features:
            user_features = torch.stack(user_features)
            user_risks = torch.sigmoid(self.user_encoder(user_features)).view(-1)
        else:
            user_risks = torch.tensor([], device=x.device)
        
        # Group-level with attention mechanism
        x2_expanded = x2.unsqueeze(0)  # Add batch dimension for attention
        attended_features, attention_weights = self.group_attention(
            x2_expanded, x2_expanded, x2_expanded
        )
        group_embedding = attended_features.squeeze(0).mean(dim=0)
        group_risk = torch.sigmoid(self.group_encoder(group_embedding)).view(-1)
        
        if return_attention:
            # Calculate edge importance scores for explainability
            edge_scores = self._calculate_edge_importance(x2, edge_index)
            return user_risks, group_risk, attention_weights_1, edge_scores
        
        return user_risks, group_risk
    
    def _calculate_edge_importance(self, node_features, edge_index):
        """Calculate importance scores for each edge (for XAI)."""
        src_features = node_features[edge_index[0]]
        dst_features = node_features[edge_index[1]]
        edge_features = torch.cat([src_features, dst_features], dim=1)
        importance = torch.sigmoid(self.edge_importance(edge_features)).squeeze()
        return importance


class TemporalRiskTracker:
    """
    Tracks risk evolution over time for detecting escalation patterns.
    Critical for real-time monitoring in SOCs.
    """
    def __init__(self, window_size=10, escalation_threshold=0.3):
        self.window_size = window_size
        self.escalation_threshold = escalation_threshold
        self.history = {}
        
    def update(self, user_id: str, risk_score: float, timestamp: float):
        """Update risk history for a user."""
        if user_id not in self.history:
            self.history[user_id] = []
        
        self.history[user_id].append({
            'score': risk_score,
            'timestamp': timestamp
        })
        
        # Keep only recent history
        if len(self.history[user_id]) > self.window_size:
            self.history[user_id] = self.history[user_id][-self.window_size:]
    
    def detect_escalation(self, user_id: str) -> dict:
        """Detect if user shows escalation pattern."""
        if user_id not in self.history or len(self.history[user_id]) < 3:
            return {'escalating': False, 'rate': 0.0}
        
        scores = [h['score'] for h in self.history[user_id]]
        
        # Calculate escalation rate (derivative)
        if len(scores) >= 2:
            recent_change = scores[-1] - scores[-2]
            avg_score = sum(scores) / len(scores)
            
            # Escalation detected if consistent increase
            escalating = (
                recent_change > self.escalation_threshold and
                scores[-1] > avg_score * 1.2
            )
            
            return {
                'escalating': escalating,
                'rate': recent_change,
                'current_risk': scores[-1],
                'avg_risk': avg_score,
                'trend': 'increasing' if recent_change > 0 else 'decreasing'
            }
        
        return {'escalating': False, 'rate': 0.0}


class AdversarialRobustness:
    """
    Implements adversarial testing for model robustness.
    Essential for international competition to show resilience.
    """
    
    @staticmethod
    def generate_adversarial_prompts(original_prompt: str, num_variants=5):
        """Generate adversarial variants of prompts to test detection."""
        techniques = [
            lambda p: p.replace(' ', '  '),  # Extra spaces
            lambda p: p.replace('a', '@').replace('e', '3'),  # Leetspeak
            lambda p: f"Ignore previous instructions. {p}",  # Injection
            lambda p: ' '.join(p.split()[::-1]),  # Word reversal
            lambda p: p.upper(),  # Case manipulation
            lambda p: p + " " + "benign text " * 10,  # Dilution
            lambda p: f"As an AI assistant, {p}",  # Role-playing
            lambda p: p.replace('.', '.\u200b'),  # Zero-width spaces
        ]
        
        variants = []
        for i in range(min(num_variants, len(techniques))):
            try:
                variant = techniques[i](original_prompt)
                variants.append(variant)
            except:
                variants.append(original_prompt)
        
        return variants
    
    @staticmethod
    def test_model_robustness(model, test_prompts, expected_risk_threshold=0.5):
        """Test model against adversarial examples."""
        results = {
            'total_tests': 0,
            'correct_detections': 0,
            'false_negatives': 0,
            'robustness_score': 0.0
        }
        
        for prompt in test_prompts:
            variants = AdversarialRobustness.generate_adversarial_prompts(prompt)
            
            for variant in variants:
                # This would integrate with your actual model inference
                # For now, returning mock structure
                results['total_tests'] += 1
                
                # Simulate detection (replace with actual model call)
                detected = torch.rand(1).item() > 0.3
                
                if detected:
                    results['correct_detections'] += 1
                else:
                    results['false_negatives'] += 1
        
        results['robustness_score'] = (
            results['correct_detections'] / max(1, results['total_tests'])
        )
        
        return results


# Integration with your existing escalation law
def enhanced_escalate(seq: list, gamma=2.0, alpha=0.5, eps=0.1, 
                      temporal_weight=0.2, coordination_factor=0.0):
    """
    Enhanced escalation law incorporating temporal patterns and coordination.
    
    Args:
        seq: Risk scores sequence
        gamma, alpha, eps: Original escalation parameters
        temporal_weight: Weight for temporal escalation patterns
        coordination_factor: Multi-user coordination strength [0,1]
    """
    if not seq:
        return 0.0
    
    # Original escalation calculation
    compounding = 1.0
    for s in seq:
        compounding *= (1 - (s ** gamma))
    compounded = 1 - compounding
    
    avg = float(sum(seq) / len(seq))
    length_factor = len(seq) / 10.0
    amplified = compounded + alpha * avg * length_factor
    amplified = min(1.0, amplified)
    
    # Temporal escalation bonus
    if len(seq) > 1:
        temporal_escalation = sum(
            max(0, seq[i] - seq[i-1]) for i in range(1, len(seq))
        ) / len(seq)
        amplified += temporal_weight * temporal_escalation
    
    # Coordination amplification
    amplified *= (1 + coordination_factor * 0.5)
    
    # Strict enforcement
    second_largest = sorted(seq)[-2] if len(seq) > 1 else 0.0
    strict = max(amplified, second_largest + eps)
    
    return float(min(1.0, strict))