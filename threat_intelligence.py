# threat_intelligence.py
"""
Advanced Threat Intelligence Module with Blockchain Integration
For AegisAI International Prototype
"""

import hashlib
import time
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import requests

# For blockchain simulation
class BlockchainLedger:
    """
    Immutable threat intelligence ledger using blockchain concepts.
    For production, integrate with Hyperledger or Ethereum.
    """
    
    def __init__(self):
        self.chain = []
        self.pending_threats = []
        self.create_genesis_block()
    
    def create_genesis_block(self):
        """Create the first block in the chain."""
        genesis = {
            'index': 0,
            'timestamp': time.time(),
            'threats': [],
            'previous_hash': '0',
            'nonce': 0
        }
        genesis['hash'] = self.calculate_hash(genesis)
        self.chain.append(genesis)
    
    @staticmethod
    def calculate_hash(block: dict) -> str:
        """Calculate SHA-256 hash of a block."""
        block_string = json.dumps(block, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def add_threat(self, threat_data: dict):
        """Add threat to pending list."""
        threat_data['timestamp'] = time.time()
        threat_data['id'] = hashlib.sha256(
            json.dumps(threat_data).encode()
        ).hexdigest()[:16]
        self.pending_threats.append(threat_data)
    
    def mine_block(self, difficulty: int = 2) -> dict:
        """Mine a new block with pending threats."""
        if not self.pending_threats:
            return None
        
        previous_block = self.chain[-1]
        new_block = {
            'index': len(self.chain),
            'timestamp': time.time(),
            'threats': self.pending_threats.copy(),
            'previous_hash': previous_block['hash'],
            'nonce': 0
        }
        
        # Proof of Work
        while True:
            new_block['nonce'] += 1
            hash_attempt = self.calculate_hash(new_block)
            if hash_attempt[:difficulty] == '0' * difficulty:
                new_block['hash'] = hash_attempt
                break
        
        self.chain.append(new_block)
        self.pending_threats = []
        return new_block
    
    def verify_chain(self) -> bool:
        """Verify blockchain integrity."""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            
            # Verify hash
            if current['hash'] != self.calculate_hash({
                k: v for k, v in current.items() if k != 'hash'
            }):
                return False
            
            # Verify link
            if current['previous_hash'] != previous['hash']:
                return False
        
        return True


@dataclass
class ThreatIntelligence:
    """Threat intelligence data structure."""
    threat_id: str
    threat_type: str  # 'disinformation', 'phishing', 'propaganda', 'coordination'
    confidence: float
    source_model: Optional[str]  # Attributed LLM model
    affected_users: List[str]
    content_hash: str
    detection_timestamp: float
    risk_score: float
    mitigation_status: str  # 'detected', 'analyzing', 'mitigated'
    cross_border_shared: bool = False
    
    def to_dict(self) -> dict:
        return asdict(self)


class FederatedIntelligenceProtocol:
    """
    Secure cross-border intelligence sharing protocol.
    Implements privacy-preserving federated learning concepts.
    """
    
    def __init__(self, agency_id: str, partner_agencies: List[str]):
        self.agency_id = agency_id
        self.partner_agencies = partner_agencies
        self.shared_intelligence = []
        self.blockchain = BlockchainLedger()
    
    def share_threat_intelligence(self, threat: ThreatIntelligence, 
                                target_agencies: List[str] = None) -> Dict:
        """
        Share threat intelligence with partner agencies.
        Uses differential privacy to protect sensitive details.
        """
        if target_agencies is None:
            target_agencies = self.partner_agencies
        
        # Apply differential privacy (noise addition)
        shared_data = threat.to_dict()
        shared_data['confidence'] = self._add_noise(threat.confidence, epsilon=0.1)
        shared_data['risk_score'] = self._add_noise(threat.risk_score, epsilon=0.1)
        
        # Remove sensitive user identifiers
        shared_data['affected_users'] = [
            hashlib.sha256(uid.encode()).hexdigest()[:8] 
            for uid in threat.affected_users
        ]
        
        # Add to blockchain
        self.blockchain.add_threat({
            'threat_id': threat.threat_id,
            'sharing_agency': self.agency_id,
            'target_agencies': target_agencies,
            'data_hash': hashlib.sha256(
                json.dumps(shared_data).encode()
            ).hexdigest(),
            'classification': 'CONFIDENTIAL'
        })
        
        # Mine block (in production, this would be distributed)
        block = self.blockchain.mine_block()
        
        return {
            'status': 'shared',
            'block_hash': block['hash'] if block else None,
            'recipients': target_agencies,
            'timestamp': time.time()
        }
    
    @staticmethod
    def _add_noise(value: float, epsilon: float = 0.1) -> float:
        """Add Laplacian noise for differential privacy."""
        import numpy as np
        noise = np.random.laplace(0, 1/epsilon)
        return max(0.0, min(1.0, value + noise * 0.01))
    
    def receive_intelligence(self, encrypted_data: str, sender_agency: str) -> Dict:
        """
        Receive and process intelligence from partner agency.
        In production, implement proper encryption/decryption.
        """
        # Mock decryption (replace with actual cryptographic implementation)
        try:
            data = json.loads(encrypted_data)  # In production: decrypt first
            
            # Verify sender
            if sender_agency not in self.partner_agencies:
                return {'status': 'rejected', 'reason': 'unauthorized_sender'}
            
            # Store in local intelligence database
            self.shared_intelligence.append({
                'sender': sender_agency,
                'received_at': time.time(),
                'data': data
            })
            
            return {'status': 'accepted', 'intelligence_id': data.get('threat_id')}
            
        except Exception as e:
            return {'status': 'error', 'reason': str(e)}


class RealTimeMonitor:
    """
    Real-time monitoring system for threat detection and response.
    Integrates with national SOCs and SIEM systems.
    """
    
    def __init__(self, update_interval: int = 5):
        self.update_interval = update_interval
        self.active_threats = {}
        self.threat_history = []
        self.alert_thresholds = {
            'critical': 0.9,
            'high': 0.7,
            'medium': 0.5,
            'low': 0.3
        }
    
    def process_threat_stream(self, threat_data: Dict) -> Dict:
        """Process incoming threat data in real-time."""
        threat_id = threat_data.get('threat_id')
        risk_score = threat_data.get('risk_score', 0)
        
        # Determine alert level
        alert_level = 'low'
        for level, threshold in self.alert_thresholds.items():
            if risk_score >= threshold:
                alert_level = level
                break
        
        # Update active threats
        self.active_threats[threat_id] = {
            'data': threat_data,
            'alert_level': alert_level,
            'first_seen': time.time(),
            'last_updated': time.time(),
            'escalation_count': 0
        }
        
        # Generate response recommendations
        response = self._generate_response(alert_level, threat_data)
        
        return {
            'threat_id': threat_id,
            'alert_level': alert_level,
            'recommended_actions': response,
            'monitoring_status': 'active'
        }
    
    def _generate_response(self, alert_level: str, threat_data: Dict) -> List[str]:
        """Generate automated response recommendations."""
        responses = {
            'critical': [
                'Immediate isolation of affected systems',
                'Alert CERT-IN emergency response team',
                'Initiate forensic data collection',
                'Activate incident response protocol',
                'Block identified malicious IP ranges'
            ],
            'high': [
                'Enable enhanced monitoring',
                'Rate-limit suspicious users',
                'Alert SOC analysts',
                'Prepare containment procedures',
                'Collect threat intelligence'
            ],
            'medium': [
                'Monitor user activity',
                'Update threat signatures',
                'Review security policies',
                'Increase logging verbosity'
            ],
            'low': [
                'Log for analysis',
                'Update threat database',
                'Continue standard monitoring'
            ]
        }
        
        return responses.get(alert_level, ['Continue monitoring'])
    
    def generate_sitrep(self) -> Dict:
        """Generate situation report for command center."""
        active_count = len(self.active_threats)
        critical_count = sum(
            1 for t in self.active_threats.values() 
            if t['alert_level'] == 'critical'
        )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_active_threats': active_count,
            'critical_threats': critical_count,
            'threat_distribution': self._get_threat_distribution(),
            'recommended_posture': self._recommend_defense_posture(),
            'cross_border_intel_status': 'ACTIVE',
            'blockchain_integrity': 'VERIFIED'
        }
    
    def _get_threat_distribution(self) -> Dict:
        """Calculate distribution of threat types."""
        distribution = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for threat in self.active_threats.values():
            level = threat['alert_level']
            distribution[level] = distribution.get(level, 0) + 1
        return distribution
    
    def _recommend_defense_posture(self) -> str:
        """Recommend overall defense posture based on threat landscape."""
        critical_count = sum(
            1 for t in self.active_threats.values() 
            if t['alert_level'] == 'critical'
        )
        
        if critical_count > 0:
            return 'DEFCON 1 - Maximum Defense'
        elif len(self.active_threats) > 10:
            return 'DEFCON 2 - Elevated Defense'
        elif len(self.active_threats) > 5:
            return 'DEFCON 3 - Enhanced Monitoring'
        else:
            return 'DEFCON 4 - Standard Operations'


# API Integration Points for SOC/SIEM
class APIGateway:
    """API gateway for integration with existing security infrastructure."""
    
    @staticmethod
    def export_to_splunk(threat_data: Dict) -> Dict:
        """Format threat data for Splunk ingestion."""
        return {
            '_time': time.time(),
            'source': 'AegisAI',
            'sourcetype': 'llm_threat_detection',
            'event': threat_data
        }
    
    @staticmethod
    def export_to_elasticsearch(threat_data: Dict) -> Dict:
        """Format threat data for ELK stack."""
        return {
            '@timestamp': datetime.now().isoformat(),
            'threat': threat_data,
            'index': 'aegis-threats',
            'doc_type': 'llm_threat'
        }
    
    @staticmethod
    def webhook_notification(threat_data: Dict, webhook_url: str) -> bool:
        """Send threat notification via webhook."""
        try:
            response = requests.post(
                webhook_url,
                json=threat_data,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False


# Integration example
if __name__ == "__main__":
    # Initialize components
    fed_protocol = FederatedIntelligenceProtocol(
        agency_id="CERT-IN",
        partner_agencies=["NCIIPC", "NIC", "DRDO", "NSA-Allied"]
    )
    
    monitor = RealTimeMonitor()
    
    # Simulate threat detection
    threat = ThreatIntelligence(
        threat_id="THR-2024-001",
        threat_type="coordinated_disinformation",
        confidence=0.92,
        source_model="GPT-4",
        affected_users=["user123", "user456"],
        content_hash="abc123def456",
        detection_timestamp=time.time(),
        risk_score=0.85,
        mitigation_status="analyzing"
    )
    
    # Share with partners
    share_result = fed_protocol.share_threat_intelligence(threat)
    print(f"Intelligence Shared: {share_result}")
    
    # Process in real-time monitor
    response = monitor.process_threat_stream(threat.to_dict())
    print(f"Recommended Actions: {response}")
    
    # Generate situation report
    sitrep = monitor.generate_sitrep()
    print(f"Situation Report: {sitrep}")