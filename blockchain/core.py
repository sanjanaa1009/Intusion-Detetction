import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

class Block:
    """Immutable block structure for log storage"""
    
    def __init__(self, log_data: Dict, previous_hash: str, nonce: int = 0):
        self.timestamp = datetime.utcnow().isoformat()
        self.log_data = self._sanitize_data(log_data)
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.calculate_hash()
        
    def _sanitize_data(self, data: Dict) -> Dict:
        """Ensure data is JSON-serializable"""
        if isinstance(data, pd.Series):
            return data.to_dict()
        return {k: str(v) for k, v in data.items()}
    
    def calculate_hash(self) -> str:
        """SHA-256 hash of block contents"""
        block_string = json.dumps({
            "timestamp": self.timestamp,
            "data": self.log_data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()
    
    def to_dict(self) -> Dict:
        return {
            "hash": self.hash,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
            "data": self.log_data
        }

class LogBlockchain:
    """Immutable ledger for security logs with proof-of-work"""
    
    def __init__(self, difficulty: int = 4):
        self.chain: List[Block] = [self._create_genesis_block()]
        self.difficulty = difficulty
        self.pending_logs = []
        
    def _create_genesis_block(self) -> Block:
        return Block(
            log_data={"message": "GENESIS BLOCK"},
            previous_hash="0"
        )
    
    def add_log(self, log_entry: Dict) -> None:
        """Add log to pending queue (batch processing)"""
        self.pending_logs.append(log_entry)
    
    def mine_pending_logs(self) -> None:
        """Mine all pending logs into a single block"""
        if not self.pending_logs:
            return
            
        # Create Merkle root of all logs
        merkle_root = self._calculate_merkle_root(self.pending_logs)
        
        # Create block with proof-of-work
        last_block = self.chain[-1]
        new_block = Block(
            log_data={"merkle_root": merkle_root, "logs": self.pending_logs},
            previous_hash=last_block.hash
        )
        
        # Proof-of-work mining
        self._proof_of_work(new_block)
        
        self.chain.append(new_block)
        self.pending_logs = []
    
    def _proof_of_work(self, block: Block) -> None:
        """Mining mechanism for immutability"""
        while not block.hash.startswith('0' * self.difficulty):
            block.nonce += 1
            block.hash = block.calculate_hash()
    
    def _calculate_merkle_root(self, logs: List[Dict]) -> str:
        """Create cryptographic summary of all logs in block"""
        hashes = [
            hashlib.sha256(json.dumps(log, sort_keys=True).encode()).hexdigest() 
            for log in logs
        ]
        
        while len(hashes) > 1:
            if len(hashes) % 2 != 0:
                hashes.append(hashes[-1])
            hashes = [
                hashlib.sha256((hashes[i] + hashes[i+1]).encode()).hexdigest()
                for i in range(0, len(hashes), 2)
            ]
        
        return hashes[0] if hashes else ""
    
    def validate_chain(self) -> bool:
        """Verify blockchain integrity"""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            
            # Check hash linkage
            if current.previous_hash != previous.hash:
                return False
                
            # Verify proof-of-work
            if not current.hash.startswith('0' * self.difficulty):
                return False
                
            # Recompute hash
            if current.hash != current.calculate_hash():
                return False
                
        return True
    
    def find_log(self, log_hash: str) -> Optional[Dict]:
        """Find log by its hash in the blockchain"""
        for block in self.chain[1:]:  # Skip genesis
            for log in block.log_data.get("logs", []):
                if self._hash_log(log) == log_hash:
                    return log
        return None
    
    def _hash_log(self, log: Dict) -> str:
        """Calculate individual log hash"""
        return hashlib.sha256(
            json.dumps(log, sort_keys=True).encode()
        ).hexdigest()
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert blockchain to analysis-ready DataFrame"""
        records = []
        for block in self.chain[1:]:  # Skip genesis
            for log in block.log_data.get("logs", []):
                records.append({
                    "block_hash": block.hash,
                    "timestamp": block.timestamp,
                    "log_hash": self._hash_log(log),
                    **log
                })
        return pd.DataFrame(records)