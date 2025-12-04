"""
Fountain Code (Luby Transform Code) Implementation
Simple implementation for QR code file transfer
"""
import random
import hashlib
from typing import Dict, Optional, List


class LTEncoder:
    """Luby Transform Encoder for fountain codes"""
    
    def __init__(self, data: bytes, block_size: int = 64):
        """
        Initialize encoder with data to encode.
        
        Args:
            data: The raw bytes to encode
            block_size: Size of each block in bytes
        """
        self.data = data
        self.block_size = block_size
        self.num_blocks = (len(data) + block_size - 1) // block_size  # Ceiling division
        
        # Split data into blocks
        self.blocks: List[bytes] = []
        for i in range(self.num_blocks):
            start = i * block_size
            end = min(start + block_size, len(data))
            block = data[start:end]
            # Pad last block if necessary
            if len(block) < block_size:
                block += b'\x00' * (block_size - len(block))
            self.blocks.append(block)
        
        self.file_size = len(data)
    
    def generate_droplet(self) -> Dict:
        """
        Generate a random droplet (encoded packet) using XOR of random blocks.
        
        Returns:
            Dictionary with seed, data (base64), num_blocks, and file_size
        """
        # Generate random seed
        seed = random.randint(0, 2**31 - 1)
        random.seed(seed)
        
        # Select random blocks to XOR together
        selected_blocks = []
        # Use a simple probability distribution (more blocks selected as we go)
        num_to_select = random.randint(1, min(self.num_blocks, 10))
        
        # Select random block indices
        block_indices = random.sample(range(self.num_blocks), num_to_select)
        
        # XOR selected blocks together
        result = bytearray(self.block_size)
        for idx in block_indices:
            block = self.blocks[idx]
            for i in range(self.block_size):
                result[i] ^= block[i]
        
        # Encode as base64 for JSON transmission
        import base64
        data_b64 = base64.b64encode(bytes(result)).decode('utf-8')
        
        return {
            "seed": seed,
            "data": data_b64,
            "num_blocks": self.num_blocks,
            "file_size": self.file_size,
            "block_size": self.block_size
        }


class LTDecoder:
    """Luby Transform Decoder for fountain codes"""
    
    def __init__(self, num_blocks: int, block_size: int = 64):
        """
        Initialize decoder.
        
        Args:
            num_blocks: Expected number of blocks
            block_size: Size of each block in bytes
        """
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.solved_blocks: Dict[int, bytes] = {}  # index -> block data
        self.num_solved = 0
        self.file_size: Optional[int] = None
        
        # System of linear equations (simplified - using XOR)
        # We'll use a greedy approach: if a droplet has only one unknown block, solve it
        self.pending_droplets: List[Dict] = []
    
    def add_droplet(self, droplet: Dict) -> bool:
        """
        Add a received droplet and attempt to solve blocks.
        
        Args:
            droplet: Dictionary with seed, data (base64), num_blocks, file_size
            
        Returns:
            True if all blocks are solved, False otherwise
        """
        if self.file_size is None:
            self.file_size = droplet.get("file_size", 0)
        
        # Decode base64 data
        import base64
        droplet_data = base64.b64decode(droplet["data"])
        seed = droplet["seed"]
        
        # Reconstruct which blocks were XORed
        random.seed(seed)
        num_to_select = random.randint(1, min(self.num_blocks, 10))
        block_indices = random.sample(range(self.num_blocks), num_to_select)
        
        # Simplified decoding: if droplet contains only one unknown block, solve it
        unknown_indices = [idx for idx in block_indices if idx not in self.solved_blocks]
        
        if len(unknown_indices) == 1:
            # We can solve this block!
            target_idx = unknown_indices[0]
            
            # XOR out all known blocks
            result = bytearray(droplet_data)
            for idx in block_indices:
                if idx != target_idx and idx in self.solved_blocks:
                    known_block = self.solved_blocks[idx]
                    for i in range(self.block_size):
                        result[i] ^= known_block[i]
            
            # Store the solved block
            self.solved_blocks[target_idx] = bytes(result)
            self.num_solved += 1
            
            # Try to solve pending droplets with this new information
            self._process_pending_droplets()
        else:
            # Store for later processing
            self.pending_droplets.append({
                "seed": seed,
                "data": droplet_data,
                "block_indices": block_indices
            })
        
        return self.is_complete()
    
    def _process_pending_droplets(self):
        """Try to solve pending droplets with newly solved blocks"""
        remaining = []
        for droplet in self.pending_droplets:
            block_indices = droplet["block_indices"]
            unknown_indices = [idx for idx in block_indices if idx not in self.solved_blocks]
            
            if len(unknown_indices) == 1:
                # Can solve now!
                target_idx = unknown_indices[0]
                result = bytearray(droplet["data"])
                
                for idx in block_indices:
                    if idx != target_idx and idx in self.solved_blocks:
                        known_block = self.solved_blocks[idx]
                        for i in range(self.block_size):
                            result[i] ^= known_block[i]
                
                self.solved_blocks[target_idx] = bytes(result)
                self.num_solved += 1
            else:
                remaining.append(droplet)
        
        self.pending_droplets = remaining
    
    def is_complete(self) -> bool:
        """Check if all blocks have been solved"""
        return self.num_solved >= self.num_blocks
    
    def get_result(self) -> bytes:
        """
        Reconstruct the original file from solved blocks.
        
        Returns:
            Reconstructed file bytes
        """
        if not self.is_complete():
            raise ValueError("Not all blocks are solved yet")
        
        # Reconstruct file in order
        result_parts = []
        for i in range(self.num_blocks):
            if i in self.solved_blocks:
                block = self.solved_blocks[i]
                # Remove padding from last block if file_size is known
                if i == self.num_blocks - 1 and self.file_size:
                    actual_size = self.file_size % self.block_size
                    if actual_size > 0:
                        block = block[:actual_size]
                    else:
                        block = block[:self.block_size]
                result_parts.append(block)
            else:
                raise ValueError(f"Block {i} is missing")
        
        return b''.join(result_parts)

