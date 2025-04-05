import torch
import math
import torch.nn.functional as F
from einops import rearrange


def create_block_mask(seq_len, block_size, causal=True, device=None):
    """
    Create a block mask for block attention.
    
    Args:
        seq_len (int): Sequence length
        block_size (int): Block size
        causal (bool): Whether to create a causal mask
        device: Device to create the mask on
        
    Returns:
        torch.Tensor: Block attention mask
    """
    num_blocks = math.ceil(seq_len / block_size)
    # Create a mask for each position in the sequence
    mask = torch.zeros(seq_len, seq_len, device=device)
    
    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = min(start_idx + block_size, seq_len)
        
        if causal:
            # Causal masking: each position can attend to itself and previous positions within the block
            for j in range(start_idx, end_idx):
                mask[j, start_idx:j+1] = 1
        else:
            # Non-causal masking: each position can attend to all positions within the block
            mask[start_idx:end_idx, start_idx:end_idx] = 1
            
    return mask


def create_sliding_window_mask(seq_len, window_size, device=None):
    """
    Create a sliding window mask.
    
    Args:
        seq_len (int): Sequence length
        window_size (int): Window size
        device: Device to create the mask on
        
    Returns:
        torch.Tensor: Sliding window mask
    """
    mask = torch.zeros(seq_len, seq_len, device=device)
    
    for i in range(seq_len):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(seq_len, i + window_size // 2 + 1)
        mask[i, start_idx:end_idx] = 1
        
    return mask


def create_global_tokens_mask(seq_len, global_token_indices, device=None):
    """
    Create a mask for global tokens.
    
    Args:
        seq_len (int): Sequence length
        global_token_indices (list): Indices of global tokens
        device: Device to create the mask on
        
    Returns:
        torch.Tensor: Global tokens mask
    """
    mask = torch.zeros(seq_len, seq_len, device=device)
    
    # Global tokens can attend to all tokens
    mask[global_token_indices, :] = 1
    
    # All tokens can attend to global tokens
    mask[:, global_token_indices] = 1
    
    return mask


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """
    Apply rotary positional embeddings to q and k tensors.
    
    Args:
        q (torch.Tensor): Query tensor
        k (torch.Tensor): Key tensor
        cos (torch.Tensor): Cosine part of rotary embeddings
        sin (torch.Tensor): Sine part of rotary embeddings
        position_ids (torch.Tensor, optional): Position indices
        
    Returns:
        tuple: Rotated q and k tensors
    """
    # Get the rotary embeddings for the given positions
    cos = cos.squeeze(1)  # [seq_len, dim]
    sin = sin.squeeze(1)  # [seq_len, dim]
    
    # Split q and k into even and odd dimensions
    q_embed_dim = q.shape[-1]
    k_embed_dim = k.shape[-1]
    
    # Handle different embedding dimensions
    cos_q = cos[..., :q_embed_dim]
    sin_q = sin[..., :q_embed_dim]
    cos_k = cos[..., :k_embed_dim]
    sin_k = sin[..., :k_embed_dim]
    
    # Apply rotary embeddings
    q_cos = q * cos_q
    q_sin = q * sin_q
    k_cos = k * cos_k
    k_sin = k * sin_k
    
    # Rotate
    q_rot = torch.cat([-q_sin, q_cos], dim=-1)
    k_rot = torch.cat([-k_sin, k_cos], dim=-1)
    
    return q_rot, k_rot 