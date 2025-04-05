import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat

from ..utils.common import create_block_mask


class MoBAAttention(nn.Module):
    """
    Mixture of Block Attention (MoBA) module based on the paper
    "MOBA: Mixture of Block Attention for Long-Context LLMs"
    """
    
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        block_size=1024,
        num_blocks=4,
        dropout_prob=0.1,
        softmax_scale=None,
        is_causal=True,
        use_rotary_emb=True,
        alpha=None  # Optional parameter for adaptive weighting
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.is_causal = is_causal
        self.use_rotary_emb = use_rotary_emb
        
        # Set default softmax scale to 1/sqrt(head_dim) if not provided
        self.softmax_scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(self.head_dim)
        
        # Projection matrices
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout_prob)
        self.resid_dropout = nn.Dropout(dropout_prob)
        
        # For adaptive weighting between blocks
        self.alpha = alpha
        if alpha is not None:
            self.block_weights = nn.Parameter(torch.ones(num_blocks))
            self.block_bias = nn.Parameter(torch.zeros(num_blocks))
        
    def _split_heads(self, x, n_head, dim_head):
        """Split the last dimension into (n_head, dim_head)"""
        new_shape = x.size()[:-1] + (n_head, dim_head)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)  # (batch, head, seq_len, head_dim)
    
    def _merge_heads(self, x, n_head, dim_head):
        """Merge the (n_head, dim_head) into hidden_size"""
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (n_head * dim_head,)
        return x.view(*new_shape)
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        output_attentions=False,
        cos=None,
        sin=None,
    ):
        """
        Forward pass for MoBA.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask of shape [batch_size, 1, seq_len, seq_len]
            position_bias: Optional positional bias
            output_attentions: Whether to output attention weights
            cos: Cosine part of rotary embeddings
            sin: Sine part of rotary embeddings
            
        Returns:
            context_layer: Output tensor of shape [batch_size, seq_len, hidden_size]
            attention_probs: Attention probabilities if output_attentions=True
        """
        batch_size, seq_length, _ = hidden_states.shape
        
        # Compute Q, K, V
        query_layer = self.q_proj(hidden_states)
        key_layer = self.k_proj(hidden_states)
        value_layer = self.v_proj(hidden_states)
        
        # Split heads
        query_layer = self._split_heads(query_layer, self.num_attention_heads, self.head_dim)
        key_layer = self._split_heads(key_layer, self.num_attention_heads, self.head_dim)
        value_layer = self._split_heads(value_layer, self.num_attention_heads, self.head_dim)
        
        # Apply rotary embeddings if provided
        if self.use_rotary_emb and cos is not None and sin is not None:
            query_layer, key_layer = self._apply_rotary_pos_emb(
                query_layer, key_layer, cos, sin, seq_length
            )
        
        # Compute block attention
        context_layers = []
        attention_probs_list = []
        
        # For each block size, compute block attention
        for block_idx in range(self.num_blocks):
            actual_block_size = self.block_size // (2 ** block_idx)
            
            # Ensure block size is at least 1
            actual_block_size = max(1, actual_block_size)
            
            block_context_layer, block_attention_probs = self._compute_block_attention(
                query_layer, key_layer, value_layer,
                actual_block_size, attention_mask,
                batch_size, seq_length
            )
            
            context_layers.append(block_context_layer)
            if output_attentions:
                attention_probs_list.append(block_attention_probs)
        
        # Combine the results from different blocks
        if self.alpha is not None:
            # Adaptively weight the blocks
            block_weights = F.softmax(self.block_weights, dim=0)
            context_layer = sum(w * ctx for w, ctx in zip(block_weights, context_layers))
        else:
            # Simple average
            context_layer = sum(context_layers) / len(context_layers)
        
        # Project back to hidden size
        context_layer = self._merge_heads(context_layer, self.num_attention_heads, self.head_dim)
        output = self.o_proj(context_layer)
        output = self.resid_dropout(output)
        
        if output_attentions:
            return output, attention_probs_list
        else:
            return output
    
    def _apply_rotary_pos_emb(self, q, k, cos, sin, seq_length):
        """Apply rotary position embeddings to q and k."""
        # q, k: [batch, heads, seq_len, head_dim]
        # cos, sin might be [seq_len, dim] or [batch, 1, seq_len, dim]
        
        # Make sure dimensions match
        head_dim = q.size(-1)
        if len(cos.shape) == 2:  # [seq_len, dim]
            cos = cos[:seq_length, :head_dim]  # Ensure proper sequence length and dimension
            sin = sin[:seq_length, :head_dim]
            cos = cos.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, head_dim]
            sin = sin.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, head_dim]
        elif len(cos.shape) == 3:  # [batch, seq_len, dim]
            cos = cos[:, :seq_length, :head_dim]  # Ensure proper sequence length and dimension
            sin = sin[:, :seq_length, :head_dim]
            cos = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
            sin = sin.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
        elif len(cos.shape) == 4:  # [batch, 1, seq_len, dim]
            cos = cos[:, :, :seq_length, :head_dim]  # Ensure proper sequence length and dimension
            sin = sin[:, :, :seq_length, :head_dim]
        
        # Ensure our head dimensions match exactly
        if cos.size(-1) != head_dim:
            # If dimensions don't match, we need to adjust
            if cos.size(-1) > head_dim:
                # If cos/sin have more dimensions, trim them
                cos = cos[..., :head_dim]
                sin = sin[..., :head_dim]
            else:
                # If cos/sin have fewer dimensions, expand and repeat
                cos = torch.cat([cos, cos], dim=-1)[..., :head_dim]
                sin = torch.cat([sin, sin], dim=-1)[..., :head_dim]
        
        # Apply rotation as a complex multiplication with real and imaginary parts
        q_cos = q * cos
        q_sin = q * sin
        k_cos = k * cos
        k_sin = k * sin
        
        q_rotated = torch.cat([-q_sin, q_cos], dim=-1)[..., :head_dim]
        k_rotated = torch.cat([-k_sin, k_cos], dim=-1)[..., :head_dim]
        
        return q_rotated, k_rotated
    
    def _compute_block_attention(
        self, 
        query_layer, 
        key_layer, 
        value_layer, 
        block_size, 
        attention_mask, 
        batch_size, 
        seq_length
    ):
        """
        Compute attention within blocks.
        
        Args:
            query_layer: [batch, heads, seq_len, head_dim]
            key_layer: [batch, heads, seq_len, head_dim]
            value_layer: [batch, heads, seq_len, head_dim]
            block_size: Size of the blocks
            attention_mask: Optional attention mask
            batch_size: Batch size
            seq_length: Sequence length
            
        Returns:
            context_layer: Block attention output
            attention_probs: Attention probabilities
        """
        # Compute number of blocks
        num_blocks = math.ceil(seq_length / block_size)
        
        # Initialize output tensors
        context_layer = torch.zeros_like(query_layer)
        attention_probs = torch.zeros(
            batch_size, 
            self.num_attention_heads, 
            seq_length, 
            seq_length, 
            device=query_layer.device
        )
        
        # Process each block
        for i in range(num_blocks):
            # Get block boundaries
            start_idx = i * block_size
            end_idx = min(start_idx + block_size, seq_length)
            block_len = end_idx - start_idx
            
            # Extract block queries, keys, and values
            block_query = query_layer[:, :, start_idx:end_idx, :]
            
            # For causal attention, only use up to the current block for keys and values
            if self.is_causal:
                block_key = key_layer[:, :, :end_idx, :]
                block_value = value_layer[:, :, :end_idx, :]
            else:
                # For non-causal, use the current block for keys and values
                block_key = key_layer[:, :, start_idx:end_idx, :]
                block_value = value_layer[:, :, start_idx:end_idx, :]
            
            # Compute attention scores
            attention_scores = torch.matmul(block_query, block_key.transpose(-1, -2))
            attention_scores = attention_scores * self.softmax_scale
            
            # Apply mask if provided
            if attention_mask is not None:
                # Handle the case where attention mask shape doesn't match 
                if self.is_causal:
                    # For causal attention, we need to make sure the mask covers up to end_idx
                    if attention_mask.size(-1) >= end_idx and attention_mask.size(-2) >= (end_idx - start_idx):
                        block_mask = attention_mask[:, :, start_idx:end_idx, :end_idx]
                        # Check if shapes match
                        if block_mask.shape[-2:] == attention_scores.shape[-2:]:
                            attention_scores = attention_scores + block_mask
                else:
                    # For non-causal, we mask within the block
                    if (attention_mask.size(-1) >= end_idx - start_idx and 
                        attention_mask.size(-2) >= end_idx - start_idx):
                        block_mask = attention_mask[:, :, start_idx:end_idx, start_idx:end_idx]
                        # Check if shapes match
                        if block_mask.shape[-2:] == attention_scores.shape[-2:]:
                            attention_scores = attention_scores + block_mask
            
            # Apply causal mask within the block if needed
            if self.is_causal and block_len > 1:
                # Create a causal mask for this block
                causal_mask = torch.tril(
                    torch.ones((block_len, block_len), device=query_layer.device)
                ).view(1, 1, block_len, block_len)
                
                if self.is_causal:
                    # For the causal case with previous blocks
                    actual_seq_len = attention_scores.size(-1)  # This might be different from full_sequence_len
                    
                    # Create a mask that allows attending to all previous positions
                    causal_mask_full = torch.zeros((1, 1, block_len, actual_seq_len), device=query_layer.device)
                    
                    # Set up the tril pattern for the actual sequence length we have
                    causal_tril = torch.tril(torch.ones((block_len, actual_seq_len), device=query_layer.device))
                    causal_mask_full[:, :, :, :actual_seq_len] = causal_tril
                    
                    # Apply the mask
                    attention_scores = attention_scores.masked_fill(
                        causal_mask_full == 0, float('-inf')
                    )
                else:
                    # Standard causal mask within the block
                    attention_scores = attention_scores.masked_fill(
                        causal_mask == 0, float('-inf')
                    )
            
            # Compute attention probabilities
            block_attention_probs = F.softmax(attention_scores, dim=-1)
            block_attention_probs = self.attn_dropout(block_attention_probs)
            
            # Compute context layer
            block_context_layer = torch.matmul(block_attention_probs, block_value)
            
            # Store the results
            context_layer[:, :, start_idx:end_idx, :] = block_context_layer
            
            # Store attention probabilities if needed
            if self.is_causal:
                # For causal attention, we've attended to all previous positions
                attention_probs_slice = attention_probs[:, :, start_idx:end_idx, :end_idx]
                if attention_probs_slice.shape[-2:] == block_attention_probs.shape[-2:]:
                    attention_probs[:, :, start_idx:end_idx, :end_idx] = block_attention_probs
            else:
                # For non-causal, we've only attended within the block
                attention_probs_slice = attention_probs[:, :, start_idx:end_idx, start_idx:end_idx]
                if attention_probs_slice.shape[-2:] == block_attention_probs.shape[-2:]:
                    attention_probs[:, :, start_idx:end_idx, start_idx:end_idx] = block_attention_probs
        
        return context_layer, attention_probs 