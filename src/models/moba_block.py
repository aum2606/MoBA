import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attention.moba_attention import MoBAAttention


class MoBABlock(nn.Module):
    """
    Transformer block using MoBA (Mixture of Block Attention).
    """
    
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        block_size=1024,
        num_blocks=4,
        dropout_prob=0.1,
        activation_function=F.gelu,
        layer_norm_epsilon=1e-5,
        is_causal=True,
        use_rotary_emb=True,
        alpha=None,
    ):
        super().__init__()
        
        # MoBA Attention
        self.attention = MoBAAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            block_size=block_size,
            num_blocks=num_blocks,
            dropout_prob=dropout_prob,
            is_causal=is_causal,
            use_rotary_emb=use_rotary_emb,
            alpha=alpha,
        )
        
        # Feed-forward network
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        
        # Layer normalization
        self.attn_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.ff_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
        
        # Activation function
        self.activation_fn = activation_function
    
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
        Forward pass for the MoBA Block.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            position_bias: Optional positional bias
            output_attentions: Whether to output attention weights
            cos: Cosine part of rotary embeddings
            sin: Sine part of rotary embeddings
            
        Returns:
            hidden_states: Output tensor
            attention_weights: Attention weights if output_attentions=True
        """
        # Pre-LayerNorm architecture (better for training stability)
        residual = hidden_states
        hidden_states = self.attn_layer_norm(hidden_states)
        
        # MoBA Self-attention
        attn_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            cos=cos,
            sin=sin,
        )
        
        if output_attentions:
            attn_output, attention_weights = attn_outputs
        else:
            attn_output = attn_outputs
            attention_weights = None
        
        # First residual connection
        hidden_states = residual + attn_output
        
        # Feed-forward network with pre-LayerNorm
        residual = hidden_states
        hidden_states = self.ff_layer_norm(hidden_states)
        
        # FFN
        hidden_states = self.intermediate(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.output(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Second residual connection
        hidden_states = residual + hidden_states
        
        if output_attentions:
            return hidden_states, attention_weights
        else:
            return hidden_states 