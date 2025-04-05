import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, List

from .moba_block import MoBABlock


class RotaryEmbedding(nn.Module):
    """
    Rotary position embeddings based on the paper 
    "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Create and register the inverse frequency buffer
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Build cache for efficiency
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, 
            device=self.inv_freq.device, 
            dtype=torch.get_default_dtype()
        )
    
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """Create a cache of cos and sin values for positions."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
        
        # Compute frequencies for different positions
        # [seq_len, dim/2]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Register the cos and sin caches
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
    
    def forward(self, x, seq_len=None):
        """Get rotary position embeddings for the given sequence length."""
        # x: [bs, seq_len, hidden_size] or [bs, num_attention_heads, seq_len, head_size]
        # We only need the sequence length from x
        
        # Get the sequence length from the tensor if not provided
        if seq_len is None:
            if len(x.shape) == 3:  # [batch, seq_len, hidden]
                seq_len = x.size(1)
            elif len(x.shape) == 4:  # [batch, heads, seq_len, dim]
                seq_len = x.size(2)
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # If sequence length exceeds cache, extend the cache
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
            
        # Return the cached values up to the required sequence length
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class MoBALMHeads(nn.Module):
    """Heads for the MoBA Language Model for Next Token Prediction"""
    
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
    def forward(self, hidden_states):
        lm_logits = self.lm_head(hidden_states)
        return lm_logits


class MoBAModel(nn.Module):
    """
    MoBA Language Model based on the paper "MOBA: Mixture of Block Attention for Long-Context LLMs"
    """
    
    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        block_size=1024,
        num_blocks=4,
        max_position_embeddings=8192,
        dropout_prob=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
        alpha=None,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        # Embeddings
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.rotary_embeddings = RotaryEmbedding(
            dim=self.head_dim, 
            max_position_embeddings=max_position_embeddings
        )
        
        # MoBA blocks
        self.blocks = nn.ModuleList([
            MoBABlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                block_size=block_size,
                num_blocks=num_blocks,
                dropout_prob=dropout_prob,
                layer_norm_epsilon=layer_norm_epsilon,
                is_causal=True,
                use_rotary_emb=True,
                alpha=alpha,
            )
            for _ in range(num_hidden_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        
        # LM head
        self.lm_head = MoBALMHeads(hidden_size, vocab_size)
        
        # Tie weights if requested
        if tie_word_embeddings:
            self.lm_head.lm_head.weight = self.word_embeddings.weight
        
        # Initialize the weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def get_input_embeddings(self):
        return self.word_embeddings
    
    def set_input_embeddings(self, value):
        self.word_embeddings = value
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Forward pass of the MoBA Model.
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            token_type_ids: Token type ids (not used in this model)
            position_ids: Position ids
            inputs_embeds: Input embeddings
            past_key_values: Past key values for caching
            use_cache: Whether to use cache
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dict or tuple
            
        Returns:
            logits: Output logits
            past_key_values: Updated past key values
            hidden_states: All hidden states if output_hidden_states=True
            attentions: All attention weights if output_attentions=True
        """
        use_cache = use_cache if use_cache is not None else self.use_cache
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        if past_key_values is None:
            past_key_values = [None] * self.num_hidden_layers
            
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        # Prepare attention mask
        if attention_mask is not None:
            # [batch_size, seq_length] -> [batch_size, 1, 1, seq_length]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Convert mask to additive mask
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Get positional embeddings
        cos, sin = self.rotary_embeddings(inputs_embeds, seq_length)
        
        # Apply transformations through the blocks
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for i, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            block_outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                cos=cos,
                sin=sin,
            )
            
            if output_attentions:
                hidden_states, attn_weights = block_outputs
                all_attentions += (attn_weights,)
            else:
                hidden_states = block_outputs
        
        # Final layer normalization
        hidden_states = self.ln_f(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        if return_dict:
            return {
                "logits": logits,
                "past_key_values": past_key_values,
                "hidden_states": all_hidden_states,
                "attentions": all_attentions,
            }
        else:
            return (logits, past_key_values, all_hidden_states, all_attentions)
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        **model_kwargs
    ):
        """
        Prepare inputs for generation.
        
        Args:
            input_ids: Input token ids
            past_key_values: Past key values
            attention_mask: Attention mask
            model_kwargs: Additional model kwargs
            
        Returns:
            dict: Prepared inputs
        """
        if past_key_values is not None:
            # Only take the last token for generation
            input_ids = input_ids[:, -1:]
            
        inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
        }
        
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask
            
        return inputs
    
    def generate(
        self,
        input_ids,
        max_length=100,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.0,
        do_sample=True,
        pad_token_id=None,
        eos_token_id=None,
        **kwargs
    ):
        """
        Generate text with the model.
        
        Args:
            input_ids: Input token ids
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep
            top_p: Nucleus sampling probability threshold
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to sample or greedy decode
            pad_token_id: Padding token id
            eos_token_id: End of sequence token id
            
        Returns:
            torch.Tensor: Generated token ids
        """
        pad_token_id = pad_token_id if pad_token_id is not None else self.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.eos_token_id
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize generated sequences with input_ids
        generated_ids = input_ids.clone()
        
        # Create attention mask if needed
        if "attention_mask" not in kwargs:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = kwargs["attention_mask"]
        
        past_key_values = None
        
        for _ in range(max_length):
            model_inputs = self.prepare_inputs_for_generation(
                generated_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                **kwargs
            )
            
            # Forward pass
            outputs = self.forward(**model_inputs)
            next_token_logits = outputs[0][:, -1, :]
            past_key_values = outputs[1]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in set(generated_ids[i].tolist()):
                        next_token_logits[i, previous_token] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k > 0:
                # Ensure top_k is not larger than the vocabulary size
                vocab_size = next_token_logits.size(-1)
                top_k = min(top_k, vocab_size - 1)
                
                # Apply top-k filtering
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                for b in range(batch_size):
                    indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                    next_token_logits[b, indices_to_remove] = -float('Inf')
            
            # Sample or argmax
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Update generated ids and attention mask
            generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((batch_size, 1), device=device)], dim=-1
            )
            
            # Check if any sequence has reached the EOS token
            if eos_token_id is not None and (next_tokens == eos_token_id).any():
                # Mask out finished sequences
                not_finished = next_tokens != eos_token_id
                
                # Check if all sequences are finished
                if not not_finished.any():
                    break
        
        return generated_ids 