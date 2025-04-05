import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import os
from tqdm import tqdm

from src.models import MoBAModel
from src.attention import MoBAAttention


def check_cuda_availability():
    """Check if CUDA is available and print detailed information."""
    print("CUDA Availability Check:")
    print(f"- torch.cuda.is_available(): {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"- CUDA Version: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"- GPU Count: {device_count}")
        
        for i in range(device_count):
            print(f"- GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  - Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            print(f"  - Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
        
        return True
    else:
        print("- No CUDA GPUs detected. Using CPU instead.")
        return False


def ensure_gpu_compatibility():
    """Configure system for optimal GPU usage if available."""
    # First, do a thorough check
    has_cuda = check_cuda_availability()
    
    if not has_cuda:
        print("CUDA not available. Using CPU.")
        return torch.device("cpu")
    
    # Set environment variables to ensure CUDA is properly detected
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    try:
        # Force CUDA initialization
        device = torch.device("cuda")
        
        # Enable cuDNN benchmarking and auto-tuner for best performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # Set default tensor type to CUDA for better performance
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
        # Print GPU info
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Pre-allocate a small tensor to initialize CUDA context
        dummy = torch.ones(1, device=device)
        del dummy
        
        # Try a simple CUDA operation to verify everything works
        test_tensor = torch.rand(100, 100, device=device)
        test_result = test_tensor @ test_tensor.T
        del test_tensor, test_result
        
        print("CUDA initialization successful!")
        return device
    except Exception as e:
        print(f"Error during CUDA initialization: {e}")
        print("Falling back to CPU.")
        torch.set_default_tensor_type('torch.FloatTensor')
        return torch.device("cpu")


def compare_attention_patterns():
    """Generate and visualize attention patterns for different attention mechanisms."""
    device = ensure_gpu_compatibility()
    seq_len = 1024
    
    # Create sample data
    x = torch.randn(1, seq_len, 512, device=device)
    
    # Traditional full attention (only for visualization, not computation)
    traditional_pattern = torch.tril(torch.ones(seq_len, seq_len, device=device))
    
    # Block attention pattern
    block_size = 128
    num_blocks = seq_len // block_size
    block_pattern = torch.zeros(seq_len, seq_len, device=device)
    
    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = (i + 1) * block_size
        block_pattern[start_idx:end_idx, :start_idx + block_size] = torch.tril(
            torch.ones(block_size, start_idx + block_size, device=device)
        )
    
    # Sliding window pattern
    window_size = 256
    sliding_pattern = torch.zeros(seq_len, seq_len, device=device)
    for i in range(seq_len):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(seq_len, i + window_size // 2 + 1)
        sliding_pattern[i, start_idx:min(i + 1, end_idx)] = 1
    
    # MoBA pattern (combination of block patterns with different sizes)
    moba_pattern = torch.zeros(seq_len, seq_len, device=device)
    for block_idx, bs in enumerate([block_size, block_size // 2, block_size // 4]):
        num_blocks = seq_len // bs
        for i in range(num_blocks):
            start_idx = i * bs
            end_idx = (i + 1) * bs
            # For simplicity, assume causal attention
            for j in range(start_idx, min(end_idx, seq_len)):
                moba_pattern[j, :j+1] += 1
    
    # Normalize the MoBA pattern
    moba_pattern = moba_pattern / moba_pattern.max()
    
    # Move tensors to CPU for plotting
    traditional_pattern = traditional_pattern.cpu()
    block_pattern = block_pattern.cpu()
    sliding_pattern = sliding_pattern.cpu()
    moba_pattern = moba_pattern.cpu()
    
    # Plot attention patterns
    plt.figure(figsize=(12, 10))
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot traditional attention
    axs[0, 0].imshow(traditional_pattern.numpy(), cmap='Blues')
    axs[0, 0].set_title('Traditional Full Attention')
    axs[0, 0].set_xlabel('Key position')
    axs[0, 0].set_ylabel('Query position')
    
    # Plot block attention
    axs[0, 1].imshow(block_pattern.numpy(), cmap='Blues')
    axs[0, 1].set_title(f'Block Attention (Size={block_size})')
    axs[0, 1].set_xlabel('Key position')
    axs[0, 1].set_ylabel('Query position')
    
    # Plot sliding window attention
    axs[1, 0].imshow(sliding_pattern.numpy(), cmap='Blues')
    axs[1, 0].set_title(f'Sliding Window Attention (Size={window_size})')
    axs[1, 0].set_xlabel('Key position')
    axs[1, 0].set_ylabel('Query position')
    
    # Plot MoBA attention
    axs[1, 1].imshow(moba_pattern.numpy(), cmap='Blues')
    axs[1, 1].set_title('MoBA Attention')
    axs[1, 1].set_xlabel('Key position')
    axs[1, 1].set_ylabel('Query position')
    
    plt.tight_layout()
    plt.savefig('attention_patterns.png')
    print("Saved attention patterns to attention_patterns.png")
    plt.close()


def compare_memory_usage(max_seq_len=8192, step=1024):
    """Compare memory usage between different attention mechanisms."""
    device = ensure_gpu_compatibility()
    
    hidden_size = 512
    num_heads = 8
    seq_lengths = list(range(1024, max_seq_len + 1, step))
    
    # Memory usage for different attention types
    standard_mem = []
    block_mem = []
    moba_mem = []
    
    for seq_len in tqdm(seq_lengths, desc="Comparing memory usage"):
        batch_size = 1
        
        # Standard attention (estimated, we don't actually run it for long sequences)
        standard_memory = (batch_size * seq_len * seq_len * 4) / (1024 * 1024)  # MB
        standard_mem.append(standard_memory)
        
        # Block attention
        if device.type == "cuda":
            torch.cuda.empty_cache()
            start_mem = torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            start_mem = 0
        
        try:
            x = torch.randn(batch_size, seq_len, hidden_size, device=device)
            # Create a standard transformer block with block attention
            model = nn.TransformerEncoderLayer(
                d_model=hidden_size, 
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                batch_first=True
            ).to(device)
            
            with torch.no_grad():
                _ = model(x)
                
            if device.type == "cuda":
                end_mem = torch.cuda.memory_allocated() / (1024 * 1024)
                block_mem.append(end_mem - start_mem)
            else:
                block_mem.append(standard_memory / 4)  # Estimated for CPU
                
        except RuntimeError as e:
            # If we run out of memory, use the last successful measurement
            if len(block_mem) > 0:
                block_mem.append(block_mem[-1])
            else:
                block_mem.append(0)
            print(f"Error with block attention at seq_len={seq_len}: {e}")
            # Try to recover if CUDA OOM
            if device.type == "cuda" and ("CUDA out of memory" in str(e) or "out of memory" in str(e)):
                torch.cuda.empty_cache()
        
        # MoBA attention
        if device.type == "cuda":
            torch.cuda.empty_cache()
            start_mem = torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            start_mem = 0
        
        try:
            x = torch.randn(batch_size, seq_len, hidden_size, device=device)
            # Create MoBA attention
            model = MoBAAttention(
                hidden_size=hidden_size, 
                num_attention_heads=num_heads,
                block_size=512,
                num_blocks=3
            ).to(device)
            
            with torch.no_grad():
                _ = model(x)
                
            if device.type == "cuda":
                end_mem = torch.cuda.memory_allocated() / (1024 * 1024)
                moba_mem.append(end_mem - start_mem)
            else:
                moba_mem.append(standard_memory / 16)  # Estimated for CPU
                
        except RuntimeError as e:
            # If we run out of memory, use the last successful measurement
            if len(moba_mem) > 0:
                moba_mem.append(moba_mem[-1])
            else:
                moba_mem.append(0)
            print(f"Error with MoBA attention at seq_len={seq_len}: {e}")
            # Try to recover if CUDA OOM
            if device.type == "cuda" and ("CUDA out of memory" in str(e) or "out of memory" in str(e)):
                torch.cuda.empty_cache()
    
    # Plot memory usage
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, standard_mem, label='Standard Attention (estimated)', marker='o')
    plt.plot(seq_lengths, block_mem, label='Block Attention', marker='s')
    plt.plot(seq_lengths, moba_mem, label='MoBA Attention', marker='^')
    plt.xlabel('Sequence Length')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage vs Sequence Length')
    plt.legend()
    plt.grid(True)
    plt.savefig('memory_usage.png')
    print("Saved memory usage comparison to memory_usage.png")
    plt.close()


def test_generation_with_long_context(context_length=6000, generation_length=50):
    """Test text generation with a long context using MoBA."""
    device = ensure_gpu_compatibility()
    
    # Create a long context string (repeat a sample text to simulate long context)
    sample_text = "MoBA (Mixture of Block Attention) is an efficient attention mechanism for long contexts. "
    long_context = sample_text * 100  # Repeat to create a long context
    
    # Create a simple character-level tokenizer
    chars = sorted(list(set(long_context)))
    char_to_idx = {ch: i + 3 for i, ch in enumerate(chars)}
    idx_to_char = {i + 3: ch for i, ch in enumerate(chars)}
    
    # Add special tokens
    char_to_idx['<pad>'] = 0
    char_to_idx['<bos>'] = 1
    char_to_idx['<eos>'] = 2
    idx_to_char[0] = '<pad>'
    idx_to_char[1] = '<bos>'
    idx_to_char[2] = '<eos>'
    
    # Calculate safe top_k (smaller than vocab size)
    vocab_size = len(char_to_idx)
    safe_top_k = min(10, max(1, vocab_size // 2))
    print(f"Vocabulary size: {vocab_size}, using top_k={safe_top_k}")
    
    # Tokenize the context
    tokens = [1]  # Start with BOS token
    for ch in long_context:
        tokens.append(char_to_idx[ch])
    tokens.append(2)  # End with EOS token
    
    # Create tensor
    input_tensor = torch.tensor(tokens, device=device)
    seq_length = len(tokens)
    
    print(f"Input sequence length: {seq_length}")
    
    # Ensure we don't try to use more context than we have
    context_length = min(context_length, seq_length - 1)
    
    # Create model with MoBA attention
    print("Creating model...")
    try:
        model = MoBAModel(
            vocab_size=len(char_to_idx) + 10,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            intermediate_size=1024,
            block_size=128,
            num_blocks=3,
            max_position_embeddings=8192,
            dropout_prob=0.1,
        ).to(device)
        
        print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")
        print(f"Model device: {next(model.parameters()).device}")
        
        # Test generation
        print(f"Generating text with {context_length} tokens of context...")
        start_time = time.time()
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_tensor.unsqueeze(0)[:, :context_length],
                max_length=generation_length,
                temperature=0.7,
                top_k=safe_top_k,
                do_sample=True,
            )
        
        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.2f} seconds")
        
        # Convert generated ids to text
        generated_text = ""
        for id in generated_ids[0].cpu().tolist():
            if id == 2:  # EOS token
                break
            if id in idx_to_char:
                char = idx_to_char[id]
                if char not in ['<pad>', '<bos>', '<eos>']:
                    generated_text += char
        
        print("\nGenerated text:")
        print(generated_text)
        
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) or "out of memory" in str(e):
            print(f"CUDA out of memory error: {e}")
            print("Try reducing context_length or model size")
            if device.type == "cuda":
                torch.cuda.empty_cache()
                print("Attempting to run on CPU instead...")
                return test_generation_with_long_context_cpu(context_length // 2, generation_length)
        else:
            print(f"Error during generation: {e}")
            raise e


def test_generation_with_long_context_cpu(context_length=2000, generation_length=50):
    """Fallback function to run on CPU if GPU runs out of memory."""
    print("Running on CPU...")
    device = torch.device("cpu")
    
    # Make sure we switch back to CPU tensor types
    torch.set_default_tensor_type('torch.FloatTensor')
    
    # Create a long context string (repeat a sample text to simulate long context)
    sample_text = "MoBA (Mixture of Block Attention) is an efficient attention mechanism for long contexts. "
    long_context = sample_text * 50  # Use less context on CPU
    
    # Create a simple character-level tokenizer
    chars = sorted(list(set(long_context)))
    char_to_idx = {ch: i + 3 for i, ch in enumerate(chars)}
    idx_to_char = {i + 3: ch for i, ch in enumerate(chars)}
    
    # Add special tokens
    char_to_idx['<pad>'] = 0
    char_to_idx['<bos>'] = 1
    char_to_idx['<eos>'] = 2
    idx_to_char[0] = '<pad>'
    idx_to_char[1] = '<bos>'
    idx_to_char[2] = '<eos>'
    
    # Calculate safe top_k (smaller than vocab size)
    vocab_size = len(char_to_idx)
    safe_top_k = min(5, max(1, vocab_size // 2))
    print(f"Vocabulary size: {vocab_size}, using top_k={safe_top_k}")
    
    # Tokenize the context
    tokens = [1]  # Start with BOS token
    for ch in long_context:
        tokens.append(char_to_idx[ch])
    tokens.append(2)  # End with EOS token
    
    # Create tensor
    input_tensor = torch.tensor(tokens, device=device)
    seq_length = len(tokens)
    
    print(f"Input sequence length: {seq_length}")
    context_length = min(context_length, seq_length - 1)
    
    # Create a smaller model for CPU
    print("Creating smaller model for CPU...")
    model = MoBAModel(
        vocab_size=len(char_to_idx) + 10,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
        block_size=64,
        num_blocks=2,
        max_position_embeddings=4096,
        dropout_prob=0.1,
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")
    
    # Test generation
    print(f"Generating text with {context_length} tokens of context...")
    start_time = time.time()
    
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_tensor.unsqueeze(0)[:, :context_length],
                max_length=generation_length,
                temperature=0.7,
                top_k=safe_top_k,
                do_sample=True,
            )
        
        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.2f} seconds")
        
        # Convert generated ids to text
        generated_text = ""
        for id in generated_ids[0].cpu().tolist():
            if id == 2:  # EOS token
                break
            if id in idx_to_char:
                char = idx_to_char[id]
                if char not in ['<pad>', '<bos>', '<eos>']:
                    generated_text += char
        
        print("\nGenerated text:")
        print(generated_text)
        
    except Exception as e:
        print(f"Error during CPU generation: {e}")
        print("Even the smaller CPU model failed. Try reducing context length further.")
        return ""


def main():
    parser = argparse.ArgumentParser(description="MoBA Long Context Examples")
    parser.add_argument("--patterns", action="store_true", help="Compare attention patterns")
    parser.add_argument("--memory", action="store_true", help="Compare memory usage")
    parser.add_argument("--generation", action="store_true", help="Test generation with long context")
    parser.add_argument("--context_length", type=int, default=6000, help="Length of context for generation")
    parser.add_argument("--generation_length", type=int, default=50, help="Length of generated text")
    parser.add_argument("--all", action="store_true", help="Run all examples")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    
    args = parser.parse_args()
    
    # Set CUDA availability based on args
    if args.no_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("CUDA disabled by user argument.")
    
    if args.all or args.patterns:
        compare_attention_patterns()
        
    if args.all or args.memory:
        compare_memory_usage()
        
    if args.all or args.generation:
        test_generation_with_long_context(args.context_length, args.generation_length)


if __name__ == "__main__":
    main() 