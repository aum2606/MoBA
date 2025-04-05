import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import os
import sys
from tqdm import tqdm

from src.models import MoBAModel


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


def sample_text_to_tensor(text, vocab_size=10000):
    """Convert a sample text to a tensor for demonstration."""
    # Create a simple tokenizer
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i + 3 for i, ch in enumerate(chars)}  # Reserve 0, 1, 2 for special tokens
    idx_to_char = {i + 3: ch for i, ch in enumerate(chars)}
    
    # Add special tokens
    char_to_idx['<pad>'] = 0
    char_to_idx['<bos>'] = 1
    char_to_idx['<eos>'] = 2
    idx_to_char[0] = '<pad>'
    idx_to_char[1] = '<bos>'
    idx_to_char[2] = '<eos>'
    
    # Convert text to tensor
    tokens = [char_to_idx.get(ch, char_to_idx.get(' ', 0)) for ch in text]
    
    # Add BOS and EOS tokens
    tokens = [1] + tokens + [2]
    
    return torch.tensor(tokens), char_to_idx, idx_to_char


def force_cuda_if_available():
    """Force CUDA usage if available."""
    # Set environment variables to make sure CUDA is visible
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    # Set default device
    if torch.cuda.is_available():
        # Force using CUDA
        torch.backends.cudnn.benchmark = True  # Optimize cudnn
        torch.backends.cudnn.enabled = True    # Enable cudnn
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
        # Print GPU info
        device = torch.device("cuda")
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        
        # Pre-allocate some memory to ensure CUDA context is initialized
        dummy = torch.ones(1, device=device)
        del dummy
        
        return device
    else:
        print("CUDA is not available. Using CPU.")
        return torch.device("cpu")


def main(args):
    # First, explicitly check CUDA availability and print details
    has_cuda = check_cuda_availability()
    
    # Force using CUDA if available and not explicitly disabled
    if not args.no_cuda and has_cuda:
        device = force_cuda_if_available()
    else:
        if args.no_cuda:
            print("CUDA disabled by user argument. Using CPU.")
        device = torch.device("cpu")
        print("Using CPU for computation.")
    
    # Sample text for demonstration
    sample_text = "MOBA stands for Mixture of Block Attention, a technique designed to handle long sequences efficiently in language models. It combines local block attention with hierarchical patterns to achieve efficient processing of long context windows."
    
    # Convert text to tensor
    input_tensor, char_to_idx, idx_to_char = sample_text_to_tensor(sample_text)
    input_tensor = input_tensor.to(device)
    
    # Create model
    print("Creating model...")
    model = MoBAModel(
        vocab_size=max(10000, len(char_to_idx) + 100),  # Ensure enough vocab size
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.hidden_size * 4,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        max_position_embeddings=args.max_len,
        dropout_prob=args.dropout,
    ).to(device)
    
    # Print model device location to verify
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")
    print(f"Model device: {next(model.parameters()).device}")
    
    if args.train:
        # For demonstration, we'll do a simple next-token prediction task
        train_model(model, input_tensor, char_to_idx, idx_to_char, args, device)
    
    # Generate text
    generated_text = generate_text(model, input_tensor[:10], char_to_idx, idx_to_char, max_length=50, device=device)
    print("Generated text:")
    print(generated_text)


def train_model(model, input_tensor, char_to_idx, idx_to_char, args, device):
    """Train the model on a simple next-token prediction task."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Simple next token prediction
    inputs = input_tensor[:-1].unsqueeze(0)  # [1, seq_len]
    targets = input_tensor[1:].unsqueeze(0)  # [1, seq_len]
    
    # Ensure inputs and targets are on the correct device
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    print("Training model...")
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        
        # Forward pass - we do a try-except to catch any CUDA OOM errors
        try:
            outputs = model(input_ids=inputs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs["logits"]
            
            # Compute loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
                
        except RuntimeError as e:
            if 'out of memory' in str(e) or 'CUDA out of memory' in str(e):
                print(f"CUDA out of memory in epoch {epoch}. Trying to recover...")
                torch.cuda.empty_cache()
                # If we're out of memory, reduce batch size or continue to next epoch
                continue
            else:
                print(f"Error during training: {e}")
                raise e
    
    print("Training completed!")


def generate_text(model, input_ids, char_to_idx, idx_to_char, max_length=100, device="cpu"):
    """Generate text using the trained model."""
    model.eval()
    input_ids = input_ids.unsqueeze(0).to(device)  # [1, seq_len]
    
    print(f"Generating text with context: {''.join([idx_to_char.get(id.item(), '') for id in input_ids[0] if id.item() > 2])}")
    
    # Calculate safe top_k
    vocab_size = len(char_to_idx)
    safe_top_k = min(50, max(1, vocab_size - 10))
    print(f"Vocabulary size: {vocab_size}, using top_k={safe_top_k}")
    
    # Generate text
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=0.7,
                top_k=safe_top_k,
                top_p=0.95,
                do_sample=True,
            )
        
        # Convert generated ids to text
        generated_text = ""
        for id in generated_ids[0].cpu().tolist():
            if id == char_to_idx['<eos>']:
                break
            if id not in idx_to_char:
                continue
            char = idx_to_char[id]
            if char not in ['<pad>', '<bos>', '<eos>']:
                generated_text += char
                
    except RuntimeError as e:
        if 'out of memory' in str(e) or 'CUDA out of memory' in str(e):
            print("CUDA out of memory during text generation. Falling back to CPU.")
            torch.cuda.empty_cache()
            # Move model to CPU and try again
            model = model.cpu()
            return generate_text(model, input_ids.cpu(), char_to_idx, idx_to_char, max_length, "cpu")
        else:
            print(f"Error during generation: {e}")
            raise e
    
    return generated_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MoBA model")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--block_size", type=int, default=16, help="Block size for MoBA")
    parser.add_argument("--num_blocks", type=int, default=2, help="Number of blocks in MoBA")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--train", action="store_true", help="Whether to train the model")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    args = parser.parse_args()
    
    main(args) 