# MoBA: Mixture of Block Attention

A PyTorch implementation of the MoBA (Mixture of Block Attention) architecture for efficient long-context sequence modeling in large language models.

## Overview

The MoBA architecture is designed to efficiently handle long sequences by using a mixture of local block attention patterns with hierarchical modeling. This implementation provides a modular and extensible codebase for research and applications involving long-context language models.

Key features of MoBA:
- Efficient attention mechanism for handling long contexts (up to thousands of tokens)
- Block-sparse attention patterns to reduce computational complexity
- Hierarchical modeling for capturing both local and global dependencies
- Rotary Position Embeddings for improved position awareness

## Installation

```bash
git clone https://github.com/aum2606/MoBA.git
cd MoBA
pip install -e .
```

### Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA (optional, but recommended for faster training and inference)

## Usage

### Quick Start

To quickly try out the MoBA model with a small example:

```bash
python run.py
```

For long-context examples:

```bash
python long_context_example.py
```

### Training

To train a MoBA model:

```bash
python run.py --train --epochs 100 --hidden_size 256 --num_layers 6
```

### Inference

For text generation:

```bash
python run.py --hidden_size 256 --num_layers 6
```

## Model Architecture

The MoBA architecture consists of:

1. **Embedding Layer**: Converts token IDs to embeddings
2. **MoBA Layers**: Multiple transformer-like layers with modified attention
3. **Block Attention**: Local attention within blocks
4. **Hierarchical Patterns**: For capturing global dependencies
5. **Rotary Position Embeddings**: For improved position awareness

## Arguments

Key parameters for the `run.py` script:

- `--hidden_size`: Dimension of hidden layers (default: 128)
- `--num_layers`: Number of MoBA layers (default: 4)
- `--num_heads`: Number of attention heads (default: 4)
- `--block_size`: Size of attention blocks (default: 16)
- `--num_blocks`: Number of hierarchical blocks (default: 2)
- `--max_len`: Maximum sequence length (default: 512)
- `--dropout`: Dropout probability (default: 0.1)
- `--lr`: Learning rate (default: 0.001)
- `--epochs`: Number of training epochs (default: 50)
- `--train`: Whether to train the model
- `--no_cuda`: Disable CUDA even if available

## Project Structure

```
MoBA/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── moba_model.py
│   │   ├── moba_layers.py
│   │   └── attention.py
│   └── utils/
│       └── __init__.py
├── run.py
├── long_context_example.py
├── README.md
└── .gitignore
```

## References

This implementation is based on the following research:

1. [Original MoBA paper citation]
2. [Related work citation]

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 