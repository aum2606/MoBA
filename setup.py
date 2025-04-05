from setuptools import setup, find_packages

setup(
    name="moba",
    version="0.1.0",
    description="Implementation of MOBA: Mixture of Block Attention for Long-Context LLMs",
    author="MOBA Team",
    author_email="info@moba.example.com",
    url="https://github.com/yourusername/moba",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "transformers>=4.20.0",
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "tqdm",
        "wandb",
        "einops",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
) 