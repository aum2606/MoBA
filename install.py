#!/usr/bin/env python
import os
import sys
import subprocess
import platform

def run_command(command):
    """Run a shell command and print output."""
    print(f"Running: {command}")
    process = subprocess.Popen(
        command, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT
    )
    
    # Print output in real-time
    for line in iter(process.stdout.readline, b''):
        sys.stdout.write(line.decode('utf-8'))
    
    process.wait()
    return process.returncode


def check_cuda():
    """Check if CUDA is available and return version if it is."""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "N/A"
            
            print(f"CUDA is available! Version: {cuda_version}")
            print(f"GPU Count: {device_count}")
            print(f"GPU Device: {device_name}")
            return True, cuda_version
        else:
            print("CUDA is not available. Will install CPU version.")
            return False, None
    except ImportError:
        print("PyTorch not installed yet. Will check CUDA availability after installation.")
        # Check if nvidia-smi is available
        try:
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                print("NVIDIA GPU detected, but PyTorch not installed yet.")
                return True, None
            else:
                print("No NVIDIA GPU detected. Will install CPU version.")
                return False, None
        except FileNotFoundError:
            print("nvidia-smi not found. Will install CPU version.")
            return False, None


def install_torch(cuda_available, cuda_version=None):
    """Install PyTorch with appropriate CUDA version."""
    if cuda_available:
        # Different install command for different CUDA versions
        if cuda_version and cuda_version.startswith('11.'):
            cmd = "pip install torch>=1.10.0 --extra-index-url https://download.pytorch.org/whl/cu116"
        elif cuda_version and cuda_version.startswith('10.'):
            cmd = "pip install torch>=1.10.0 --extra-index-url https://download.pytorch.org/whl/cu102"
        else:
            # Latest version
            cmd = "pip install torch>=1.10.0"
    else:
        cmd = "pip install torch>=1.10.0 --extra-index-url https://download.pytorch.org/whl/cpu"
    
    return run_command(cmd)


def install_dependencies():
    """Install all required dependencies."""
    # Install from requirements.txt
    result = run_command("pip install -r requirements.txt")
    
    # Install the package in development mode
    result = run_command("pip install -e .")
    
    return result == 0


def main():
    print("Setting up MOBA: Mixture of Block Attention")
    print("-" * 50)
    
    # Check CUDA availability
    cuda_available, cuda_version = check_cuda()
    
    # Create virtual environment if requested
    if "--venv" in sys.argv:
        venv_name = ".venv"
        print(f"Creating virtual environment: {venv_name}")
        run_command(f"python -m venv {venv_name}")
        
        # Activate virtual environment commands
        if platform.system() == "Windows":
            activate_cmd = f"{venv_name}\\Scripts\\activate"
        else:
            activate_cmd = f"source {venv_name}/bin/activate"
        
        print(f"Activate the environment with: {activate_cmd}")
        print("Then run this script again without --venv")
        return
    
    # Install PyTorch with appropriate CUDA support
    install_torch(cuda_available, cuda_version)
    
    # Install other dependencies
    success = install_dependencies()
    
    if success:
        print("\nSetup completed successfully!")
        print("\nYou can now run the MOBA examples:")
        print("  python run.py --train")
        print("  python long_context_example.py --all")
    else:
        print("\nSetup encountered some errors. Please check the output above.")


if __name__ == "__main__":
    main() 