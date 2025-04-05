@echo off
echo Running MoBA with GPU...
echo.

:: Check if Python is in PATH
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python not found in PATH!
    echo Please install Python or ensure it's in your PATH.
    pause
    exit /b 1
)

:: Check if PyTorch with CUDA is installed
python -c "import torch; print('CUDA Available: ' + str(torch.cuda.is_available()))" 
if %ERRORLEVEL% neq 0 (
    echo Error checking PyTorch! Make sure PyTorch is installed.
    echo You can install it with: pip install torch
    pause
    exit /b 1
)

:: Run the quick demo script
echo.
python quickrun.py %*

echo.
echo Completed execution.
pause 