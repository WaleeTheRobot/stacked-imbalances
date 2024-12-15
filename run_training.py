import xgboost as xgb
import numpy as np
from model import TradingModel
from config import PATHS
import time
import psutil
import GPUtil


def check_system_resources():
    """Check available system resources"""
    print("\n=== System Resource Check ===")

    # CPU info
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU Cores: {cpu_count}")
    print(f"CPU Usage: {cpu_percent}%")

    # Memory info
    memory = psutil.virtual_memory()
    print(f"Available Memory: {memory.available / (1024 ** 3):.1f} GB")

    # GPU info
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"\nGPU: {gpu.name}")
            print(f"Memory Used: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
            print(f"GPU Load: {gpu.load*100}%")
        use_gpu = len(gpus) > 0
    except:
        print("\nNo GPU information available")
        use_gpu = False

    return use_gpu


def main():
    start_time = time.time()
    print("Starting training pipeline...")

    try:
        use_gpu = check_system_resources()

        # Initialize model (now without base_win_rate parameter)
        model = TradingModel(use_gpu=use_gpu)
        model.train(PATHS['training_directory'])

        # Print execution time
        duration = time.time() - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        print(f"\nTotal execution time: {hours}h {minutes}m {seconds}s")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
