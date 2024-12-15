**_Don't Use This_**

This is an example model for the stacked imbalances for OrderFlowBot using several days of one minute bars. Move the trained model and scaler to the models directory.

https://github.com/WaleeTheRobot/order-flow-bot

# Model Setup Instructions

## Prerequisites

1. **NVIDIA GPU Support**

   - NVIDIA GPU with compute capability 3.5 or higher
   - NVIDIA drivers installed and working
   - CUDA Toolkit (11.x or 12.x)

2. To verify CUDA installation: `nvcc --version`
3. Install dependencies: `pip install -r requirements.txt`

## Training

1. The training files should be in `C:\training`.
2. Train: `python run_training.py`
3. Move the trained models into the `models` directory.
4. Make sure the path and file names are correct in the `.env` file.

## Running

1. Run `python server.py` to run the server at `http://localhost:5000`.
