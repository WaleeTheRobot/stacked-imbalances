from flask import Flask, request, jsonify
from model import TradingModel
from dotenv import load_dotenv
import traceback
import os

load_dotenv()

app = Flask(__name__)
model = TradingModel()


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        print("Received request data:", request.json)

        data = request.json
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data received'
            }), 400

        pre_trade_bars = data.get('PreTradeBars', [])
        current_bar = data.get('CurrentBar', {})
        trade_direction = data.get('TradeDirection')

        if trade_direction is None:
            return jsonify({
                'status': 'error',
                'message': 'TradeDirection is required'
            }), 400

        print("Processing prediction...")

        prediction = model.predict_proba(
            pre_trade_bars, current_bar, trade_direction)

        print("Converted prediction:", prediction)

        return jsonify({
            'status': 'success',
            'prediction': prediction
        })

    except Exception as e:
        print("Error:", str(e))
        print("Detailed traceback:", traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    try:
        print("Loading model...")

        model_path = os.getenv('MODEL_PATH')
        if not model_path:
            raise EnvironmentError(
                "MODEL_PATH environment variable is not set")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        model.load_model(model_path)
        print("Model and scaler loaded successfully")

        # Verify scaler is available
        if not hasattr(model, 'feature_scaler'):
            raise ValueError("Model loaded but scaler is missing")

        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print("Error during startup:", str(e))
        raise
