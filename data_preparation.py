import numpy as np
from data_validation import validate_bar_data
from feature_extraction import extract_all_features, calculate_momentum_features


def get_class_label(trade_direction, trade_outcome):
    """Convert trade direction and outcome to class label

    Classes:
    0: Failed Long (direction=0, outcome=0)
    1: Successful Long (direction=0, outcome=1)
    2: Failed Short (direction=1, outcome=0)
    3: Successful Short (direction=1, outcome=1)
    """
    return (trade_direction * 2) + trade_outcome


def prepare_training_data(directory_path):
    """Prepare training data from historical trades"""
    from data_loader import load_training_data

    # Load trades from directory
    historical_trades = load_training_data(directory_path)

    features = []
    momentum_metrics = []
    labels = []
    direction_counts = {0: 0, 1: 0}  # Count of long (0) and short (1) trades
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # Count of all class combinations

    for trade in historical_trades:
        try:
            # Extract features and momentum metrics
            trade_features = extract_all_features(
                trade['PreTradeBars'], trade['CurrentBar'])

            momentum_features = calculate_momentum_features(
                trade['PreTradeBars'] + [trade['CurrentBar']])

            if any(np.isnan(feature) for feature in trade_features) or \
               any(np.isnan(feature) for feature in momentum_features):
                continue

            # Get direction and outcome
            direction = trade['TradeDirection']
            outcome = trade['TradeOutcome']

            # Calculate proper class label
            class_label = get_class_label(direction, outcome)

            features.append(trade_features)
            momentum_metrics.append(momentum_features)
            labels.append(class_label)

            # Update counts
            direction_counts[direction] += 1
            class_counts[class_label] += 1

        except Exception as e:
            print(f"Error processing trade for training: {str(e)}")
            continue

    print(f"\nFeature extraction summary:")
    print(f"Total samples: {len(features)}")
    print(f"Feature vector size: {len(features[0]) if features else 0}")
    print(
        f"Momentum metrics size: {len(momentum_metrics[0]) if momentum_metrics else 0}")

    print(f"\nTrade Direction Distribution:")
    total = sum(direction_counts.values())
    for direction, count in direction_counts.items():
        print(f"{'Long' if direction == 0 else 'Short'}: {count} ({count/total:.1%})")

    print(f"\nClass Distribution:")
    for label, count in class_counts.items():
        class_name = {
            0: "Failed Long",
            1: "Successful Long",
            2: "Failed Short",
            3: "Successful Short"
        }[label]
        print(f"{class_name}: {count} ({count/total:.1%})")

    return np.array(features), np.array(momentum_metrics), np.array(labels)


def prepare_realtime_data(pre_trade_bars, current_bar):
    """Prepare real-time data for prediction"""
    cleaned_pre_bars = [validate_bar_data(bar) for bar in pre_trade_bars]
    cleaned_current_bar = validate_bar_data(current_bar)

    # Extract features and momentum metrics
    features = extract_all_features(cleaned_pre_bars, cleaned_current_bar)
    momentum_features = calculate_momentum_features(
        cleaned_pre_bars + [cleaned_current_bar])

    # Handle any NaN values
    features = np.nan_to_num(features, nan=0.0)
    momentum_features = np.nan_to_num(momentum_features, nan=0.0)

    return {
        'features': np.array([features]),
        'momentum_metrics': {
            'delta_trend': momentum_features[0],
            'current_delta': momentum_features[1],
            'delta_divergence': momentum_features[2],
            'imbalance_shift': momentum_features[3],
            'imbalance_trend': momentum_features[4],
            'relative_bar_strength': momentum_features[5],
            'atr_ratio': momentum_features[6],
            'atr_trend': momentum_features[7],
            'close_vs_open': momentum_features[8],
            'price_momentum': momentum_features[9],
        }
    }
