import numpy as np


def extract_bar_features(bars):
    """Extract basic features from a list of bars"""
    features = []
    for bar in bars:
        features.extend([
            bar['DataBar']['BarTypeTraits'],
            bar['DataBar']['OpenTraits'],
            bar['DataBar']['CloseTraits'],
            bar['DataBar']['Atr'],
            bar['DeltaBar']['BarTypeTraits'],
            bar['DeltaBar']['OpenTraits'],
            bar['DeltaBar']['CloseTraits'],
            bar['DeltaBar']['Atr'],
            bar['OrderFlowData']['DeltaPercentage'],
            bar['OrderFlowData']['ImbalanceMetric']
        ])
    return features


def calculate_momentum_features(bars):
    """Extract momentum-specific features from bar data"""
    if len(bars) < 2:
        return np.zeros(11)

    current_bar = bars[-1]
    prev_bar = bars[-2]

    features = {
        # Trend and Momentum
        'delta_trend': sum(b['OrderFlowData']['DeltaPercentage'] for b in bars[:-1]) / len(bars[:-1]),
        'current_delta': current_bar['OrderFlowData']['DeltaPercentage'],
        'delta_divergence': current_bar['OrderFlowData']['DeltaPercentage'] - prev_bar['OrderFlowData']['DeltaPercentage'],

        # Imbalance Analysis
        'imbalance_shift': current_bar['OrderFlowData']['ImbalanceMetric'] - prev_bar['OrderFlowData']['ImbalanceMetric'],
        'imbalance_trend': sum(b['OrderFlowData']['ImbalanceMetric'] for b in bars[:-1]) / len(bars[:-1]),

        # Bar Characteristics
        'relative_bar_strength': abs(current_bar['DataBar']['CloseTraits'] - current_bar['DataBar']['OpenTraits']) /
        max(abs(prev_bar['DataBar']['CloseTraits'] - \
            prev_bar['DataBar']['OpenTraits']), 0.0001),

        # Volatility Context
        'atr_ratio': current_bar['DataBar']['Atr'] / max(np.mean([b['DataBar']['Atr'] for b in bars]), 0.0001),
        'atr_trend': (current_bar['DataBar']['Atr'] - prev_bar['DataBar']['Atr']) / max(prev_bar['DataBar']['Atr'], 0.0001),

        # Price Action
        'close_vs_open': current_bar['DataBar']['CloseTraits'] - current_bar['DataBar']['OpenTraits'],
        'price_momentum': (current_bar['DataBar']['CloseTraits'] - prev_bar['DataBar']['CloseTraits']) /
        max(prev_bar['DataBar']['Atr'], 0.0001)
    }

    return np.array(list(features.values()))


def extract_all_features(pre_trade_bars, current_bar):
    """Extract all features including basic and momentum features"""
    # Extract basic features
    original_features = []
    for bar in pre_trade_bars:
        original_features.extend(extract_bar_features([bar]))
    original_features.extend(extract_bar_features([current_bar]))

    # Extract momentum features
    all_bars = pre_trade_bars + [current_bar]
    momentum_features = calculate_momentum_features(all_bars)

    # Combine features
    return np.concatenate([original_features, momentum_features])
