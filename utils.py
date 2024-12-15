import numpy as np


def safe_float_convert(value):
    """Safely convert a value to float, handling NaN strings and other invalid values"""
    if isinstance(value, str) and value.lower() == 'nan':
        return np.nan
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def map_label_to_name(label):
    """Map numeric label to descriptive name"""
    label_map = {
        0: "Buy Fail",
        1: "Buy Success",
        2: "Sell Fail",
        3: "Sell Success"
    }
    return label_map.get(label, f"Unknown ({label})")
