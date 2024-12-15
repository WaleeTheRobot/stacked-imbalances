import numpy as np


def validate_bar_data(bar):
    """Validate and clean bar data, replacing any invalid values with np.nan"""
    from utils import safe_float_convert

    cleaned_bar = {
        'DataBar': {
            'BarTypeTraits': safe_float_convert(bar['DataBar']['BarTypeTraits']),
            'OpenTraits': safe_float_convert(bar['DataBar']['OpenTraits']),
            'CloseTraits': safe_float_convert(bar['DataBar']['CloseTraits']),
            'Atr': safe_float_convert(bar['DataBar'].get('Atr', np.nan))
        },
        'DeltaBar': {
            'BarTypeTraits': safe_float_convert(bar['DeltaBar']['BarTypeTraits']),
            'OpenTraits': safe_float_convert(bar['DeltaBar']['OpenTraits']),
            'CloseTraits': safe_float_convert(bar['DeltaBar']['CloseTraits']),
            'Atr': safe_float_convert(bar['DeltaBar'].get('Atr', np.nan))
        },
        'OrderFlowData': {
            'DeltaPercentage': safe_float_convert(bar['OrderFlowData']['DeltaPercentage']),
            'ImbalanceMetric': safe_float_convert(bar['OrderFlowData']['ImbalanceMetric']),
        }
    }
    return cleaned_bar
