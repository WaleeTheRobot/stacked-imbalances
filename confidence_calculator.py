import numpy as np
from config import CONFIDENCE_CONFIG


class ConfidenceCalculator:
    def __init__(self):
        self.config = CONFIDENCE_CONFIG
        self.base_win_rate = self.config['base_win_rate']
        self.continuation_base_rate = self.config['continuation_base_rate']

    def analyze_confidence_distribution(self, predictions):
        """Analyze the distribution of confidence scores for both trade directions"""
        long_confidences = []
        short_confidences = []

        for pred in predictions:
            # Calculate long confidence
            long_conf = self.calculate_calibrated_confidence(
                float(pred[1]), float(pred[0]))
            long_confidences.append(long_conf)

            # Calculate short confidence
            short_conf = self.calculate_calibrated_confidence(
                float(pred[3]), float(pred[2]))
            short_confidences.append(short_conf)

        long_confidences = np.array(long_confidences)
        short_confidences = np.array(short_confidences)

        return {
            'long': {
                'mean': np.mean(long_confidences),
                'std': np.std(long_confidences),
                'min': np.min(long_confidences),
                'max': np.max(long_confidences),
                'percentiles': {
                    '25': np.percentile(long_confidences, 25),
                    '50': np.percentile(long_confidences, 50),
                    '75': np.percentile(long_confidences, 75)
                }
            },
            'short': {
                'mean': np.mean(short_confidences),
                'std': np.std(short_confidences),
                'min': np.min(short_confidences),
                'max': np.max(short_confidences),
                'percentiles': {
                    '25': np.percentile(short_confidences, 25),
                    '50': np.percentile(short_confidences, 50),
                    '75': np.percentile(short_confidences, 75)
                }
            }
        }

    def analyze_confidence_thresholds(self, predictions, y, thresholds=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]):
        """Analyze prediction accuracy at different confidence thresholds"""
        results = {}

        for threshold in thresholds:
            long_correct = 0
            long_total = 0
            short_correct = 0
            short_total = 0

            for i, pred in enumerate(predictions):
                # Analyze long trades
                long_conf = self.calculate_calibrated_confidence(
                    float(pred[1]), float(pred[0]))
                if long_conf >= threshold:
                    long_total += 1
                    if y[i] == 1:  # Successful long
                        long_correct += 1

                # Analyze short trades
                short_conf = self.calculate_calibrated_confidence(
                    float(pred[3]), float(pred[2]))
                if short_conf >= threshold:
                    short_total += 1
                    if y[i] == 3:  # Successful short
                        short_correct += 1

            results[threshold] = {
                'long': {
                    'count': long_total,
                    'accuracy': long_correct / max(long_total, 1)
                },
                'short': {
                    'count': short_total,
                    'accuracy': short_correct / max(short_total, 1)
                }
            }

        return results

    def calculate_calibrated_confidence(self, success_prob, fail_prob):
        """Calculate calibrated confidence score"""
        if success_prob + fail_prob == 0:
            return 0.0

        raw_confidence = success_prob / (success_prob + fail_prob)
        return (raw_confidence - self.base_win_rate) / (1 - self.base_win_rate)

    def calculate_trade_confidences(self, probas, momentum_metrics):
        """Calculate trade continuation probabilities based on configured weights"""
        def normalize_metric(value, max_value):
            return min(max(value / max_value, -1), 1)

        # Normalize metrics
        delta = normalize_metric(
            momentum_metrics['current_delta'], self.config['max_delta'])
        delta_trend = normalize_metric(
            momentum_metrics['delta_trend'], self.config['max_delta'])
        imbalance = normalize_metric(
            momentum_metrics['imbalance_trend'], self.config['max_imbalance'])
        momentum = normalize_metric(
            momentum_metrics['price_momentum'], self.config['max_momentum'])

        # Calculate weighted components
        delta_alignment = (
            delta * self.config['momentum_weights']['current_delta'] +
            delta_trend * self.config['momentum_weights']['delta_trend']
        ) / sum(self.config['momentum_weights'].values())

        momentum_strength = momentum * \
            self.config['momentum_weights']['price_momentum']

        volume_confirmation = (
            delta * self.config['volume_weights']['volume_pressure'] +
            imbalance * self.config['volume_weights']['imbalance_trend']
        ) / sum(self.config['volume_weights'].values())

        # Calculate final scores
        weights = self.config['weights']
        long_score = (
            weights['delta_alignment'] * max(0, delta_alignment) +
            weights['momentum_strength'] * max(0, momentum_strength) +
            weights['volume_confirmation'] * max(0, volume_confirmation)
        )

        short_score = (
            weights['delta_alignment'] * max(0, -delta_alignment) +
            weights['momentum_strength'] * max(0, -momentum_strength) +
            weights['volume_confirmation'] * max(0, -volume_confirmation)
        )

        return {
            'long': {
                'continuation_prob': self._adjust_probability(
                    float(probas[0][1]), long_score)
            },
            'short': {
                'continuation_prob': self._adjust_probability(
                    float(probas[0][3]), short_score)
            }
        }

    def _adjust_probability(self, base_prob, score):
        """Adjust probability based on score and configured bounds"""
        multiplier = 1.0 + (score * self.config['max_adjustment'])
        multiplier = max(self.config['min_prob_multiplier'],
                         min(self.config['max_prob_multiplier'], multiplier))
        return min(1.0, base_prob * multiplier)
