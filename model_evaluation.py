import numpy as np
from confidence_calculator import ConfidenceCalculator


class ModelEvaluator:
    def __init__(self):
        """Initialize ModelEvaluator with feature names mapping and confidence calculator"""
        self.feature_names = self._get_feature_names()
        self.confidence_calculator = ConfidenceCalculator()

    def _get_feature_names(self):
        """Create mapping of feature indices to names"""
        # Basic bar features (repeated for each bar)
        base_features = [
            'BarTypeTraits',
            'OpenTraits',
            'CloseTraits',
            'Atr',
            'DeltaBarType',
            'DeltaBarOpen',
            'DeltaBarClose',
            'DeltaBarAtr',
            'DeltaPercentage',
            'ImbalanceMetric'
        ]

        # Momentum specific features
        momentum_features = [
            'delta_trend',
            'current_delta',
            'delta_divergence',
            'imbalance_shift',
            'imbalance_trend',
            'relative_bar_strength',
            'atr_ratio',
            'atr_trend',
            'close_vs_open',
            'price_momentum'
        ]

        all_features = []

        # Add features for historical bars
        for i in range(3):  # Assuming 3 historical bars
            all_features.extend([f'bar{i+1}_{feat}' for feat in base_features])

        # Add current bar features
        all_features.extend([f'current_{feat}' for feat in base_features])

        # Add momentum features
        all_features.extend(momentum_features)

        return {f'f{i}': name for i, name in enumerate(all_features)}

    def evaluate_model(self, model, X_scaled, y, momentum_metrics=None):
        """Evaluate the model and print comprehensive statistics"""
        try:
            dtrain = model.prepare_data(X_scaled, y)
            predictions = model.model.predict(dtrain)

            # Class distribution
            class_counts = np.bincount(y)
            total_samples = len(y)

            print("\nClass Distribution:")
            print("------------------")
            for i, count in enumerate(class_counts):
                class_name = {
                    0: 'fail_long',
                    1: 'success_long',
                    2: 'fail_short',
                    3: 'success_short'
                }.get(i, f'Class {i}')
                print(f"{class_name}: {count} samples ({count/total_samples:.2%})")

            # Probability distribution for each class
            print("\nProbability Distribution:")
            print("------------------------")
            for i in range(4):
                class_name = {
                    0: 'fail_long',
                    1: 'success_long',
                    2: 'fail_short',
                    3: 'success_short'
                }.get(i, f'Class {i}')
                probs = predictions[:, i]
                print(f"\n{class_name}:")
                print(f"  Mean probability: {probs.mean():.4f}")
                print(f"  Std deviation: {probs.std():.4f}")
                print(f"  25th percentile: {np.percentile(probs, 25):.4f}")
                print(f"  Median: {np.percentile(probs, 50):.4f}")
                print(f"  75th percentile: {np.percentile(probs, 75):.4f}")

            # Confidence Analysis
            print("\nCalibrated Confidence Analysis:")
            print("------------------------------")
            confidence_dist = self.confidence_calculator.analyze_confidence_distribution(
                predictions)

            print("\nLong Trade Confidence:")
            print(f"  Mean: {confidence_dist['long']['mean']:.4f}")
            print(f"  Std: {confidence_dist['long']['std']:.4f}")
            print(
                f"  Range: [{confidence_dist['long']['min']:.4f}, {confidence_dist['long']['max']:.4f}]")

            print("\nShort Trade Confidence:")
            print(f"  Mean: {confidence_dist['short']['mean']:.4f}")
            print(f"  Std: {confidence_dist['short']['std']:.4f}")
            print(
                f"  Range: [{confidence_dist['short']['min']:.4f}, {confidence_dist['short']['max']:.4f}]")

            # Accuracy by confidence threshold
            print("\nAccuracy by Calibrated Confidence Threshold:")
            print("------------------------------------------")
            threshold_results = self.confidence_calculator.analyze_confidence_thresholds(
                predictions, y)

            for threshold, results in threshold_results.items():
                print(f"\nThreshold: {threshold:.1f}")

                long_results = results['long']
                print(f"Long trades:")
                print(f"  Count: {long_results['count']}")
                print(f"  Accuracy: {long_results['accuracy']:.2%}")

                short_results = results['short']
                print(f"Short trades:")
                print(f"  Count: {short_results['count']}")
                print(f"  Accuracy: {short_results['accuracy']:.2%}")

            # Feature importance analysis
            print("\nFeature Importance Analysis:")
            print("---------------------------")
            importance_scores = model.model.get_score(importance_type='gain')

            # Sort features by importance
            sorted_features = sorted(importance_scores.items(),
                                     key=lambda x: x[1],
                                     reverse=True)

            # Continuation Probability Analysis (if momentum metrics available)
            if momentum_metrics is not None:
                print("\nContinuation Probability Analysis:")
                print("--------------------------------")
                sample_idx = 0  # Use first sample as example
                momentum_dict = {
                    'delta_trend': momentum_metrics[sample_idx][0],
                    'current_delta': momentum_metrics[sample_idx][1],
                    'delta_divergence': momentum_metrics[sample_idx][2],
                    'imbalance_shift': momentum_metrics[sample_idx][3],
                    'imbalance_trend': momentum_metrics[sample_idx][4],
                    'relative_bar_strength': momentum_metrics[sample_idx][5],
                    'atr_ratio': momentum_metrics[sample_idx][6],
                    'atr_trend': momentum_metrics[sample_idx][7],
                    'close_vs_open': momentum_metrics[sample_idx][8],
                    'price_momentum': momentum_metrics[sample_idx][9]
                }

                trade_confidences = self.confidence_calculator.calculate_trade_confidences(
                    predictions, momentum_dict)

                print("\nLong Trade Continuation:")
                print(
                    f"  Mean probability: {trade_confidences['long']['continuation_prob']:.4f}")

                print("\nShort Trade Continuation:")
                print(
                    f"  Mean probability: {trade_confidences['short']['continuation_prob']:.4f}")

                print("\nMomentum Metrics Summary:")
                for metric, value in momentum_dict.items():
                    print(f"  {metric}: {value:.4f}")

            print("\nTop 5 Most Important Features:")
            for feat, score in sorted_features[:5]:
                feature_name = self.feature_names.get(feat, feat)
                print(f"  {feature_name}: {score:.4f}")

            return {
                'class_distribution': class_counts,
                'predictions': predictions,
                'confidence_distribution': confidence_dist,
                'threshold_results': threshold_results,
                'feature_importance': dict(sorted_features)
            }

        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            raise
