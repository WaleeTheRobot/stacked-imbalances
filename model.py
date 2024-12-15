import os
import joblib
import numpy as np
import xgboost as xgb
import optuna
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import parallel_backend
import time
from data_preparation import prepare_training_data, prepare_realtime_data
from feature_extraction import calculate_momentum_features
from confidence_calculator import ConfidenceCalculator
from model_evaluation import ModelEvaluator
from config import MODEL_CONFIG, TRAIN_CONFIG, PARAM_RANGES, PATHS

warnings.filterwarnings('ignore', category=UserWarning)


class TradingModel:
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu
        self.model_params = MODEL_CONFIG['train_params'].copy()
        self.model = None
        self.confidence_calculator = ConfidenceCalculator()
        self.model_evaluator = ModelEvaluator()

    def prepare_data(self, X, y=None, sample_weight=None):
        return xgb.DMatrix(
            X,
            label=y,
            weight=sample_weight,
            enable_categorical=True,
            nthread=4
        )

    def train(self, training_directory=PATHS['training_directory']):
        try:
            print("Loading and preparing data...")
            X, momentum_metrics, y = prepare_training_data(training_directory)
            print(f"Dataset shape: {X.shape}")

            print("\nScaling features...")
            self.feature_scaler = StandardScaler()
            X_scaled = self.feature_scaler.fit_transform(X)

            print("\nCalculating class weights...")
            class_weights = self._calculate_class_weights(y)

            print("\nStarting hyperparameter optimization...")
            storage_string = f"{TRAIN_CONFIG['optimization']['storage_type']}:///{TRAIN_CONFIG['optimization']['storage_path']}"

            study = optuna.create_study(
                direction="minimize",
                storage=storage_string,
                study_name=TRAIN_CONFIG['optimization']['study_name'],
                load_if_exists=TRAIN_CONFIG['optimization']['load_if_exists']
            )

            objective = self._create_objective(X_scaled, y, class_weights)

            with parallel_backend(
                TRAIN_CONFIG['optimization']['backend'],
                n_jobs=TRAIN_CONFIG['optimization']['n_jobs']
            ):
                study.optimize(
                    objective,
                    n_trials=TRAIN_CONFIG['n_trials'],
                    timeout=TRAIN_CONFIG['timeout'],
                    n_jobs=TRAIN_CONFIG['optimization']['n_jobs'],
                    gc_after_trial=TRAIN_CONFIG['optimization']['gc_after_trial'],
                    show_progress_bar=TRAIN_CONFIG['optimization']['show_progress_bar']
                )

            print("\nHyperparameter optimization results:")
            print(f"Best trial value: {study.best_value}")
            print(f"Best parameters: {study.best_params}")

            print("\nTraining final model...")
            best_params = {**self.model_params, **study.best_params}
            sample_weight = np.array([class_weights[yi] for yi in y])
            dtrain = self.prepare_data(X_scaled, y, sample_weight)

            self.model = xgb.train(best_params, dtrain,
                                   num_boost_round=best_params['n_estimators'])

            print("\nEvaluating model performance...")
            evaluation_results = self.model_evaluator.evaluate_model(
                self, X_scaled, y, momentum_metrics)

            # Store evaluation results
            self.evaluation_results = evaluation_results

            print("\nSaving model...")
            self.save_model(PATHS['model_save_path'])

            return self.model

        except Exception as e:
            print(f"\nError during training: {str(e)}")
            raise

    def predict_proba(self, pre_trade_bars, current_bar, trade_direction=None):
        if self.model is None:
            raise ValueError("Model not loaded or trained")

        try:
            prepared_data = prepare_realtime_data(pre_trade_bars, current_bar)
            features_scaled = self.feature_scaler.transform(
                prepared_data['features'])
            dtest = self.prepare_data(features_scaled)
            probas = self.model.predict(dtest)

            if len(probas.shape) == 1:
                probas = probas.reshape(1, -1)

            # Calculate confidences using configured parameters
            long_confidence = self.confidence_calculator.calculate_calibrated_confidence(
                float(probas[0][1]), float(probas[0][0]))
            short_confidence = self.confidence_calculator.calculate_calibrated_confidence(
                float(probas[0][3]), float(probas[0][2]))

            # Calculate continuation probabilities
            trade_confidences = self.confidence_calculator.calculate_trade_confidences(
                probas, prepared_data['momentum_metrics'])

            result = {
                'buy': {
                    'success_prob': float(probas[0][1]),
                    'fail_prob': float(probas[0][0]),
                    'confidence': long_confidence,
                    'continuation_prob': trade_confidences['long']['continuation_prob']
                },
                'sell': {
                    'success_prob': float(probas[0][3]),
                    'fail_prob': float(probas[0][2]),
                    'confidence': short_confidence,
                    'continuation_prob': trade_confidences['short']['continuation_prob']
                },
                'momentum_metrics': prepared_data['momentum_metrics']
            }

            return result

        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise

    def _calculate_class_weights(self, y):
        class_counts = np.bincount(y)
        total_samples = len(y)
        max_weight = 10.0

        class_weights = {}
        for i, count in enumerate(class_counts):
            raw_weight = total_samples / (len(class_counts) * count)
            class_weights[i] = min(raw_weight, max_weight)
            print(
                f"Class {i}: {count} samples ({count/total_samples:.2%}), weight: {class_weights[i]:.3f}")

        return class_weights

    def _create_objective(self, X_scaled, y, class_weights):
        skf = StratifiedKFold(
            n_splits=TRAIN_CONFIG['cv_folds'],
            shuffle=True,
            random_state=TRAIN_CONFIG['random_state']
        )

        def objective(trial):
            start_time = time.time()

            param = {
                'max_depth': trial.suggest_int('max_depth', *PARAM_RANGES['max_depth']),
                'learning_rate': trial.suggest_float('learning_rate', *PARAM_RANGES['learning_rate'], log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', *PARAM_RANGES['min_child_weight']),
                'gamma': trial.suggest_float('gamma', *PARAM_RANGES['gamma']),
                'subsample': trial.suggest_float('subsample', *PARAM_RANGES['subsample']),
                'colsample_bytree': trial.suggest_float('colsample_bytree', *PARAM_RANGES['colsample_bytree']),
                'reg_alpha': trial.suggest_float('reg_alpha', *PARAM_RANGES['reg_alpha']),
                'reg_lambda': trial.suggest_float('reg_lambda', *PARAM_RANGES['reg_lambda']),
            }
            n_estimators = trial.suggest_int(
                'n_estimators', *PARAM_RANGES['n_estimators'])

            param.update(self.model_params)

            scores = []
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                sample_weight = np.array([class_weights[yi] for yi in y_train])
                dtrain = self.prepare_data(X_train, y_train, sample_weight)
                dval = self.prepare_data(X_val, y_val)

                evals_result = {}
                model = xgb.train(
                    param,
                    dtrain,
                    num_boost_round=n_estimators,
                    early_stopping_rounds=TRAIN_CONFIG['early_stopping_rounds'],
                    evals=[(dtrain, 'train'), (dval, 'val')],
                    evals_result=evals_result,
                    verbose_eval=False
                )

                val_score = evals_result['val']['mlogloss'][-1]
                scores.append(val_score)

            avg_score = np.mean(scores)
            print(
                f"Trial {trial.number} completed in {time.time() - start_time:.2f}s, score: {avg_score:.4f}")
            return avg_score

        return objective

    def save_model(self, filename):
        """Save the model and scaler files"""
        if self.model is not None:
            # Save model
            model_filename = filename
            self.model.save_model(model_filename)
            print(f"Model saved to: {model_filename}")

            # Save scaler
            scaler_filename = filename.replace('.json', '_scaler.pkl')
            if hasattr(self, 'feature_scaler'):
                joblib.dump(self.feature_scaler, scaler_filename)
                print(f"Scaler saved to: {scaler_filename}")
        else:
            raise ValueError("No model to save. Train or load a model first.")

    def load_model(self, filename):
        """Load the model and scaler files"""
        try:
            # Load model
            self.model = xgb.Booster()
            self.model.load_model(filename)

            # Load scaler
            scaler_filename = filename.replace('.json', '_scaler.pkl')
            if os.path.exists(scaler_filename):
                self.feature_scaler = joblib.load(scaler_filename)
                print(f"Model and scaler loaded from: {filename}")
            else:
                raise ValueError(f"Scaler file not found: {scaler_filename}")

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
