PATHS = {
    'training_directory': r"C:\training",
    'model_save_path': "stackedimbalances_model.json",
    'study_save_path': "stackedimbalances_scaler.pkl"
}

MODEL_CONFIG = {
    'train_params': {
        # Core parameters
        'objective': 'multi:softprob',
        'num_class': 4,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'device': 'cuda',

        # Stability enhancement parameters
        'min_child_weight': 5,
        'max_depth': 4,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'reg_alpha': 0.15,
        'reg_lambda': 0.25,
        'gamma': 0.15,
        'learning_rate': 0.03,

        # Sampling parameters
        'sampling_method': 'gradient_based',
        'grow_policy': 'lossguide',

        # Performance parameters
        'n_jobs': 1,  # Set to 1 since we're using GPU
        'max_bin': 64,
        'max_leaves': 16,
    },

    'base_win_rate': 0.63,
}

TRAIN_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'n_trials': 1,  # Adjust when ready
    'timeout': 14400,
    'n_startup_trials': 1,  # Adjust when ready. Maybe use 25% of n_trials
    'early_stopping_rounds': 50,
    # Parallel optimization settings
    'optimization': {
        'storage_type': 'sqlite',
        'storage_path': 'optuna_study.db',
        'study_name': 'trading_model_optimization',
        'load_if_exists': True,
        'n_jobs': -1,  # Use all CPU cores for parallel trials
        'gc_after_trial': True,
        'show_progress_bar': True,
        'backend': 'multiprocessing'
    }
}

# More conservative parameter ranges
PARAM_RANGES = {
    'max_depth': (3, 5),             # Narrower range
    'learning_rate': (0.01, 0.05),   # Lower learning rates
    'n_estimators': (300, 2000),     # More trees
    'min_child_weight': (3, 6),      # Higher minimum weights
    'gamma': (0.1, 0.2),            # Narrower gamma range
    'subsample': (0.8, 0.9),        # Higher subsample range
    'colsample_bytree': (0.8, 0.9),  # Higher column sample range
    'reg_alpha': (0.1, 0.2),        # Higher regularization
    'reg_lambda': (0.2, 0.3)        # Higher regularization
}

CONFIDENCE_CONFIG = {
    # Base rates
    'base_win_rate': 0.63,
    'continuation_base_rate': 0.75,

    # Normalization constants
    'max_delta': 30.0,
    'max_imbalance': 8.0,
    'max_atr_ratio': 1.5,
    'max_momentum': 1.0,

    # Continuation probability weights
    'weights': {
        'delta_alignment': 0.40,
        'momentum_strength': 0.30,
        'volume_confirmation': 0.20,
        'divergence_penalty': 0.10
    },

    # Momentum strength weights
    'momentum_weights': {
        'current_delta': 3,
        'delta_trend': 2,
        'price_momentum': 1
    },

    # Volume confirmation weights
    'volume_weights': {
        'volume_pressure': 2,
        'imbalance_trend': 1
    },

    # Probability adjustment bounds
    'min_prob_multiplier': 0.8,
    'max_prob_multiplier': 1.2,
    'max_adjustment': 0.20
}
