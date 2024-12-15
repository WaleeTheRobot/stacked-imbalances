import json
import os
from glob import glob
import numpy as np
from data_validation import validate_bar_data
from feature_extraction import extract_bar_features


def load_training_data(directory_path):
    """Load all JSON files from a directory and combine them into a single dataset"""
    all_trades = []

    json_files = glob(os.path.join(directory_path, "*.json"))

    if not json_files:
        raise ValueError(f"No JSON files found in {directory_path}")

    print(f"Found {len(json_files)} JSON files")

    for file_path in json_files:
        try:
            with open(file_path, 'r') as file:
                trades = json.load(file)

                if not isinstance(trades, list):
                    print(f"Warning: Content in {file_path} is not a list")
                    continue

                for trade in trades:
                    try:
                        # Validate trade structure
                        if not all(key in trade for key in ['PreTradeBars', 'CurrentBar', 'TradeDirection', 'TradeOutcome']):
                            continue

                        # Clean pre-trade bars
                        cleaned_pre_bars = []
                        for bar in trade['PreTradeBars']:
                            cleaned_bar = validate_bar_data(bar)
                            cleaned_pre_bars.append(cleaned_bar)

                        # Clean current bar
                        cleaned_current_bar = validate_bar_data(
                            trade['CurrentBar'])

                        # Create cleaned trade
                        cleaned_trade = {
                            'PreTradeBars': cleaned_pre_bars,
                            'CurrentBar': cleaned_current_bar,
                            'TradeDirection': trade['TradeDirection'],
                            'TradeOutcome': trade['TradeOutcome']
                        }

                        all_trades.append(cleaned_trade)

                    except Exception as e:
                        print(
                            f"Error processing trade in {file_path}: {str(e)}")
                        continue

        except json.JSONDecodeError:
            print(f"Error: Could not parse JSON in {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")

    print(f"\nData Loading Summary:")
    print(f"Total trades loaded: {len(all_trades)}")

    if not all_trades:
        raise ValueError("No valid trades were loaded")

    # Print sample trade structure
    print("\nSample trade structure:")
    sample_trade = all_trades[0]
    print(f"PreTradeBars count: {len(sample_trade['PreTradeBars'])}")
    print(f"TradeDirection: {sample_trade['TradeDirection']}")
    print(f"TradeOutcome: {sample_trade['TradeOutcome']}")

    return all_trades
