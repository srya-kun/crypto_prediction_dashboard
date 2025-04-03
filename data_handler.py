# crypto_prediction_dashboard/data_handler.py

import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
import os # Used to construct file paths reliably
import datetime # Needed for timestamp in filename potentially

# --- PLACEHOLDER for API Key ---
# IMPORTANT: Replace "YOUR_API_KEY_HERE" with your actual CoinGecko API key
COINGECKO_API_KEY = "YOUR_API_KEY_HERE"
# --- END PLACEHOLDER ---

# Define base directories relative to this script's location
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_DIR = os.path.join(BASE_DIR, 'logs') # <-- ADDED: Directory for log files

# Ensure the log directory exists
os.makedirs(LOG_DIR, exist_ok=True) # <-- ADDED: Create log dir if it doesn't exist

# --- load_historical_data function remains the same as before ---
def load_historical_data(coin_name):
    """
    Loads historical cryptocurrency data from a CSV file.
    (Code is identical to the previous version)
    """
    # Construct the expected filename (e.g., bitcoin_data.csv)
    filename = f"{coin_name.lower()}_data.csv"
    file_path = os.path.join(DATA_DIR, filename)

    print(f"Attempting to load data from: {file_path}")

    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return None

    try:
        df = pd.read_csv(file_path)
        df.rename(columns={
            'Date': 'date', 'High': 'high', 'Low': 'low', 'Open': 'open',
            'Close': 'close', 'Volume': 'volume', 'Marketcap': 'marketcap'
        }, inplace=True)
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
        df.set_index('date', inplace=True)
        relevant_columns = ['open', 'high', 'low', 'close', 'volume', 'marketcap']
        df = df[relevant_columns]
        df.sort_index(ascending=True, inplace=True)
        df.ffill(inplace=True)
        df.dropna(inplace=True)

        if df.empty:
            print(f"Warning: DataFrame is empty after loading and cleaning {file_path}")
            return None
        print(f"Successfully loaded and formatted data for {coin_name}.")
        return df

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except KeyError as e:
        print(f"Error: Column mismatch in {file_path}. Missing column: {e}. Check CSV header.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")
        return None


# --- add_technical_features function remains the same as before ---
def add_technical_features(df):
    """
    Adds technical indicators as features to the DataFrame.
    (Code is identical to the previous version)
    """
    if df is None or df.empty:
        print("Cannot add features: Input DataFrame is None or empty.")
        return df

    print("Adding technical features...")
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    # Add other indicators (e.g., using pandas_ta) here if desired

    original_rows = len(df)
    df.dropna(inplace=True)
    rows_after_dropna = len(df)
    print(f"Removed {original_rows - rows_after_dropna} rows with NaN values after adding features.")
    print("Finished adding features.")
    return df


# --- NEW Function: save_fetched_data ---
def save_fetched_data(coin_id, data_point):
    """
    Saves the fetched data point to a separate log file for the coin.

    Args:
        coin_id (str): The CoinGecko API ID for the coin (e.g., 'bitcoin').
        data_point (dict): The dictionary containing the fetched data point.
                           Expected keys match the output of get_latest_coingecko_data.
    """
    if not data_point:
        print("No data point provided to save.")
        return

    log_filename = f"{coin_id}_fetch_log.csv"
    log_file_path = os.path.join(LOG_DIR, log_filename)

    # Define the exact order of columns for the CSV
    # Include the timestamp as the first column for clarity
    columns_order = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'marketcap']

    # Convert the single data point dictionary to a DataFrame
    # Make sure the 'timestamp' key exists in data_point
    try:
        # Ensure timestamp is timezone-aware (UTC recommended) before saving
        if isinstance(data_point.get('timestamp'), pd.Timestamp):
             ts = data_point['timestamp'].tz_convert('UTC') if data_point['timestamp'].tz is not None else data_point['timestamp'].tz_localize('UTC')
        else:
             ts = pd.Timestamp(data_point.get('timestamp', pd.Timestamp.now(tz='UTC'))).tz_convert('UTC')

        df_to_save = pd.DataFrame([{
             'timestamp': ts,
             'open': data_point.get('open'),
             'high': data_point.get('high'),
             'low': data_point.get('low'),
             'close': data_point.get('close'),
             'volume': data_point.get('volume'),
             'marketcap': data_point.get('marketcap')
        }], columns=columns_order) # Ensure columns are in the defined order

    except Exception as e:
        print(f"Error creating DataFrame for logging: {e}")
        print(f"Data point received: {data_point}")
        return

    try:
        # Check if file exists to determine if header should be written
        file_exists = os.path.exists(log_file_path)

        # Append to CSV. Write header only if file doesn't exist yet.
        df_to_save.to_csv(log_file_path, mode='a', header=not file_exists, index=False)
        # print(f"Successfully saved fetched data for {coin_id} to {log_file_path}") # Optional: verbose logging

    except IOError as e:
        print(f"Error saving fetched data to {log_file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during saving fetched data: {e}")


# --- MODIFIED Function: get_latest_coingecko_data ---
def get_latest_coingecko_data(coin_id):
    """
    Fetches the latest market data for a specific coin from CoinGecko API
    AND saves the fetched data point to a log file.

    Args:
        coin_id (str): The CoinGecko API ID for the coin (e.g., 'bitcoin', 'ethereum', 'dogecoin').

    Returns:
        dict: A dictionary containing the latest market data (approximating one row
              of the historical data format), or None if an error occurs.
              Keys: 'open', 'high', 'low', 'close', 'volume', 'marketcap', 'timestamp'
    """
    api_url = f"https://api.coingecko.com/api/v3/coins/markets"
    params = {'vs_currency': 'usd', 'ids': coin_id}
    headers = {'X-CG-API-Key': COINGECKO_API_KEY}

    print(f"Fetching latest data for {coin_id} from CoinGecko...")

    try:
        response = requests.get(api_url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()

        if not data:
            print(f"Error: No data returned from CoinGecko for {coin_id}")
            return None

        coin_data = data[0]
        latest_timestamp = pd.to_datetime(coin_data.get('last_updated', pd.Timestamp.now(tz='UTC')))

        latest_data_point = {
            'open': coin_data.get('current_price'),
            'high': coin_data.get('high_24h'),
            'low': coin_data.get('low_24h'),
            'close': coin_data.get('current_price'),
            'volume': coin_data.get('total_volume'),
            'marketcap': coin_data.get('market_cap'),
            'timestamp': latest_timestamp # Keep the timestamp!
        }
        print(f"Successfully fetched latest data point for {coin_id} at {latest_timestamp}")

        # --- ADDED: Call the function to save the fetched data ---
        save_fetched_data(coin_id, latest_data_point)
        # --- END ADDED ---

        return latest_data_point # Return the data AFTER attempting to save it

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from CoinGecko for {coin_id}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response text: {e.response.text}") # Helps debug API key/rate limit issues
            if e.response.status_code == 401: print("Authentication Error: Check API Key.")
            elif e.response.status_code == 429: print("Rate Limit Error.")
        return None
    except (KeyError, IndexError) as e:
        print(f"Error parsing CoinGecko response for {coin_id}: Missing expected data - {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while fetching/processing latest data for {coin_id}: {e}")
        return None


# --- scale_data function remains the same as before ---
def scale_data(df, features_to_scale, scaler=None):
    """
    Scales the specified features of the DataFrame using MinMaxScaler.
    (Code is identical to the previous version)
    """
    if df is None or df.empty:
        print("Cannot scale data: Input DataFrame is None or empty.")
        return None, None

    print(f"Scaling features: {features_to_scale}")
    df_scaled = df.copy()

    if scaler is None:
        print("No scaler provided, fitting a new MinMaxScaler.")
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])
        print("Scaler fitted.")
    else:
        print("Using provided scaler to transform data.")
        try:
             df_scaled[features_to_scale] = scaler.transform(df[features_to_scale])
        except ValueError as e:
             print(f"Error applying scaler: {e}. Features mismatch? Have {df[features_to_scale].shape[1]}, Scaler expects: {scaler.n_features_in_}")
             return None, scaler
        except Exception as e:
             print(f"An unexpected error occurred during scaling: {e}")
             return None, scaler

    print("Scaling complete.")
    return df_scaled, scaler


# --- Example Usage (for testing this script directly) ---
if __name__ == '__main__':
    # Create the logs directory if it doesn't exist when testing
    os.makedirs(LOG_DIR, exist_ok=True)

    # Test fetching and saving for Bitcoin
    print("\n--- Testing Fetch & Save for Bitcoin ---")
    latest_btc_data = get_latest_coingecko_data('bitcoin') # This now also saves
    if latest_btc_data:
        print("\nLatest BTC Data Point Fetched (and saved to logs/bitcoin_fetch_log.csv):")
        print(latest_btc_data)
    else:
        print("\nFailed to fetch latest BTC data.")

    # Test fetching and saving for Ethereum
    print("\n--- Testing Fetch & Save for Ethereum ---")
    latest_eth_data = get_latest_coingecko_data('ethereum') # This now also saves
    if latest_eth_data:
        print("\nLatest ETH Data Point Fetched (and saved to logs/ethereum_fetch_log.csv):")
        print(latest_eth_data)
    else:
        print("\nFailed to fetch latest ETH data.")


    # You can also still test loading historical data etc. here if needed
    # btc_hist_df = load_historical_data('bitcoin')
    # if btc_hist_df is not None:
    #     # ... rest of the original test block ...
    #     pass