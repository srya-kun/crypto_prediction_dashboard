# crypto_prediction_dashboard/app.py

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from flask import Flask, request, jsonify, render_template

# Import functions from our data handler script
from data_handler import (
    load_historical_data,
    add_technical_features,
    get_latest_coingecko_data
)

# --- Configuration ---
# List of supported coin names (lowercase, matching filenames and model names)
SUPPORTED_COINS = ['bitcoin', 'ethereum', 'dogecoin']
# Must match the sequence length used during training in train.py
SEQUENCE_LENGTH = 60
# Must match the features used during training in train.py
FEATURES_TO_USE = ['open', 'high', 'low', 'close', 'volume', 'marketcap', 'sma_10', 'sma_50']
# Must match the targets predicted during training in train.py
TARGETS_PREDICTED = ['close', 'next_open'] # Used for interpreting the output
# Directory where models and scalers are saved
MODEL_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'models')
# Determine buffer needed for feature calculation (e.g., max window size)
# If using sma_50, we need at least 50 points before the sequence starts.
FEATURE_BUFFER = 50 # Adjust if using features with longer lookbacks
# --- End Configuration ---

# Initialize Flask app
app = Flask(__name__)

# --- Load Models and Scalers on Startup ---
models = {}
feature_scalers = {}
target_scalers = {}

def load_resources():
    """Loads trained models and scalers for supported coins into memory."""
    print("Loading models and scalers...")
    for coin in SUPPORTED_COINS:
        print(f"  Loading resources for {coin.capitalize()}...")
        model_path = os.path.join(MODEL_SAVE_DIR, f"{coin}_best_model.h5")
        f_scaler_path = os.path.join(MODEL_SAVE_DIR, f"{coin}_feature_scaler.pkl")
        t_scaler_path = os.path.join(MODEL_SAVE_DIR, f"{coin}_target_scaler.pkl")

        try:
            if not os.path.exists(model_path):
                print(f"    Error: Model file not found at {model_path}")
                continue
            if not os.path.exists(f_scaler_path):
                print(f"    Error: Feature scaler not found at {f_scaler_path}")
                continue
            if not os.path.exists(t_scaler_path):
                print(f"    Error: Target scaler not found at {t_scaler_path}")
                continue

            # Load model (compile=False can speed up loading if optimizer state isn't needed for inference)
            models[coin] = tf.keras.models.load_model(model_path, compile=False)
            # Load scalers
            feature_scalers[coin] = joblib.load(f_scaler_path)
            target_scalers[coin] = joblib.load(t_scaler_path)
            print(f"    Successfully loaded model and scalers for {coin.capitalize()}.")

        except Exception as e:
            print(f"    Error loading resources for {coin.capitalize()}: {e}")
            # Ensure partial loads don't cause issues later
            if coin in models: del models[coin]
            if coin in feature_scalers: del feature_scalers[coin]
            if coin in target_scalers: del target_scalers[coin]

    print("Finished loading resources.")
    print(f"Loaded resources for: {list(models.keys())}")

# Call loading function when the script starts
load_resources()
# --- End Loading ---


# --- Flask Routes ---

@app.route('/')
def home():
    """Renders the main dashboard page."""
    # Pass the list of successfully loaded coins to the template
    available_coins = list(models.keys())
    return render_template('index.html', coins=available_coins)

@app.route('/historical_data/<coin_name>')
def get_historical_data_route(coin_name):
    """Provides historical data for charting."""
    if coin_name not in SUPPORTED_COINS:
        return jsonify({"error": f"Unsupported coin: {coin_name}"}), 400

    print(f"Request received for historical data: {coin_name}")
    df_hist = load_historical_data(coin_name)

    if df_hist is None or df_hist.empty:
        return jsonify({"error": f"Could not load historical data for {coin_name}"}), 404

    try:
        # Select Date (index) and Close price for the chart
        chart_df = df_hist[['close']].copy()
        # Reset index to make date a column for easier JSON conversion
        chart_df.reset_index(inplace=True)
        chart_df.rename(columns={'index': 'date', 'close': 'price'}, inplace=True)
        # Convert date to string format (e.g., ISO) for JavaScript compatibility
        chart_df['date'] = chart_df['date'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

        # Convert DataFrame to list of dictionaries
        chart_data = chart_df.to_dict(orient='records')
        return jsonify(chart_data)

    except Exception as e:
        app.logger.error(f"Error processing historical data for {coin_name}: {e}", exc_info=True)
        return jsonify({"error": f"Error processing historical data for {coin_name}"}), 500


@app.route('/predict/<coin_name>')
def predict_route(coin_name):
    """Handles prediction requests for a given coin."""
    coin_name = coin_name.lower()
    print(f"Request received for prediction: {coin_name}")

    # 1. Check if coin is supported and resources are loaded
    if coin_name not in SUPPORTED_COINS:
        return jsonify({"error": f"Unsupported coin: {coin_name}"}), 400
    if coin_name not in models or coin_name not in feature_scalers or coin_name not in target_scalers:
         return jsonify({"error": f"Model/scalers not loaded for {coin_name}. Check server logs."}), 500

    try:
        # 2. Get Data for Prediction Input
        # --- Load Recent Historical Data ---
        # Load enough history to calculate features + the sequence length
        # Example: Need SEQUENCE_LENGTH days + FEATURE_BUFFER days for calculations
        required_hist_len = SEQUENCE_LENGTH + FEATURE_BUFFER
        df_hist_raw = load_historical_data(coin_name)
        if df_hist_raw is None or len(df_hist_raw) < required_hist_len:
             return jsonify({"error": f"Not enough historical data to make prediction for {coin_name}"}), 500
        df_recent_hist = df_hist_raw.tail(required_hist_len).copy() # Get the most recent segment

        # --- Fetch Latest Data Point ---
        latest_data = get_latest_coingecko_data(coin_name) # This is a dict
        if latest_data is None:
            return jsonify({"error": f"Could not fetch latest data for {coin_name} from API."}), 500

        # Convert latest data dict to a DataFrame row with correct timestamp index
        latest_timestamp = latest_data['timestamp']
        latest_df_row = pd.DataFrame([{
             'open': latest_data.get('open'), 'high': latest_data.get('high'),
             'low': latest_data.get('low'), 'close': latest_data.get('close'),
             'volume': latest_data.get('volume'), 'marketcap': latest_data.get('marketcap')
        }], index=[latest_timestamp])

        # --- Combine Historical and Latest ---
        # Ensure columns match before concatenating
        latest_df_row = latest_df_row[df_recent_hist.columns] # Match column order/names
        # Use pd.concat instead of append
        df_combined = pd.concat([df_recent_hist, latest_df_row])
        # Remove potential duplicate index (if API fetch happens exactly at timestamp boundary)
        df_combined = df_combined[~df_combined.index.duplicated(keep='last')]


        # 3. Add Technical Features to Combined Data
        # Features will be calculated on the combined recent history + latest point
        df_combined_featured = add_technical_features(df_combined.copy())
        if df_combined_featured is None or df_combined_featured.empty:
             return jsonify({"error": f"Failed to calculate features on combined data for {coin_name}"}), 500

        # 4. Prepare Final Sequence for Model
        # Select the last SEQUENCE_LENGTH rows after features are calculated
        sequence_df = df_combined_featured.tail(SEQUENCE_LENGTH)

        # Verify shape
        if len(sequence_df) < SEQUENCE_LENGTH:
             print(f"Warning: Final sequence length is {len(sequence_df)}, expected {SEQUENCE_LENGTH}")
             return jsonify({"error": f"Could not form complete input sequence ({len(sequence_df)}/{SEQUENCE_LENGTH}) for {coin_name}"}), 500

        # Select only the features the model was trained on
        sequence_features = sequence_df[FEATURES_TO_USE].astype(np.float32) # Ensure float type

        # 5. Scale the Sequence using Loaded Scaler
        f_scaler = feature_scalers[coin_name]
        scaled_features = f_scaler.transform(sequence_features)

        # 6. Reshape for LSTM Input (samples, time_steps, features)
        input_sequence = np.reshape(scaled_features, (1, SEQUENCE_LENGTH, len(FEATURES_TO_USE)))

        # 7. Predict using Loaded Model
        model = models[coin_name]
        scaled_prediction = model.predict(input_sequence) # Shape: (1, num_targets)

        # 8. Inverse Scale the Prediction using Loaded Target Scaler
        t_scaler = target_scalers[coin_name]
        final_prediction = t_scaler.inverse_transform(scaled_prediction) # Shape: (1, num_targets)

        # 9. Format and Return Prediction
        # Assuming the model outputs ['close', 'next_open'] in that order
        predicted_eod_close = float(final_prediction[0, 0]) # First target
        predicted_next_open = float(final_prediction[0, 1]) # Second target

        result = {
            "coin": coin_name,
            "predicted_eod_close": round(predicted_eod_close, 2), # Round to 2 decimal places
            "predicted_next_open": round(predicted_next_open, 2)
        }
        print(f"Prediction successful for {coin_name}: {result}")
        return jsonify(result)

    except Exception as e:
        # Log the full error for debugging on the server side
        app.logger.error(f"Error during prediction for {coin_name}: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred during prediction for {coin_name}. Check server logs."}), 500

# --- Run Flask App ---
if __name__ == '__main__':
    # Set debug=False for production deployment
    app.run(debug=True, host='0.0.0.0', port=5000) # Runs on localhost:5000 by default