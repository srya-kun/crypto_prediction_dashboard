# crypto_prediction_dashboard/train.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split # Used for splitting, but we won't shuffle
import joblib # For saving the scaler object
import os

# Import functions from our data handler script
from data_handler import load_historical_data, add_technical_features

# --- Configuration ---
# List of coin names (must match filenames in data/ e.g., bitcoin_data.csv)
COINS_TO_TRAIN = ['bitcoin', 'ethereum', 'dogecoin']
# How many past days of data to use for predicting the next step
SEQUENCE_LENGTH = 60
# Which features from the dataframe to use as input for the model
# Ensure these columns exist after feature engineering in data_handler.py
FEATURES_TO_USE = ['open', 'high', 'low', 'close', 'volume', 'marketcap', 'sma_10', 'sma_50']
# Which columns we want to predict
# 'close' -> Current Day Estimated Closing Price
# 'next_open' -> Next Day Opening Price (Engineered below)
TARGETS_TO_PREDICT = ['close', 'next_open']
# Directory to save trained models and scalers
MODEL_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'models')
# Create directory if it doesn't exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
# --- End Configuration ---

def create_sequences(features, targets, sequence_length):
    """
    Creates sequences of data for LSTM training.

    Args:
        features (np.array): Array of scaled features.
        targets (np.array): Array of scaled targets.
        sequence_length (int): The number of time steps in each input sequence.

    Returns:
        tuple: (np.array for X (sequences), np.array for y (targets corresponding to sequences))
    """
    X, y = [], []
    for i in range(len(features) - sequence_length):
        # Input sequence: features from index i to i + sequence_length - 1
        X.append(features[i:(i + sequence_length)])
        # Target: targets at index i + sequence_length
        y.append(targets[i + sequence_length])
    return np.array(X), np.array(y)

# --- Main Training Loop ---
print("Starting model training process...")

for coin in COINS_TO_TRAIN:
    print(f"\n--- Training for {coin.capitalize()} ---")

    # 1. Load Data
    print("Loading historical data...")
    df_hist = load_historical_data(coin)
    if df_hist is None:
        print(f"Skipping {coin}: Could not load data.")
        continue

    # 2. Add Features
    print("Adding technical features...")
    df_featured = add_technical_features(df_hist.copy()) # Use copy
    if df_featured is None or df_featured.empty:
        print(f"Skipping {coin}: Failed to add features or data became empty.")
        continue

    # 3. Engineer Target Variable ('next_open')
    print("Engineering target variable 'next_open'...")
    # Shift the 'open' price column up by one row to get the next day's open price as a target for the current row
    df_featured['next_open'] = df_featured['open'].shift(-1)
    # Drop the last row, as it will have NaN for 'next_open'
    df_featured.dropna(subset=['next_open'], inplace=True)
    print(f"DataFrame shape after adding 'next_open' and dropping last row: {df_featured.shape}")

    # 4. Select Final Features and Targets
    # Ensure all selected features and targets exist in the DataFrame
    missing_features = [f for f in FEATURES_TO_USE if f not in df_featured.columns]
    missing_targets = [t for t in TARGETS_TO_PREDICT if t not in df_featured.columns]

    if missing_features or missing_targets:
        print(f"Error: Missing required columns for {coin}:")
        if missing_features: print(f"  Missing Features: {missing_features}")
        if missing_targets: print(f"  Missing Targets: {missing_targets}")
        print(f"Skipping {coin}.")
        continue

    features_df = df_featured[FEATURES_TO_USE]
    targets_df = df_featured[TARGETS_TO_PREDICT]

    # 5. Split Data (Chronological)
    # Determine split point (e.g., 80% train, 20% validation)
    split_index = int(len(features_df) * 0.8)
    train_features = features_df[:split_index]
    train_targets = targets_df[:split_index]
    val_features = features_df[split_index:]
    val_targets = targets_df[split_index:]

    print(f"Training set size: {len(train_features)} samples")
    print(f"Validation set size: {len(val_features)} samples")

    if len(val_features) < SEQUENCE_LENGTH:
         print(f"Warning: Validation set size ({len(val_features)}) is smaller than SEQUENCE_LENGTH ({SEQUENCE_LENGTH}).")
         print("Consider using more data or a shorter sequence length.")
         if len(val_features) == 0:
              print(f"Skipping {coin}: Validation set is empty.")
              continue


    # 6. Scale Data
    print("Scaling data...")
    # Scale features (fit on training data only)
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit scaler on training features and transform both train and validation features
    scaled_train_features = feature_scaler.fit_transform(train_features)
    scaled_val_features = feature_scaler.transform(val_features)

    # Scale targets (fit on training data only) - using a separate scaler is often best
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit scaler on training targets and transform both train and validation targets
    scaled_train_targets = target_scaler.fit_transform(train_targets)
    scaled_val_targets = target_scaler.transform(val_targets)

    # Save the scalers for later use during prediction
    feature_scaler_path = os.path.join(MODEL_SAVE_DIR, f"{coin}_feature_scaler.pkl")
    target_scaler_path = os.path.join(MODEL_SAVE_DIR, f"{coin}_target_scaler.pkl")
    try:
        joblib.dump(feature_scaler, feature_scaler_path)
        joblib.dump(target_scaler, target_scaler_path)
        print(f"Feature scaler saved to {feature_scaler_path}")
        print(f"Target scaler saved to {target_scaler_path}")
    except Exception as e:
        print(f"Error saving scalers for {coin}: {e}")
        print(f"Skipping model training for {coin}.")
        continue


    # 7. Create Sequences
    print("Creating sequences for LSTM...")
    X_train, y_train = create_sequences(scaled_train_features, scaled_train_targets, SEQUENCE_LENGTH)
    X_val, y_val = create_sequences(scaled_val_features, scaled_val_targets, SEQUENCE_LENGTH)

    # Check if sequence creation was successful and yielded data
    if X_train.size == 0 or y_train.size == 0 or X_val.size == 0 or y_val.size == 0:
        print(f"Error: Sequence creation resulted in empty arrays for {coin}. ")
        print(f"Check SEQUENCE_LENGTH ({SEQUENCE_LENGTH}) relative to dataset sizes after splitting.")
        print(f"Training features shape: {scaled_train_features.shape}, Validation features shape: {scaled_val_features.shape}")
        print(f"Skipping model training for {coin}.")
        continue

    print(f"Training sequences shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation sequences shape: X={X_val.shape}, y={y_val.shape}")


    # 8. Build LSTM Model
    print("Building LSTM model...")
    model = Sequential()
    # Input LSTM layer
    # input_shape=(SEQUENCE_LENGTH, num_features)
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2)) # Dropout for regularization
    # Second LSTM layer
    model.add(LSTM(units=50, return_sequences=False)) # return_sequences=False as it's the last LSTM layer
    model.add(Dropout(0.2))
    # Dense layer before the final output
    model.add(Dense(units=25))
    # Output layer: Dense layer with 'n' neurons, where n = number of target variables (2 in our case)
    model.add(Dense(units=y_train.shape[1])) # Output shape should match the number of targets

    # Compile the model
    # Using Mean Squared Error loss as it's a regression problem
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()


    # 9. Train Model
    print("Training model...")
    # Define callbacks
    # Stop training early if validation loss doesn't improve for 'patience' epochs
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    # Save the best model during training based on validation loss
    model_checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"{coin}_best_model.h5")
    model_checkpoint = ModelCheckpoint(model_checkpoint_path, monitor='val_loss', save_best_only=True)

    # Fit the model
    history = model.fit(
        X_train, y_train,
        epochs=200, # Adjust number of epochs as needed
        batch_size=32, # Adjust batch size as needed
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1 # Set to 0 for less output, 1 for progress bar
    )

    print(f"Model training completed for {coin}.")

    # 10. Save Final Model (optional, as ModelCheckpoint saves the best one)
    # final_model_path = os.path.join(MODEL_SAVE_DIR, f"{coin}_final_model.h5")
    # model.save(final_model_path)
    # print(f"Final model saved to {final_model_path}")
    print(f"Best model during training saved to {model_checkpoint_path}")


print("\n--- Model training process finished for all specified coins. ---")
print(f"Trained models and scalers saved in: {MODEL_SAVE_DIR}")