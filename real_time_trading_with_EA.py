import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import sqlite3
import MetaTrader5 as mt5
import ta
import time
import talib
import threading
import os
import signal
from datetime import datetime, timedelta

# Ruta para guardar los archivos de entrenamiento
MODEL_SAVE_PATH = "C:\\Users\\Contabilidad\\Desktop\\Tradeando\\PRUEBAS DE FONDEO\\model.keras"

# Conectar a MetaTrader 5
def initialize_mt5():
    if not mt5.initialize():
        print("initialize() failed")
        mt5.shutdown()
        return False

    login = 5026733189
    password = '0wMwD-Vr'
    server = 'MetaQuotes-Demo'

    authorized = mt5.login(login, password=password, server=server)
    if not authorized:
        print(f"Failed to connect at account #{login}, error code: {mt5.last_error()}")
        return False
    else:
        print(f"Connected to account #{login}")
        return True

def fetch_historical_data(symbol, timeframe, duration):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, duration)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def calculate_indicators(df):
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['candlestick_pattern'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
    df.dropna(inplace=True)
    return df

def save_data_to_db(symbol, timeframe, df):
    conn = sqlite3.connect('market_data.db')
    df.to_sql(f"{symbol}_{timeframe}", conn, if_exists='replace', index=True)
    conn.close()

def load_and_prepare_data(symbol, timeframe):
    conn = sqlite3.connect('market_data.db')
    query = f"SELECT * FROM {symbol}_{timeframe}"
    df = pd.read_sql(query, conn)
    conn.close()
    
    df = df.sort_values('time')
    df['time'] = pd.to_datetime(df['time'])
    
    features = ['close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_signal', 'candlestick_pattern']
    target = 'close'
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    sequence_length = 60
    X = []
    y = []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, features.index(target)])
    
    X, y = np.array(X), np.array(y)
    
    split = int(0.8 * len(X))
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    
    return X_train, X_test, y_train, y_test, scaler, features

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def load_or_create_model(input_shape):
    if os.path.exists(MODEL_SAVE_PATH):
        print("Loading existing model...")
        model = load_model(MODEL_SAVE_PATH)
    else:
        print("Creating new model...")
        model = build_model(input_shape)
    return model

def save_model_periodically(model):
    while True:
        time.sleep(600)  # Guardar el modelo cada 10 minutos
        model.save(MODEL_SAVE_PATH)
        print("Model saved periodically.")

def handle_exit(signal, frame, model):
    print("Saving model before exit...")
    model.save(MODEL_SAVE_PATH)
    print("Model saved. Exiting...")
    exit(0)

def calculate_confidence_level(prediction_direction, indicators, pattern_confidence):
    confidence = 0.5  # Base confidence
    if prediction_direction > 0:
        if indicators['RSI'] < 30:
            confidence += 0.1
        if indicators['MACD'] > indicators['MACD_signal']:
            confidence += 0.1
        if indicators['candlestick_pattern'] != 0:
            confidence += 0.1
    else:
        if indicators['RSI'] > 70:
            confidence += 0.1
        if indicators['MACD'] < indicators['MACD_signal']:
            confidence += 0.1
        if indicators['candlestick_pattern'] != 0:
            confidence += 0.1
    confidence += pattern_confidence
    return min(confidence, 1)

def determine_lot_size(confidence, max_lot):
    return round(0.1 + (max_lot - 0.1) * confidence, 2)

def get_trade_profit(position_ticket, open_price, close_price, lot_size, position_type):
    if position_type == mt5.ORDER_TYPE_BUY:
        return (close_price - open_price) * lot_size
    else:
        return (open_price - close_price) * lot_size

def calculate_distrust_level(indicators):
    distrust = 0  # Base distrust level
    if indicators['RSI'] < 30:
        distrust -= 0.2
    elif indicators['RSI'] > 70:
        distrust += 0.2
    
    if indicators['MACD'] > indicators['MACD_signal']:
        distrust -= 0.2
    elif indicators['MACD'] < indicators['MACD_signal']:
        distrust += 0.2
    
    if indicators['candlestick_pattern'] != 0:
        distrust -= 0.1
    else:
        distrust += 0.1
    
    return distrust

def monitor_and_close_position(symbol, position_ticket, open_time, indicators, model, lot_size, confidence, timeframe, max_retries=5):
    max_profit = 0
    retries = 0
    while retries < max_retries:
        time.sleep(30)
        open_positions = mt5.positions_get(ticket=position_ticket)
        if not open_positions:
            break
        position = open_positions[0]
        current_price = mt5.symbol_info_tick(symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
        current_profit = get_trade_profit(position.ticket, position.price_open, current_price, lot_size, position.type)
        max_profit = max(max_profit, current_profit)

        current_time = time.time()
        time_elapsed = current_time - open_time

        if time_elapsed < 60:
            print("Waiting for at least 1 minute before evaluating close conditions.")
            continue

        distrust_level = calculate_distrust_level(indicators)
        print(f"Distrust Level: {distrust_level}")

        close_reason = ""

        if current_profit > 0:
            if current_profit < max_profit * 0.7:
                close_reason = "profit dropped"
            elif indicators['RSI'] > 70:
                close_reason = "RSI > 70"
            elif indicators['MACD'] < indicators['MACD_signal']:
                close_reason = "MACD < MACD_signal"
            elif indicators['candlestick_pattern'] != 0:
                close_reason = "candlestick pattern"
        else:
            if current_profit < 0:
                close_reason = "negative profit"
            elif distrust_level > 0.2:
                close_reason = "high distrust level"

        if close_reason:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": position.volume,
                "position": position.ticket,
                "type": mt5.ORDER_TYPE_BUY if position.type == mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_SELL,
                "price": current_price,
                "deviation": 100,
                "magic": 234000,
                "comment": "AI trade - close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN
            }
            print(f"Sending close request at market price {current_price}")
            result = mt5.order_send(request)
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                trade_profit = get_trade_profit(position.ticket, position.price_open, current_price, lot_size, position.type)
                print(f"Trade closed for {symbol}, profit: {trade_profit}")
                close_time = time.time()
                log_trade(symbol, position.ticket, position.price_open, current_price, trade_profit, open_time, close_time, close_reason, lot_size, confidence, timeframe)
                monitor_after_close(symbol, position.price_open, current_price, open_time, close_time, indicators)
                return trade_profit
            else:
                print(f"Close trade failed for {symbol}, retcode={result.retcode}, comment={result.comment}, retrying...")
                retries += 1

    return None

def log_trade(symbol, ticket, open_price, close_price, profit, open_time, close_time, close_reason, lot_size, confidence, timeframe):
    conn = sqlite3.connect('market_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            lot_size REAL,
            ticket INTEGER,
            open_price REAL,
            close_price REAL,
            profit REAL,
            open_time TEXT,
            close_time TEXT,
            duration TEXT,
            confidence REAL,
            timeframe TEXT,
            close_reason TEXT,
            comment TEXT
        )
    ''')
    open_time_utc = datetime.utcfromtimestamp(open_time) + timedelta(hours=2)
    close_time_utc = datetime.utcfromtimestamp(close_time) + timedelta(hours=2)
    duration = str(timedelta(seconds=(close_time - open_time)))
    
    cursor.execute('''
        INSERT INTO trades (symbol, lot_size, ticket, open_price, close_price, profit, open_time, close_time, duration, confidence, timeframe, close_reason, comment)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (symbol, lot_size, ticket, open_price, close_price, profit, 
          open_time_utc.strftime('%Y-%m-%d %H:%M:%S'), 
          close_time_utc.strftime('%Y-%m-%d %H:%M:%S'), 
          duration, confidence, timeframe, close_reason, ""))
    conn.commit()
    conn.close()

def monitor_after_close(symbol, open_price, close_price, open_time, close_time, indicators):
    print("Monitoring after close for additional 10 minutes...")
    time.sleep(600)  # Esperar 10 minutos

    # Evaluar la situación actual del mercado
    new_indicators = calculate_indicators(fetch_historical_data(symbol, mt5.TIMEFRAME_M1, 1000))
    new_price = new_indicators['close'].iloc[-1]
    new_profit = (new_price - open_price) if close_price > open_price else (open_price - new_price)
    
    # Decidir si fue un buen cierre o no
    if new_profit > (close_price - open_price):
        comment = "Good close: avoided potential loss or missed out on more profit"
    else:
        comment = "Bad close: could have lost less or gained more"

    # Actualizar la tabla trades con el comentario
    conn = sqlite3.connect('market_data.db')
    cursor = conn.cursor()
    close_time_utc = datetime.utcfromtimestamp(close_time) + timedelta(hours=2)
    cursor.execute('''
        UPDATE trades
        SET comment = ?
        WHERE close_time = ?
    ''', (comment, close_time_utc.strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()

def detect_double_bottom(df):
    prices = df['close'].values
    pattern_indices = []
    for i in range(1, len(prices) - 1):
        if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
            pattern_indices.append(i)
    return len(pattern_indices) >= 2

def detect_double_top(df):
    prices = df['close'].values
    pattern_indices = []
    for i in range(1, len(prices) - 1):
        if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
            pattern_indices.append(i)
    return len(pattern_indices) >= 2

def detect_hch(df):
    prices = df['close'].values
    for i in range(1, len(prices) - 2):
        if prices[i - 1] < prices[i] > prices[i + 1] and prices[i + 1] < prices[i + 2]:
            return True
    return False

def detect_hchi(df):
    prices = df['close'].values
    for i in range(1, len(prices) - 2):
        if prices[i - 1] > prices[i] < prices[i + 1] and prices[i + 1] > prices[i + 2]:
            return True
    return False

def detect_wedge(df):
    prices = df['close'].values
    upward = all(prices[i] < prices[i + 1] for i in range(len(prices) - 1))
    downward = all(prices[i] > prices[i + 1] for i in range(len(prices) - 1))
    return upward or downward

def detect_triangle(df):
    highs = df['high'].values
    lows = df['low'].values
    for i in range(len(highs) - 1):
        if highs[i] > highs[i + 1] and lows[i] < lows[i + 1]:
            return True
    return False

def detect_darvas_box(df):
    prices = df['close'].values
    for i in range(1, len(prices) - 1):
        if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
            return True
    return False

def monitor_trade(symbol, ticket, open_price, tp, sl, timeframe, confidence, indicators, model):
    max_profit = 0
    open_time = time.time()
    while True:
        open_positions = mt5.positions_get(ticket=ticket)
        if not open_positions:
            break
        position = open_positions[0]
        current_price = mt5.symbol_info_tick(symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
        profit = get_trade_profit(position.ticket, open_price, current_price, position.volume, position.type)
        max_profit = max(max_profit, profit)
        print(f"Ticket: {ticket}, Symbol: {symbol}, Open Price: {open_price}, Current Price: {current_price}, TP: {tp}, SL: {sl}, Profit: {profit}, Timeframe: {timeframe}, Confidence: {confidence}, Indicators: {indicators}")

        current_time = time.time()
        time_elapsed = current_time - open_time

        if time_elapsed < 60:
            print("Waiting for at least 1 minute before evaluating close conditions.")
            time.sleep(60 - time_elapsed)
            continue

        distrust_level = calculate_distrust_level(indicators)
        print(f"Distrust Level: {distrust_level}")

        close_reason = ""

        if profit > 0:
            if profit < max_profit * 0.7:
                close_reason = "profit dropped"
            elif indicators['RSI'] > 70:
                close_reason = "RSI > 70"
            elif indicators['MACD'] < indicators['MACD_signal']:
                close_reason = "MACD < MACD_signal"
            elif indicators['candlestick_pattern'] != 0:
                close_reason = "candlestick pattern"
        else:
            if profit < 0:
                close_reason = "negative profit"
            elif distrust_level > 0.2:
                close_reason = "high distrust level"

        if close_reason:
            close_price = current_price
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": position.volume,
                "position": position.ticket,
                "type": mt5.ORDER_TYPE_BUY if position.type == mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_SELL,
                "price": close_price,
                "deviation": 100,
                "magic": 234000,
                "comment": "AI trade - close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN
            }
            print(f"Sending close request at market price {close_price}")
            result = mt5.order_send(request)
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                trade_profit = get_trade_profit(position.ticket, position.price_open, close_price, position.volume, position.type)
                print(f"Trade closed for {symbol}, profit: {trade_profit}")
                close_time = time.time()
                log_trade(symbol, position.ticket, position.price_open, close_price, trade_profit, open_time, close_time, close_reason, position.volume, confidence, timeframe)
                monitor_after_close(symbol, position.price_open, close_price, open_time, close_time, indicators)
                return trade_profit
            else:
                print(f"Close trade failed for {symbol}, retcode={result.retcode}, comment={result.comment}, retrying...")
                time.sleep(30)

        time.sleep(60)  # Reevaluar cada minuto

    return None

def main():
    if not initialize_mt5():
        return

    symbols = ['EURUSD', 'GBPUSD', 'XAUUSD']
    timeframes_to_open = {
        '15_mins': mt5.TIMEFRAME_M15,
        '30_mins': mt5.TIMEFRAME_M30,
        '1_hour': mt5.TIMEFRAME_H1
    }
    all_timeframes = {
        '1_min': mt5.TIMEFRAME_M1,
        '15_mins': mt5.TIMEFRAME_M15,
        '30_mins': mt5.TIMEFRAME_M30,
        '1_hour': mt5.TIMEFRAME_H1,
        '4_hours': mt5.TIMEFRAME_H4,
        '1_day': mt5.TIMEFRAME_D1
    }

    account_info = mt5.account_info()
    balance = account_info.balance
    daily_target = balance * 0.02
    total_profit = 0

    # Preparar el modelo y el manejo de señales para guardar el modelo antes de salir
    input_shape = (60, 7)  # Esta es la forma de entrada para el modelo LSTM
    model = load_or_create_model(input_shape)
    signal.signal(signal.SIGINT, lambda s, f: handle_exit(s, f, model))

    # Iniciar un hilo para guardar el modelo periódicamente
    threading.Thread(target=save_model_periodically, args=(model,)).start()

    while True:
        current_hour = datetime.now().hour
        if 22 <= current_hour or current_hour < 6:
            print("No new trades will be opened between 22:00 and 06:00 (Spain time).")
        else:
            for symbol in symbols:
                positions = mt5.positions_get(symbol=symbol)
                if positions:
                    print(f"Already have an open position for {symbol}. Skipping.")
                    for position in positions:
                        df = fetch_historical_data(symbol, mt5.TIMEFRAME_M1, 1000)
                        df = calculate_indicators(df)
                        prediction_direction = 0  # Este valor debe ser definido de manera adecuada
                        pattern_confidence = 0  # Este valor debe ser definido de manera adecuada
                        indicators = {
                            'RSI': df['RSI'].iloc[-1],
                            'MACD': df['MACD'].iloc[-1],
                            'MACD_signal': df['MACD_signal'].iloc[-1],
                            'candlestick_pattern': df['candlestick_pattern'].iloc[-1]
                        }
                        confidence = calculate_confidence_level(prediction_direction, indicators, pattern_confidence)
                        timeframe = 'current_timeframe'  # Ajusta esto a la variable de temporalidad correcta
                        profit = monitor_and_close_position(symbol, position.ticket, position.time, indicators, model, position.volume, confidence, timeframe)
                        if profit is not None:
                            total_profit += profit
                            if total_profit >= daily_target:
                                print(f"Daily target of 2% reached with a profit of {total_profit}")
                                handle_exit(signal.SIGINT, None, model)
                    continue

                for tf_name, tf_value in timeframes_to_open.items():
                    df = fetch_historical_data(symbol, tf_value, 1000)
                    df = calculate_indicators(df)
                    save_data_to_db(symbol, tf_name, df)
                    print(f"Data for {symbol} with {tf_name} updated and saved.")
                    
                    X_train, X_test, y_train, y_test, scaler, features = load_and_prepare_data(symbol, tf_name)
                    model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)  # Entrenar el modelo incrementalmente
                    predictions = model.predict(X_test)
                    
                    predicted_scaled = np.zeros((1, len(features)))
                    predicted_scaled[0][features.index('close')] = predictions[-1].item()
                    predicted_price = scaler.inverse_transform(predicted_scaled)[0][features.index('close')]
                    
                    current_price = df['close'].iloc[-1]
                    prediction_direction = predicted_price - current_price
                    
                    indicators = {
                        'RSI': df['RSI'].iloc[-1],
                        'MACD': df['MACD'].iloc[-1],
                        'MACD_signal': df['MACD_signal'].iloc[-1],
                        'candlestick_pattern': df['candlestick_pattern'].iloc[-1]
                    }
                    
                    double_bottoms = detect_double_bottom(df)
                    double_tops = detect_double_top(df)
                    hch_patterns = detect_hch(df)
                    hchi_patterns = detect_hchi(df)
                    wedge_patterns = detect_wedge(df)
                    triangle_patterns = detect_triangle(df)
                    darvas_box_patterns = detect_darvas_box(df)

                    pattern_confidence = 0
                    if double_bottoms: pattern_confidence += 0.1
                    if double_tops: pattern_confidence -= 0.1
                    if hch_patterns: pattern_confidence -= 0.1
                    if hchi_patterns: pattern_confidence += 0.1
                    if wedge_patterns: pattern_confidence += 0.1
                    if triangle_patterns: pattern_confidence += 0.1
                    if darvas_box_patterns: pattern_confidence += 0.1

                    confidence = calculate_confidence_level(prediction_direction, indicators, pattern_confidence)
                    lot_size = determine_lot_size(confidence, 5)
                    print(f"Confidence: {confidence}, Lot Size: {lot_size}")
                    
                    if confidence >= 0.79:  # Nivel de confianza aumentado al 79%
                        confirm = True
                        for confirm_tf_name, confirm_tf_value in all_timeframes.items():
                            confirm_df = fetch_historical_data(symbol, confirm_tf_value, 1000)
                            confirm_df = calculate_indicators(confirm_df)
                            save_data_to_db(symbol, confirm_tf_name, confirm_df)
                            confirm_X_train, confirm_X_test, confirm_y_train, confirm_y_test, confirm_scaler, confirm_features = load_and_prepare_data(symbol, confirm_tf_name)
                            model.fit(confirm_X_train, confirm_y_train, epochs=1, batch_size=32, verbose=0)  # Entrenar el modelo incrementalmente
                            confirm_predictions = model.predict(confirm_X_test)
                            
                            confirm_predicted_scaled = np.zeros((1, len(confirm_features)))
                            confirm_predicted_scaled[0][confirm_features.index('close')] = confirm_predictions[-1].item()
                            confirm_predicted_price = confirm_scaler.inverse_transform(confirm_predicted_scaled)[0][confirm_features.index('close')]
                            
                            confirm_current_price = confirm_df['close'].iloc[-1]
                            confirm_prediction_direction = confirm_predicted_price - confirm_current_price

                            confirm_indicators = {
                                'RSI': confirm_df['RSI'].iloc[-1],
                                'MACD': confirm_df['MACD'].iloc[-1],
                                'MACD_signal': confirm_df['MACD_signal'].iloc[-1],
                                'candlestick_pattern': confirm_df['candlestick_pattern'].iloc[-1]
                            }

                            confirm_double_bottoms = detect_double_bottom(confirm_df)
                            confirm_double_tops = detect_double_top(confirm_df)
                            confirm_hch_patterns = detect_hch(confirm_df)
                            confirm_hchi_patterns = detect_hchi(confirm_df)
                            confirm_wedge_patterns = detect_wedge(confirm_df)
                            confirm_triangle_patterns = detect_triangle(confirm_df)
                            confirm_darvas_box_patterns = detect_darvas_box(confirm_df)

                            confirm_pattern_confidence = 0
                            if confirm_double_bottoms: confirm_pattern_confidence += 0.1
                            if confirm_double_tops: confirm_pattern_confidence -= 0.1
                            if confirm_hch_patterns: confirm_pattern_confidence -= 0.1
                            if confirm_hchi_patterns: confirm_pattern_confidence += 0.1
                            if confirm_wedge_patterns: confirm_pattern_confidence += 0.1
                            if confirm_triangle_patterns: confirm_pattern_confidence += 0.1
                            if confirm_darvas_box_patterns: confirm_pattern_confidence += 0.1

                            confirm_confidence = calculate_confidence_level(confirm_prediction_direction, confirm_indicators, confirm_pattern_confidence)
                            if confirm_confidence < 0.79:  # Nivel de confianza aumentado al 79%
                                confirm = False
                                break

                        if confirm:
                            tick = mt5.symbol_info_tick(symbol)
                            current_price = tick.ask if prediction_direction > 0 else tick.bid
                            tp = current_price + (current_price * 0.002) if prediction_direction > 0 else current_price - (current_price * 0.002)
                            sl = current_price - (current_price * 0.002) if prediction_direction > 0 else current_price + (current_price * 0.002)
                            request = {
                                "action": mt5.TRADE_ACTION_DEAL,
                                "symbol": symbol,
                                "volume": lot_size,
                                "type": mt5.ORDER_TYPE_BUY if prediction_direction > 0 else mt5.ORDER_TYPE_SELL,
                                "price": current_price,
                                "deviation": 10,
                                "magic": 234000,
                                "comment": "AI trade",
                                "type_time": mt5.ORDER_TIME_GTC,
                                "type_filling": mt5.ORDER_FILLING_RETURN
                            }
                            result = mt5.order_send(request)
                            while result is not None and result.retcode == mt5.TRADE_RETCODE_NO_MONEY and lot_size > 0.1:
                                lot_size -= 1
                                request["volume"] = lot_size
                                result = mt5.order_send(request)
                            
                            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                                print(f"Trade opened for {symbol}, ticket={result.order}, open price={current_price}")
                                threading.Thread(target=monitor_trade, args=(symbol, result.order, current_price, tp, sl, tf_name, confidence, indicators, model)).start()
                            else:
                                print(f"Trade failed for {symbol}, retcode={result.retcode}")

        time.sleep(30)
        print("Sleeping for 30 seconds.")

if __name__ == "__main__":
    print("Starting script...")
    main()
