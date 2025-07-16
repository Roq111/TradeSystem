#!/usr/bin/env python3
"""
AI Prediction Module for Stock Analysis
Calculates AI-based predictions using multiple models
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class AIPredictor:
    def __init__(self, db_path='unified_trading.db'):
        self.db_path = db_path
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # Ensure AI predictions table exists
        self._create_ai_tables()
    
    def _create_ai_tables(self):
        """Create AI-related tables"""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_predictions (
                symbol TEXT,
                prediction_date TEXT,
                model_type TEXT,
                prediction_1d REAL,
                prediction_5d REAL,
                prediction_10d REAL,
                confidence_score REAL,
                momentum_score REAL,
                reversal_score REAL,
                pattern_score REAL,
                features_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, prediction_date, model_type)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_model_performance (
                symbol TEXT,
                model_type TEXT,
                train_start_date TEXT,
                train_end_date TEXT,
                accuracy_1d REAL,
                accuracy_5d REAL,
                accuracy_10d REAL,
                mse REAL,
                r2_score REAL,
                feature_importance_json TEXT,
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, model_type)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def prepare_features(self, symbol, lookback_days=60):
        """Prepare features for AI prediction"""
        conn = sqlite3.connect(self.db_path)
        
        # Get price and technical indicator data
        query = """
            SELECT 
                p.trade_date,
                p.open, p.high, p.low, p.close, p.volume,
                t.rsi_14, t.macd, t.macd_signal, t.macd_histogram,
                t.bb_upper, t.bb_middle, t.bb_lower, t.bb_position,
                t.sma_5, t.sma_10, t.sma_20, t.sma_50,
                t.ema_12, t.ema_26, t.atr_14,
                t.volume_ratio, t.obv
            FROM stock_prices p
            LEFT JOIN technical_indicators t ON p.symbol = t.symbol AND p.trade_date = t.trade_date
            WHERE p.symbol = ?
            ORDER BY p.trade_date DESC
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(symbol, lookback_days + 50))
        conn.close()
        
        if len(df) < 50:
            return None
        
        # Sort by date ascending for calculations
        df = df.sort_values('trade_date')
        
        # Calculate additional features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volume_change'] = df['volume'].pct_change()
        
        # Price patterns
        df['high_low_spread'] = (df['high'] - df['low']) / df['close']
        df['close_open_spread'] = (df['close'] - df['open']) / df['open']
        
        # Momentum indicators
        df['momentum_3'] = df['close'] / df['close'].shift(3) - 1
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Volatility
        df['volatility_5'] = df['returns'].rolling(5).std()
        df['volatility_10'] = df['returns'].rolling(10).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        
        # Support/Resistance levels
        df['resistance_distance'] = (df['bb_upper'] - df['close']) / df['close']
        df['support_distance'] = (df['close'] - df['bb_lower']) / df['close']
        
        # Trend strength
        df['trend_strength'] = (df['sma_5'] - df['sma_20']) / df['sma_20']
        
        # Market regime features (if SPY data available)
        try:
            spy_conn = sqlite3.connect(self.db_path)
            spy_query = """
                SELECT trade_date, close
                FROM stock_prices
                WHERE symbol = 'SPY'
                ORDER BY trade_date DESC
                LIMIT ?
            """
            spy_df = pd.read_sql_query(spy_query, spy_conn, params=(lookback_days + 50,))
            spy_conn.close()
            
            if len(spy_df) > 0:
                spy_df = spy_df.sort_values('trade_date')
                spy_df['spy_returns'] = spy_df['close'].pct_change()
                
                # Merge SPY returns
                df = df.merge(spy_df[['trade_date', 'spy_returns']], on='trade_date', how='left')
                
                # Calculate beta
                if 'spy_returns' in df.columns:
                    df['beta'] = df['returns'].rolling(20).cov(df['spy_returns']) / df['spy_returns'].rolling(20).var()
        except:
            pass
        
        # Target variables (future returns)
        df['target_1d'] = df['close'].shift(-1) / df['close'] - 1
        df['target_5d'] = df['close'].shift(-5) / df['close'] - 1
        df['target_10d'] = df['close'].shift(-10) / df['close'] - 1
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def train_models(self, symbol):
        """Train multiple AI models for a symbol"""
        print(f"Training AI models for {symbol}...")
        
        # Prepare features
        df = self.prepare_features(symbol, lookback_days=500)
        if df is None or len(df) < 100:
            print(f"  Insufficient data for {symbol}")
            return False
        
        # Define feature columns
        feature_cols = [
            'returns', 'log_returns', 'volume_change', 'high_low_spread', 'close_open_spread',
            'momentum_3', 'momentum_5', 'momentum_10', 'momentum_20',
            'volatility_5', 'volatility_10', 'volatility_20',
            'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
            'bb_position', 'resistance_distance', 'support_distance',
            'trend_strength', 'volume_ratio'
        ]
        
        # Add beta if available
        if 'beta' in df.columns:
            feature_cols.append('beta')
        
        # Filter existing features
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[feature_cols].values
        
        # Train models for different prediction horizons
        models_trained = False
        
        for horizon, target_col in [(1, 'target_1d'), (5, 'target_5d'), (10, 'target_10d')]:
            if target_col not in df.columns:
                continue
            
            y = df[target_col].values
            
            # Remove samples with NaN targets
            valid_idx = ~np.isnan(y)
            X_valid = X[valid_idx]
            y_valid = y[valid_idx]
            
            if len(X_valid) < 50:
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_valid, y_valid, test_size=0.2, shuffle=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train_scaled, y_train)
            
            # Train Gradient Boosting
            gb_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.01,
                random_state=42
            )
            gb_model.fit(X_train_scaled, y_train)
            
            # Store models
            model_key = f"{symbol}_{horizon}d"
            self.models[f"{model_key}_rf"] = rf_model
            self.models[f"{model_key}_gb"] = gb_model
            self.scalers[model_key] = scaler
            
            # Store feature importance
            feature_importance = dict(zip(feature_cols, rf_model.feature_importances_))
            self.feature_importance[model_key] = feature_importance
            
            models_trained = True
        
        if models_trained:
            print(f"  ✓ Models trained successfully")
            
            # Generate predictions for recent data
            self.generate_predictions(symbol, df, feature_cols)
            
            return True
        else:
            print(f"  ✗ Failed to train models")
            return False
    
    def generate_predictions(self, symbol, df=None, feature_cols=None):
        """Generate AI predictions for a symbol"""
        if df is None:
            df = self.prepare_features(symbol, lookback_days=60)
            if df is None:
                return
        
        if feature_cols is None:
            feature_cols = [
                'returns', 'log_returns', 'volume_change', 'high_low_spread', 'close_open_spread',
                'momentum_3', 'momentum_5', 'momentum_10', 'momentum_20',
                'volatility_5', 'volatility_10', 'volatility_20',
                'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
                'bb_position', 'resistance_distance', 'support_distance',
                'trend_strength', 'volume_ratio'
            ]
            if 'beta' in df.columns:
                feature_cols.append('beta')
            feature_cols = [col for col in feature_cols if col in df.columns]
        
        # Get latest data point
        latest_data = df.iloc[-1]
        latest_features = latest_data[feature_cols].values.reshape(1, -1)
        
        predictions = {}
        confidence_scores = []
        
        # Generate predictions for each horizon
        for horizon in [1, 5, 10]:
            model_key = f"{symbol}_{horizon}d"
            
            if f"{model_key}_rf" in self.models and model_key in self.scalers:
                # Scale features
                features_scaled = self.scalers[model_key].transform(latest_features)
                
                # Get predictions from both models
                rf_pred = self.models[f"{model_key}_rf"].predict(features_scaled)[0]
                gb_pred = self.models[f"{model_key}_gb"].predict(features_scaled)[0]
                
                # Ensemble prediction
                ensemble_pred = (rf_pred + gb_pred) / 2
                predictions[f'prediction_{horizon}d'] = ensemble_pred
                
                # Calculate confidence based on model agreement
                model_agreement = 1 - abs(rf_pred - gb_pred) / (abs(rf_pred) + abs(gb_pred) + 0.0001)
                confidence_scores.append(model_agreement)
        
        if not predictions:
            return
        
        # Calculate overall confidence
        confidence_score = np.mean(confidence_scores) if confidence_scores else 0.5
        
        # Calculate pattern scores
        momentum_score = self._calculate_momentum_score(latest_data)
        reversal_score = self._calculate_reversal_score(latest_data)
        pattern_score = self._calculate_pattern_score(df)
        
        # Store predictions in database
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            INSERT OR REPLACE INTO ai_predictions
            (symbol, prediction_date, model_type, prediction_1d, prediction_5d, prediction_10d,
             confidence_score, momentum_score, reversal_score, pattern_score, features_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol,
            latest_data['trade_date'],
            'ensemble',
            predictions.get('prediction_1d', 0),
            predictions.get('prediction_5d', 0),
            predictions.get('prediction_10d', 0),
            confidence_score,
            momentum_score,
            reversal_score,
            pattern_score,
            pd.Series(latest_data[feature_cols]).to_json()
        ))
        
        conn.commit()
        conn.close()
    
    def _calculate_momentum_score(self, data):
        """Calculate momentum score based on technical indicators"""
        score = 0.5  # Neutral
        
        # Price above moving averages
        if 'close' in data and 'sma_20' in data and data['sma_20'] > 0:
            if data['close'] > data['sma_20']:
                score += 0.1
        
        if 'sma_5' in data and 'sma_10' in data and data['sma_10'] > 0:
            if data['sma_5'] > data['sma_10']:
                score += 0.1
        
        # Positive momentum
        if 'momentum_5' in data and data['momentum_5'] > 0:
            score += min(0.2, data['momentum_5'] * 2)
        
        # MACD positive
        if 'macd' in data and 'macd_signal' in data:
            if data['macd'] > data['macd_signal']:
                score += 0.1
        
        # Volume confirmation
        if 'volume_ratio' in data and data['volume_ratio'] > 1.2:
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _calculate_reversal_score(self, data):
        """Calculate reversal probability score"""
        score = 0.5  # Neutral
        
        # RSI extremes
        if 'rsi_14' in data:
            if data['rsi_14'] < 30:
                score += 0.2  # Oversold
            elif data['rsi_14'] > 70:
                score += 0.2  # Overbought
        
        # Bollinger Band extremes
        if 'bb_position' in data:
            if data['bb_position'] < 0.1:
                score += 0.15  # Near lower band
            elif data['bb_position'] > 0.9:
                score += 0.15  # Near upper band
        
        # High volatility
        if 'volatility_20' in data and data['volatility_20'] > 0.02:
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _calculate_pattern_score(self, df):
        """Calculate pattern recognition score"""
        if len(df) < 20:
            return 0.5
        
        score = 0.5
        recent = df.tail(20)
        
        # Detect support/resistance breaks
        recent_high = recent['high'].max()
        recent_low = recent['low'].min()
        current_close = recent['close'].iloc[-1]
        
        if current_close > recent_high * 0.98:
            score += 0.2  # Breaking resistance
        elif current_close < recent_low * 1.02:
            score -= 0.2  # Breaking support
        
        # Trend consistency
        sma_5_trend = recent['sma_5'].diff().mean()
        sma_20_trend = recent['sma_20'].diff().mean()
        
        if sma_5_trend > 0 and sma_20_trend > 0:
            score += 0.1  # Uptrend
        elif sma_5_trend < 0 and sma_20_trend < 0:
            score -= 0.1  # Downtrend
        
        return min(1.0, max(0.0, score))
    
    def update_all_predictions(self, symbols=None):
        """Update AI predictions for all or specified symbols"""
        conn = sqlite3.connect(self.db_path)
        
        if symbols is None:
            # Get all active symbols with sufficient data
            symbols_df = pd.read_sql_query("""
                SELECT DISTINCT s.symbol
                FROM stocks s
                JOIN stock_prices p ON s.symbol = p.symbol
                WHERE s.is_active = 1
                GROUP BY s.symbol
                HAVING COUNT(DISTINCT p.trade_date) >= 100
            """, conn)
            symbols = symbols_df['symbol'].tolist()
        
        conn.close()
        
        print(f"\nUpdating AI predictions for {len(symbols)} symbols...")
        
        success_count = 0
        for i, symbol in enumerate(symbols):
            try:
                print(f"\r[{i+1}/{len(symbols)}] Processing {symbol}...", end='', flush=True)
                
                # Check if models exist for this symbol
                model_key = f"{symbol}_1d_rf"
                if model_key not in self.models:
                    # Train models if not exist
                    if self.train_models(symbol):
                        success_count += 1
                else:
                    # Generate new predictions
                    self.generate_predictions(symbol)
                    success_count += 1
                
            except Exception as e:
                print(f"\nError processing {symbol}: {str(e)}")
                continue
        
        print(f"\n\nAI predictions updated for {success_count}/{len(symbols)} symbols")
        
        # Show summary
        conn = sqlite3.connect(self.db_path)
        summary = pd.read_sql_query("""
            SELECT 
                COUNT(DISTINCT symbol) as symbols_with_predictions,
                COUNT(*) as total_predictions,
                AVG(confidence_score) as avg_confidence,
                MAX(prediction_date) as latest_prediction
            FROM ai_predictions
        """, conn)
        
        print("\nAI Predictions Summary:")
        print(f"  Symbols with predictions: {summary['symbols_with_predictions'].iloc[0]}")
        print(f"  Total predictions: {summary['total_predictions'].iloc[0]}")
        print(f"  Average confidence: {summary['avg_confidence'].iloc[0]:.3f}")
        print(f"  Latest prediction: {summary['latest_prediction'].iloc[0]}")
        
        conn.close()

def integrate_ai_predictions():
    """Integrate AI predictions into the import process"""
    predictor = AIPredictor('unified_trading.db')
    
    # Update predictions for all symbols
    predictor.update_all_predictions()
    
    print("\n✅ AI predictions integrated successfully!")

if __name__ == "__main__":
    integrate_ai_predictions()