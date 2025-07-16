#!/usr/bin/env python3
"""
ENHANCED SIGNAL GENERATOR - DATABASE CONFIGURATION VERSION
Connects to optimizer_configurations table for dynamic scoring
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MaxProfitSignals')

@dataclass
class MaxProfitSignal:
    """Enhanced trading signal with detailed scoring"""
    symbol: str
    date: str
    action: str  # BUY, SELL, HOLD
    current_price: float
    market_cap: float
    
    # Component Scores (0-100 scale)
    technical_score: float
    ai_prediction_score: float
    sentiment_score: float
    regime_score: float
    big_trader_score: float
    fear_greed_score: float
    momentum_score: float
    
    # Weighted scores
    weighted_technical: float
    weighted_ai: float
    weighted_sentiment: float
    weighted_regime: float
    weighted_big_trader: float
    weighted_fear_greed: float
    
    # Final analysis
    final_score: float
    confidence: float
    profit_potential: float
    
    # Risk management
    stop_loss_price: float
    target_price: float
    position_size_pct: float
    max_holding_days: int
    
    # Detailed scoring breakdown
    score_details: Dict[str, float] = field(default_factory=dict)
    action_reasons: List[str] = field(default_factory=list)
    technical_indicators: Dict = field(default_factory=dict)
    big_trader_activity: Dict = field(default_factory=dict)
    ai_predictions: Dict = field(default_factory=dict)
    
    # Configuration used
    config_id: Optional[int] = None
    
    def get_detailed_report(self) -> str:
        """Generate detailed scoring report"""
        report = f"""
{'='*80}
STOCK: {self.symbol} | DATE: {self.date}
{'='*80}
Current Price: ${self.current_price:.2f} | Market Cap: ${self.market_cap:,.0f}

FINAL DECISION: {self.action}
FINAL SCORE: {self.final_score:.2f}/100
CONFIDENCE: {self.confidence:.1%}
PROFIT POTENTIAL: {self.profit_potential:.1%}

DETAILED SCORING BREAKDOWN:
{'='*80}
Component          Raw Score   Weight    Weighted Score
---------------------------------------------------------
Technical          {self.technical_score:6.2f}     {self.score_details.get('technical_weight', 0):.1%}      {self.weighted_technical:6.2f}
AI Prediction      {self.ai_prediction_score:6.2f}     {self.score_details.get('ai_weight', 0):.1%}      {self.weighted_ai:6.2f}
News Sentiment     {self.sentiment_score:6.2f}     {self.score_details.get('sentiment_weight', 0):.1%}      {self.weighted_sentiment:6.2f}
Market Regime      {self.regime_score:6.2f}     {self.score_details.get('regime_weight', 0):.1%}      {self.weighted_regime:6.2f}
Big Traders        {self.big_trader_score:6.2f}     {self.score_details.get('big_trader_weight', 0):.1%}      {self.weighted_big_trader:6.2f}
Fear/Greed         {self.fear_greed_score:6.2f}     {self.score_details.get('fear_greed_weight', 0):.1%}      {self.weighted_fear_greed:6.2f}
---------------------------------------------------------
TOTAL                              100.0%      {self.final_score:6.2f}

DECISION THRESHOLDS:
Buy Score: {self.score_details.get('min_buy_score', 0):.0f} | Strong Buy: {self.score_details.get('min_strong_buy_score', 0):.0f}
Sell Score: {self.score_details.get('max_sell_score', 0):.0f} | Strong Sell: {self.score_details.get('max_strong_sell_score', 0):.0f}

TARGET PRICES:
Stop Loss: ${self.stop_loss_price:.2f} ({((self.stop_loss_price/self.current_price - 1)*100):.1f}%)
Take Profit: ${self.target_price:.2f} ({((self.target_price/self.current_price - 1)*100):.1f}%)
Position Size: {self.position_size_pct:.1%} of portfolio

ACTION REASONS:
"""
        for i, reason in enumerate(self.action_reasons, 1):
            report += f"{i}. {reason}\n"
        
        report += f"\nConfiguration ID: {self.config_id}"
        return report

class ConfigurationLoader:
    """Loads configuration from optimizer_configurations table"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def load_configuration(self, config_id: Optional[int] = None) -> Dict:
        """Load configuration by ID or get the best one"""
        conn = sqlite3.connect(self.db_path)
        
        if config_id:
            query = """
                SELECT * FROM optimizer_configurations 
                WHERE id = ?
            """
            params = (config_id,)
        else:
            query = """
                SELECT * FROM optimizer_configurations 
                WHERE is_best = TRUE 
                ORDER BY id DESC 
                LIMIT 1
            """
            params = ()
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if df.empty:
            raise ValueError("No configuration found")
        
        row = df.iloc[0]
        
        # Convert row to configuration dictionary
        config = {
            'config_id': row['id'],
            # Weights
            'technical_weight': row['technical_weight'],
            'ai_weight': row['ai_weight'],
            'sentiment_weight': row['sentiment_weight'],
            'regime_weight': row['regime_weight'],
            'big_trader_weight': row['big_trader_weight'],
            'fear_greed_weight': row['fear_greed_weight'],
            
            # Entry/Exit
            'min_buy_score': row['min_buy_score'],
            'min_strong_buy_score': row['min_strong_buy_score'],
            'max_sell_score': row['max_sell_score'],
            'max_strong_sell_score': row['max_strong_sell_score'],
            'min_confidence': row['min_confidence'],
            
            # Profit/Risk
            'base_profit_target': row['base_profit_target'],
            'stop_loss_pct': row['stop_loss_pct'],
            'trailing_stop_pct': row['trailing_stop_pct'],
            'take_profit_multiplier': row['take_profit_multiplier'],
            
            # Position sizing
            'base_position_size': row['base_position_size'],
            'max_position_size': row['max_position_size'],
            'kelly_fraction': row['kelly_fraction'],
            
            # Filters
            'min_market_cap': row['min_market_cap'],
            'min_price': row['min_price'],
            'min_volume': row['min_volume'],
            'max_positions': row['max_positions'],
            
            # Big trader parameters
            'institutional_volume_multiplier': row['institutional_volume_multiplier'],
            'accumulation_threshold': row['accumulation_threshold'],
            'distribution_threshold': row['distribution_threshold'],
            
            # Technical parameters
            'rsi_oversold_threshold': row['rsi_oversold_threshold'],
            'rsi_overbought_threshold': row['rsi_overbought_threshold'],
            'macd_signal_threshold': row['macd_signal_threshold'],
            
            # AI parameters
            'ai_momentum_weight': row['ai_momentum_weight'],
            'ai_reversion_weight': row['ai_reversion_weight'],
            'ai_pattern_weight': row['ai_pattern_weight'],
            'ai_max_prediction': row['ai_max_prediction'],
            
            # News parameters
            'news_lookback_days': row['news_lookback_days'],
            'news_relevance_threshold': row['news_relevance_threshold'],
            
            # Regime parameters
            'regime_trend_threshold': row['regime_trend_threshold'],
            'regime_volatility_threshold': row['regime_volatility_threshold'],
            
            # Momentum parameters
            'momentum_short_period': row['momentum_short_period'],
            'momentum_long_period': row['momentum_long_period'],
            
            # Trade fee
            'trade_fee': 1.0
        }
        
        logger.info(f"Loaded configuration ID: {config['config_id']}")
        logger.info(f"Big Trader Weight: {config['big_trader_weight']:.1%}")
        logger.info(f"Buy Threshold: {config['min_buy_score']:.0f}")
        
        return config

class BigTraderTracker:
    """Institutional flow tracking with configurable thresholds"""
    
    def __init__(self, db_path: str, config: Dict):
        self.db_path = db_path
        self.config = config
        
    def analyze_big_money_flows(self, symbol: str, date: str) -> Dict:
        """Detect institutional accumulation/distribution"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            df = pd.read_sql_query("""
                SELECT trade_date, open, high, low, close, volume,
                       volume * close as dollar_volume
                FROM stock_prices
                WHERE symbol = ? AND trade_date <= ?
                ORDER BY trade_date DESC
                LIMIT 60
            """, conn, params=(symbol, date))
            
            conn.close()
            
            if len(df) < 20:
                return {'score': 50, 'signal': 'insufficient_data', 'confidence': 0}
            
            df = df.iloc[::-1].reset_index(drop=True)
            
            # Volume analysis with config thresholds
            avg_volume_20 = df['volume'].rolling(20).mean()
            recent_volume = df['volume'].tail(5).mean()
            volume_ratio = recent_volume / avg_volume_20.iloc[-5:].mean()
            
            # Price-volume correlation
            price_change_5d = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1)
            
            # Money flow with configurable periods
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']
            
            # Calculate score using configuration
            score = 50
            signal_type = 'neutral'
            confidence = 0.5
            
            # Use configured thresholds
            if (volume_ratio > self.config['institutional_volume_multiplier'] and 
                price_change_5d > 0.02):
                score = 85 + min(15, price_change_5d * 200)
                signal_type = 'heavy_accumulation'
                confidence = 0.85
                
            elif (volume_ratio > 2.0 and price_change_5d < -0.02):
                score = 15
                signal_type = 'heavy_distribution'
                confidence = 0.80
                
            elif (volume_ratio > 1.5 and price_change_5d > 0.01):
                score = 70
                signal_type = 'accumulation'
                confidence = 0.70
                
            # Accumulation/Distribution scoring
            money_flow_ratio = money_flow.tail(5).mean() / money_flow.tail(20).mean()
            if money_flow_ratio > self.config['accumulation_threshold']:
                score = min(100, score + 10)
            elif money_flow_ratio < self.config['distribution_threshold']:
                score = max(0, score - 10)
            
            return {
                'score': score,
                'signal': signal_type,
                'confidence': confidence,
                'volume_ratio': volume_ratio,
                'price_change': price_change_5d,
                'money_flow_ratio': money_flow_ratio
            }
            
        except Exception as e:
            logger.error(f"Big trader analysis error for {symbol}: {e}")
            return {'score': 50, 'signal': 'error', 'confidence': 0}

class AIPredictor:
    """AI prediction with configurable model weights"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """Generate AI prediction score"""
        
        if len(df) < 50:
            return {'score': 50, 'confidence': 0, 'predicted_move': 0}
        
        close = df['close'].values
        returns = pd.Series(close).pct_change().dropna()
        
        # Momentum analysis with configured periods
        momentum_score = self._momentum_analysis(returns)
        
        # Mean reversion analysis  
        reversion_score = self._mean_reversion_analysis(close)
        
        # Pattern detection
        pattern_score = self._pattern_detection(df)
        
        # Apply configured weights
        final_score = (
            momentum_score * self.config['ai_momentum_weight'] + 
            reversion_score * self.config['ai_reversion_weight'] + 
            pattern_score * self.config['ai_pattern_weight']
        )
        
        # Predicted move with configured maximum
        predicted_move = (final_score - 50) / 100 * self.config['ai_max_prediction']
        
        return {
            'score': final_score,
            'confidence': min(0.9, abs(final_score - 50) / 40),
            'predicted_move': predicted_move,
            'components': {
                'momentum': momentum_score,
                'reversion': reversion_score,
                'pattern': pattern_score
            }
        }
    
    def _momentum_analysis(self, returns: pd.Series) -> float:
        """Momentum scoring with configured periods"""
        short_period = self.config['momentum_short_period']
        long_period = self.config['momentum_long_period']
        
        mom_short = returns.tail(short_period).mean()
        mom_long = returns.tail(long_period).mean()
        
        if mom_short > 0.01 and mom_long > 0.005 and mom_short > mom_long:
            return 75 + min(25, mom_short * 500)
        elif mom_short < -0.01 and mom_long < -0.005:
            return 25 - min(25, abs(mom_short) * 500)
        return 50
    
    def _mean_reversion_analysis(self, close: np.ndarray) -> float:
        """Mean reversion with z-score analysis"""
        sma_20 = pd.Series(close).rolling(20).mean().iloc[-1]
        std_20 = pd.Series(close).rolling(20).std().iloc[-1]
        
        if std_20 > 0:
            z_score = (close[-1] - sma_20) / std_20
            if z_score < -2:
                return 80  # Oversold
            elif z_score > 2:
                return 20  # Overbought
            return 50 - z_score * 15
        return 50
    
    def _pattern_detection(self, df: pd.DataFrame) -> float:
        """Pattern detection scoring"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        score = 50
        
        # Breakout detection
        if len(close) >= 20:
            recent_high = high[-20:-1].max()
            if close[-1] > recent_high * 1.02:
                score += 30
        
        # Support bounce
        if len(close) >= 10:
            recent_low = low[-10:-1].min()
            if recent_low < close[-1] < recent_low * 1.05:
                score += 20
        
        return min(100, score)

class MarketSentimentAnalyzer:
    """Sentiment analysis with configurable parameters"""
    
    def __init__(self, db_path: str, config: Dict):
        self.db_path = db_path
        self.config = config
        
    def get_fear_greed_score(self, date: str) -> float:
        """Calculate market fear/greed index"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            spy_data = pd.read_sql_query("""
                SELECT close FROM stock_prices
                WHERE symbol = 'SPY' AND trade_date <= ?
                ORDER BY trade_date DESC
                LIMIT 20
            """, conn, params=(date,))
            
            conn.close()
            
            if len(spy_data) < 10:
                return 50
            
            # Market momentum
            returns = spy_data['close'].pct_change()
            momentum = returns.tail(5).mean()
            
            # Calculate score with volatility consideration
            volatility = returns.std()
            
            score = 50 + momentum * 2000
            
            # Adjust for volatility
            if volatility > 0.02:
                score -= 10
            elif volatility < 0.01:
                score += 10
            
            return max(0, min(100, score))
            
        except:
            return 50
    
    def analyze_news_sentiment(self, symbol: str, date: str) -> Dict:
        """Analyze news with configured lookback and relevance"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Use configured lookback days
            news_data = pd.read_sql_query("""
                SELECT sentiment_score, relevance_score, published_at
                FROM stock_news
                WHERE symbol = ?
                AND relevance_score >= ?
                AND date(published_at) >= date(?, '-' || ? || ' days')
                AND date(published_at) <= ?
                ORDER BY published_at DESC
            """, conn, params=(
                symbol, 
                self.config['news_relevance_threshold'],
                date, 
                self.config['news_lookback_days'], 
                date
            ))
            
            conn.close()
            
            if len(news_data) == 0:
                return {'score': 50, 'confidence': 0, 'news_count': 0}
            
            # Weighted average with time decay
            news_data['days_ago'] = (pd.to_datetime(date) - pd.to_datetime(news_data['published_at'])).dt.days
            news_data['weight'] = news_data['relevance_score'] * np.exp(-news_data['days_ago'] / 3)
            
            weighted_sentiment = (news_data['sentiment_score'] * news_data['weight']).sum() / news_data['weight'].sum()
            
            # Convert to 0-100 scale
            score = 50 + weighted_sentiment * 10
            
            return {
                'score': max(0, min(100, score)),
                'confidence': min(0.9, len(news_data) / 10),
                'news_count': len(news_data),
                'avg_sentiment': weighted_sentiment
            }
            
        except:
            return {'score': 50, 'confidence': 0, 'news_count': 0}

class MaxProfitSignalGenerator:
    """Signal generator with database configuration support"""
    
    def __init__(self, db_path: str, config_id: Optional[int] = None):
        self.db_path = db_path
        
        # Load configuration from database
        config_loader = ConfigurationLoader(db_path)
        self.config = config_loader.load_configuration(config_id)
        self.config_id = self.config['config_id']
        
        # Initialize components with configuration
        self.big_trader_tracker = BigTraderTracker(db_path, self.config)
        self.ai_predictor = AIPredictor(self.config)
        self.sentiment_analyzer = MarketSentimentAnalyzer(db_path, self.config)
        
        logger.info(f"Signal Generator initialized with Configuration ID: {self.config_id}")
        self._print_configuration_summary()
    
    def _print_configuration_summary(self):
        """Print configuration summary"""
        logger.info("\nCONFIGURATION SUMMARY:")
        logger.info(f"Weights: Tech={self.config['technical_weight']:.1%}, "
                   f"AI={self.config['ai_weight']:.1%}, "
                   f"BigTrader={self.config['big_trader_weight']:.1%}")
        logger.info(f"Buy Threshold: {self.config['min_buy_score']:.0f}, "
                   f"Sell Threshold: {self.config['max_sell_score']:.0f}")
        logger.info(f"Profit Target: {self.config['base_profit_target']:.1%}, "
                   f"Stop Loss: {self.config['stop_loss_pct']:.1%}")
    
    def generate_all_signals(self, date: str = None, show_details: bool = True) -> List[MaxProfitSignal]:
        """Generate signals for all eligible stocks with detailed scoring"""
        
        if date is None:
            #date = datetime.now().strftime('%Y-%m-%d')
            conn = sqlite3.connect(self.db_path)
            date = pd.read_sql_query(
                "SELECT MAX(trade_date) as latest FROM stock_prices", conn
            ).iloc[0]['latest']
            conn.close()
            
        logger.info(f"\n{'='*80}")
        logger.info(f"GENERATING SIGNALS - CONFIG ID: {self.config_id}")
        logger.info(f"DATE: {date}")
        logger.info(f"{'='*80}")
        
        # Get eligible stocks
        eligible_stocks = self._get_eligible_stocks(date)
        logger.info(f"Analyzing {len(eligible_stocks)} stocks")
        
        # Get market sentiment
        fear_greed = self.sentiment_analyzer.get_fear_greed_score(date)
        logger.info(f"Market Fear/Greed Index: {fear_greed:.1f}")
        
        signals = []
        
        for i, stock_info in enumerate(eligible_stocks):
            if i % 50 == 0 and i > 0:
                logger.info(f"Progress: {i}/{len(eligible_stocks)} stocks analyzed")
            
            try:
                signal = self._analyze_stock(stock_info, date, fear_greed)
                if signal:
                    signals.append(signal)
                    if show_details and signal.action != 'HOLD':
                        logger.info(f"\n{signal.symbol}: {signal.action} "
                                   f"(Score: {signal.final_score:.1f}, "
                                   f"Profit: {signal.profit_potential:.1%})")
            except Exception as e:
                if "no such table" in str(e) or "no data" in str(e).lower():
                    logger.warning(f"Missing data for {stock_info['symbol']}, needs import")
                else:
                    logger.error(f"Error analyzing {stock_info['symbol']}: {e}")
        
        # Sort by score and profit potential
        signals.sort(key=lambda x: (x.final_score, x.profit_potential), reverse=True)
        
        # Filter to show only actionable signals
        actionable_signals = [s for s in signals if s.action != 'HOLD']
        
        # Print summary
        self._print_summary(actionable_signals)
        
        # Print top detailed reports if requested
        if show_details and actionable_signals:
            self._print_detailed_reports(actionable_signals[:5])
        
        return actionable_signals
    
    def _get_eligible_stocks(self, date: str) -> List[Dict]:
        """Get all stocks meeting criteria"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Get all stocks meeting basic criteria
            query = """
                SELECT DISTINCT p.symbol, s.market_cap, s.company_name, s.sector,
                       p.close as current_price, p.volume
                FROM stock_prices p
                JOIN stocks s ON p.symbol = s.symbol
                WHERE p.trade_date = ?
                AND p.close >= ?
                AND p.volume >= ?
                AND (s.market_cap >= ? OR s.market_cap IS NULL)
                AND s.is_active = 1
                ORDER BY p.volume * p.close DESC
                LIMIT 500
            """
            
            df = pd.read_sql_query(
                query, conn, 
                params=(date, self.config['min_price'], 
                       self.config['min_volume'], 
                       self.config['min_market_cap'])
            )
            
            stocks = []
            for _, row in df.iterrows():
                stocks.append({
                    'symbol': row['symbol'],
                    'market_cap': row['market_cap'] or 0,
                    'name': row['company_name'],
                    'sector': row['sector'],
                    'current_price': row['current_price'],
                    'volume': row['volume']
                })
            
            conn.close()
            return stocks
            
        except Exception as e:
            conn.close()
            logger.error(f"Error getting eligible stocks: {e}")
            return []
    
    def _analyze_stock(self, stock_info: Dict, date: str, market_fear_greed: float) -> Optional[MaxProfitSignal]:
        """Comprehensive stock analysis with detailed scoring"""
        
        symbol = stock_info['symbol']
        
        try:
            # Get price data
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query("""
                SELECT trade_date, open, high, low, close, volume
                FROM stock_prices
                WHERE symbol = ? AND trade_date <= ?
                ORDER BY trade_date DESC
                LIMIT 100
            """, conn, params=(symbol, date))
            
            conn.close()
            
            if len(df) < 50:
                return None
            
            # Prepare data
            df = df.iloc[::-1].reset_index(drop=True)
            current_price = df['close'].iloc[-1]
            
            # Calculate all component scores
            technical_score = self._calculate_technical_score(df)
            ai_analysis = self.ai_predictor.predict(df)
            sentiment_analysis = self.sentiment_analyzer.analyze_news_sentiment(symbol, date)
            big_trader_analysis = self.big_trader_tracker.analyze_big_money_flows(symbol, date)
            regime_score = self._calculate_regime_score(df, market_fear_greed)
            momentum_score = self._calculate_momentum_score(df)
            
            # Calculate weighted scores
            weighted_technical = technical_score * self.config['technical_weight']
            weighted_ai = ai_analysis['score'] * self.config['ai_weight']
            weighted_sentiment = sentiment_analysis['score'] * self.config['sentiment_weight']
            weighted_regime = regime_score * self.config['regime_weight']
            weighted_big_trader = big_trader_analysis['score'] * self.config['big_trader_weight']
            weighted_fear_greed = market_fear_greed * self.config['fear_greed_weight']
            
            # Final score
            final_score = (weighted_technical + weighted_ai + weighted_sentiment + 
                          weighted_regime + weighted_big_trader + weighted_fear_greed)
            
            # Confidence calculation
            confidence = min(0.95, 
                           (final_score / 100) * 
                           big_trader_analysis['confidence'] * 
                           ai_analysis['confidence'])
            
            # Determine action
            action = self._determine_action(final_score, big_trader_analysis)
            
            # Skip HOLD signals
            if action == 'HOLD':
                return None
            
            # Calculate profit potential
            profit_potential = self._calculate_profit_potential(
                ai_analysis, big_trader_analysis, momentum_score, final_score
            )
            
            # Set targets
            stop_loss_price, target_price = self._calculate_targets(
                current_price, action, profit_potential, df
            )
            
            # Position sizing
            position_size = self._calculate_position_size(
                final_score, profit_potential, confidence
            )
            
            # Build detailed reasoning
            action_reasons = self._build_detailed_reasons(
                action, technical_score, ai_analysis, sentiment_analysis,
                big_trader_analysis, regime_score, momentum_score, final_score
            )
            
            # Score details for reporting
            score_details = {
                'technical_weight': self.config['technical_weight'],
                'ai_weight': self.config['ai_weight'],
                'sentiment_weight': self.config['sentiment_weight'],
                'regime_weight': self.config['regime_weight'],
                'big_trader_weight': self.config['big_trader_weight'],
                'fear_greed_weight': self.config['fear_greed_weight'],
                'min_buy_score': self.config['min_buy_score'],
                'min_strong_buy_score': self.config['min_strong_buy_score'],
                'max_sell_score': self.config['max_sell_score'],
                'max_strong_sell_score': self.config['max_strong_sell_score']
            }
            
            return MaxProfitSignal(
                symbol=symbol,
                date=date,
                action=action,
                current_price=current_price,
                market_cap=stock_info['market_cap'],
                technical_score=technical_score,
                ai_prediction_score=ai_analysis['score'],
                sentiment_score=sentiment_analysis['score'],
                regime_score=regime_score,
                big_trader_score=big_trader_analysis['score'],
                fear_greed_score=market_fear_greed,
                momentum_score=momentum_score,
                weighted_technical=weighted_technical,
                weighted_ai=weighted_ai,
                weighted_sentiment=weighted_sentiment,
                weighted_regime=weighted_regime,
                weighted_big_trader=weighted_big_trader,
                weighted_fear_greed=weighted_fear_greed,
                final_score=final_score,
                confidence=confidence,
                profit_potential=profit_potential,
                stop_loss_price=stop_loss_price,
                target_price=target_price,
                position_size_pct=position_size,
                max_holding_days=self._calculate_holding_period(profit_potential),
                score_details=score_details,
                action_reasons=action_reasons,
                technical_indicators=self._get_technical_indicators(df),
                big_trader_activity=big_trader_analysis,
                ai_predictions=ai_analysis,
                config_id=self.config_id
            )
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def _calculate_technical_score(self, df: pd.DataFrame) -> float:
        """Technical scoring using configured parameters"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        score = 40  # Base score
        
        # Moving averages
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        
        # Price position
        if close.iloc[-1] > sma_20.iloc[-1]:
            score += 10
        if close.iloc[-1] > sma_50.iloc[-1]:
            score += 10
            
        # Trend alignment
        if sma_20.iloc[-1] > sma_50.iloc[-1]:
            score += 10
        
        # RSI with configured thresholds
        rsi = self._calculate_rsi(close)
        if rsi < self.config['rsi_oversold_threshold']:
            score += 20  # Oversold opportunity
        elif rsi > self.config['rsi_overbought_threshold']:
            score -= 10  # Overbought warning
        else:
            score += 5   # Neutral range
        
        # MACD with configured threshold
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        
        if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-1] > self.config['macd_signal_threshold']:
            score += 10
        
        # Volume confirmation
        vol_ratio = volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]
        if vol_ratio > 1.5:
            score += 10
        elif vol_ratio > 1.2:
            score += 5
        
        # ATR for volatility
        atr = self._calculate_atr(df)
        atr_ratio = atr / close.iloc[-1]
        if atr_ratio < 0.02:  # Low volatility
            score += 5
        
        return min(100, max(0, score))
    
    def _calculate_momentum_score(self, df: pd.DataFrame) -> float:
        """Momentum scoring with configured periods"""
        close = df['close']
        
        short_period = int(self.config['momentum_short_period'])
        long_period = int(self.config['momentum_long_period'])
        
        # Price momentum
        mom_short = (close.iloc[-1] / close.iloc[-short_period] - 1)
        mom_long = (close.iloc[-1] / close.iloc[-long_period] - 1)
        
        score = 50
        
        # Strong momentum
        if mom_short > 0.05 and mom_long > 0.10 and mom_short > mom_long:
            score = 85 + min(15, mom_short * 200)
        # Moderate momentum
        elif mom_short > 0.02 and mom_long > 0.05:
            score = 70 + min(10, mom_short * 300)
        # Negative momentum
        elif mom_short < -0.05:
            score = 20 - min(20, abs(mom_short) * 200)
        else:
            score = 50 + mom_short * 400
        
        return max(0, min(100, score))
    
    def _calculate_regime_score(self, df: pd.DataFrame, market_sentiment: float) -> float:
        """Market regime scoring with configured thresholds"""
        close = df['close']
        returns = close.pct_change()
        
        # Volatility analysis
        current_vol = returns.tail(20).std()
        historical_vol = returns.std()
        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1
        
        # Trend analysis
        trend_correlation = close.tail(20).corr(pd.Series(range(20)))
        
        score = 50
        
        # Strong uptrend in bullish market
        if (trend_correlation > self.config['regime_trend_threshold'] and 
            vol_ratio < self.config['regime_volatility_threshold'] and 
            market_sentiment > 60):
            score = 80
            
        # Downtrend in bearish market
        elif (trend_correlation < -self.config['regime_trend_threshold'] and 
              market_sentiment < 40):
            score = 20
            
        # High volatility regime
        elif vol_ratio > self.config['regime_volatility_threshold']:
            score = 35
            
        # Stable trend
        elif abs(trend_correlation) > 0.5:
            score = 50 + trend_correlation * 30
        
        # Market sentiment adjustment
        score += (market_sentiment - 50) / 5
        
        return max(0, min(100, score))
    
    def _determine_action(self, final_score: float, big_trader_analysis: Dict) -> str:
        """Determine action based on score and thresholds"""
        
        # Strong signals from big traders
        if big_trader_analysis['signal'] == 'heavy_accumulation' and final_score >= 55:
            return 'BUY'
        
        if big_trader_analysis['signal'] == 'heavy_distribution' and final_score <= 45:
            return 'SELL'
        
        # Score-based decisions
        if final_score >= self.config['min_strong_buy_score']:
            return 'BUY'
        elif final_score >= self.config['min_buy_score']:
            if big_trader_analysis['confidence'] >= self.config['min_confidence']:
                return 'BUY'
            return 'HOLD'
        elif final_score <= self.config['max_strong_sell_score']:
            return 'SELL'
        elif final_score <= self.config['max_sell_score']:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _calculate_profit_potential(self, ai_analysis: Dict, big_trader: Dict,
                                   momentum: float, final_score: float) -> float:
        """Calculate profit potential using configuration"""
        
        base = self.config['base_profit_target']
        
        # AI prediction bonus
        ai_bonus = max(0, ai_analysis['predicted_move'] * 0.5)
        
        # Big trader bonus
        trader_bonus = 0
        if big_trader['signal'] == 'heavy_accumulation':
            trader_bonus = 0.10
        elif big_trader['signal'] == 'institutional_breakout':
            trader_bonus = 0.15
        elif big_trader['signal'] == 'accumulation':
            trader_bonus = 0.05
        
        # Momentum bonus
        momentum_bonus = 0
        if momentum > 80:
            momentum_bonus = 0.08
        elif momentum > 70:
            momentum_bonus = 0.04
        
        # Score multiplier
        score_factor = (final_score - 50) / 50  # -1 to 1 range
        
        # Calculate total
        potential = base + ai_bonus + trader_bonus + momentum_bonus
        potential *= (1 + score_factor * 0.3)  # Adjust by score
        
        return min(0.50, max(0.05, potential))
    
    def _calculate_targets(self, current_price: float, action: str,
                          profit_potential: float, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate price targets with configuration"""
        
        # ATR for volatility-based stops
        atr = self._calculate_atr(df)
        
        if action == 'BUY':
            # Stop loss
            stop_loss = max(
                current_price * (1 - self.config['stop_loss_pct']),
                current_price - 2.5 * atr
            )
            
            # Take profit
            target = current_price * (1 + profit_potential * self.config['take_profit_multiplier'])
            
        else:  # SELL
            stop_loss = current_price * 0.98
            target = current_price * 0.95
        
        return stop_loss, target
    
    def _calculate_position_size(self, score: float, profit_potential: float,
                                confidence: float) -> float:
        """Kelly Criterion position sizing"""
        
        # Win probability based on score and confidence
        win_prob = min(0.80, (score / 100) * confidence)
        
        # Expected win/loss amounts
        win_amount = profit_potential
        loss_amount = self.config['stop_loss_pct']
        
        # Kelly calculation
        if win_amount > 0:
            kelly = (win_prob * win_amount - (1 - win_prob) * loss_amount) / win_amount
            kelly = max(0, kelly) * self.config['kelly_fraction']
        else:
            kelly = self.config['base_position_size']
        
        # Score-based adjustment
        if score >= 85:
            kelly *= 1.3
        elif score >= 75:
            kelly *= 1.1
        elif score < 50:
            kelly *= 0.5
        
        # Apply limits
        position_size = max(self.config['base_position_size'] * 0.5, kelly)
        position_size = min(self.config['max_position_size'], position_size)
        
        return position_size
    
    def _calculate_holding_period(self, profit_potential: float) -> int:
        """Dynamic holding period based on profit potential"""
        if profit_potential > 0.30:
            return 30
        elif profit_potential > 0.20:
            return 20
        elif profit_potential > 0.15:
            return 15
        else:
            return 10
    
    def _build_detailed_reasons(self, action: str, technical: float, ai_analysis: Dict,
                               sentiment: Dict, big_trader: Dict, regime: float,
                               momentum: float, final_score: float) -> List[str]:
        """Build detailed reasoning for the action"""
        reasons = []
        
        # Main action reason
        if action == 'BUY':
            if final_score >= self.config['min_strong_buy_score']:
                reasons.append(f"STRONG BUY: Score {final_score:.1f} exceeds strong buy threshold {self.config['min_strong_buy_score']:.0f}")
            else:
                reasons.append(f"BUY: Score {final_score:.1f} exceeds buy threshold {self.config['min_buy_score']:.0f}")
        else:
            if final_score <= self.config['max_strong_sell_score']:
                reasons.append(f"STRONG SELL: Score {final_score:.1f} below strong sell threshold {self.config['max_strong_sell_score']:.0f}")
            else:
                reasons.append(f"SELL: Score {final_score:.1f} below sell threshold {self.config['max_sell_score']:.0f}")
        
        # Component analysis
        if big_trader['signal'] == 'heavy_accumulation':
            reasons.append(f"🔥 INSTITUTIONAL ACCUMULATION: Volume {big_trader['volume_ratio']:.1f}x normal, money flow {big_trader.get('money_flow_ratio', 0):.2f}")
        elif big_trader['signal'] == 'heavy_distribution':
            reasons.append(f"⚠️ INSTITUTIONAL DISTRIBUTION: Heavy selling detected")
        
        if ai_analysis['predicted_move'] > 0.10:
            reasons.append(f"AI BULLISH: Predicts {ai_analysis['predicted_move']:.1%} upside (Momentum: {ai_analysis['components']['momentum']:.0f}, Pattern: {ai_analysis['components']['pattern']:.0f})")
        elif ai_analysis['predicted_move'] < -0.10:
            reasons.append(f"AI BEARISH: Predicts {ai_analysis['predicted_move']:.1%} downside")
        
        if technical > 70:
            reasons.append(f"TECHNICAL STRENGTH: Score {technical:.0f} - Strong setup confirmed")
        elif technical < 30:
            reasons.append(f"TECHNICAL WEAKNESS: Score {technical:.0f} - Breakdown detected")
        
        if sentiment['confidence'] > 0.5:
            if sentiment['score'] > 70:
                reasons.append(f"POSITIVE SENTIMENT: {sentiment['news_count']} news articles, avg sentiment {sentiment.get('avg_sentiment', 0):.1f}")
            elif sentiment['score'] < 30:
                reasons.append(f"NEGATIVE SENTIMENT: {sentiment['news_count']} news articles")
        
        if momentum > 80:
            reasons.append(f"STRONG MOMENTUM: Score {momentum:.0f} - Price trending strongly")
        
        if regime > 70:
            reasons.append(f"FAVORABLE REGIME: Market conditions supportive")
        
        return reasons
    
    def _get_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Get current technical indicators"""
        close = df['close']
        
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        
        indicators = {
            'price': close.iloc[-1],
            'sma_20': sma_20,
            'sma_50': sma_50,
            'price_vs_sma20': (close.iloc[-1] / sma_20 - 1) * 100,
            'price_vs_sma50': (close.iloc[-1] / sma_50 - 1) * 100,
            'rsi': self._calculate_rsi(close),
            'volume_ratio': df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1],
            'atr': self._calculate_atr(df),
            'volatility': close.pct_change().tail(20).std() * np.sqrt(252) * 100  # Annualized
        }
        
        return indicators
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """RSI calculation"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        
        if loss.iloc[-1] == 0:
            return 100
        
        rs = gain.iloc[-1] / loss.iloc[-1]
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Average True Range calculation"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]
    
    def _print_summary(self, signals: List[MaxProfitSignal]):
        """Print signal summary"""
        if not signals:
            logger.info("\nNo actionable signals generated")
            return
        
        buy_signals = [s for s in signals if s.action == 'BUY']
        sell_signals = [s for s in signals if s.action == 'SELL']
        
        logger.info(f"\n📊 SIGNAL SUMMARY - CONFIG ID: {self.config_id}")
        logger.info(f"Total Actionable Signals: {len(signals)}")
        logger.info(f"BUY Signals: {len(buy_signals)}")
        logger.info(f"SELL Signals: {len(sell_signals)}")
        
        if buy_signals:
            avg_buy_score = sum(s.final_score for s in buy_signals) / len(buy_signals)
            avg_buy_profit = sum(s.profit_potential for s in buy_signals) / len(buy_signals)
            logger.info(f"Average BUY Score: {avg_buy_score:.1f}")
            logger.info(f"Average BUY Profit Potential: {avg_buy_profit:.1%}")
        
        if sell_signals:
            avg_sell_score = sum(s.final_score for s in sell_signals) / len(sell_signals)
            logger.info(f"Average SELL Score: {avg_sell_score:.1f}")
    
    def _print_detailed_reports(self, signals: List[MaxProfitSignal]):
        """Print detailed reports for top signals"""
        logger.info(f"\n{'='*80}")
        logger.info("TOP SIGNALS - DETAILED REPORTS")
        logger.info(f"{'='*80}")
        
        for signal in signals[:5]:
            print(signal.get_detailed_report())
            print()

# Maintain compatibility
@dataclass  
class TradingSignal:
    """Legacy compatibility class"""
    symbol: str
    date: str
    signal_type: str
    technical_score: float
    ai_score: float
    sentiment_score: float
    regime_score: float
    final_score: float
    confidence: float
    position_size_pct: float
    stop_loss_price: float
    take_profit_price: float
    current_price: float
    risk_reward_ratio: float
    reasoning: Dict[str, Any]
    big_trader_score: float = 0

# Main entry point
if __name__ == "__main__":
    import sys
    
    db_path = sys.argv[1] if len(sys.argv) > 1 else "unified_trading.db"
    config_id = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else None
    date = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Initialize generator with configuration
    generator = MaxProfitSignalGenerator(db_path, config_id)
    
    # Generate signals
    signals = generator.generate_all_signals(date, show_details=True)
    
    print(f"\nGenerated {len(signals)} actionable trading signals")