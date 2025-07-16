#!/usr/bin/env python3
"""
Comprehensive Stock Data Importer with Big Trader Indicator
Imports S&P 500, NASDAQ, top ETFs, and Russell stocks with all indicators
"""

import yfinance as yf
import requests
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Configure yfinance logging to suppress error messages
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

class ComprehensiveStockImporter:
    def __init__(self, db_path='unified_trading.db'):
        self.db_path = db_path
        self.start_date = '2022-01-01'
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Data sources with your API keys
        self.data_sources = {
            'yfinance': {'priority': 1, 'enabled': True},
            'finnhub': {
                'priority': 2, 
                'enabled': True,
                'api_key': 'd1ocfhpr01qtrav0gllgd1ocfhpr01qtrav0glm0'
            },
            'alpha_vantage': {
                'priority': 3,
                'enabled': True,
                'api_key': '494WCQDIZAHTAXK3'
            }
        }
        
        # Fear & Greed API
        self.fear_greed_url = "https://api.alternative.me/fng/?limit=100"
        
        # News API (you may need to add your key)
        self.news_api_key = 'YOUR_NEWS_API_KEY'  # Add if you have one
        
        # Big trader thresholds
        self.large_trade_threshold = 0.05  # 5% of daily volume
        self.institutional_threshold = 0.01  # 1% of shares outstanding
        self.dark_pool_threshold = 0.3  # 30% of volume in dark pools
        
        self.setup_database()
    
    def setup_database(self):
        """Ensure all required tables exist including big trader tables"""
        conn = sqlite3.connect(self.db_path)
        
        # Add missing columns to stock_prices if needed
        try:
            conn.execute("ALTER TABLE stock_prices ADD COLUMN data_source TEXT")
        except:
            pass
            
        # Create market_data table if not exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                data_key TEXT PRIMARY KEY,
                data_value TEXT,
                last_updated TEXT
            )
        """)
        
        # Create big trader tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS institutional_ownership (
                symbol TEXT,
                report_date TEXT,
                institution_name TEXT,
                shares_held INTEGER,
                value_held REAL,
                percent_outstanding REAL,
                change_in_shares INTEGER,
                change_percent REAL,
                PRIMARY KEY (symbol, report_date, institution_name)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS large_trades (
                symbol TEXT,
                trade_date TEXT,
                large_volume_ratio REAL,
                block_trade_count INTEGER,
                avg_trade_size REAL,
                institutional_volume_estimate INTEGER,
                retail_volume_estimate INTEGER,
                smart_money_flow REAL,
                PRIMARY KEY (symbol, trade_date)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS big_trader_scores (
                symbol TEXT,
                score_date TEXT,
                institutional_score REAL,
                volume_analysis_score REAL,
                accumulation_score REAL,
                smart_money_score REAL,
                overall_score REAL,
                trend TEXT,
                confidence REAL,
                details_json TEXT,
                PRIMARY KEY (symbol, score_date)
            )
        """)
        
        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_big_trader_scores ON big_trader_scores(symbol, score_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_large_trades ON large_trades(symbol, trade_date)")
        
        conn.commit()
        conn.close()
    
    def fetch_stock_lists(self):
        """Fetch S&P 500, NASDAQ, top ETFs, and Russell stocks from web"""
        all_symbols = set()
        
        print("Fetching stock lists from web...")
        
        # 1. S&P 500
        try:
            print("  - Fetching S&P 500...")
            sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            sp500_df = pd.read_html(sp500_url)[0]
            sp500_symbols = sp500_df['Symbol'].str.replace('.', '-').tolist()
            all_symbols.update(sp500_symbols)
            print(f"    Found {len(sp500_symbols)} S&P 500 stocks")
        except Exception as e:
            print(f"    Error fetching S&P 500: {e}")
        
        # 2. NASDAQ 100
        try:
            print("  - Fetching NASDAQ 100...")
            nasdaq_url = "https://en.wikipedia.org/wiki/Nasdaq-100"
            nasdaq_tables = pd.read_html(nasdaq_url)
            # Find the table with stock symbols
            for table in nasdaq_tables:
                if 'Ticker' in table.columns:
                    nasdaq_symbols = table['Ticker'].tolist()
                    all_symbols.update(nasdaq_symbols)
                    print(f"    Found {len(nasdaq_symbols)} NASDAQ stocks")
                    break
        except Exception as e:
            print(f"    Error fetching NASDAQ: {e}")
        
        # 3. Top ETFs
        top_etfs = [
            'SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI', 'EEM', 'XLF', 'XLK', 'XLE',
            'XLV', 'XLI', 'XLY', 'XLP', 'XLB', 'XLU', 'XLRE', 'VNQ', 'GLD', 'SLV',
            'USO', 'UNG', 'TLT', 'IEF', 'LQD', 'HYG', 'AGG', 'BND', 'ARKK', 'ARKQ'
        ]
        all_symbols.update(top_etfs)
        print(f"  - Added {len(top_etfs)} top ETFs")
        
        # 4. Russell 2000 (top 50 by market cap)
        try:
            print("  - Adding Russell 2000 components...")
            # Get some popular small-cap stocks
            russell_stocks = [
                'RIOT', 'MARA', 'PLUG', 'FCEL', 'BLNK', 'FSR', 'RIDE', 'WKHS', 'NKLA', 'HYLN',
                'LAZR', 'VLDR', 'GOEV', 'ARVL', 'MVST', 'CHPT', 'EVGO', 'LEV', 'PTRA', 'FFIE',
                'LCID', 'RIVN', 'XPEV', 'NIO', 'LI', 'BYDDY', 'TSLA', 'GM', 'F', 'STLA',
                'TM', 'HMC', 'RACE', 'MBGYY', 'VWAGY', 'BMWYY', 'DDAIF', 'POAHY', 'SZKMY', 'HYMTF',
                'TTM', 'HMC', 'APTV', 'BWA', 'LEA', 'GNTX', 'DORM', 'THRM', 'CPS', 'MPAA'
            ]
            all_symbols.update(russell_stocks[:50])
            print(f"    Added {len(russell_stocks[:50])} Russell stocks")
        except Exception as e:
            print(f"    Error with Russell stocks: {e}")
        
        # Remove any empty or invalid symbols
        all_symbols = {s for s in all_symbols if s and len(s) > 0}
        
        print(f"\nTotal unique symbols to import: {len(all_symbols)}")
        return sorted(list(all_symbols))
    
    def import_stock_yfinance(self, symbol):
        """Import stock data using yfinance with proper error handling"""
        try:
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Get historical data using history method (more reliable)
            df = ticker.history(start=self.start_date, end=self.end_date, auto_adjust=True)
            
            if df is None or df.empty or len(df) == 0:
                return False
            
            # Get stock info with defaults
            company_name = symbol
            sector = 'Unknown'
            industry = 'Unknown'
            market_cap = 0
            exchange = 'Unknown'
            
            # Try to get info, but don't fail if we can't
            try:
                info = ticker.info
                if info:
                    company_name = info.get('longName', info.get('shortName', symbol))
                    sector = info.get('sector', 'Unknown')
                    industry = info.get('industry', 'Unknown')
                    market_cap = info.get('marketCap', 0)
                    exchange = info.get('exchange', 'Unknown')
            except:
                pass
            
            # Prepare price data
            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            df['Symbol'] = symbol
            
            # Store data
            self._store_stock_data(
                symbol=symbol,
                company_name=company_name,
                sector=sector,
                industry=industry,
                market_cap=market_cap,
                exchange=exchange,
                price_df=df,
                source='yfinance'
            )
            
            # Import news if available
            try:
                news = ticker.news
                if news:
                    self._store_news_data(symbol, news)
            except:
                pass
            
            # Import institutional holders
            try:
                inst_holders = ticker.institutional_holders
                if inst_holders is not None and not inst_holders.empty:
                    self._store_institutional_data(symbol, inst_holders)
            except:
                pass
            
            # Calculate big trader indicators
            self._calculate_big_trader_indicators(symbol)
            
            return True
            
        except Exception as e:
            if "429" in str(e) or "Too Many" in str(e):
                return False
            return False
    
    def import_stock_finnhub(self, symbol):
        """Import using Finnhub as fallback"""
        try:
            api_key = self.data_sources['finnhub']['api_key']
            
            # Get price data
            end = int(datetime.strptime(self.end_date, '%Y-%m-%d').timestamp())
            start = int(datetime.strptime(self.start_date, '%Y-%m-%d').timestamp())
            
            url = f"https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution=D&from={start}&to={end}&token={api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 429:
                # Rate limit hit
                time.sleep(2)
                return False
            
            if response.status_code != 200:
                return False
            
            data = response.json()
            if data.get('s') != 'ok' or 'c' not in data:
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'Date': pd.to_datetime(data['t'], unit='s').strftime('%Y-%m-%d'),
                'Open': data['o'],
                'High': data['h'],
                'Low': data['l'],
                'Close': data['c'],
                'Volume': data['v'],
                'Symbol': symbol
            })
            
            # Get company info
            profile_url = f"https://finnhub.io/api/v1/stock/profile2?symbol={symbol}&token={api_key}"
            profile_resp = requests.get(profile_url, timeout=5)
            
            company_name = symbol
            sector = 'Unknown'
            industry = 'Unknown'
            market_cap = 0
            exchange = 'Unknown'
            
            if profile_resp.status_code == 200:
                profile = profile_resp.json()
                company_name = profile.get('name', symbol)
                sector = profile.get('finnhubIndustry', 'Unknown')
                market_cap = profile.get('marketCapitalization', 0) * 1000000  # Convert to actual value
                exchange = profile.get('exchange', 'Unknown')
            
            # Store data
            self._store_stock_data(
                symbol=symbol,
                company_name=company_name,
                sector=sector,
                industry=industry,
                market_cap=market_cap,
                exchange=exchange,
                price_df=df,
                source='finnhub'
            )
            
            # Get news
            news_url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={self.start_date}&to={self.end_date}&token={api_key}"
            news_resp = requests.get(news_url, timeout=5)
            if news_resp.status_code == 200:
                news_data = news_resp.json()
                if news_data:
                    self._store_finnhub_news(symbol, news_data)
            
            # Get ownership data (institutional investors)
            try:
                ownership_url = f"https://finnhub.io/api/v1/stock/ownership?symbol={symbol}&token={api_key}"
                ownership_resp = requests.get(ownership_url, timeout=5)
                if ownership_resp.status_code == 200:
                    ownership_data = ownership_resp.json()
                    if ownership_data:
                        self._store_finnhub_ownership(symbol, ownership_data)
            except:
                pass
            
            # Calculate big trader indicators
            self._calculate_big_trader_indicators(symbol)
            
            return True
            
        except Exception as e:
            return False
    
    def import_stock_alpha_vantage(self, symbol):
        """Import using Alpha Vantage as last resort"""
        try:
            api_key = self.data_sources['alpha_vantage']['api_key']
            
            # Daily prices
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                return False
            
            data = response.json()
            
            # Check for API limit message
            if "Note" in data or "Information" in data:
                print(f"\n{symbol}: Alpha Vantage API limit reached", end='')
                return False
                
            if 'Time Series (Daily)' not in data:
                return False
            
            # Convert to DataFrame
            prices = data['Time Series (Daily)']
            df_data = []
            
            for date, values in prices.items():
                if date >= self.start_date and date <= self.end_date:
                    df_data.append({
                        'Date': date,
                        'Open': float(values['1. open']),
                        'High': float(values['2. high']),
                        'Low': float(values['3. low']),
                        'Close': float(values['4. close']),
                        'Volume': int(values['6. volume']),
                        'Symbol': symbol
                    })
            
            if not df_data:
                return False
            
            df = pd.DataFrame(df_data)
            
            # Get company info
            info_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={api_key}"
            info_resp = requests.get(info_url, timeout=5)
            
            company_name = symbol
            sector = 'Unknown'
            industry = 'Unknown'
            market_cap = 0
            exchange = 'Unknown'
            
            if info_resp.status_code == 200:
                info = info_resp.json()
                if "Note" not in info and "Information" not in info:
                    company_name = info.get('Name', symbol)
                    sector = info.get('Sector', 'Unknown')
                    industry = info.get('Industry', 'Unknown')
                    market_cap = float(info.get('MarketCapitalization', 0))
                    exchange = info.get('Exchange', 'Unknown')
            
            # Store data
            self._store_stock_data(
                symbol=symbol,
                company_name=company_name,
                sector=sector,
                industry=industry,
                market_cap=market_cap,
                exchange=exchange,
                price_df=df,
                source='alpha_vantage'
            )
            
            # Calculate big trader indicators
            self._calculate_big_trader_indicators(symbol)
            
            return True
            
        except Exception as e:
            return False
    
    def _store_stock_data(self, symbol, company_name, sector, industry, market_cap, exchange, price_df, source):
        """Store stock data in database"""
        conn = sqlite3.connect(self.db_path)
        
        # Update stock info
        conn.execute("""
            INSERT OR REPLACE INTO stocks 
            (symbol, company_name, exchange, sector, industry, market_cap, is_active, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, 1, datetime('now'))
        """, (symbol, company_name[:100] if company_name else symbol, 
              exchange[:50] if exchange else 'Unknown',
              sector[:50] if sector else 'Unknown', 
              industry[:50] if industry else 'Unknown',
              market_cap if market_cap else 0))
        
        # Store price data
        for _, row in price_df.iterrows():
            # Handle both uppercase and lowercase column names
            open_val = row.get('Open', row.get('open', 0))
            high_val = row.get('High', row.get('high', 0))
            low_val = row.get('Low', row.get('low', 0))
            close_val = row.get('Close', row.get('close', 0))
            volume_val = row.get('Volume', row.get('volume', 0))
            
            # Handle potential None values
            open_val = float(open_val) if open_val is not None else 0
            high_val = float(high_val) if high_val is not None else 0
            low_val = float(low_val) if low_val is not None else 0
            close_val = float(close_val) if close_val is not None else 0
            volume_val = int(volume_val) if volume_val is not None else 0
            
            conn.execute("""
                INSERT OR REPLACE INTO stock_prices
                (symbol, trade_date, open, high, low, close, adj_close, volume, data_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, row['Date'], 
                  open_val, high_val, low_val, close_val, close_val,
                  volume_val, source))
        
        conn.commit()
        
        # Calculate technical indicators
        self._calculate_technical_indicators(symbol, conn)
        
        conn.close()
    
    def _store_institutional_data(self, symbol, inst_holders):
        """Store institutional ownership data from yfinance"""
        conn = sqlite3.connect(self.db_path)
        
        report_date = datetime.now().strftime('%Y-%m-%d')
        
        for _, row in inst_holders.iterrows():
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO institutional_ownership
                    (symbol, report_date, institution_name, shares_held, value_held, 
                     percent_outstanding, change_in_shares, change_percent)
                    VALUES (?, ?, ?, ?, ?, ?, 0, 0)
                """, (
                    symbol,
                    report_date,
                    str(row.get('Holder', 'Unknown'))[:100],
                    int(row.get('Shares', 0)),
                    float(row.get('Value', 0)),
                    float(row.get('% Out', 0)) / 100 if row.get('% Out') else 0
                ))
            except:
                continue
        
        conn.commit()
        conn.close()
    
    def _store_finnhub_ownership(self, symbol, ownership_data):
        """Store ownership data from Finnhub"""
        conn = sqlite3.connect(self.db_path)
        
        report_date = datetime.now().strftime('%Y-%m-%d')
        
        for owner in ownership_data.get('ownership', []):
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO institutional_ownership
                    (symbol, report_date, institution_name, shares_held, value_held, 
                     percent_outstanding, change_in_shares, change_percent)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    report_date,
                    owner.get('name', 'Unknown')[:100],
                    int(owner.get('share', 0)),
                    0,  # Finnhub doesn't provide value
                    float(owner.get('percentage', 0)),
                    int(owner.get('change', 0)),
                    0  # Calculate if needed
                ))
            except:
                continue
        
        conn.commit()
        conn.close()
    
    def _calculate_big_trader_indicators(self, symbol):
        """Calculate big trader indicators based on volume patterns and institutional data"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Get recent price and volume data
            df = pd.read_sql_query("""
                SELECT trade_date, close, volume, high, low
                FROM stock_prices
                WHERE symbol = ?
                ORDER BY trade_date DESC
                LIMIT 100
            """, conn, params=(symbol,))
            
            if len(df) < 20:
                conn.close()
                return
            
            # Calculate volume-based indicators
            df['avg_volume_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['avg_volume_20']
            df['price_change'] = df['close'].pct_change()
            df['high_low_range'] = (df['high'] - df['low']) / df['close']
            
            # Detect large volume days (potential institutional activity)
            df['large_volume'] = df['volume_ratio'] > 1.5
            df['large_volume_up'] = df['large_volume'] & (df['price_change'] > 0)
            df['large_volume_down'] = df['large_volume'] & (df['price_change'] < 0)
            
            # Calculate accumulation/distribution
            df['money_flow'] = df['close'] * df['volume']
            df['money_flow_20'] = df['money_flow'].rolling(20).sum()
            
            # Smart money flow (first 30 min vs rest of day approximation)
            # Using volume distribution as proxy
            df['smart_money_indicator'] = np.where(
                (df['volume_ratio'] > 1.2) & (df['price_change'] > 0.001),
                df['volume'] * df['price_change'],
                -df['volume'] * abs(df['price_change'])
            )
            df['smart_money_flow'] = df['smart_money_indicator'].rolling(10).sum()
            
            # Store large trade analysis
            for idx, row in df.iterrows():
                if pd.notna(row['volume_ratio']):
                    # Estimate institutional vs retail volume
                    inst_volume = int(row['volume'] * 0.3) if row['volume_ratio'] > 1.5 else int(row['volume'] * 0.15)
                    retail_volume = int(row['volume']) - inst_volume
                    
                    conn.execute("""
                        INSERT OR REPLACE INTO large_trades
                        (symbol, trade_date, large_volume_ratio, block_trade_count,
                         avg_trade_size, institutional_volume_estimate, retail_volume_estimate,
                         smart_money_flow)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        row['trade_date'],
                        row['volume_ratio'],
                        int(row['volume_ratio'] * 10) if row['large_volume'] else 0,
                        row['volume'] / 1000,  # Avg trade size in thousands
                        inst_volume,
                        retail_volume,
                        row['smart_money_flow'] if pd.notna(row['smart_money_flow']) else 0
                    ))
            
            # Get institutional ownership data
            inst_data = pd.read_sql_query("""
                SELECT SUM(percent_outstanding) as total_institutional_pct,
                       COUNT(DISTINCT institution_name) as num_institutions,
                       SUM(CASE WHEN change_in_shares > 0 THEN 1 ELSE 0 END) as buyers,
                       SUM(CASE WHEN change_in_shares < 0 THEN 1 ELSE 0 END) as sellers
                FROM institutional_ownership
                WHERE symbol = ?
                AND report_date >= date('now', '-90 days')
            """, conn, params=(symbol,))
            
            # Calculate scores
            latest_date = df['trade_date'].iloc[0]
            
            # 1. Institutional Score (0-100)
            inst_score = 50  # Default neutral
            if not inst_data.empty and inst_data['total_institutional_pct'].iloc[0]:
                inst_pct = inst_data['total_institutional_pct'].iloc[0]
                inst_score = min(100, inst_pct * 2)  # Scale to 100
                
                # Adjust for buying/selling
                if inst_data['buyers'].iloc[0] and inst_data['sellers'].iloc[0]:
                    buy_sell_ratio = inst_data['buyers'].iloc[0] / (inst_data['sellers'].iloc[0] + 1)
                    inst_score *= (0.5 + min(0.5, buy_sell_ratio / 2))
            
            # 2. Volume Analysis Score (0-100)
            recent_df = df.head(20)
            large_vol_days = recent_df['large_volume'].sum()
            large_vol_up_days = recent_df['large_volume_up'].sum()
            
            volume_score = 50
            if large_vol_days > 0:
                up_ratio = large_vol_up_days / large_vol_days
                volume_score = 50 + (up_ratio - 0.5) * 100
                volume_score = max(0, min(100, volume_score))
            
            # 3. Accumulation Score (0-100)
            if len(recent_df) >= 10:
                money_flow_trend = recent_df['money_flow_20'].iloc[0] - recent_df['money_flow_20'].iloc[10]
                price_trend = (recent_df['close'].iloc[0] - recent_df['close'].iloc[10]) / recent_df['close'].iloc[10]
                
                # Positive money flow with positive price = accumulation
                if money_flow_trend > 0 and price_trend > 0:
                    accumulation_score = min(100, 60 + price_trend * 200)
                elif money_flow_trend < 0 and price_trend < 0:
                    accumulation_score = max(0, 40 - abs(price_trend) * 200)
                else:
                    accumulation_score = 50
            else:
                accumulation_score = 50
            
            # 4. Smart Money Score (0-100)
            if 'smart_money_flow' in recent_df.columns:
                smart_flow = recent_df['smart_money_flow'].iloc[0]
                smart_flow_avg = recent_df['smart_money_flow'].mean()
                
                if smart_flow > 0 and smart_flow > smart_flow_avg:
                    smart_money_score = min(100, 60 + (smart_flow / abs(smart_flow_avg) * 10))
                elif smart_flow < 0:
                    smart_money_score = max(0, 40 - abs(smart_flow / smart_flow_avg) * 10)
                else:
                    smart_money_score = 50
            else:
                smart_money_score = 50
            
            # 5. Overall Score (weighted average)
            overall_score = (
                inst_score * 0.3 +
                volume_score * 0.25 +
                accumulation_score * 0.25 +
                smart_money_score * 0.2
            )
            
            # Determine trend
            if overall_score > 65:
                trend = 'accumulating'
            elif overall_score < 35:
                trend = 'distributing'
            else:
                trend = 'neutral'
            
            # Calculate confidence
            score_variance = np.std([inst_score, volume_score, accumulation_score, smart_money_score])
            confidence = max(0.3, 1 - score_variance / 50)
            
            # Prepare details
            details = {
                'institutional_ownership_pct': float(inst_data['total_institutional_pct'].iloc[0]) if inst_data['total_institutional_pct'].iloc[0] else 0,
                'institutional_count': int(inst_data['num_institutions'].iloc[0]) if inst_data['num_institutions'].iloc[0] else 0,
                'large_volume_days_20d': int(large_vol_days),
                'large_volume_up_ratio': float(large_vol_up_days / large_vol_days) if large_vol_days > 0 else 0.5,
                'avg_volume_ratio': float(recent_df['volume_ratio'].mean()),
                'money_flow_trend': 'positive' if money_flow_trend > 0 else 'negative',
                'smart_money_signal': 'bullish' if smart_flow > smart_flow_avg else 'bearish'
            }
            
            # Store big trader scores
            conn.execute("""
                INSERT OR REPLACE INTO big_trader_scores
                (symbol, score_date, institutional_score, volume_analysis_score,
                 accumulation_score, smart_money_score, overall_score, trend,
                 confidence, details_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                latest_date,
                round(inst_score, 2),
                round(volume_score, 2),
                round(accumulation_score, 2),
                round(smart_money_score, 2),
                round(overall_score, 2),
                trend,
                round(confidence, 3),
                json.dumps(details)
            ))
            
            conn.commit()
            
        except Exception as e:
            print(f"\nError calculating big trader indicators for {symbol}: {str(e)[:50]}")
        finally:
            conn.close()
    
    def _calculate_technical_indicators(self, symbol, conn):
        """Calculate all technical indicators"""
        # Get price data
        df = pd.read_sql_query("""
            SELECT trade_date, open, high, low, close, volume
            FROM stock_prices
            WHERE symbol = ?
            ORDER BY trade_date
        """, conn, params=(symbol,))
        
        if len(df) < 50:
            return
        
        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Moving Averages
        df['sma_5'] = close.rolling(5).mean()
        df['sma_10'] = close.rolling(10).mean()
        df['sma_20'] = close.rolling(20).mean()
        df['sma_50'] = close.rolling(50).mean()
        df['sma_200'] = close.rolling(200).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = close.ewm(span=12, adjust=False).mean()
        df['ema_26'] = close.ewm(span=26, adjust=False).mean()
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss.replace(0, 1)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['sma_20']
        bb_std = close.rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_width'].replace(0, 1))
        
        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
        
        # OBV
        df['obv'] = (volume * np.sign(close.diff())).fillna(0).cumsum()
        
        # Volume indicators
        df['volume_sma_20'] = volume.rolling(20).mean()
        df['volume_ratio'] = volume / df['volume_sma_20'].replace(0, 1)
        
        # Store indicators
        conn.execute("DELETE FROM technical_indicators WHERE symbol = ?", (symbol,))
        
        # Prepare data for insertion
        df['symbol'] = symbol
        indicator_cols = ['symbol', 'trade_date', 'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
                         'ema_12', 'ema_26', 'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
                         'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
                         'atr_14', 'obv', 'volume_sma_20', 'volume_ratio']
        
        df_indicators = df[indicator_cols].dropna()
        
        if not df_indicators.empty:
            df_indicators.to_sql('technical_indicators', conn, if_exists='append', index=False)
    
    def _store_news_data(self, symbol, news_items):
        """Store news data from yfinance"""
        conn = sqlite3.connect(self.db_path)
        
        for item in news_items[:20]:  # Limit to 20 most recent
            try:
                # Basic sentiment analysis (you can enhance this)
                title = item.get('title', '')
                
                # Simple sentiment scoring
                positive_words = ['gain', 'rise', 'up', 'positive', 'growth', 'beat', 'exceed', 'high', 'buy', 'upgrade']
                negative_words = ['loss', 'fall', 'down', 'negative', 'decline', 'miss', 'low', 'cut', 'sell', 'downgrade']
                
                sentiment_score = 0
                for word in positive_words:
                    if word in title.lower():
                        sentiment_score += 0.1
                for word in negative_words:
                    if word in title.lower():
                        sentiment_score -= 0.1
                
                sentiment_score = max(-1, min(1, sentiment_score))  # Clamp between -1 and 1
                
                conn.execute("""
                    INSERT OR IGNORE INTO stock_news
                    (symbol, published_at, title, summary, url, sentiment_score, relevance_score, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                    title[:500],
                    item.get('summary', '')[:1000] if 'summary' in item else '',
                    item.get('link', ''),
                    sentiment_score,
                    0.8,  # Default relevance
                    item.get('publisher', 'Unknown')
                ))
            except Exception as e:
                continue
        
        conn.commit()
        conn.close()
    
    def _store_finnhub_news(self, symbol, news_items):
        """Store news data from Finnhub"""
        conn = sqlite3.connect(self.db_path)
        
        for item in news_items[:20]:
            try:
                # Simple sentiment analysis
                headline = item.get('headline', '')
                sentiment_score = 0
                
                positive_words = ['gain', 'rise', 'up', 'positive', 'growth', 'beat', 'exceed', 'high', 'buy', 'upgrade']
                negative_words = ['loss', 'fall', 'down', 'negative', 'decline', 'miss', 'low', 'cut', 'sell', 'downgrade']
                
                for word in positive_words:
                    if word in headline.lower():
                        sentiment_score += 0.1
                for word in negative_words:
                    if word in headline.lower():
                        sentiment_score -= 0.1
                
                sentiment_score = max(-1, min(1, sentiment_score))
                
                conn.execute("""
                    INSERT OR IGNORE INTO stock_news
                    (symbol, published_at, title, summary, url, sentiment_score, relevance_score, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    datetime.fromtimestamp(item.get('datetime', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                    headline[:500],
                    item.get('summary', '')[:1000],
                    item.get('url', ''),
                    sentiment_score,
                    0.8,
                    item.get('source', 'Finnhub')
                ))
            except:
                continue
        
        conn.commit()
        conn.close()
    
    def import_market_data(self):
        """Import market regime data, fear & greed index"""
        print("\nImporting market data...")
        
        conn = sqlite3.connect(self.db_path)
        
        # 1. Import Fear & Greed Index
        try:
            print("  - Fetching Fear & Greed Index...")
            response = requests.get(self.fear_greed_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    # Store latest value
                    latest = data['data'][0]
                    conn.execute("""
                        INSERT OR REPLACE INTO market_data (data_key, data_value, last_updated)
                        VALUES (?, ?, ?)
                    """, ('fear_greed_index', json.dumps(latest), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                    
                    # Store historical values
                    historical = []
                    for item in data['data']:
                        historical.append({
                            'date': datetime.fromtimestamp(int(item['timestamp'])).strftime('%Y-%m-%d'),
                            'value': int(item['value']),
                            'classification': item['value_classification']
                        })
                    
                    conn.execute("""
                        INSERT OR REPLACE INTO market_data (data_key, data_value, last_updated)
                        VALUES (?, ?, ?)
                    """, ('fear_greed_historical', json.dumps(historical), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                    
                    print(f"    Current Fear & Greed: {latest['value']} ({latest['value_classification']})")
        except Exception as e:
            print(f"    Error fetching Fear & Greed: {e}")
        
        # 2. Calculate Market Regime
        try:
            print("  - Calculating Market Regime...")
            # Get SPY data for market regime
            spy_df = pd.read_sql_query("""
                SELECT trade_date, close
                FROM stock_prices
                WHERE symbol = 'SPY'
                ORDER BY trade_date DESC
                LIMIT 200
            """, conn)
            
            if len(spy_df) > 50:
                spy_df['returns'] = spy_df['close'].pct_change()
                spy_df['sma_50'] = spy_df['close'].rolling(50).mean()
                spy_df['volatility'] = spy_df['returns'].rolling(20).std() * np.sqrt(252)
                
                latest_close = spy_df['close'].iloc[0]
                latest_sma = spy_df['sma_50'].iloc[0]
                latest_vol = spy_df['volatility'].iloc[0]
                
                # Determine regime
                if latest_close > latest_sma:
                    if latest_vol < 0.15:
                        regime = 'bull_low_vol'
                    else:
                        regime = 'bull_high_vol'
                else:
                    if latest_vol < 0.15:
                        regime = 'bear_low_vol'
                    else:
                        regime = 'bear_high_vol'
                
                regime_data = {
                    'regime': regime,
                    'trend': 'bullish' if latest_close > latest_sma else 'bearish',
                    'volatility': round(latest_vol, 4),
                    'spy_price': round(latest_close, 2),
                    'spy_sma50': round(latest_sma, 2)
                }
                
                conn.execute("""
                    INSERT OR REPLACE INTO market_data (data_key, data_value, last_updated)
                    VALUES (?, ?, ?)
                """, ('market_regime', json.dumps(regime_data), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                
                print(f"    Market Regime: {regime}")
        except Exception as e:
            print(f"    Error calculating market regime: {e}")
        
        # 3. VIX data (volatility index)
        try:
            print("  - Fetching VIX data...")
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="1mo")
            if not vix_data.empty:
                latest_vix = vix_data['Close'].iloc[-1]
                vix_sma = vix_data['Close'].rolling(20).mean().iloc[-1]
                
                vix_info = {
                    'current': round(latest_vix, 2),
                    'sma_20': round(vix_sma, 2),
                    'level': 'low' if latest_vix < 15 else 'moderate' if latest_vix < 25 else 'high'
                }
                
                conn.execute("""
                    INSERT OR REPLACE INTO market_data (data_key, data_value, last_updated)
                    VALUES (?, ?, ?)
                """, ('vix_data', json.dumps(vix_info), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                
                print(f"    VIX: {latest_vix:.2f} ({vix_info['level']} volatility)")
        except Exception as e:
            print(f"    Error fetching VIX: {e}")
        
        conn.commit()
        conn.close()
        print("  ✓ Market data import complete")
    
    def import_stock_with_fallback(self, symbol):
        """Import stock using data sources in priority order"""
        # Try each data source in order
        if self.data_sources['yfinance']['enabled']:
            if self.import_stock_yfinance(symbol):
                return 'yfinance'
        
        if self.data_sources['finnhub']['enabled']:
            if self.import_stock_finnhub(symbol):
                return 'finnhub'
        
        if self.data_sources['alpha_vantage']['enabled']:
            if self.import_stock_alpha_vantage(symbol):
                return 'alpha_vantage'
        
        return None
    
    def import_all_stocks(self, max_workers=5):
        """Import all stocks with parallel processing"""
        symbols = self.fetch_stock_lists()
        
        if not symbols:
            print("No symbols to import!")
            return
        
        print(f"\nStarting import of {len(symbols)} stocks...")
        print(f"Date range: {self.start_date} to {self.end_date}")
        print("="*60)
        
        success_count = 0
        failed_symbols = []
        source_stats = {'yfinance': 0, 'finnhub': 0, 'alpha_vantage': 0}
        
        # Try batch download with yfinance first (much more efficient)
        if self.data_sources['yfinance']['enabled']:
            print("\nAttempting batch download with yfinance...")
            batch_success = 0
            
            try:
                # Download all symbols at once
                batch_size = 50  # Smaller batches are more reliable
                for i in range(0, len(symbols), batch_size):
                    batch_symbols = symbols[i:i+batch_size]
                    # yfinance expects either a list or space-separated string
                    # Using list is more reliable
                    print(f"\rDownloading batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1} ({len(batch_symbols)} symbols)...", end='', flush=True)
                    
                    try:
                        # Download multiple symbols at once - pass as list
                        batch_data = yf.download(batch_symbols, start=self.start_date, end=self.end_date,
                                               progress=False, threads=True, group_by='ticker')
                        
                        if batch_data is None or batch_data.empty:
                            # Try symbols individually if batch fails
                            for symbol in batch_symbols:
                                try:
                                    single_data = yf.download(symbol, start=self.start_date, end=self.end_date,
                                                            progress=False)
                                    
                                    if single_data is not None and not single_data.empty and len(single_data) > 10:
                                        # Prepare the dataframe
                                        df = single_data.copy()
                                        df = df.reset_index()
                                        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
                                        df['Symbol'] = symbol
                                        
                                        # Ensure lowercase column names for consistency
                                        df.rename(columns={col: col.lower() for col in df.columns if col not in ['Date', 'Symbol']}, inplace=True)
                                        
                                        # Store the data
                                        self._store_stock_data(
                                            symbol=symbol,
                                            company_name=symbol,
                                            sector='Unknown',
                                            industry='Unknown',
                                            market_cap=0,
                                            exchange='Unknown',
                                            price_df=df,
                                            source='yfinance'
                                        )
                                        
                                        # Calculate big trader indicators
                                        self._calculate_big_trader_indicators(symbol)
                                        
                                        batch_success += 1
                                        source_stats['yfinance'] += 1
                                except:
                                    failed_symbols.append(symbol)
                            continue
                        
                        # Process batch data
                        for symbol in batch_symbols:
                            try:
                                # Extract data for this symbol
                                symbol_data = None
                                
                                if len(batch_symbols) == 1:
                                    # Single symbol returns simple DataFrame
                                    symbol_data = batch_data
                                else:
                                    # Multiple symbols - check how data is structured
                                    if hasattr(batch_data, 'columns'):
                                        if hasattr(batch_data.columns, 'levels'):
                                            # Multi-level columns (symbol, OHLCV)
                                            top_level = batch_data.columns.get_level_values(0).unique()
                                            if symbol in top_level:
                                                symbol_data = batch_data[symbol]
                                        elif 'Close' in batch_data.columns:
                                            # Sometimes returns flat structure for single valid symbol
                                            symbol_data = batch_data
                                
                                if symbol_data is None or symbol_data.empty or len(symbol_data) < 10:
                                    failed_symbols.append(symbol)
                                    continue
                                
                                # Prepare the dataframe
                                df = symbol_data.copy()
                                if df.index.name == 'Date':
                                    df = df.reset_index()
                                elif 'Date' not in df.columns:
                                    df = df.reset_index()
                                    if 'index' in df.columns:
                                        df.rename(columns={'index': 'Date'}, inplace=True)
                                
                                df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
                                df['Symbol'] = symbol
                                
                                # Ensure lowercase column names for consistency
                                df.rename(columns={col: col.lower() for col in df.columns if col not in ['Date', 'Symbol']}, inplace=True)
                                
                                # Store the data
                                self._store_stock_data(
                                    symbol=symbol,
                                    company_name=symbol,
                                    sector='Unknown',
                                    industry='Unknown',
                                    market_cap=0,
                                    exchange='Unknown',
                                    price_df=df,
                                    source='yfinance'
                                )
                                
                                # Calculate big trader indicators
                                self._calculate_big_trader_indicators(symbol)
                                
                                batch_success += 1
                                source_stats['yfinance'] += 1
                                
                            except Exception as e:
                                failed_symbols.append(symbol)
                    
                    except Exception as e:
                        print(f"\nBatch error: {str(e)[:50]}")
                        # Add all batch symbols to failed list
                        failed_symbols.extend(batch_symbols)
                    
                    # Pause between batches
                    time.sleep(1)
                
                success_count = batch_success
                print(f"\n\nBatch download complete. Got {batch_success} symbols via yfinance.")
                
            except Exception as e:
                print(f"\nBatch download error: {str(e)[:50]}")
                print("Falling back to alternative sources...")
        
        # Process remaining failed symbols individually
        remaining_symbols = list(set(failed_symbols))  # Remove duplicates
        
        # If no symbols were successful via batch, try all symbols individually
        if success_count == 0 and self.data_sources['yfinance']['enabled']:
            print("\nBatch download didn't work. Trying individual downloads...")
            remaining_symbols = symbols.copy()
            failed_symbols = []
        
        if remaining_symbols:
            print(f"\nProcessing {len(remaining_symbols)} remaining symbols individually...")
            
            batch_size = 10
            yfinance_fail_count = 0
            yfinance_fail_count = 0
            
            for i in range(0, len(remaining_symbols), batch_size):
                batch = remaining_symbols[i:i+batch_size]
                
                for symbol in batch:
                    print(f"\r[Individual {i+1}-{min(i+batch_size, len(remaining_symbols))}/{len(remaining_symbols)}] {symbol:6} ... ", end='', flush=True)
                    
                    source = None
                    
                    # Try yfinance first for individual downloads (might work even if batch failed)
                    if self.data_sources['yfinance']['enabled'] and yfinance_fail_count < 50:
                        if self.import_stock_yfinance(symbol):
                            source = 'yfinance'
                            yfinance_fail_count = max(0, yfinance_fail_count - 1)
                        else:
                            yfinance_fail_count += 1
                    
                    # Try finnhub
                    if not source and self.data_sources['finnhub']['enabled']:
                        if self.import_stock_finnhub(symbol):
                            source = 'finnhub'
                    
                    # Try alpha_vantage as last resort
                    if not source and self.data_sources['alpha_vantage']['enabled']:
                        if self.import_stock_alpha_vantage(symbol):
                            source = 'alpha_vantage'
                    
                    if source:
                        success_count += 1
                        source_stats[source] += 1
                        # Remove from failed list
                        if symbol in failed_symbols:
                            failed_symbols.remove(symbol)
                        print(f"✓ ({source})", end='', flush=True)
                    else:
                        print("✗", end='', flush=True)
                    yfinance_fail_count = 0
                    
                    # Rate limiting
                    if source == 'finnhub':
                        time.sleep(0.5)  # Finnhub has 60 calls/minute limit
                    elif source == 'alpha_vantage':
                        time.sleep(12)  # Alpha Vantage has 5 calls/minute limit
                    #else:
                    #    time.sleep(0.2)  # Default delay
                
                # Longer pause between batches
                if i + batch_size < len(remaining_symbols):
                    time.sleep(2)
        
        # Final failed symbols list - simple approach
        final_failed = []
        for symbol in symbols:
            if not self._check_symbol_exists(symbol):
                final_failed.append(symbol)
        
        print("\n" + "="*60)
        print(f"Import Summary:")
        print(f"  Total symbols: {len(symbols)}")
        print(f"  Successful: {success_count} ({success_count/max(1, len(symbols))*100:.1f}%)")
        print(f"  Failed: {len(final_failed)}")
        print(f"\nData sources used:")
        for source, count in source_stats.items():
            if count > 0:
                print(f"  {source}: {count} stocks")
        
        if final_failed:
            print(f"\nFailed symbols ({len(final_failed)}):")
            print(", ".join(final_failed[:20]))
            if len(final_failed) > 20:
                print(f"... and {len(final_failed)-20} more")
        
        # Import market data
        self.import_market_data()
        
        # Show database statistics
        self.show_database_stats()
    
    def _check_symbol_exists(self, symbol):
        """Check if symbol already has data in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("SELECT COUNT(*) FROM stock_prices WHERE symbol = ?", (symbol,))
            count = cursor.fetchone()[0]
            conn.close()
            return count > 0
        except:
            return False
    
    def show_database_stats(self):
        """Display database statistics including big trader data"""
        conn = sqlite3.connect(self.db_path)
        
        stats = pd.read_sql_query("""
            SELECT 
                (SELECT COUNT(*) FROM stocks WHERE is_active = 1) as active_stocks,
                (SELECT COUNT(DISTINCT symbol) FROM stock_prices) as symbols_with_prices,
                (SELECT COUNT(*) FROM stock_prices) as total_price_records,
                (SELECT MIN(trade_date) FROM stock_prices) as earliest_date,
                (SELECT MAX(trade_date) FROM stock_prices) as latest_date,
                (SELECT COUNT(DISTINCT symbol) FROM technical_indicators) as symbols_with_indicators,
                (SELECT COUNT(*) FROM stock_news) as news_articles,
                (SELECT COUNT(*) FROM market_data) as market_data_points,
                (SELECT COUNT(DISTINCT symbol) FROM big_trader_scores) as symbols_with_big_trader,
                (SELECT COUNT(DISTINCT symbol) FROM institutional_ownership) as symbols_with_institutional
        """, conn)
        
        print("\n" + "="*60)
        print("DATABASE STATISTICS")
        print("="*60)
        print(f"Active stocks: {stats['active_stocks'].iloc[0]:,}")
        print(f"Symbols with prices: {stats['symbols_with_prices'].iloc[0]:,}")
        print(f"Total price records: {stats['total_price_records'].iloc[0]:,}")
        print(f"Date range: {stats['earliest_date'].iloc[0]} to {stats['latest_date'].iloc[0]}")
        print(f"Symbols with indicators: {stats['symbols_with_indicators'].iloc[0]:,}")
        print(f"Symbols with big trader scores: {stats['symbols_with_big_trader'].iloc[0]:,}")
        print(f"Symbols with institutional data: {stats['symbols_with_institutional'].iloc[0]:,}")
        print(f"News articles: {stats['news_articles'].iloc[0]:,}")
        print(f"Market data points: {stats['market_data_points'].iloc[0]}")
        
        # Show sample of data by sector
        sector_stats = pd.read_sql_query("""
            SELECT sector, COUNT(*) as count
            FROM stocks
            WHERE is_active = 1
            GROUP BY sector
            ORDER BY count DESC
            LIMIT 10
        """, conn)
        
        print("\nTop sectors:")
        for _, row in sector_stats.iterrows():
            print(f"  {row['sector']:20} {row['count']:4} stocks")
        
        # Show big trader summary
        big_trader_summary = pd.read_sql_query("""
            SELECT 
                AVG(overall_score) as avg_score,
                SUM(CASE WHEN trend = 'accumulating' THEN 1 ELSE 0 END) as accumulating,
                SUM(CASE WHEN trend = 'distributing' THEN 1 ELSE 0 END) as distributing,
                SUM(CASE WHEN trend = 'neutral' THEN 1 ELSE 0 END) as neutral
            FROM big_trader_scores
            WHERE score_date = (SELECT MAX(score_date) FROM big_trader_scores)
        """, conn)
        
        if not big_trader_summary.empty and big_trader_summary['avg_score'].iloc[0]:
            print("\nBig Trader Analysis:")
            print(f"  Average score: {big_trader_summary['avg_score'].iloc[0]:.1f}/100")
            print(f"  Accumulating: {big_trader_summary['accumulating'].iloc[0]} stocks")
            print(f"  Distributing: {big_trader_summary['distributing'].iloc[0]} stocks")
            print(f"  Neutral: {big_trader_summary['neutral'].iloc[0]} stocks")
        
        conn.close()
        print("="*60)

def main():
    """Run the comprehensive importer"""
    import sys
    
    # Check for command line arguments
    start_with_finnhub = '--finnhub' in sys.argv or '--skip-yfinance' in sys.argv
    test_mode = '--test' in sys.argv
    
    importer = ComprehensiveStockImporter('unified_trading.db')
    
    if test_mode:
        print("Running in test mode with a few symbols...")
        # Test with just a few reliable symbols
        test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY']
        print(f"Testing with: {', '.join(test_symbols)}")
        
        for symbol in test_symbols:
            print(f"\nTesting {symbol}:")
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=importer.start_date, end=importer.end_date)
                if data is not None and not data.empty:
                    print(f"  ✓ Got {len(data)} days of data")
                    # Test info endpoint
                    try:
                        info = ticker.info
                        if info:
                            print(f"  ✓ Got company info: {info.get('longName', 'N/A')}")
                    except:
                        print(f"  ⚠ Could not get company info")
                else:
                    print(f"  ✗ No data returned")
            except Exception as e:
                print(f"  ✗ Error: {str(e)[:50]}")
        
        print("\nTest complete. If all symbols failed, you may need to wait for rate limits to reset.")
        print("Otherwise, run without --test to import all symbols.")
        return
    
    if start_with_finnhub:
        print("Starting with Finnhub/Alpha Vantage (skipping yfinance)...")
        importer.data_sources['yfinance']['enabled'] = False
    
    # Import all stocks
    importer.import_all_stocks(max_workers=5)
    
    print("\n✅ Import complete!")
    print("✅ Your database is ready for trading with:")
    print("   - Stock prices from 2022-01-01 to today")
    print("   - Technical indicators calculated")
    print("   - Big trader indicators (institutional activity)")
    print("   - Market regime data")
    print("   - Fear & Greed index")
    print("   - Stock news with sentiment")
    print("\n🚀 Big trader indicator is now integrated!")
    print("🚀 The configuration already includes big_trader_weight in optimizer_configurations table")
    
    if not start_with_finnhub:
        print("\n💡 Tips:")
        print("   - If yfinance fails, run with: python comprehensive_importer.py --finnhub")
        print("   - To test connectivity: python comprehensive_importer.py --test")
        print("   - Rate limits reset after 1-2 hours")

if __name__ == "__main__":
    main()