BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS "backtest_results" (
	"backtest_id"	INTEGER,
	"run_date"	TEXT DEFAULT CURRENT_TIMESTAMP,
	"start_date"	TEXT NOT NULL,
	"end_date"	TEXT NOT NULL,
	"initial_capital"	REAL NOT NULL,
	"final_capital"	REAL NOT NULL,
	"total_profit"	REAL NOT NULL,
	"total_return_pct"	REAL NOT NULL,
	"annual_return_pct"	REAL,
	"sharpe_ratio"	REAL,
	"max_drawdown_pct"	REAL,
	"profit_factor"	REAL,
	"winning_trades"	INTEGER,
	"losing_trades"	INTEGER,
	"win_rate"	REAL,
	"avg_win"	REAL,
	"avg_loss"	REAL,
	"avg_profit_per_trade"	REAL,
	"best_trade_return_pct"	REAL,
	"worst_trade_return_pct"	REAL,
	"avg_holding_days"	REAL,
	"configuration"	TEXT,
	"monthly_returns_json"	TEXT,
	PRIMARY KEY("backtest_id" AUTOINCREMENT)
);
CREATE TABLE IF NOT EXISTS "backtest_trades" (
	"trade_id"	INTEGER,
	"backtest_id"	INTEGER NOT NULL,
	"symbol"	TEXT NOT NULL,
	"entry_date"	TEXT NOT NULL,
	"exit_date"	TEXT NOT NULL,
	"entry_price"	REAL NOT NULL,
	"exit_price"	REAL NOT NULL,
	"shares"	INTEGER NOT NULL,
	"profit_loss"	REAL NOT NULL,
	"return_pct"	REAL NOT NULL,
	"holding_days"	INTEGER NOT NULL,
	"exit_reason"	TEXT,
	"entry_score"	REAL,
	"exit_score"	REAL,
	"entry_position_value"	REAL,
	"exit_position_value"	REAL,
	"gross_profit"	REAL,
	"net_profit"	REAL,
	"commission_paid"	REAL,
	"trade_type"	TEXT,
	"peak_return_pct"	REAL,
	"trade_exit_reasons"	TEXT,
	PRIMARY KEY("trade_id" AUTOINCREMENT),
	FOREIGN KEY("backtest_id") REFERENCES "backtest_results"("backtest_id"),
	FOREIGN KEY("symbol") REFERENCES "stocks"("symbol")
);
CREATE TABLE IF NOT EXISTS "big_trader_scores" (
	"symbol"	TEXT,
	"score_date"	TEXT,
	"institutional_score"	REAL,
	"volume_analysis_score"	REAL,
	"accumulation_score"	REAL,
	"smart_money_score"	REAL,
	"overall_score"	REAL,
	"trend"	TEXT,
	"confidence"	REAL,
	"details_json"	TEXT,
	PRIMARY KEY("symbol","score_date")
);
CREATE TABLE IF NOT EXISTS "indicators" (
	"symbol"	TEXT,
	"trade_date"	TEXT,
	"rsi"	REAL,
	"macd"	REAL,
	"macd_signal"	REAL,
	"macd_hist"	REAL,
	"bollinger_upper"	REAL,
	"bollinger_middle"	REAL,
	"bollinger_lower"	REAL,
	"sma_50"	REAL,
	"sma_200"	REAL,
	PRIMARY KEY("symbol","trade_date"),
	FOREIGN KEY("symbol") REFERENCES "stocks"("symbol")
);
CREATE TABLE IF NOT EXISTS "institutional_ownership" (
	"symbol"	TEXT,
	"report_date"	TEXT,
	"institution_name"	TEXT,
	"shares_held"	INTEGER,
	"value_held"	REAL,
	"percent_outstanding"	REAL,
	"change_in_shares"	INTEGER,
	"change_percent"	REAL,
	PRIMARY KEY("symbol","report_date","institution_name")
);
CREATE TABLE IF NOT EXISTS "large_trades" (
	"symbol"	TEXT,
	"trade_date"	TEXT,
	"large_volume_ratio"	REAL,
	"block_trade_count"	INTEGER,
	"avg_trade_size"	REAL,
	"institutional_volume_estimate"	INTEGER,
	"retail_volume_estimate"	INTEGER,
	"smart_money_flow"	REAL,
	PRIMARY KEY("symbol","trade_date")
);
CREATE TABLE IF NOT EXISTS "market_data" (
	"data_key"	TEXT,
	"data_value"	TEXT,
	"last_updated"	TEXT,
	PRIMARY KEY("data_key")
);
CREATE TABLE IF NOT EXISTS "optimizer_configurations" (
	"id"	INTEGER,
	"technical_weight"	REAL NOT NULL CHECK("technical_weight" >= 0 AND "technical_weight" <= 1),
	"ai_weight"	REAL NOT NULL CHECK("ai_weight" >= 0 AND "ai_weight" <= 1),
	"sentiment_weight"	REAL NOT NULL CHECK("sentiment_weight" >= 0 AND "sentiment_weight" <= 1),
	"regime_weight"	REAL NOT NULL CHECK("regime_weight" >= 0 AND "regime_weight" <= 1),
	"big_trader_weight"	REAL NOT NULL CHECK("big_trader_weight" >= 0 AND "big_trader_weight" <= 1),
	"fear_greed_weight"	REAL NOT NULL CHECK("fear_greed_weight" >= 0 AND "fear_greed_weight" <= 1),
	"min_buy_score"	REAL NOT NULL CHECK("min_buy_score" >= 0 AND "min_buy_score" <= 100),
	"min_strong_buy_score"	REAL NOT NULL CHECK("min_strong_buy_score" >= 0 AND "min_strong_buy_score" <= 100),
	"max_sell_score"	REAL NOT NULL CHECK("max_sell_score" >= 0 AND "max_sell_score" <= 100),
	"max_strong_sell_score"	REAL NOT NULL CHECK("max_strong_sell_score" >= 0 AND "max_strong_sell_score" <= 100),
	"min_confidence"	REAL NOT NULL CHECK("min_confidence" >= 0 AND "min_confidence" <= 1),
	"base_profit_target"	REAL NOT NULL CHECK("base_profit_target" > 0 AND "base_profit_target" <= 1),
	"stop_loss_pct"	REAL NOT NULL CHECK("stop_loss_pct" > 0 AND "stop_loss_pct" <= 0.5),
	"trailing_stop_pct"	REAL NOT NULL CHECK("trailing_stop_pct" > 0 AND "trailing_stop_pct" <= 0.5),
	"take_profit_multiplier"	REAL NOT NULL CHECK("take_profit_multiplier" >= 1 AND "take_profit_multiplier" <= 5),
	"base_position_size"	REAL NOT NULL CHECK("base_position_size" > 0 AND "base_position_size" <= 1),
	"max_position_size"	REAL NOT NULL CHECK("max_position_size" > 0 AND "max_position_size" <= 1),
	"kelly_fraction"	REAL NOT NULL CHECK("kelly_fraction" > 0 AND "kelly_fraction" <= 1),
	"min_market_cap"	REAL NOT NULL CHECK("min_market_cap" >= 0),
	"min_price"	REAL NOT NULL CHECK("min_price" >= 0),
	"min_volume"	INTEGER NOT NULL CHECK("min_volume" >= 0),
	"max_positions"	INTEGER NOT NULL CHECK("max_positions" > 0),
	"institutional_volume_multiplier"	REAL NOT NULL CHECK("institutional_volume_multiplier" >= 1),
	"accumulation_threshold"	REAL NOT NULL CHECK("accumulation_threshold" >= 0 AND "accumulation_threshold" <= 1),
	"distribution_threshold"	REAL NOT NULL CHECK("distribution_threshold" >= 0 AND "distribution_threshold" <= 1),
	"rsi_oversold_threshold"	REAL NOT NULL DEFAULT 30 CHECK("rsi_oversold_threshold" >= 0 AND "rsi_oversold_threshold" <= 50),
	"rsi_overbought_threshold"	REAL NOT NULL DEFAULT 70 CHECK("rsi_overbought_threshold" >= 50 AND "rsi_overbought_threshold" <= 100),
	"macd_signal_threshold"	REAL NOT NULL DEFAULT 0,
	"ai_momentum_weight"	REAL NOT NULL DEFAULT 0.4 CHECK("ai_momentum_weight" >= 0 AND "ai_momentum_weight" <= 1),
	"ai_reversion_weight"	REAL NOT NULL DEFAULT 0.3 CHECK("ai_reversion_weight" >= 0 AND "ai_reversion_weight" <= 1),
	"ai_pattern_weight"	REAL NOT NULL DEFAULT 0.3 CHECK("ai_pattern_weight" >= 0 AND "ai_pattern_weight" <= 1),
	"ai_max_prediction"	REAL NOT NULL DEFAULT 0.2 CHECK("ai_max_prediction" > 0 AND "ai_max_prediction" <= 1),
	"news_lookback_days"	INTEGER NOT NULL DEFAULT 7 CHECK("news_lookback_days" >= 1 AND "news_lookback_days" <= 30),
	"news_relevance_threshold"	REAL NOT NULL DEFAULT 0.5 CHECK("news_relevance_threshold" >= 0 AND "news_relevance_threshold" <= 1),
	"regime_trend_threshold"	REAL NOT NULL DEFAULT 0.7 CHECK("regime_trend_threshold" >= 0 AND "regime_trend_threshold" <= 1),
	"regime_volatility_threshold"	REAL NOT NULL DEFAULT 1.5 CHECK("regime_volatility_threshold" > 0),
	"momentum_short_period"	INTEGER NOT NULL DEFAULT 5 CHECK("momentum_short_period" >= 1),
	"momentum_long_period"	INTEGER NOT NULL DEFAULT 20 CHECK("momentum_long_period" > "momentum_short_period"),
	"sum_revenues"	REAL DEFAULT 0,
	"total_trades"	INTEGER DEFAULT 0,
	"win_rate"	REAL DEFAULT 0,
	"avg_profit_per_trade"	REAL DEFAULT 0,
	"max_drawdown"	REAL DEFAULT 0,
	"sharpe_ratio"	REAL DEFAULT 0,
	"is_best"	BOOLEAN DEFAULT FALSE,
	"optimization_date"	DATETIME DEFAULT CURRENT_TIMESTAMP,
	"backtest_start_date"	DATE,
	"backtest_end_date"	DATE,
	PRIMARY KEY("id" AUTOINCREMENT),
	CHECK("technical_weight" + "ai_weight" + "sentiment_weight" + "regime_weight" + "big_trader_weight" + "fear_greed_weight" BETWEEN 0.99 AND 1.01),
	CHECK("min_buy_score" < "min_strong_buy_score"),
	CHECK("max_strong_sell_score" < "max_sell_score"),
	CHECK("base_position_size" < "max_position_size"),
	CHECK("ai_momentum_weight" + "ai_reversion_weight" + "ai_pattern_weight" BETWEEN 0.99 AND 1.01)
);
CREATE TABLE IF NOT EXISTS "signals" (
	"signal_id"	INTEGER,
	"symbol"	TEXT NOT NULL,
	"signal_date"	TEXT NOT NULL,
	"action"	TEXT NOT NULL,
	"current_price"	REAL NOT NULL,
	"market_cap"	REAL,
	"technical_score"	REAL,
	"ai_prediction_score"	REAL,
	"sentiment_score"	REAL,
	"regime_score"	REAL,
	"big_trader_score"	REAL,
	"fear_greed_score"	REAL,
	"momentum_score"	REAL,
	"final_score"	REAL NOT NULL,
	"confidence"	REAL,
	"profit_potential"	REAL,
	"stop_loss_price"	REAL,
	"target_price"	REAL,
	"position_size_pct"	REAL,
	"max_holding_days"	INTEGER,
	"entry_price"	REAL,
	"pnl"	REAL,
	"holding_days"	INTEGER,
	"action_reasons"	TEXT,
	"created_at"	TEXT DEFAULT CURRENT_TIMESTAMP,
	"market_regime_details_json"	TEXT,
	"big_trader_activity_json"	TEXT,
	"sentiment_details_json"	TEXT,
	"ai_details_json"	TEXT,
	"technical_details_json"	TEXT,
	"momentum_details_json"	TEXT,
	"fear_greed_details_json"	TEXT,
	PRIMARY KEY("signal_id" AUTOINCREMENT),
	FOREIGN KEY("symbol") REFERENCES "stocks"("symbol")
);
CREATE TABLE IF NOT EXISTS "sqlite_stat4" (
	"tbl"	,
	"idx"	,
	"neq"	,
	"nlt"	,
	"ndlt"	,
	"sample"	
);
CREATE TABLE IF NOT EXISTS "stock_discoveries" (
	"symbol"	TEXT,
	"discovery_type"	TEXT,
	"discovery_date"	TEXT,
	"imported"	INTEGER DEFAULT 0,
	PRIMARY KEY("symbol","discovery_type","discovery_date")
);
CREATE TABLE IF NOT EXISTS "stock_news" (
	"id"	INTEGER,
	"symbol"	TEXT,
	"published_at"	TEXT,
	"title"	TEXT,
	"summary"	TEXT,
	"url"	TEXT,
	"sentiment_score"	REAL DEFAULT 0,
	"relevance_score"	REAL DEFAULT 0.5,
	"source"	TEXT,
	"created_at"	DATETIME DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY("id" AUTOINCREMENT)
);
CREATE TABLE IF NOT EXISTS "stock_prices" (
	"symbol"	TEXT,
	"trade_date"	TEXT,
	"open"	REAL,
	"high"	REAL,
	"low"	REAL,
	"close"	REAL,
	"adj_close"	REAL,
	"volume"	INTEGER,
	"data_source"	TEXT,
	PRIMARY KEY("symbol","trade_date"),
	FOREIGN KEY("symbol") REFERENCES "stocks"("symbol")
);
CREATE TABLE IF NOT EXISTS "stocks" (
	"symbol"	TEXT NOT NULL,
	"company_name"	TEXT,
	"exchange"	TEXT,
	"sector"	TEXT,
	"industry"	TEXT,
	"market_cap"	REAL,
	"is_active"	INTEGER DEFAULT 1,
	"last_updated"	TEXT,
	"is_delisted"	INTEGER DEFAULT 0,
	"last_error"	TEXT,
	"error_count"	INTEGER DEFAULT 0,
	PRIMARY KEY("symbol")
);
CREATE TABLE IF NOT EXISTS "stocks_data" (
	"symbol"	TEXT NOT NULL,
	"trade_date"	TEXT NOT NULL,
	"open_price"	REAL,
	"high_price"	REAL,
	"low_price"	REAL,
	"close_price"	REAL NOT NULL,
	"adj_close_price"	REAL,
	"volume"	INTEGER,
	PRIMARY KEY("symbol","trade_date")
);
CREATE TABLE IF NOT EXISTS "symbol_universe" (
	"symbol"	TEXT,
	"added_date"	TEXT,
	"last_seen"	TEXT,
	"is_active"	INTEGER DEFAULT 1,
	PRIMARY KEY("symbol")
);
CREATE TABLE IF NOT EXISTS "technical_indicators" (
	"symbol"	TEXT,
	"trade_date"	TEXT,
	"sma_5"	REAL,
	"sma_10"	REAL,
	"sma_20"	REAL,
	"sma_50"	REAL,
	"sma_200"	REAL,
	"ema_12"	REAL,
	"ema_26"	REAL,
	"rsi_14"	REAL,
	"macd"	REAL,
	"macd_signal"	REAL,
	"macd_histogram"	REAL,
	"bb_upper"	REAL,
	"bb_middle"	REAL,
	"bb_lower"	REAL,
	"bb_width"	REAL,
	"bb_position"	REAL,
	"atr_14"	REAL,
	"obv"	REAL,
	"volume_sma_20"	REAL,
	"volume_ratio"	REAL,
	PRIMARY KEY("symbol","trade_date")
);
CREATE TABLE IF NOT EXISTS "user_positions" (
	"position_id"	INTEGER,
	"portfolio_id"	INTEGER NOT NULL,
	"symbol"	TEXT NOT NULL,
	"position_type"	TEXT DEFAULT 'long',
	"shares"	REAL NOT NULL,
	"avg_entry_price"	REAL NOT NULL,
	"current_price"	REAL,
	"market_value"	REAL,
	"cost_basis"	REAL NOT NULL,
	"unrealized_pnl"	REAL DEFAULT 0,
	"unrealized_pnl_pct"	REAL DEFAULT 0,
	"realized_pnl"	REAL DEFAULT 0,
	"realized_pnl_pct"	REAL DEFAULT 0,
	"stop_loss_price"	REAL,
	"take_profit_price"	REAL,
	"margin_used"	REAL DEFAULT 0,
	"entry_date"	TEXT NOT NULL,
	"exit_date"	TEXT,
	"exit_price"	REAL,
	"entry_signal_id"	INTEGER,
	"exit_signal_id"	INTEGER,
	"status"	TEXT DEFAULT 'open',
	"last_updated"	TEXT DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY("position_id" AUTOINCREMENT),
	FOREIGN KEY("entry_signal_id") REFERENCES "trading_signals"("signal_id"),
	FOREIGN KEY("exit_signal_id") REFERENCES "trading_signals"("signal_id"),
	FOREIGN KEY("portfolio_id") REFERENCES "user_portfolios"("portfolio_id"),
	FOREIGN KEY("symbol") REFERENCES "stocks"("symbol")
);
CREATE VIEW best_configuration AS
SELECT * FROM optimizer_configurations
WHERE is_best = TRUE
ORDER BY id DESC
LIMIT 1;
CREATE VIEW top_configurations AS
SELECT 
    id,
    sum_revenues,
    total_trades,
    win_rate,
    avg_profit_per_trade,
    sharpe_ratio,
    max_drawdown,
    ROUND(big_trader_weight * 100, 1) || '%' as big_trader_pct,
    ROUND(ai_weight * 100, 1) || '%' as ai_pct,
    min_buy_score,
    base_profit_target,
    stop_loss_pct,
    optimization_date
FROM optimizer_configurations
WHERE sum_revenues > 0
ORDER BY sum_revenues DESC
LIMIT 20;
CREATE VIEW v_big_trader_insights AS
SELECT 
    b.symbol,
    s.company_name,
    s.sector,
    b.overall_score,
    b.trend,
    b.confidence,
    p.close as current_price,
    p.volume,
    json_extract(b.details_json, '$.institutional_ownership_pct') as inst_ownership_pct,
    json_extract(b.details_json, '$.smart_money_signal') as smart_money_signal
FROM big_trader_scores b
JOIN stocks s ON b.symbol = s.symbol
JOIN (
    SELECT symbol, close, volume, trade_date 
    FROM stock_prices 
    WHERE (symbol, trade_date) IN (
        SELECT symbol, MAX(trade_date) 
        FROM stock_prices 
        GROUP BY symbol
    )
) p ON b.symbol = p.symbol
WHERE b.score_date = (SELECT MAX(score_date) FROM big_trader_scores WHERE symbol = b.symbol);
CREATE VIEW v_data_quality AS
SELECT 
    s.symbol,
    s.company_name,
    s.is_active,
    s.is_delisted,
    s.error_count,
    COUNT(DISTINCT p.trade_date) as price_days,
    MAX(p.trade_date) as latest_price_date,
    COUNT(DISTINCT i.trade_date) as indicator_days
FROM stocks s
LEFT JOIN stock_prices p ON s.symbol = p.symbol
LEFT JOIN technical_indicators i ON s.symbol = i.symbol
GROUP BY s.symbol;
CREATE VIEW v_problematic_symbols AS
SELECT symbol, company_name, last_error, error_count, last_updated
FROM stocks
WHERE error_count > 0 OR is_delisted = 1
ORDER BY error_count DESC;
CREATE INDEX IF NOT EXISTS "idx_backtest_trades_backtest_id" ON "backtest_trades" (
	"backtest_id"
);
CREATE INDEX IF NOT EXISTS "idx_big_trader_scores" ON "big_trader_scores" (
	"symbol",
	"score_date"
);
CREATE INDEX IF NOT EXISTS "idx_indicators" ON "technical_indicators" (
	"symbol",
	"trade_date"
);
CREATE INDEX IF NOT EXISTS "idx_large_trades" ON "large_trades" (
	"symbol",
	"trade_date"
);
CREATE INDEX IF NOT EXISTS "idx_optimizer_best" ON "optimizer_configurations" (
	"is_best"
);
CREATE INDEX IF NOT EXISTS "idx_optimizer_revenues" ON "optimizer_configurations" (
	"sum_revenues"	DESC
);
CREATE INDEX IF NOT EXISTS "idx_optimizer_sharpe" ON "optimizer_configurations" (
	"sharpe_ratio"	DESC
);
CREATE INDEX IF NOT EXISTS "idx_prices" ON "stock_prices" (
	"symbol",
	"trade_date"
);
CREATE INDEX IF NOT EXISTS "idx_signals_symbol_date" ON "signals" (
	"symbol",
	"signal_date"
);
CREATE INDEX IF NOT EXISTS "idx_stock_prices_symbol_date" ON "stock_prices" (
	"symbol",
	"trade_date"	DESC
);
CREATE INDEX IF NOT EXISTS "idx_stocks_active" ON "stocks" (
	"is_active",
	"is_delisted"
);
CREATE INDEX IF NOT EXISTS "idx_stocks_data_symbol_date" ON "stocks_data" (
	"symbol",
	"trade_date"
);
CREATE INDEX IF NOT EXISTS "idx_technical_indicators_symbol_date" ON "technical_indicators" (
	"symbol",
	"trade_date"	DESC
);
COMMIT;
