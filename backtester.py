#!/usr/bin/env python3
"""
BACKTEST MODULE - Historical Performance Analysis
Tests trading strategies over 3 years with detailed metrics
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json
import logging
from collections import defaultdict

# Import the signal generator
from signal_generator import MaxProfitSignalGenerator, MaxProfitSignal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Backtest')

@dataclass
class Position:
    """Represents a stock position"""
    symbol: str
    shares: int
    entry_price: float
    entry_date: str
    entry_signal_score: float
    stop_loss: float
    take_profit: float
    
    def current_value(self, current_price: float) -> float:
        return self.shares * current_price
    
    def unrealized_pnl(self, current_price: float) -> float:
        return self.shares * (current_price - self.entry_price)
    
    def realized_pnl(self, exit_price: float, trade_fee: float = 1.0) -> float:
        gross_pnl = self.shares * (exit_price - self.entry_price)
        return gross_pnl - (trade_fee * 2)  # Entry + Exit fees

@dataclass
class Trade:
    """Completed trade record"""
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: int
    gross_pnl: float
    net_pnl: float  # After fees
    return_pct: float
    holding_days: int
    entry_score: float
    exit_score: float
    exit_reason: str  # 'signal', 'stop_loss', 'take_profit'

@dataclass
class BacktestResults:
    """Complete backtest results"""
    # Summary metrics
    total_return: float
    annual_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # P&L metrics
    gross_pnl: float
    total_fees: float
    net_pnl: float
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_duration_days: int
    sharpe_ratio: float
    volatility: float
    
    # Trade statistics
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    avg_holding_days: float
    max_holding_days: int
    
    # Position metrics
    max_positions_held: int
    max_portfolio_value: float
    avg_position_size: float
    
    # Best/Worst trades
    best_trade: Optional[Trade]
    worst_trade: Optional[Trade]
    
    # Time series data
    daily_returns: List[float] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    
    # Configuration
    config_id: int = None
    start_date: str = None
    end_date: str = None
    
    def get_summary_report(self) -> str:
        """Generate comprehensive summary report"""
        return f"""
{'='*80}
BACKTEST RESULTS - Configuration ID: {self.config_id}
Period: {self.start_date} to {self.end_date}
{'='*80}

RETURNS:
Total Return: {self.total_return:.2%}
Annual Return: {self.annual_return:.2%}
Sharpe Ratio: {self.sharpe_ratio:.2f}
Max Drawdown: {self.max_drawdown:.2%} ({self.max_drawdown_duration_days} days)

TRADING STATISTICS:
Total Trades: {self.total_trades}
Win Rate: {self.win_rate:.1%} ({self.winning_trades}W / {self.losing_trades}L)
Avg Win: ${self.avg_win:,.2f}
Avg Loss: ${self.avg_loss:,.2f}
Win/Loss Ratio: {self.win_loss_ratio:.2f}

P&L BREAKDOWN:
Gross P&L: ${self.gross_pnl:,.2f}
Trading Fees: ${self.total_fees:,.2f}
Net P&L: ${self.net_pnl:,.2f}

POSITION METRICS:
Max Positions Held: {self.max_positions_held}
Max Portfolio Value: ${self.max_portfolio_value:,.2f}
Avg Position Size: ${self.avg_position_size:,.2f}
Avg Holding Days: {self.avg_holding_days:.1f}
Max Holding Days: {self.max_holding_days}

BEST TRADE:
{self._format_trade(self.best_trade) if self.best_trade else 'N/A'}

WORST TRADE:
{self._format_trade(self.worst_trade) if self.worst_trade else 'N/A'}
{'='*80}
"""
    
    def _format_trade(self, trade: Trade) -> str:
        return f"{trade.symbol}: ${trade.net_pnl:,.2f} ({trade.return_pct:.1%}) in {trade.holding_days} days"

class Portfolio:
    """Manages positions and tracks performance"""
    
    def __init__(self, initial_capital: float = 100000, max_positions: int = 25):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.max_positions = max_positions
        self.trades: List[Trade] = []
        self.daily_values: List[Tuple[str, float]] = []
        
    def can_buy(self) -> bool:
        """Check if we can open new positions"""
        return len(self.positions) < self.max_positions and self.cash > 0
    
    def buy(self, signal: MaxProfitSignal, trade_fee: float = 1.0) -> bool:
        """Execute buy order"""
        if signal.symbol in self.positions:
            return False  # Already have position
        
        # Calculate position size
        position_value = self.get_total_value() * signal.position_size_pct
        max_affordable = self.cash - trade_fee
        
        if max_affordable <= 0:
            return False
        
        actual_position_value = min(position_value, max_affordable)
        shares = int(actual_position_value / signal.current_price)
        
        if shares <= 0:
            return False
        
        # Create position
        cost = shares * signal.current_price + trade_fee
        
        if cost > self.cash:
            shares = int((self.cash - trade_fee) / signal.current_price)
            cost = shares * signal.current_price + trade_fee
        
        if shares <= 0:
            return False
        
        self.positions[signal.symbol] = Position(
            symbol=signal.symbol,
            shares=shares,
            entry_price=signal.current_price,
            entry_date=signal.date,
            entry_signal_score=signal.final_score,
            stop_loss=signal.stop_loss_price,
            take_profit=signal.target_price
        )
        
        self.cash -= cost
        
        logger.debug(f"BUY {signal.symbol}: {shares} shares @ ${signal.current_price:.2f}, Cost: ${cost:.2f}")
        return True
    
    def sell(self, symbol: str, current_price: float, exit_date: str, 
             exit_score: float, exit_reason: str, trade_fee: float = 1.0) -> Optional[Trade]:
        """Execute sell order"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Calculate P&L
        gross_pnl = position.shares * (current_price - position.entry_price)
        net_pnl = gross_pnl - (trade_fee * 2)  # Entry + Exit fees
        return_pct = (current_price / position.entry_price - 1) * 100
        
        # Calculate holding period
        entry_date = datetime.strptime(position.entry_date, '%Y-%m-%d')
        exit_date_dt = datetime.strptime(exit_date, '%Y-%m-%d')
        holding_days = (exit_date_dt - entry_date).days
        
        # Create trade record
        trade = Trade(
            symbol=symbol,
            entry_date=position.entry_date,
            exit_date=exit_date,
            entry_price=position.entry_price,
            exit_price=current_price,
            shares=position.shares,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            return_pct=return_pct,
            holding_days=holding_days,
            entry_score=position.entry_signal_score,
            exit_score=exit_score,
            exit_reason=exit_reason
        )
        
        # Update cash and remove position
        self.cash += position.shares * current_price - trade_fee
        del self.positions[symbol]
        self.trades.append(trade)
        
        logger.debug(f"SELL {symbol}: {position.shares} shares @ ${current_price:.2f}, Net P&L: ${net_pnl:.2f}")
        return trade
    
    def check_stops(self, date: str, prices: Dict[str, float]) -> List[Trade]:
        """Check and execute stop losses and take profits"""
        trades = []
        
        for symbol, position in list(self.positions.items()):
            if symbol not in prices:
                continue
                
            current_price = prices[symbol]
            
            # Check stop loss
            if current_price <= position.stop_loss:
                trade = self.sell(symbol, current_price, date, 0, 'stop_loss')
                if trade:
                    trades.append(trade)
                    logger.info(f"STOP LOSS hit for {symbol} @ ${current_price:.2f}")
                    
            # Check take profit
            elif current_price >= position.take_profit:
                trade = self.sell(symbol, current_price, date, 100, 'take_profit')
                if trade:
                    trades.append(trade)
                    logger.info(f"TAKE PROFIT hit for {symbol} @ ${current_price:.2f}")
        
        return trades
    
    def get_total_value(self, prices: Optional[Dict[str, float]] = None) -> float:
        """Calculate total portfolio value"""
        positions_value = 0
        
        if prices:
            for symbol, position in self.positions.items():
                if symbol in prices:
                    positions_value += position.shares * prices[symbol]
        
        return self.cash + positions_value
    
    def record_daily_value(self, date: str, prices: Dict[str, float]):
        """Record daily portfolio value"""
        total_value = self.get_total_value(prices)
        self.daily_values.append((date, total_value))

class Backtester:
    """Main backtest engine"""
    
    def __init__(self, db_path: str, initial_capital: float = 100000):
        self.db_path = db_path
        self.initial_capital = initial_capital
        self.trade_fee = 1.0  # $1 per trade
        
    def run_backtest(self, config_id: Optional[int] = None, 
                     years_back: int = 3,
                     end_date: Optional[str] = None) -> BacktestResults:
        """Run complete backtest"""
        
        # Set date range
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=years_back * 365)
        start_date = start_dt.strftime('%Y-%m-%d')
        
        logger.info(f"\n{'='*80}")
        logger.info(f"STARTING BACKTEST")
        logger.info(f"Configuration ID: {config_id if config_id else 'Best'}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"{'='*80}\n")
        
        # Initialize components
        signal_generator = MaxProfitSignalGenerator(self.db_path, config_id)
        portfolio = Portfolio(self.initial_capital, signal_generator.config['max_positions'])
        
        # Get trading days
        trading_days = self._get_trading_days(start_date, end_date)
        logger.info(f"Total trading days: {len(trading_days)}")
        
        # Track metrics
        daily_trades = defaultdict(int)
        max_positions = 0
        max_value = self.initial_capital
        
        # Process each trading day
        for i, date in enumerate(trading_days):
            if i % 60 == 0:  # Progress every ~3 months
                progress = (i / len(trading_days)) * 100
                logger.info(f"Progress: {progress:.1f}% - Date: {date}")
            
            # Get current prices for portfolio
            prices = self._get_prices_for_date(date)
            
            # Check stops first
            stop_trades = portfolio.check_stops(date, prices)
            daily_trades[date] += len(stop_trades)
            
            # Generate signals
            try:
                signals = signal_generator.generate_all_signals(date, show_details=False)
                
                # Process SELL signals
                sell_signals = [s for s in signals if s.action == 'SELL']
                for signal in sell_signals:
                    if signal.symbol in portfolio.positions:
                        trade = portfolio.sell(
                            signal.symbol, 
                            signal.current_price, 
                            date, 
                            signal.final_score,
                            'signal',
                            self.trade_fee
                        )
                        if trade:
                            daily_trades[date] += 1
                
                # Process BUY signals
                buy_signals = sorted(
                    [s for s in signals if s.action == 'BUY'],
                    key=lambda x: x.final_score,
                    reverse=True
                )
                
                for signal in buy_signals:
                    if portfolio.can_buy():
                        if portfolio.buy(signal, self.trade_fee):
                            daily_trades[date] += 1
                    else:
                        break
                
            except Exception as e:
                logger.error(f"Error processing {date}: {e}")
                continue
            
            # Update daily metrics
            portfolio.record_daily_value(date, prices)
            current_value = portfolio.get_total_value(prices)
            
            if len(portfolio.positions) > max_positions:
                max_positions = len(portfolio.positions)
            
            if current_value > max_value:
                max_value = current_value
        
        # Final liquidation
        logger.info("\nLiquidating remaining positions...")
        final_prices = self._get_prices_for_date(end_date)
        
        for symbol in list(portfolio.positions.keys()):
            if symbol in final_prices:
                portfolio.sell(
                    symbol,
                    final_prices[symbol],
                    end_date,
                    0,
                    'liquidation',
                    self.trade_fee
                )
        
        # Calculate results
        results = self._calculate_results(
            portfolio, 
            start_date, 
            end_date,
            max_positions,
            max_value,
            config_id or signal_generator.config_id
        )
        
        # Save results to database
        self._save_results_to_db(results, config_id or signal_generator.config_id)
        
        return results
    
    def _get_trading_days(self, start_date: str, end_date: str) -> List[str]:
        """Get all trading days in date range"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT DISTINCT trade_date 
            FROM stock_prices 
            WHERE trade_date >= ? AND trade_date <= ?
            ORDER BY trade_date
        """
        
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()
        
        return df['trade_date'].tolist()
    
    def _get_prices_for_date(self, date: str) -> Dict[str, float]:
        """Get all stock prices for a given date"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT symbol, close 
            FROM stock_prices 
            WHERE trade_date = ?
        """
        
        df = pd.read_sql_query(query, conn, params=(date,))
        conn.close()
        
        return dict(zip(df['symbol'], df['close']))
    
    def _calculate_results(self, portfolio: Portfolio, start_date: str, 
                          end_date: str, max_positions: int, 
                          max_value: float, config_id: int) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        
        # Basic metrics
        final_value = portfolio.cash
        total_return = (final_value / self.initial_capital - 1)
        
        # Annual return
        days = (datetime.strptime(end_date, '%Y-%m-%d') - 
                datetime.strptime(start_date, '%Y-%m-%d')).days
        years = days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Trade statistics
        trades = portfolio.trades
        total_trades = len(trades)
        
        if total_trades > 0:
            winning_trades = [t for t in trades if t.net_pnl > 0]
            losing_trades = [t for t in trades if t.net_pnl <= 0]
            
            win_rate = len(winning_trades) / total_trades
            avg_win = sum(t.net_pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(t.net_pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0
            win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            avg_holding_days = sum(t.holding_days for t in trades) / total_trades
            max_holding_days = max(t.holding_days for t in trades)
            
            # Best and worst trades
            best_trade = max(trades, key=lambda t: t.net_pnl)
            worst_trade = min(trades, key=lambda t: t.net_pnl)
            
            # P&L metrics
            gross_pnl = sum(t.gross_pnl for t in trades)
            total_fees = total_trades * 2 * self.trade_fee  # Buy + Sell
            net_pnl = gross_pnl - total_fees
            
        else:
            win_rate = avg_win = avg_loss = win_loss_ratio = 0
            avg_holding_days = max_holding_days = 0
            best_trade = worst_trade = None
            gross_pnl = total_fees = net_pnl = 0
            winning_trades = losing_trades = []
        
        # Calculate daily returns and risk metrics
        daily_returns = []
        equity_curve = []
        
        if portfolio.daily_values:
            values = [v[1] for v in portfolio.daily_values]
            equity_curve = values
            
            for i in range(1, len(values)):
                daily_return = (values[i] / values[i-1] - 1)
                daily_returns.append(daily_return)
            
            # Sharpe ratio (assuming 0% risk-free rate)
            if daily_returns:
                avg_daily_return = np.mean(daily_returns)
                daily_volatility = np.std(daily_returns)
                sharpe_ratio = np.sqrt(252) * avg_daily_return / daily_volatility if daily_volatility > 0 else 0
                volatility = daily_volatility * np.sqrt(252)
            else:
                sharpe_ratio = volatility = 0
            
            # Maximum drawdown
            peak = values[0]
            max_drawdown = 0
            max_dd_duration = 0
            current_dd_duration = 0
            
            for value in values:
                if value > peak:
                    peak = value
                    current_dd_duration = 0
                else:
                    current_dd_duration += 1
                    drawdown = (peak - value) / peak
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                        max_dd_duration = current_dd_duration
        else:
            sharpe_ratio = volatility = max_drawdown = max_dd_duration = 0
        
        # Average position size
        avg_position_size = sum(t.shares * t.entry_price for t in trades) / total_trades if total_trades > 0 else 0
        
        return BacktestResults(
            total_return=total_return,
            annual_return=annual_return,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            gross_pnl=gross_pnl,
            total_fees=total_fees,
            net_pnl=net_pnl,
            max_drawdown=max_drawdown,
            max_drawdown_duration_days=max_dd_duration,
            sharpe_ratio=sharpe_ratio,
            volatility=volatility,
            avg_win=avg_win,
            avg_loss=avg_loss,
            win_loss_ratio=win_loss_ratio,
            avg_holding_days=avg_holding_days,
            max_holding_days=max_holding_days,
            max_positions_held=max_positions,
            max_portfolio_value=max_value,
            avg_position_size=avg_position_size,
            best_trade=best_trade,
            worst_trade=worst_trade,
            daily_returns=daily_returns,
            equity_curve=equity_curve,
            trades=trades,
            config_id=config_id,
            start_date=start_date,
            end_date=end_date
        )
    
    def _save_results_to_db(self, results: BacktestResults, config_id: int):
        """Save backtest results to optimizer_configurations table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update configuration with performance metrics
        cursor.execute("""
            UPDATE optimizer_configurations
            SET sum_revenues = ?,
                total_trades = ?,
                win_rate = ?,
                avg_profit_per_trade = ?,
                max_drawdown = ?,
                sharpe_ratio = ?,
                backtest_start_date = ?,
                backtest_end_date = ?
            WHERE id = ?
        """, (
            results.net_pnl,
            results.total_trades,
            results.win_rate,
            results.net_pnl / results.total_trades if results.total_trades > 0 else 0,
            results.max_drawdown,
            results.sharpe_ratio,
            results.start_date,
            results.end_date,
            config_id
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved backtest results to configuration ID: {config_id}")

def run_backtest_analysis(db_path: str, config_id: Optional[int] = None,
                         years_back: int = 3, initial_capital: float = 100000):
    """Main entry point for backtesting"""
    
    backtester = Backtester(db_path, initial_capital)
    results = backtester.run_backtest(config_id, years_back)
    
    # Print detailed report
    print(results.get_summary_report())
    
    # Print top 10 trades
    print("\nTOP 10 TRADES BY PROFIT:")
    print("-" * 80)
    sorted_trades = sorted(results.trades, key=lambda t: t.net_pnl, reverse=True)
    for i, trade in enumerate(sorted_trades[:10], 1):
        print(f"{i}. {trade.symbol}: ${trade.net_pnl:,.2f} ({trade.return_pct:.1f}%) "
              f"in {trade.holding_days} days | {trade.entry_date} -> {trade.exit_date}")
    
    # Print monthly returns
    print("\nMONTHLY RETURNS:")
    print("-" * 80)
    monthly_returns = calculate_monthly_returns(results.trades)
    for month, return_pct in monthly_returns:
        print(f"{month}: {return_pct:>6.2f}%")
    
    return results

def calculate_monthly_returns(trades: List[Trade]) -> List[Tuple[str, float]]:
    """Calculate returns by month"""
    monthly_pnl = defaultdict(float)
    
    for trade in trades:
        month = trade.exit_date[:7]  # YYYY-MM
        monthly_pnl[month] += trade.net_pnl
    
    # Sort by month
    sorted_months = sorted(monthly_pnl.items())
    
    # Calculate returns (simplified - assumes even capital distribution)
    monthly_returns = []
    for month, pnl in sorted_months:
        # Rough approximation of monthly return
        return_pct = (pnl / 100000) * 100  # As percentage of initial capital
        monthly_returns.append((month, return_pct))
    
    return monthly_returns

# Main entry point
if __name__ == "__main__":
    import sys
    
    db_path = sys.argv[1] if len(sys.argv) > 1 else "unified_trading.db"
    config_id = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else None
    years = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    capital = float(sys.argv[4]) if len(sys.argv) > 4 else 100000
    
    # Run backtest
    results = run_backtest_analysis(db_path, config_id, years, capital)
    
    # Export results to CSV if requested
    if len(sys.argv) > 5 and sys.argv[5] == "--export":
        trades_df = pd.DataFrame([vars(t) for t in results.trades])
        trades_df.to_csv(f"backtest_trades_{config_id or 'best'}.csv", index=False)
        print(f"\nExported {len(results.trades)} trades to CSV")