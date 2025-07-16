#!/usr/bin/env python3
"""
OPTIMIZER MODULE - Automated Configuration Optimization
Finds the most profitable trading parameters through iterative testing
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import json
import time

# Import backtest module
from backtester import Backtester, run_backtest_analysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Optimizer')

@dataclass
class OptimizationResult:
    """Results from optimization run"""
    best_config_id: int
    best_revenue: float
    total_iterations: int
    improvement_pct: float
    runtime_hours: float
    parameters_tested: Dict[str, int]
    improvement_history: List[Tuple[str, float]]

class ParameterOptimizer:
    """Optimizes trading parameters for maximum profit"""
    
    def __init__(self, db_path: str, initial_capital: float = 100000):
        self.db_path = db_path
        self.initial_capital = initial_capital
        self.backtester = Backtester(db_path, initial_capital)
        
        # Define parameter step sizes
        self.step_sizes = {
            # Weights (must maintain sum = 1.0)
            'technical_weight': 0.02,
            'ai_weight': 0.02,
            'sentiment_weight': 0.02,
            'regime_weight': 0.02,
            'big_trader_weight': 0.02,
            'fear_greed_weight': 0.02,
            
            # Scores
            'min_buy_score': 2.0,
            'min_strong_buy_score': 2.0,
            'max_sell_score': 2.0,
            'max_strong_sell_score': 2.0,
            'min_confidence': 0.05,
            
            # Profit/Risk
            'base_profit_target': 0.02,
            'stop_loss_pct': 0.01,
            'trailing_stop_pct': 0.01,
            'take_profit_multiplier': 0.1,
            
            # Position sizing
            'base_position_size': 0.01,
            'max_position_size': 0.02,
            'kelly_fraction': 0.05,
            
            # Big trader
            'institutional_volume_multiplier': 0.1,
            'accumulation_threshold': 0.05,
            'distribution_threshold': 0.05,
            
            # Technical
            'rsi_oversold_threshold': 2.0,
            'rsi_overbought_threshold': 2.0,
            
            # AI
            'ai_momentum_weight': 0.05,
            'ai_reversion_weight': 0.05,
            'ai_pattern_weight': 0.05,
            'ai_max_prediction': 0.02,
            
            # Momentum
            'momentum_short_period': 1,
            'momentum_long_period': 2
        }
        
        # Parameters to optimize (in order)
        self.parameters_to_optimize = [
            # Most important first
            'big_trader_weight',
            'min_buy_score',
            'base_profit_target',
            'stop_loss_pct',
            'ai_weight',
            'technical_weight',
            'min_strong_buy_score',
            'max_sell_score',
            'kelly_fraction',
            'base_position_size',
            'institutional_volume_multiplier',
            'min_confidence',
            'sentiment_weight',
            'regime_weight',
            'take_profit_multiplier',
            'accumulation_threshold',
            'rsi_oversold_threshold',
            'ai_momentum_weight',
            'ai_reversion_weight',
            'momentum_short_period'
        ]
    
    def optimize(self, base_config_id: Optional[int] = None, 
                years_back: int = 3,
                max_iterations: int = 100) -> OptimizationResult:
        """Run optimization process"""
        
        start_time = time.time()
        
        logger.info(f"\n{'='*80}")
        logger.info("STARTING PARAMETER OPTIMIZATION")
        logger.info(f"{'='*80}")
        
        # Load base configuration
        base_config = self._load_configuration(base_config_id)
        logger.info(f"Base Configuration ID: {base_config['id']}")
        
        # Run initial backtest
        logger.info("\nRunning initial backtest...")
        initial_results = self.backtester.run_backtest(base_config['id'], years_back)
        best_revenue = initial_results.net_pnl
        best_config_id = base_config['id']
        
        logger.info(f"Initial Revenue: ${best_revenue:,.2f}")
        logger.info(f"Initial Sharpe: {initial_results.sharpe_ratio:.2f}")
        
        # Track optimization progress
        iteration = 0
        improvements = []
        parameters_tested = {param: 0 for param in self.parameters_to_optimize}
        
        # Optimization loop
        current_config = base_config.copy()
        
        for param in self.parameters_to_optimize:
            if iteration >= max_iterations:
                logger.info(f"\nReached maximum iterations ({max_iterations})")
                break
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Optimizing: {param}")
            logger.info(f"Current value: {current_config[param]}")
            
            # Try increasing the parameter
            improved = True
            direction = 1  # 1 for increase, -1 for decrease
            
            while improved and iteration < max_iterations:
                # Create new configuration
                new_config = self._create_modified_config(
                    current_config, param, direction * self.step_sizes.get(param, 0.01)
                )
                
                if new_config is None:
                    logger.info(f"Cannot modify {param} further in this direction")
                    break
                
                # Save configuration to database
                config_id = self._save_configuration(new_config)
                
                # Run backtest
                logger.info(f"\nIteration {iteration + 1}: Testing {param} = {new_config[param]:.4f}")
                results = self.backtester.run_backtest(config_id, years_back)
                revenue = results.net_pnl
                
                parameters_tested[param] += 1
                iteration += 1
                
                # Check if improved
                if revenue > best_revenue:
                    improvement = (revenue - best_revenue) / abs(best_revenue) * 100
                    logger.info(f"✓ IMPROVED! Revenue: ${revenue:,.2f} (+{improvement:.1f}%)")
                    logger.info(f"  Sharpe: {results.sharpe_ratio:.2f}, Win Rate: {results.win_rate:.1%}")
                    
                    best_revenue = revenue
                    best_config_id = config_id
                    current_config = new_config.copy()
                    improvements.append((param, improvement))
                    
                    # Mark as new best
                    self._update_best_configuration(config_id)
                else:
                    logger.info(f"✗ No improvement. Revenue: ${revenue:,.2f}")
                    improved = False
            
            # Try decreasing if we haven't found improvement
            if not improved and direction == 1:
                logger.info(f"\nTrying to decrease {param}")
                direction = -1
                improved = True
                
                # Reset to best known value
                current_config = self._load_configuration(best_config_id)
                
                while improved and iteration < max_iterations:
                    new_config = self._create_modified_config(
                        current_config, param, direction * self.step_sizes.get(param, 0.01)
                    )
                    
                    if new_config is None:
                        break
                    
                    config_id = self._save_configuration(new_config)
                    
                    logger.info(f"\nIteration {iteration + 1}: Testing {param} = {new_config[param]:.4f}")
                    results = self.backtester.run_backtest(config_id, years_back)
                    revenue = results.net_pnl
                    
                    parameters_tested[param] += 1
                    iteration += 1
                    
                    if revenue > best_revenue:
                        improvement = (revenue - best_revenue) / abs(best_revenue) * 100
                        logger.info(f"✓ IMPROVED! Revenue: ${revenue:,.2f} (+{improvement:.1f}%)")
                        
                        best_revenue = revenue
                        best_config_id = config_id
                        current_config = new_config.copy()
                        improvements.append((param, improvement))
                        
                        self._update_best_configuration(config_id)
                    else:
                        logger.info(f"✗ No improvement. Revenue: ${revenue:,.2f}")
                        improved = False
            
            # Reload best configuration for next parameter
            current_config = self._load_configuration(best_config_id)
        
        # Calculate total improvement
        total_improvement = (best_revenue - initial_results.net_pnl) / abs(initial_results.net_pnl) * 100
        runtime_hours = (time.time() - start_time) / 3600
        
        # Create result summary
        result = OptimizationResult(
            best_config_id=best_config_id,
            best_revenue=best_revenue,
            total_iterations=iteration,
            improvement_pct=total_improvement,
            runtime_hours=runtime_hours,
            parameters_tested=parameters_tested,
            improvement_history=improvements
        )
        
        # Print final summary
        self._print_optimization_summary(result, initial_results.net_pnl)
        
        return result
    
    def _load_configuration(self, config_id: Optional[int] = None) -> Dict:
        """Load configuration from database"""
        conn = sqlite3.connect(self.db_path)
        
        if config_id:
            query = "SELECT * FROM optimizer_configurations WHERE id = ?"
            params = (config_id,)
        else:
            query = "SELECT * FROM optimizer_configurations WHERE is_best = TRUE ORDER BY id DESC LIMIT 1"
            params = ()
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if df.empty:
            raise ValueError("No configuration found")
        
        return df.iloc[0].to_dict()
    
    def _create_modified_config(self, base_config: Dict, param: str, delta: float) -> Optional[Dict]:
        """Create modified configuration with parameter change"""
        new_config = base_config.copy()
        
        # Special handling for weight parameters (must sum to 1)
        if param.endswith('_weight'):
            return self._modify_weight_parameter(new_config, param, delta)
        
        # Special handling for integer parameters
        if param in ['momentum_short_period', 'momentum_long_period']:
            delta = int(delta)
            if delta == 0:
                delta = 1 if delta >= 0 else -1
        
        # Modify parameter
        new_value = new_config[param] + delta
        
        # Check bounds
        if param == 'min_buy_score':
            if new_value < 30 or new_value >= new_config['min_strong_buy_score']:
                return None
        elif param == 'min_strong_buy_score':
            if new_value <= new_config['min_buy_score'] or new_value > 95:
                return None
        elif param == 'max_sell_score':
            if new_value <= new_config['max_strong_sell_score'] or new_value > 70:
                return None
        elif param == 'max_strong_sell_score':
            if new_value < 5 or new_value >= new_config['max_sell_score']:
                return None
        elif param in ['stop_loss_pct', 'base_profit_target', 'trailing_stop_pct']:
            if new_value <= 0 or new_value > 0.5:
                return None
        elif param in ['base_position_size', 'max_position_size', 'kelly_fraction']:
            if new_value <= 0 or new_value > 1:
                return None
        elif param == 'momentum_short_period':
            if new_value < 2 or new_value >= new_config['momentum_long_period']:
                return None
        elif param == 'momentum_long_period':
            if new_value <= new_config['momentum_short_period'] or new_value > 50:
                return None
        elif new_value < 0:
            return None
        
        new_config[param] = new_value
        return new_config
    
    def _modify_weight_parameter(self, config: Dict, param: str, delta: float) -> Optional[Dict]:
        """Modify weight parameter while maintaining sum = 1"""
        weight_params = ['technical_weight', 'ai_weight', 'sentiment_weight',
                        'regime_weight', 'big_trader_weight', 'fear_greed_weight']
        
        new_config = config.copy()
        
        # Check if new value would be valid
        new_value = new_config[param] + delta
        if new_value < 0 or new_value > 0.7:  # Max 70% for any weight
            return None
        
        # Apply change
        new_config[param] = new_value
        
        # Redistribute the difference among other weights
        other_weights = [w for w in weight_params if w != param]
        total_other = sum(new_config[w] for w in other_weights)
        
        if total_other > 0:
            # Scale other weights proportionally
            scale_factor = (1 - new_value) / total_other
            for w in other_weights:
                new_config[w] = new_config[w] * scale_factor
        else:
            return None
        
        # Verify sum is 1
        total = sum(new_config[w] for w in weight_params)
        if abs(total - 1.0) > 0.001:
            # Normalize
            for w in weight_params:
                new_config[w] /= total
        
        return new_config
    
    def _save_configuration(self, config: Dict) -> int:
        """Save configuration to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Remove non-configuration fields
        config_copy = config.copy()
        for field in ['id', 'sum_revenues', 'total_trades', 'win_rate', 
                     'avg_profit_per_trade', 'max_drawdown', 'sharpe_ratio',
                     'is_best', 'optimization_date', 'backtest_start_date', 
                     'backtest_end_date']:
            config_copy.pop(field, None)
        
        # Build insert query
        columns = list(config_copy.keys())
        placeholders = ['?' for _ in columns]
        
        query = f"""
            INSERT INTO optimizer_configurations ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
        """
        
        cursor.execute(query, list(config_copy.values()))
        config_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return config_id
    
    def _update_best_configuration(self, config_id: int):
        """Update the best configuration flag"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Reset all is_best flags
        cursor.execute("UPDATE optimizer_configurations SET is_best = FALSE")
        
        # Set new best
        cursor.execute(
            "UPDATE optimizer_configurations SET is_best = TRUE WHERE id = ?",
            (config_id,)
        )
        
        conn.commit()
        conn.close()
    
    def _print_optimization_summary(self, result: OptimizationResult, initial_revenue: float):
        """Print optimization summary"""
        
        print(f"\n{'='*80}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*80}")
        print(f"\nBest Configuration ID: {result.best_config_id}")
        print(f"Initial Revenue: ${initial_revenue:,.2f}")
        print(f"Final Revenue: ${result.best_revenue:,.2f}")
        print(f"Total Improvement: {result.improvement_pct:+.1f}%")
        print(f"Total Iterations: {result.total_iterations}")
        print(f"Runtime: {result.runtime_hours:.1f} hours")
        
        print("\nParameters Tested:")
        for param, count in sorted(result.parameters_tested.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"  {param}: {count} iterations")
        
        if result.improvement_history:
            print("\nTop Improvements:")
            for param, improvement in sorted(result.improvement_history, key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {param}: +{improvement:.1f}%")
        
        # Load and display best configuration
        best_config = self._load_configuration(result.best_config_id)
        
        print("\nBest Configuration Summary:")
        print(f"  Big Trader Weight: {best_config['big_trader_weight']:.1%}")
        print(f"  AI Weight: {best_config['ai_weight']:.1%}")
        print(f"  Technical Weight: {best_config['technical_weight']:.1%}")
        print(f"  Buy Threshold: {best_config['min_buy_score']:.0f}")
        print(f"  Profit Target: {best_config['base_profit_target']:.1%}")
        print(f"  Stop Loss: {best_config['stop_loss_pct']:.1%}")
        print(f"  Kelly Fraction: {best_config['kelly_fraction']:.1%}")

def run_optimization(db_path: str, base_config_id: Optional[int] = None,
                    years_back: int = 3, max_iterations: int = 100):
    """Main entry point for optimization"""
    
    optimizer = ParameterOptimizer(db_path)
    result = optimizer.optimize(base_config_id, years_back, max_iterations)
    
    # Save optimization report
    report_filename = f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_filename, 'w') as f:
        json.dump({
            'best_config_id': result.best_config_id,
            'best_revenue': result.best_revenue,
            'improvement_pct': result.improvement_pct,
            'runtime_hours': result.runtime_hours,
            'parameters_tested': result.parameters_tested,
            'improvement_history': result.improvement_history
        }, f, indent=2)
    
    print(f"\nOptimization report saved to: {report_filename}")
    
    return result

# Main entry point
if __name__ == "__main__":
    import sys
    
    db_path = sys.argv[1] if len(sys.argv) > 1 else "unified_trading.db"
    config_id = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else None
    years = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    max_iter = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    
    # Run optimization
    result = run_optimization(db_path, config_id, years, max_iter)
    
    print(f"\n✓ Optimization complete! Best configuration ID: {result.best_config_id}")