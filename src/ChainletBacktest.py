import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from datetime import datetime
from tqdm import tqdm

from processor import BitcoinDataProcessor
from visuals import Visualizer

"""
===================================================================
ChainletBacktest.py
Description: Run backtests for the chainlet price-prediction trading algorithm
Pipeline: 

- ChainletBacktester class must be initialized with trading fees and choice of chainlets
    Selecting the entire 20x20 matrix will usually lead to poor outcomes
- Choice of model (RF or SVM) is tuned with respect to the base period data (l = 250)
- Following optimization, model is retrained either on simple rolling window or SPred approach
    (defaults set as l=250, w=5, h=1)
===================================================================
"""



def calculate_sharpe(returns: np.array, risk_free_rate: float = 0.02) -> float:
    """Calculate annualized Sharpe ratio"""
    if len(returns) == 0:
        return 0
    excess_returns = returns - risk_free_rate/252
    if np.std(returns) == 0:
        return 0
    return np.sqrt(252) * np.mean(excess_returns) / np.std(returns)

class ChainletBacktester:
    def __init__(self, maker_fee: float = 0.0015, taker_fee: float = 0.0025):
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.data_processor = BitcoinDataProcessor()
        self.chainlet_patterns = ['C_1_7', 'C_6_1', 'C_3_3', 'C_20_12']
        self.visualizer = Visualizer()
        self.best_params = None
   
    def prepare_data(self, price_path: str, chainlet_path: str, use_lagged_chainlets: bool = True) -> pd.DataFrame:
        price_data = self.data_processor.load_price_data(price_path)
        chainlet_data = self.data_processor.process_chainlet_data(chainlet_path)
        return self.data_processor.align_and_prepare_data(price_data, chainlet_data, use_lagged_chainlets)
    
    def optimize_random_forest(self, train_data: pd.DataFrame) -> Dict:
        """Optimize Random Forest hyperparameters using time series cross-validation"""
        print("\nOptimizing Random Forest parameters...")
        
        X = train_data[self.chainlet_patterns]
        y = train_data['target']
        
        param_grid = {
            'n_estimators': [300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        tscv = TimeSeriesSplit(n_splits=5)
        best_score = -np.inf
        best_params = None
        
        for n_estimators in param_grid['n_estimators']:
            for max_depth in param_grid['max_depth']:
                for min_samples_split in param_grid['min_samples_split']:
                    for min_samples_leaf in param_grid['min_samples_leaf']:
                        model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            random_state=42
                        )
                        
                        scores = []
                        for train_idx, val_idx in tscv.split(X):
                            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                            
                            model.fit(X_train, y_train)
                            score = accuracy_score(y_val, model.predict(X_val))
                            scores.append(score)
                        
                        avg_score = np.mean(scores)
                        if avg_score > best_score:
                            best_score = avg_score
                            best_params = {
                                'n_estimators': n_estimators,
                                'max_depth': max_depth,
                                'min_samples_split': min_samples_split,
                                'min_samples_leaf': min_samples_leaf
                            }
        
        print(f"Best Random Forest parameters found: {best_params}")
        print(f"Best cross-validation score: {best_score:.4f}")
        return best_params

    def optimize_svm(self, train_data: pd.DataFrame) -> Dict:
        """Optimize SVM hyperparameters using time series cross-validation"""
        print("\nOptimizing SVM parameters...")
        
        X = train_data[self.chainlet_patterns]
        y = train_data['target']
        
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear']
        }
        
        tscv = TimeSeriesSplit(n_splits=5)
        best_score = -np.inf
        best_params = None
        
        for C in param_grid['C']:
            for gamma in param_grid['gamma']:
                for kernel in param_grid['kernel']:
                    model = SVC(
                        C=C,
                        gamma=gamma,
                        kernel=kernel,
                        random_state=42
                    )
                    
                    scores = []
                    for train_idx, val_idx in tscv.split(X):
                        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                        
                        model.fit(X_train, y_train)
                        score = accuracy_score(y_val, model.predict(X_val))
                        scores.append(score)
                    
                    avg_score = np.mean(scores)
                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = {
                            'C': C,
                            'gamma': gamma,
                            'kernel': kernel
                        }
        
        print(f"Best SVM parameters found: {best_params}")
        print(f"Best cross-validation score: {best_score:.4f}")
        return best_params

    def train_random_forest(self, train_data: pd.DataFrame) -> RandomForestClassifier:
        model = RandomForestClassifier(
            **self.best_params if self.best_params else {'n_estimators': 300},
            random_state=42
        )
        X = train_data[self.chainlet_patterns]
        y = train_data['target']
        model.fit(X, y)
        return model

    def train_svm(self, train_data: pd.DataFrame) -> SVC:
        model = SVC(
            **self.best_params if self.best_params else {'kernel': 'rbf', 'C': 1.0},
            random_state=42
        )
        X = train_data[self.chainlet_patterns]
        y = train_data['target']
        model.fit(X, y)
        return model
        
    def simple_rolling_backtest(self, data: pd.DataFrame, model_type: str, 
                              training_days: int = 250, pbar: Optional[tqdm] = None) -> Dict:
        """
        Simple rolling window backtest where each prediction uses previous training_days.
        """
        if len(data) < training_days + 1:
            raise ValueError(f"Insufficient data for backtesting. Need at least {training_days + 1} days")
        
        initial_train = data.iloc[:training_days]
        if model_type == 'random_forest':
            self.best_params = self.optimize_random_forest(initial_train)
        elif model_type == 'svm':
            self.best_params = self.optimize_svm(initial_train)
            
        results = []
        cumulative_return = 1.0
        in_position = False
        
        start_idx = training_days
        initial_price = data.iloc[start_idx]['price']
        
        trade_returns = []
        y_true = []
        y_pred = []
        
        if pbar is not None:
            pbar.total = len(data) - start_idx - 1
            
        for i in range(start_idx, len(data)-1):
            current_price = data.iloc[i]['price']
            next_price = data.iloc[i+1]['price']
            daily_return = (next_price / current_price) - 1
            
            train_data = data.iloc[i-training_days:i]
            
            if model_type == 'random_forest':
                model = self.train_random_forest(train_data)
            elif model_type == 'svm':
                model = self.train_svm(train_data)
            
            pred = model.predict(data.iloc[i:i+1][self.chainlet_patterns])[0]
            y_true.append(data.iloc[i]['target'])
            y_pred.append(pred)
            
            realized_return = 0
            
            if pred == 1:
                if not in_position:
                    realized_return = daily_return - self.taker_fee
                else:
                    realized_return = daily_return
                in_position = True
                trade_returns.append(realized_return)
                cumulative_return *= (1 + realized_return)
            else:
                if in_position:
                    cumulative_return *= (1 - self.maker_fee)
                in_position = False
                
            buy_hold_return = (current_price / initial_price)
                
            results.append({
                'date': data.index[i],
                'price': current_price,
                'prediction': pred,
                'strategy_return': cumulative_return,
                'buy_hold_return': buy_hold_return,
                'in_position': in_position,
                'daily_return': daily_return,
                'realized_return': realized_return if pred == 1 else 0
            })
            
            if pbar is not None:
                pbar.update(1)
                
        return self._prepare_results(results, trade_returns, initial_price, data, y_true, y_pred)
    
    def spred_backtest(self, data: pd.DataFrame, model_type: str, 
                       training_length: int = 250, window: int = 5, horizon: int = 1,
                       pbar: Optional[tqdm] = None) -> Dict:
        """
        SPred algorithm implementation from the ChainNet paper.
        
        Args:
            data: Full dataset
            model_type: 'random_forest' or 'svm'
            training_length: l parameter - training length
            window: w parameter - sliding window length
            horizon: h parameter - prediction horizon
            pbar: Optional progress bar
        """
        if len(data) < training_length + window + horizon:
            raise ValueError(f"Insufficient data. Need at least {training_length + window + horizon} days")
        
        initial_train = data.iloc[:training_length]
        if model_type == 'random_forest':
            self.best_params = self.optimize_random_forest(initial_train)
        elif model_type == 'svm':
            self.best_params = self.optimize_svm(initial_train)
            
        results = []
        cumulative_return = 1.0
        in_position = False
        
        start_idx = training_length + window + horizon
        initial_price = data.iloc[start_idx]['price']
        
        trade_returns = []
        y_true = []
        y_pred = []
        
        if pbar is not None:
            pbar.total = len(data) - start_idx - 1
            
        for i in range(start_idx, len(data)-horizon):
            current_price = data.iloc[i]['price']
            next_price = data.iloc[i+horizon]['price']
            daily_return = (next_price / current_price) - 1
            
            base_train = data.iloc[i-training_length-window:i-window]
            window_train = data.iloc[i-window:i]
            train_data = pd.concat([base_train, window_train])
            
            if model_type == 'random_forest':
                model = self.train_random_forest(train_data)
            elif model_type == 'svm':
                model = self.train_svm(train_data)
            
            pred = model.predict(data.iloc[i:i+1][self.chainlet_patterns])[0]
            y_true.append(data.iloc[i]['target'])
            y_pred.append(pred)
            
            realized_return = 0
            
            if pred == 1:
                if not in_position:
                    realized_return = daily_return - self.taker_fee
                else:
                    realized_return = daily_return
                in_position = True
                trade_returns.append(realized_return)
                cumulative_return *= (1 + realized_return)
            else:
                if in_position:
                    cumulative_return *= (1 - self.maker_fee)
                in_position = False
                
            buy_hold_return = (current_price / initial_price)
                
            results.append({
                'date': data.index[i],
                'price': current_price,
                'prediction': pred,
                'strategy_return': cumulative_return,
                'buy_hold_return': buy_hold_return,
                'in_position': in_position,
                'daily_return': daily_return,
                'realized_return': realized_return if pred == 1 else 0
            })
            
            if pbar is not None:
                pbar.update(1)
                
        return self._prepare_results(results, trade_returns, initial_price, data, y_true, y_pred)
    
    def _prepare_results(self, results, trade_returns, initial_price, data, y_true, y_pred):
        """Helper method to prepare final results dictionary"""
        results_df = pd.DataFrame(results)
        win_rate = np.mean([ret > 0 for ret in trade_returns]) if trade_returns else 0
        strategy_return = (results_df['strategy_return'] - 1)
        buy_hold_return = ((data.iloc[-1]['price'] / initial_price) - 1) * 100
        trading_sharpe = calculate_sharpe(np.array(trade_returns)) if trade_returns else 0
        
        return {
            'results': results_df,
            'total_return': (strategy_return*100),
            'buy_hold_return': buy_hold_return,
            'win_rate': win_rate,
            'trading_sharpe': trading_sharpe,
            'performance_plot': self.visualizer.plot_performance(results_df),
            'confusion_matrix': self.visualizer.plot_confusion_matrix(y_true, y_pred),
            'roc_curve': self.visualizer.plot_roc_curve(y_true, y_pred)
        }