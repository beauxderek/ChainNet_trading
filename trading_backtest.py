import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import numpy as np
from numpy.linalg import LinAlgError
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, Tuple, List
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, root_mean_squared_error, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from tqdm import tqdm
!pip install nbformat 
import warnings
warnings.filterwarnings('ignore')

class BitcoinDataProcessor:
    def __init__(self, n_jobs: int = 5):
        self.chainlet_scaler = StandardScaler()
        self.price_scaler = StandardScaler()
        self.n_jobs = n_jobs
   
    def load_price_data(self, price_path: str) -> pd.DataFrame:
        """
        Load price data from text file with either timestamp or date format
        
        Args:
            price_path: Path to price data file
            
        Returns:
            DataFrame with datetime index, price and totaltx columns
        """
        try:
            data = []
            with open(price_path, 'r') as file:
                lines = file.readlines()
                
                # Try to detect format by checking first line
                first_line = lines[0].strip()
                
                if ',' in first_line:  # CSV format with headers
                    headers = first_line.split(',')
                    for line in lines[1:]:
                        if line.strip():  # Skip empty lines
                            values = line.strip().split(',')
                            data.append(values)
                            
                    df = pd.DataFrame(data, columns=headers)
                    df['date'] = pd.to_datetime(df['date'])
                    df['price'] = df['price'].astype(float)
                    df['totaltx'] = df['totaltx'].astype(float)
                    df.set_index('date', inplace=True)
                    
                else:  # Timestamp format
                    for line in lines:
                        if line.strip():
                            timestamp, price = line.strip().split()
                            data.append({
                                'date': pd.Timestamp(int(timestamp), unit='s'),
                                'price': float(price),
                                'totaltx': 0  # Will be updated from matrices
                            })
                    
                    df = pd.DataFrame(data)
                    df.set_index('date', inplace=True)
            
            df = df[['price', 'totaltx']]
            print(f"Loaded price data: {df.shape[0]} periods")
            return df
                
        except Exception as e:
            raise ValueError(f"Error loading price data: {str(e)}")
       
    def process_chainlet_data(self, chainlet_path: str, data_type: str, patterns: list) -> pd.DataFrame:
        """
        Process chainlet data from tab-delimited text file
        
        Args:
            chainlet_path: Path to chainlet data file
            data_type: Type of chainlet data ('occ' for occurrences or 'amount' for amounts)
            patterns: List of chainlet patterns to include (e.g., ['1_7', '6_1'])
        """
        try:
            with open(chainlet_path, 'r') as file:
                lines = file.readlines()
                headers = lines[0].strip().split('\t')
                data = []
                for line in lines[1:]:
                    if line.strip():
                        values = line.strip().split('\t')
                        data.append(values)
                
            df = pd.DataFrame(data, columns=headers)
            numeric_cols = df.columns[df.columns != 'date']
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            df['date'] = pd.to_datetime(
                df.apply(lambda row: f"{int(row['year'])}-{int(row['day']):03d}", axis=1),
                format='%Y-%j'
            )
            df.set_index('date', inplace=True)
            
            # Filter for specified patterns with exact matching
            chainlet_cols = []
            for pattern in patterns:
                in_val, out_val = pattern.split('_')
                # Create exact pattern match
                exact_pattern = f"{in_val}:{out_val}"
                matching_cols = [col for col in df.columns if ':' in col and 
                            col.split(':')[0] == in_val and 
                            col.split(':')[1] == out_val]
                chainlet_cols.extend(matching_cols)
            
            chainlet_features = df[chainlet_cols].copy()
            
            # Add prefix based on data type
            prefix = 'COCC_' if data_type == 'occ' else 'CAMT_'
            new_cols = {col: f'{prefix}{col.replace(":", "_")}' for col in chainlet_cols}
            chainlet_features.rename(columns=new_cols, inplace=True)
            
            print(f"Loaded {data_type} chainlet data: {chainlet_features.shape[0]} days, {len(chainlet_cols)} chainlet features")
            print(f"Selected patterns: {patterns}")
            print(f"Actual columns: {sorted(chainlet_features.columns)}")
            
            return chainlet_features
            
        except Exception as e:
            raise ValueError(f"Error processing chainlet data: {str(e)}")

    def process_hourly_chainlet_data(self, hourly_path: str, feature_config: dict) -> dict:
        """
        Process hourly chainlet data from the matrix format
        
        Args:
            hourly_path: Path to hourly data file
            feature_config: Dictionary specifying which features to extract
                Example:
                {
                    'occ': {
                        'enabled': True,
                        'patterns': ['1_7', ...]
                    },
                    'amount': {
                        'enabled': True,
                        'patterns': ['1_7', ...]
                    }
                }
        
        Returns:
            Dictionary with DataFrames for each feature type
        """
        try:
            with open(hourly_path, 'r') as file:
                lines = file.readlines()

            data = []
            i = 0
            while i < len(lines):
                # Parse block header
                height, timestamp = map(int, lines[i].strip().split())
                
                # Parse occurrence matrix
                occ_entries = lines[i + 1].strip().split()[1:]  # Skip "tx_count_matrix"
                occ_dict = {}
                total_tx = 0  # Track total transactions
                for entry in occ_entries:
                    coords, count = entry.split(':')
                    inputs, outputs = map(int, coords.split(','))
                    count = int(count)
                    occ_dict[(inputs, outputs)] = count
                    total_tx += count  # Add to total
                
                # Parse weight matrix
                weight_entries = lines[i + 2].strip().split()[1:]  # Skip "tx_weight_matrix"
                weight_dict = {}
                for entry in weight_entries:
                    coords, weight = entry.split(':')
                    inputs, outputs = map(int, coords.split(','))
                    weight_dict[(inputs, outputs)] = float(weight)
                
                # Create row data
                row = {
                    'timestamp': timestamp,
                    'datetime': pd.to_datetime(timestamp, unit='s'),
                    'height': height,
                    'totaltx': total_tx  # Add total transactions
                }
                
                # Add occurrence features if enabled
                if feature_config.get('occ', {}).get('enabled', False):
                    patterns = feature_config['occ'].get('patterns', [])
                    for pattern in patterns:
                        x, y = map(int, pattern.split('_'))
                        row[f'COCC_{pattern}'] = occ_dict.get((x, y), 0)
                            
                # Add amount features if enabled
                if feature_config.get('amount', {}).get('enabled', False):
                    patterns = feature_config['amount'].get('patterns', [])
                    for pattern in patterns:
                        x, y = map(int, pattern.split('_'))
                        row[f'CAMT_{pattern}'] = weight_dict.get((x, y), 0.0)
                
                data.append(row)
                i += 3  # Move to next block
            
            # Create DataFrame
            df = pd.DataFrame(data)
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            
            # Split into separate DataFrames by feature type
            result = {}
            result['totaltx'] = df[['totaltx']]  # Add totaltx to result
            
            if feature_config.get('occ', {}).get('enabled', False):
                occ_cols = [col for col in df.columns if col.startswith('COCC_')]
                result['occ'] = df[occ_cols]
                print(f"Extracted occurrence features: {len(occ_cols)} chainlet patterns")
                print(f"Patterns: {[col.replace('COCC_', '') for col in occ_cols]}")
                
            if feature_config.get('amount', {}).get('enabled', False):
                amt_cols = [col for col in df.columns if col.startswith('CAMT_')]
                result['amount'] = df[amt_cols]
                print(f"Extracted amount features: {len(amt_cols)} chainlet patterns")
                print(f"Patterns: {[col.replace('CAMT_', '') for col in amt_cols]}")
            
            print(f"Loaded hourly data: {df.shape[0]} hours")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            print(f"Average transactions per hour: {df['totaltx'].mean():.2f}")
            
            return result
            
        except Exception as e:
            raise ValueError(f"Error processing hourly chainlet data: {str(e)}")

    def create_features(self, price_data: pd.DataFrame, chainlet_data_dict: dict,
                    use_lagged_chainlets: bool = True, lookback: int = 3,
                    horizon: int = 1) -> pd.DataFrame:
        """Create features from price and chainlet data"""
        data = price_data.copy()
        
        # Update totaltx from chainlet data if provided
        if 'totaltx' in chainlet_data_dict:
            # Instead of direct assignment, use merge for totaltx update
            totaltx_df = chainlet_data_dict['totaltx']
            data = pd.merge(data, totaltx_df, left_index=True, right_index=True, how='left')
            # If we got _x and _y columns, combine them
            if 'totaltx_x' in data.columns and 'totaltx_y' in data.columns:
                data['totaltx'] = data['totaltx_y'].fillna(data['totaltx_x'])
                data = data.drop(columns=['totaltx_x', 'totaltx_y'])
        
        # Merge chainlet features
        for data_type, chainlet_df in chainlet_data_dict.items():
            if data_type not in ['totaltx']:  # Skip totaltx as we handled it above
                data = pd.merge(data, chainlet_df, left_index=True, right_index=True, how='inner')
        
        # Price returns for features
        price_log = np.log(data['price'])
        price_returns = price_log.diff()
        
        # Create lagged returns
        for i in range(1, lookback + 1):
            data[f'return_lag_{i}'] = price_returns.shift(i)
        
        # Scale chainlet features
        chainlet_cols = [col for col in data.columns if col.startswith(('COCC_', 'CAMT_'))]
        if chainlet_cols:
            data[chainlet_cols] = self.chainlet_scaler.fit_transform(data[chainlet_cols])
        
        if use_lagged_chainlets:
            for col in chainlet_cols:
                data[f"{col}_lag1"] = data[col].shift(1)
            data = data.drop(columns=chainlet_cols)
            data = data.rename(columns={f"{col}_lag1": col for col in chainlet_cols})
        
        # Target: return from t+h-1 to t+h
        next_price = price_log.shift(-horizon)
        prior_price = price_log.shift(-(horizon-1))
        future_return = next_price - prior_price
        data['target'] = (future_return > 0).astype(int)
        
        # Ensure columns are unique
        assert len(data.columns) == len(set(data.columns)), "Duplicate column names found"
        
        return data.dropna()
   
    def align_and_prepare_data(self, price_data: pd.DataFrame, chainlet_paths: dict,
                            use_lagged_chainlets: bool = True, horizon: int = 1) -> pd.DataFrame:
        """Align and prepare data for modeling"""
        try:
            print("\nProcessing chainlet data...")
            chainlet_data = {}
            
            if chainlet_paths.get('hourly', {}).get('enabled', False):
                hourly_path = chainlet_paths['hourly']['path']
                feature_config = {
                    'occ': chainlet_paths.get('occ', {}),
                    'amount': chainlet_paths.get('amount', {})
                }
                
                if not any(cfg.get('enabled', False) for cfg in feature_config.values()):
                    raise ValueError("Hourly data enabled but no feature types selected")
                    
                hourly_data = self.process_hourly_chainlet_data(
                    hourly_path,
                    feature_config=feature_config
                )
                
                chainlet_data.update(hourly_data)
                
            else:
                for data_type in ['occ', 'amount']:
                    config = chainlet_paths.get(data_type, {})
                    if config.get('enabled', False):
                        chainlet_data[data_type] = self.process_chainlet_data(
                            config['path'],
                            data_type=data_type,
                            patterns=config['patterns']  # Pass patterns to process_chainlet_data
                        )
            
            if not chainlet_data:
                raise ValueError("No valid chainlet data provided")
            
            print("\nAligning data...")
            print(f"Price data dates: {price_data.index.min()} to {price_data.index.max()}")
            for data_type, df in chainlet_data.items():
                if data_type != 'totaltx':
                    print(f"{data_type.title()} chainlet data dates: {df.index.min()} to {df.index.max()}")
            
            data = self.create_features(
                price_data, 
                chainlet_data,
                use_lagged_chainlets,
                lookback=3,
                horizon=horizon
            )
            
            print(f"\nFinal dataset shape: {data.shape}")
            print(f"Date range: {data.index.min()} to {data.index.max()}")
            
            return data
            
        except Exception as e:
            raise ValueError(f"Error aligning data: {str(e)}")
        
# Initialize processor
processor = BitcoinDataProcessor()

# Define chainlet patterns to use for both cases
chainlet_patterns = ['1_7', '6_1', '3_3', '20_2', '20_3', '20_12', '20_17', '1_1']

# Example 1: Hourly Data
print("="*80)
print("HOURLY DATA PROCESSING")
print("="*80)

# Configure hourly data processing
hourly_config = {
    'hourly': {
        'enabled': True,
        'path': '2024_output_matrices.txt'
    },
    'occ': {
        'enabled': True,
        'patterns': chainlet_patterns
    },
    'amount': {
        'enabled': True,
        'patterns': chainlet_patterns
    }
}

# Load and process hourly data
hourly_price_data = processor.load_price_data('1h_interval_price_data.csv')
hourly_features = processor.align_and_prepare_data(
    price_data=hourly_price_data,
    chainlet_paths=hourly_config,
    use_lagged_chainlets=True,
    horizon=1
)

print("\nHourly Features Head:")
print(hourly_features.head())
print("\nHourly Features Info:")
print(hourly_features.info())

# Example 2: Daily Data
print("\n" + "="*80)
print("DAILY DATA PROCESSING")
print("="*80)

# Configure daily data processing
daily_config = {
    'hourly': {
        'enabled': False
    },
    'occ': {
        'enabled': True,
        'path': 'chainlet_occurrence.txt',
        'patterns': chainlet_patterns
    },
    'amount': {
        'enabled': True,
        'path': 'chainlet_amounts.txt',
        'patterns': chainlet_patterns
    }
}

# Load and process daily data
daily_price_data = processor.load_price_data('price_data.csv')
daily_features = processor.align_and_prepare_data(
    price_data=daily_price_data,
    chainlet_paths=daily_config,
    use_lagged_chainlets=True,
    horizon=1
)

print("\nDaily Features Head:")
print(daily_features.head())
print("\nDaily Features Info:")
print(daily_features.info())

# Compare feature sets
print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"Hourly data shape: {hourly_features.shape}")
print(f"Daily data shape: {daily_features.shape}")

class Visualizer:
    def plot_performance(self, results: pd.DataFrame) -> go.Figure:
        """
        Plot strategy performance with trade entry/exit arrows
        
        Args:
            results: DataFrame containing trading results with columns for:
                    date, strategy_return, buy_hold_return, in_position
        
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Plot strategy returns
        fig.add_trace(
            go.Scatter(
                x=results['date'],
                y=(results['strategy_return'] - 1) * 100,
                name='Strategy',
                line=dict(color='blue')
            )
        )
        
        # Plot buy & hold returns
        fig.add_trace(
            go.Scatter(
                x=results['date'],
                y=(results['buy_hold_return'] - 1) * 100,
                name='Buy & Hold',
                line=dict(color='gray', dash='dash')
            )
        )
        
        # Add trade arrows
        # Get points where trades occur
        trades = []
        prev_position = False  # Initialize previous position
        
        for i in range(len(results)):
            current_position = results.iloc[i]['in_position']
            if current_position != prev_position:
                trade_type = 'buy' if current_position else 'sell'
                trades.append({
                    'date': results.iloc[i]['date'],
                    'return': (results.iloc[i]['strategy_return'] - 1) * 100,
                    'type': trade_type
                })
            prev_position = current_position
        
        # Split trades into buys and sells
        buy_trades = [t for t in trades if t['type'] == 'buy']
        sell_trades = [t for t in trades if t['type'] == 'sell']
        
        # Print debug information
        print(f"Total trades detected: {len(trades)}")
        print(f"Buy trades: {len(buy_trades)}")
        print(f"Sell trades: {len(sell_trades)}")
        
        # Add buy arrows (green, pointing up)
        if buy_trades:
            fig.add_trace(
                go.Scatter(
                    x=[t['date'] for t in buy_trades],
                    y=[t['return'] for t in buy_trades],
                    mode='markers',
                    name='Buy',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='green',
                        line=dict(width=2)
                    ),
                    showlegend=True
                )
            )
        
        # Add sell arrows (red, pointing down)
        if sell_trades:
            fig.add_trace(
                go.Scatter(
                    x=[t['date'] for t in sell_trades],
                    y=[t['return'] for t in sell_trades],
                    mode='markers',
                    name='Sell',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='red',
                        line=dict(width=2)
                    ),
                    showlegend=True
                )
            )
        
        # Update layout
        fig.update_layout(
            title='Strategy vs Buy & Hold Returns (%)',
            xaxis_title='Date',
            yaxis_title='Return (%)',
            template='plotly_white',
            hovermode='x unified'
        )
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fig.write_html(f'strategy_performance_{timestamp}.html')
        
        return fig

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        
        # Create text annotations for the cells
        annotations = []
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=str(cm[i, j]),
                        showarrow=False,
                        font=dict(
                            color='white' if cm[i, j] > cm.mean() else 'black',
                            size=16
                        ),
                    )
                )
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted 0', 'Predicted 1'],
            y=['Actual 0', 'Actual 1'],
            colorscale='Blues'
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            xaxis_side='top',
            annotations=annotations
        )
        return fig

    def plot_roc(self, y_true, y_score):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name='ROC Curve (AUC = %0.2f)' % roc_auc,
            mode='lines',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random Classifier',
            mode='lines',
            line=dict(color='red', dash='dash')
        ))
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate (1 - Specificity)',
            yaxis_title='True Positive Rate (Sensitivity)',
            xaxis_range=[0, 1],
            yaxis_range=[0, 1]
        )
        return fig
    
    def _create_price_prediction_plot(self, results_df: pd.DataFrame, rmse: float, 
                                    mae: float, model_name: str) -> go.Figure:
        """Create visualization for price predictions"""
        fig = go.Figure()
        
        # Add actual prices
        fig.add_trace(go.Scatter(
            x=results_df['date'],
            y=results_df['price'],
            name='Actual Price',
            line=dict(color='blue', width=2)
        ))
        
        # Add predictions
        fig.add_trace(go.Scatter(
            x=results_df['date'],
            y=results_df['prediction'],
            name=f'Predicted Price (RMSE: ${rmse:.2f}, MAE: ${mae:.2f})',
            line=dict(color='red', width=2, dash='dot')
        ))
        
        fig.update_layout(
            title=f'Bitcoin Price Predictions - {model_name}',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    def plot_confidence_accuracy(self, y_true, y_score):
        """Plot histogram of prediction confidence with overlaid accuracy line."""
        bins = np.linspace(0, 1, 100)  # 10% bins
        bin_indices = np.digitize(y_score, bins) - 1
        
        accuracies = []
        counts = []
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        for i in range(len(bins) - 1):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                accuracy = np.mean(y_true[mask] == (y_score[mask] > 0.5))
                accuracies.append(accuracy)
                counts.append(np.sum(mask))
            else:
                accuracies.append(np.nan)
                counts.append(0)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(
                x=bin_centers * 100,
                y=counts,
                name="Prediction Count",
                marker_color="lightblue",
                opacity=0.7
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=bin_centers * 100,
                y=np.array(accuracies) * 100,
                name="Accuracy",
                line=dict(color="red", width=2),
                mode="lines+markers"
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title="Prediction Confidence vs Accuracy",
            xaxis_title="Confidence (%)",
            barmode="overlay",
            template="plotly_white"
        )
        
        fig.update_yaxes(
            title_text="Number of Predictions",
            secondary_y=False
        )
        fig.update_yaxes(
            title_text="Accuracy (%)",
            secondary_y=True,
            range=[0, 100]
        )
        
        return fig
    
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Base class for all models to ensure consistent interface"""
    def __init__(self, params=None):
        self.params = params or {}
        self.model = None
        self.feature_scaler = MinMaxScaler()
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train model on scaled data"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using scaled data"""
        pass
    
    def scale_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Scale features, optionally fitting scaler"""
        if fit:
            return self.feature_scaler.fit_transform(X)
        return self.feature_scaler.transform(X)

class RFModel(BaseModel):
    def __init__(self, params=None, task='classification'):
        super().__init__(params)
        default_params = {
            'n_estimators': 300,
            'max_depth': 10,
            'random_state': 42
        }
        self.params = {**default_params, **(params or {})}
        self.task = task
        
        # Initialize appropriate model based on task
        if self.task == 'classification':
            self.model = RandomForestClassifier(**self.params)
        else:  # regression
            self.model = RandomForestRegressor(**self.params)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

class SVMModel(BaseModel):
    def __init__(self, params=None, task='classification'):
        super().__init__(params)
        default_params = {
            'C': 1.0,
            'kernel': 'rbf',
            'probability': True,  # Only used for classification
        }
        self.params = {**default_params, **(params or {})}
        self.task = task
        
        # Remove probability parameter for regression task
        if self.task == 'regression' and 'probability' in self.params:
            del self.params['probability']
        
        # Initialize appropriate model based on task
        if self.task == 'classification':
            self.model = SVC(**self.params)
        else:  # regression
            self.model = SVR(**self.params)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float, task: str = 'classification'):
        super().__init__()
        self.task = task
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        # Only use sigmoid activation for classification
        self.sigmoid = nn.Sigmoid() if task == 'classification' else None
        
    def forward(self, x):
        # Add sequence dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])  # Take last timestep
        out = self.fc(out)
        
        # Apply sigmoid only for classification
        if self.task == 'classification':
            out = self.sigmoid(out)
            
        return out

class LSTMModel(BaseModel):
    def __init__(self, params=None, task='classification'):
        super().__init__(params)
        default_params = {
            'input_dim': None,  # Must be set during training
            'hidden_dim': 64,
            'num_layers': 10,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'epochs': 50
        }
        self.params = {**default_params, **(params or {})}
        self.task = task
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        if self.params['input_dim'] is None:
            self.params['input_dim'] = X_train.shape[1]
        
        self.model = LSTMPredictor(
            input_dim=self.params['input_dim'],
            hidden_dim=self.params['hidden_dim'],
            num_layers=self.params['num_layers'],
            dropout=self.params['dropout'],
            task=self.task
        ).to(self.device)
        
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['learning_rate'])
        
        # Use appropriate loss function based on task
        criterion = (nn.BCELoss() if self.task == 'classification' 
                    else nn.MSELoss())
        
        self.model.train()
        for _ in range(self.params['epochs']):
            optimizer.zero_grad()
            output = self.model(X_tensor)
            
            # Adjust target dimensions for loss calculation
            if self.task == 'classification':
                loss = criterion(output.squeeze(), y_tensor)
            else:
                loss = criterion(output, y_tensor.unsqueeze(1))
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
            if self.task == 'classification':
                return (predictions > 0.5).astype(int).flatten()
            return predictions.flatten()

class TradingStrategy:
    """Handles trading logic and portfolio calculations"""
    def __init__(self, maker_fee: float = 0.0015, taker_fee: float = 0.0025):
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.reset()
    
    def reset(self):
        """Reset portfolio state"""
        self.portfolio_value = 1.0
        self.in_position = False
        self.trades = []
    
    def execute_trade(self, prediction: int, current_price: float, next_price: float) -> dict:
        """Execute single trade based on prediction and update portfolio"""
        daily_return = (next_price / current_price) - 1
        
        if prediction == 1:  # Want to be long
            if not self.in_position:  # Need to enter
                self.portfolio_value *= (1 - self.taker_fee)  # Entry fee
                self.portfolio_value *= (1 + daily_return)
                self.in_position = True
            else:  # Already in position
                self.portfolio_value *= (1 + daily_return)
        else:  # Want to be out
            if self.in_position:  # Need to exit
                self.portfolio_value *= (1 + daily_return)  # Apply return first
                self.portfolio_value *= (1 - self.maker_fee)  # Exit fee
                self.in_position = False
        
        trade_result = {
            'in_position': self.in_position,
            'portfolio_value': self.portfolio_value,
            'strategy_return': self.portfolio_value - 1
        }
        self.trades.append(trade_result)
        return trade_result

class BacktestResult:
    """Container for backtest results"""
    def __init__(self, results_df: pd.DataFrame, y_true: list, y_pred: list):
        self.results_df = results_df
        self.y_true = y_true
        self.y_pred = y_pred
        self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        valid_returns = [r['strategy_return'] for r in self.results_df.to_dict('records') 
                        if r['in_position']]
        
        self.total_return = (self.results_df['portfolio_value'].iloc[-1] - 1) * 100
        self.buy_hold_return = self.results_df['buy_hold_return'].iloc[-1] * 100
        self.win_rate = np.mean([1 if p == a else 0 
                               for p, a in zip(self.y_pred, self.y_true)]) * 100
        self.trading_sharpe = calculate_sharpe(np.array(valid_returns))
        
class BacktestEngine:
    """Handles the rolling window backtesting process"""
    
    def __init__(self, strategy: TradingStrategy = None):
        self.strategy = strategy or TradingStrategy()
        self.visualizer = Visualizer()
    
    def run_backtest(self, 
                    data: pd.DataFrame,
                    model: BaseModel,
                    start_date: str = None,
                    end_date: str = None,
                    training_length: int = 250,
                    window: int = 5,
                    horizon: int = 1,
                    feature_columns: list = None) -> BacktestResult:
        """
        Run rolling window backtest with exact original methodology
        
        Args:
            data: DataFrame with features and price data
            model: Model instance (RF, SVM, or LSTM)
            start_date: Start date for backtest in format 'YYYY-MM-DD'
            end_date: End date for backtest in format 'YYYY-MM-DD'
            training_length: Number of days for training window
            window: Number of days for prediction window
            horizon: Days ahead to predict
            feature_columns: List of feature columns to use
        """
        # Filter data for date range if specified
        if start_date and end_date:
            mask = (data.index >= start_date) & (data.index <= end_date)
            data = data[mask].copy()
            print(f"Using data from {data.index.min()} to {data.index.max()}")
        
        if len(data) < training_length + horizon:
            raise ValueError(f"Insufficient data: Need at least {training_length + horizon} days")
        
        feature_columns = feature_columns or [col for col in data.columns 
                                            if col not in ['price', 'target', 'totaltx']]
        
        # Reset strategy
        self.strategy.reset()
        
        # Initialize tracking
        results = []
        y_true = []
        y_pred = []
        
        # Main backtesting loop
        for i in range(training_length, len(data) - horizon):
            # Get training data
            train_start = i - training_length
            train_end = i
            train_data = data.iloc[train_start:train_end]
            
            # Scale features
            X_train = train_data[feature_columns]
            X_train_scaled = model.scale_features(X_train, fit=True)
            y_train = train_data['target'].values
            
            # Train model
            model.train(X_train_scaled, y_train)
            
            # Get prediction window data (last w days)
            predict_start = i - window + 1
            predict_end = i + 1
            X_predict = data.iloc[predict_start:predict_end][feature_columns]
            X_predict_scaled = model.scale_features(X_predict)
            
            # Make prediction
            prediction = model.predict(X_predict_scaled)[-1]
            
            # Record predictions
            y_pred.append(prediction)
            actual = data.iloc[i + horizon]['target']
            y_true.append(actual)
            
            # Execute trading strategy
            current_price = data.iloc[i]['price']
            next_price = data.iloc[i + horizon]['price']
            
            trade_result = self.strategy.execute_trade(prediction, current_price, next_price)
            
            # Record results
            results.append({
                'date': data.index[i + horizon],
                'price': next_price,
                'prediction': prediction,
                'actual_target': actual,
                **trade_result,
                'buy_hold_return': (next_price / data.iloc[training_length]['price']) - 1
            })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        return BacktestResult(results_df, y_true, y_pred)
    
    def run_price_prediction(self,
                        data: pd.DataFrame,
                        models: Dict[str, BaseModel],
                        start_date: str = None,
                        end_date: str = None,
                        training_length: int = 250,
                        window: int = 5,
                        horizon: int = 1,
                        feature_columns: list = None) -> Dict:
        """Run absolute price prediction with date filtering"""
        # Filter data for date range if specified
        if start_date and end_date:
            mask = (data.index >= start_date) & (data.index <= end_date)
            data = data[mask].copy()
            print(f"Using data from {data.index.min()} to {data.index.max()}")
        if len(data) < training_length + horizon:
            raise ValueError(f"Insufficient data: Need at least {training_length + horizon} days")
        
        feature_columns = feature_columns or [col for col in data.columns 
                                            if col not in ['price', 'target', 'totaltx']]
        
        results = {}
        metrics = {}
        plots = {}
        
        price_scaler = MinMaxScaler()
        
        for name, model in models.items():
            model_results = []
            
            for i in range(training_length, len(data) - horizon):
                # Get training data
                train_start = i - training_length
                train_end = i
                train_data = data.iloc[train_start:train_end]
                
                # Scale data
                train_prices = train_data['price'].values.reshape(-1, 1)
                price_scaler.fit(train_prices)
                train_prices_scaled = price_scaler.transform(train_prices)
                
                X_train = train_data[feature_columns]
                X_train_scaled = model.scale_features(X_train, fit=True)
                
                # Train model
                model.train(X_train_scaled, train_prices_scaled.ravel())
                
                # Get prediction window data
                predict_start = i - window + 1
                predict_end = i + 1
                X_predict = data.iloc[predict_start:predict_end][feature_columns]
                X_predict_scaled = model.scale_features(X_predict)
                
                # Make prediction
                prediction_scaled = model.predict(X_predict_scaled)[-1]
                prediction = price_scaler.inverse_transform([[prediction_scaled]])[0][0]
                
                # Record results
                model_results.append({
                    'date': data.index[i + horizon],
                    'price': data.iloc[i + horizon]['price'],
                    'prediction': prediction
                })
            
            results_df = pd.DataFrame(model_results)
            
            # Calculate metrics
            valid_mask = ~np.isnan(results_df['prediction'])
            if valid_mask.any():
                rmse = root_mean_squared_error(
                    results_df.loc[valid_mask, 'price'],
                    results_df.loc[valid_mask, 'prediction']
                )
                mae = mean_absolute_error(
                    results_df.loc[valid_mask, 'price'],
                    results_df.loc[valid_mask, 'prediction']
                )
            else:
                rmse = mae = np.nan
            
            results[name] = results_df
            metrics[name] = {'RMSE': rmse, 'MAE': mae}
            plots[name] = self.visualizer._create_price_prediction_plot(results_df, rmse, mae, name)
        
        return {
            'results': results,
            'metrics': metrics,
            'plots': plots
        }
        
# Initialize components
strategy = TradingStrategy(maker_fee=0.0015, taker_fee=0.0025)
engine = BacktestEngine(strategy)
visualizer = Visualizer()

# Prepare data
processor = BitcoinDataProcessor()
price_data = processor.load_price_data('price_data.csv')

config = {
    'hourly': {
        'enabled': False
    },
    'occ': {
        'enabled': False,
        'path': 'chainlet_occurrence.txt',
        'patterns': [f'{i}_{j}' for i in range(1, 21) for j in range(1, 21)]
    },
    'amount': {
        'enabled': True,
        'path': 'chainlet_amounts.txt',
        'patterns': [f'{i}_{j}' for i in range(1, 21) for j in range(1, 21)]
    }
}

# Get feature data
data = processor.align_and_prepare_data(
    price_data=price_data,
    chainlet_paths=config,
    horizon=1
)

# Define backtest parameters
params = {
    'start_date': '2013-01-01',
    'end_date': '2015-01-01',
    'training_length': 250,
    'window': 5,
    'horizon': 1
}

# Initialize classification models for trading backtest
classification_models = {
    'SVM': SVMModel(task='classification')
}

print("\n" + "="*50)
print("RUNNING TRADING BACKTESTS")
print("="*50)

# Change this part in the main loop
for model_name, model in classification_models.items():
    print(f"\nRunning trading backtest for {model_name}...")
    
    backtest_result = engine.run_backtest(
        data=data,
        model=model,
        **params
    )
    
    # Generate trading strategy visualizations
    perf_plot = visualizer.plot_performance(backtest_result.results_df)
    conf_mat_plot = visualizer.plot_confusion_matrix(backtest_result.y_true, backtest_result.y_pred)
    roc_fig = visualizer.plot_roc(backtest_result.y_true, backtest_result.y_pred)
    
    # Save plots
    perf_plot.write_html(f'{model_name}_performance.html')
    conf_mat_plot.write_html(f'{model_name}_confusion_matrix.html')  # Updated here as well
    roc_fig.write_html(f'{model_name}_roc_fig.html')
    
    # Print results
    print(f"\n{model_name} Trading Strategy Results:")
    print(f"Total Return: {backtest_result.total_return:.2f}%")
    print(f"Buy & Hold Return: {backtest_result.buy_hold_return:.2f}%")
    print(f"Win Rate: {backtest_result.win_rate:.2f}%")
    print(f"Trading Sharpe: {backtest_result.trading_sharpe:.2f}")

print("\n" + "="*50)
print("RUNNING PRICE PREDICTIONS")
print("="*50)

# Initialize regression models for price prediction
regression_models = {
    'SVM': SVMModel(task='regression')
}

# Run price prediction
price_results = engine.run_price_prediction(
    data=data,
    models=regression_models,
    **params
)

# Display price prediction results and save plots
for model_name, metrics in price_results['metrics'].items():
    print(f"\n{model_name} Price Prediction Results:")
    print(f"RMSE: ${metrics['RMSE']:.2f}")
    print(f"MAE: ${metrics['MAE']:.2f}")
    
    # Save price prediction plot
    price_results['plots'][model_name].write_html(f'{model_name}_price_prediction.html')

print("\nAll results have been saved as HTML files in the current directory.")

