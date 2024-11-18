import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

"""
===================================================================
processor.py
Description: Process raw chainlet and price data 

===================================================================
"""



class BitcoinDataProcessor:
    def __init__(self, n_jobs: int = 5):
        self.chainlet_scaler = StandardScaler()
        self.price_scaler = StandardScaler()
        self.n_jobs = n_jobs
    
    def load_price_data(self, price_path: str) -> pd.DataFrame:
        """Load price data from text file"""
        try:
            with open(price_path, 'r') as file:
                lines = file.readlines()
                
                headers = lines[0].strip().split(',')
                data = []
                for line in lines[1:]:
                    if line.strip():  # Skip empty lines
                        values = line.strip().split(',')
                        data.append(values)
                
            df = pd.DataFrame(data, columns=headers)
            
            df['date'] = pd.to_datetime(df['date'])
            df['price'] = df['price'].astype(float)
            df['totaltx'] = df['totaltx'].astype(float)
            
            df.set_index('date', inplace=True)
            df = df[['price', 'totaltx']]
            
            print(f"Loaded price data: {df.shape[0]} days")
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading price data: {str(e)}")
        
    def process_chainlet_data(self, chainlet_path: str, binary_mode: bool = False) -> pd.DataFrame:
        """
        Process chainlet data from tab-delimited text file
        
        Args:
            chainlet_path: Path to chainlet data file
            binary_mode: If True, convert chainlet values to binary (0/1), otherwise keep counts
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
            
            chainlet_cols = [col for col in df.columns if ':' in col]
            chainlet_features = df[chainlet_cols].copy()
            
            new_cols = {col: f'C_{col.replace(":", "_")}' for col in chainlet_cols}
            chainlet_features.rename(columns=new_cols, inplace=True)
            
            if binary_mode:
                chainlet_features = (chainlet_features > 0).astype(int)
            
            print(f"Loaded chainlet data: {chainlet_features.shape[0]} days, {len(chainlet_cols)} chainlet features")
            print(f"Mode: {'Binary (0/1)' if binary_mode else 'Counts'}")
            return chainlet_features
            
        except Exception as e:
            raise ValueError(f"Error processing chainlet data: {str(e)}")

    def create_features(self, 
                       price_data: pd.DataFrame, 
                       chainlet_data: pd.DataFrame,
                       use_lagged_chainlets: bool = True,
                       lookback: int = 3) -> pd.DataFrame:
        """Create feature matrix with price lags and chainlet features"""
        try:
            if price_data.empty or chainlet_data.empty:
                raise ValueError("Price or chainlet data is empty")
                
            data = pd.merge(
                price_data,
                chainlet_data,
                left_index=True,
                right_index=True,
                how='inner'
            )
            
            if data.empty:
                raise ValueError("No overlapping dates between price and chainlet data")
            
            #Create price lags (using log returns as per paper)
            price_returns = np.log(data['price']).diff()
            for i in range(1, lookback + 1):
                data[f'return_lag_{i}'] = price_returns.shift(i)
            
            #Scale chainlet features
            chainlet_cols = [col for col in data.columns if col.startswith('C_')]
            if chainlet_cols:
                data[chainlet_cols] = self.chainlet_scaler.fit_transform(data[chainlet_cols])
            
            if use_lagged_chainlets:
                for col in chainlet_cols:
                    data[f"{col}_lag1"] = data[col].shift(1)
                    
                data = data.drop(columns=chainlet_cols)
                data = data.rename(columns={f"{col}_lag1": col for col in chainlet_cols})
            data['target'] = (price_returns.shift(-1) > 0).astype(int)
            data = data.dropna()
            
            print(f"Created feature matrix: {data.shape[0]} days, {data.shape[1]} features")
            print("Features include:")
            print(f"- {lookback} price return lags")
            print(f"- {len(chainlet_cols)} chainlet features")
            print(f"Using {'lagged' if use_lagged_chainlets else 'same-day'} chainlet data")
            
            return data
            
        except Exception as e:
            raise ValueError(f"Error creating features: {str(e)}")

    def prepare_training_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare feature matrix X and target y for training"""
        feature_cols = ([f'return_lag_{i}' for i in range(1, 4)] +
                       [col for col in data.columns if col.startswith('C_')])
        
        X = data[feature_cols]
        y = data['target']
        
        return X, y
    
    def align_and_prepare_data(self,
                             price_data: pd.DataFrame,
                             chainlet_data: pd.DataFrame,
                             use_lagged_chainlets: bool = True) -> pd.DataFrame:
        """Align price and chainlet data and prepare features"""
        try:
            print("\nAligning data...")
            print(f"Price data dates: {price_data.index.min()} to {price_data.index.max()}")
            print(f"Chainlet data dates: {chainlet_data.index.min()} to {chainlet_data.index.max()}")
            data = self.create_features(price_data, chainlet_data, use_lagged_chainlets)
            
            print(f"\nFinal dataset shape: {data.shape}")
            print(f"Date range: {data.index.min()} to {data.index.max()}")
            
            return data
            
        except Exception as e:
            raise ValueError(f"Error aligning data: {str(e)}")