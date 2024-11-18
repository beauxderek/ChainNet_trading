from datetime import datetime, timedelta
from tqdm import tqdm
from ChainletBacktest import ChainletBacktester, calculate_sharpe

def main():
    print(f"\nStarting chainlet trading strategy backtest at {datetime.now()}")
    print("Loading and processing data...")
    
    price_path = "price_data.txt"
    chainlet_path = "chainlet_data.txt"
    
    #Get chainlet timing approach from user
    while True:
        chainlet_timing = input("\nChoose chainlet timing:\n1: Use previous day's chainlets (more conservative)\n2: Use same-day chainlets (assumes reliable real-time chainlet data)\nEnter choice (1 or 2): ")
        if chainlet_timing in ['1','2']:
            use_lagged_chainlets = (chainlet_timing == '1')
            break
        print("Invalid choice. Please enter 1 or 2.")  
          
    #Get approach selection from user
    while True:
        approach = input("\nChoose backtesting approach:\n1: Simple rolling window\n2: SPred algorithm\nEnter choice (1 or 2): ")
        if approach in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    #Get parameters based on approach
    if approach == '1':
        training_days = int(input("\nEnter training days (default 250): ") or "250")
    else:
        training_length = int(input("\nEnter training length l (default 250): ") or "250")
        window = int(input("Enter window length w (default 5): ") or "5")
        horizon = int(input("Enter prediction horizon h (default 1): ") or "1")
    
    #Get chainlet processing mode from user
    while True:
        chainlet_mode = input("\nChoose chainlet processing mode:\n1: Count mode (use actual chainlet counts)\n2: Binary mode (0/1 for chainlet presence)\nEnter choice (1 or 2): ")
        if chainlet_mode in ['1', '2']:
            binary_mode = (chainlet_mode == '2')
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    backtester = ChainletBacktester()
    data_processor = backtester.data_processor

    price_data = data_processor.load_price_data(price_path)
    chainlet_data = data_processor.process_chainlet_data(chainlet_path, binary_mode=binary_mode)
    data = data_processor.align_and_prepare_data(price_data, chainlet_data, use_lagged_chainlets)
    
    print(f"\nAvailable date range: {data.index[0]} to {data.index[-1]}")
    
    while True:
        try:
            backtest_start = input("Enter backtest start date (YYYY-MM-DD): ")
            end_date = input("Enter end date (YYYY-MM-DD): ")
            
            backtest_start_date = datetime.strptime(backtest_start, "%Y-%m-%d")
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
            if approach == '1':
                required_days = training_days
            else:
                required_days = training_length + window + horizon
            
            training_start_date = backtest_start_date - timedelta(days=required_days)
            
            training_start_str = training_start_date.strftime("%Y-%m-%d")
            backtest_start_str = backtest_start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            if backtest_start_date >= end_date:
                print("Start date must be before end date")
                continue
                
            if (training_start_str not in data.index or 
                backtest_start_str not in data.index or 
                end_str not in data.index):
                print(f"Dates must be within the available date range")
                print(f"Need data from {training_start_str} for training period")
                continue
            
            #Filter data to include training period and backtest period
            filtered_data = data[training_start_str:end_str]
            
            #Check if we have enough data
            if len(filtered_data) < required_days + 1:
                print(f"Error: Insufficient data. Need at least {required_days} days for training plus backtest period.")
                continue
                
            break
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD")
    
    model_type = input("Choose model type (random_forest or svm): ")
    while model_type not in ['random_forest', 'svm']:
        print("Invalid model type. Please choose 'random_forest' or 'svm'")
        model_type = input("Choose model type (random_forest or svm): ")
    
    print("\nRunning backtest simulation...")
    if approach == '1':
        print(f"Approach: Simple rolling window")
        print(f"Training days: {training_days}")
        with tqdm(total=len(filtered_data)-training_days-1) as pbar:
            results = backtester.simple_rolling_backtest(
                filtered_data,
                model_type,
                training_days=training_days,
                pbar=pbar
            )
    else:
        print(f"Approach: SPred algorithm")
        print(f"Training length (l): {training_length}")
        print(f"Window length (w): {window}")
        print(f"Prediction horizon (h): {horizon}")
        with tqdm(total=len(filtered_data)-training_length-window-horizon-1) as pbar:
            results = backtester.spred_backtest(
                filtered_data,
                model_type,
                training_length=training_length,
                window=window,
                horizon=horizon,
                pbar=pbar
            )
    
    metrics = {
        'Total Strategy Return': f"{results['total_return'].iloc[-1]:.2f}%",  # Take final value
        'Buy & Hold Return': f"{results['buy_hold_return']:.2f}%",
        'Sharpe Ratio': f"{results['trading_sharpe']:.2f}",
        'Win Rate': f"{results['win_rate']*100:.2f}%",
        'Number of Trades': len(results['results'][results['results']['in_position'].diff() != 0]),
        'Simulation Period': f"{results['results']['date'].iloc[0]} to {results['results']['date'].iloc[-1]}",
        'Model Type': model_type,
        'Training Approach': 'Simple rolling window' if approach == '1' else 'SPred algorithm',
        'Chainlet Timing': 'Previous day' if use_lagged_chainlets else 'Same day',
        'Chainlet Mode': 'Binary (0/1)' if binary_mode else 'Count',
        'Training Parameters': (f"Training days: {training_days}" if approach == '1' else 
                             f"l: {training_length}, w: {window}, h: {horizon}"),
        'Positive Returns Days': sum(results['results']['prediction'] == 1),
        'Negative Returns Days': sum(results['results']['prediction'] == 0)
    }
    
    print("\n=== Backtest Results ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
        
    results['performance_plot'].show()
    results['confusion_matrix'].show()
    results['roc_curve'].show()

if __name__ == "__main__":
    main()