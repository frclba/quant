"""
Long & Short Operations with Cointegration Analysis

This script implements Long & Short trading operations based on spread analysis between asset pairs.
The strategy focuses on spread reduction from distortion points, using cointegration and correlation
as quantitative measures for pair selection.

Key Components:
1. Data Acquisition from market sources
2. Cointegrated Asset Pair Identification
3. Long & Short Trading Results (In-Sample)
4. Long & Short Trading Results (Out-of-Sample)
5. Final Considerations and Performance Analysis

University of Brasília - UnB
Campus UnB Gama
Subject: Financial Signal Digital Processing
Prof. Marcelino Monteiro de Andrade Dr.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import matplotlib.dates as mdates
from scipy import signal
from itertools import combinations
from scipy.stats.stats import pearsonr
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class LongShortCointegrationStrategy:
    """
    A comprehensive Long & Short trading strategy based on cointegration analysis.
    
    This class implements a pairs trading strategy that:
    - Identifies cointegrated asset pairs
    - Generates trading signals based on residual z-scores
    - Executes Long & Short operations
    - Analyzes performance metrics
    """
    
    def __init__(self, symbols=None, start_date='25/10/2016', end_date='25/10/2020'):
        """
        Initialize the Long & Short Cointegration Strategy.
        
        Parameters:
        -----------
        symbols : list, optional
            List of asset symbols to analyze. If None, will fetch from index.
        start_date : str
            Start date for data collection (format: 'DD/MM/YYYY')
        end_date : str
            End date for data collection (format: 'DD/MM/YYYY')
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.data_is = None  # In-sample data
        self.data_oos = None  # Out-of-sample data
        self.cointegrated_pairs = []
        self.correlation_ranked_pairs = []
        
    def acquire_market_data(self):
        """
        Acquire price signals from market sources.
        
        This function fetches historical price data for the specified symbols
        and handles any errors during data collection.
        
        Returns:
        --------
        pd.DataFrame : Price data for all successfully retrieved symbols
        """
        print("=== 1) Market Data Acquisition ===")
        
        # For demonstration purposes, we'll create synthetic data
        # In practice, you would replace this with actual market data APIs
        print("Note: Using synthetic data for demonstration. Replace with actual market data API.")
        
        # Generate synthetic price data that mimics real market behavior
        np.random.seed(42)
        dates = pd.date_range(start='2016-10-25', end='2020-10-25', freq='D')
        
        # Create synthetic symbols if none provided
        if self.symbols is None:
            self.symbols = [f'ASSET{i:02d}' for i in range(1, 21)]  # 20 synthetic assets
        
        # Generate correlated price series
        n_assets = len(self.symbols)
        n_days = len(dates)
        
        # Create base random walks
        returns = np.random.normal(0.0005, 0.02, (n_days, n_assets))
        
        # Add some correlation structure
        for i in range(n_assets):
            if i > 0:
                returns[:, i] += 0.3 * returns[:, i-1]  # Add correlation with previous asset
        
        # Convert returns to prices
        prices = np.exp(np.cumsum(returns, axis=0)) * 100
        
        # Create DataFrame
        price_data = pd.DataFrame(prices, index=dates, columns=self.symbols)
        
        # Add some missing values to simulate real data issues
        price_data.iloc[100:105, 0] = np.nan
        price_data.iloc[200:202, 3] = np.nan
        
        # Drop columns with too many missing values
        price_data = price_data.dropna(axis=1, thresh=len(price_data)*0.95)
        
        self.data = price_data.fillna(method='forward')
        
        print(f"Successfully collected data for {len(self.data.columns)} assets")
        print(f"Data period: {self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}")
        
        return self.data
    
    def split_sample_data(self, train_ratio=0.7):
        """
        Split data into in-sample (training) and out-of-sample (testing) periods.
        
        Parameters:
        -----------
        train_ratio : float
            Proportion of data to use for in-sample analysis
        """
        print("\n=== 1.1) Sample Data Separation ===")
        
        split_point = int(len(self.data) * train_ratio)
        self.data_is = self.data.iloc[:split_point]
        self.data_oos = self.data.iloc[split_point:]
        
        print(f"In-sample period: {self.data_is.index[0].strftime('%Y-%m-%d')} to {self.data_is.index[-1].strftime('%Y-%m-%d')}")
        print(f"Out-of-sample period: {self.data_oos.index[0].strftime('%Y-%m-%d')} to {self.data_oos.index[-1].strftime('%Y-%m-%d')}")
    
    def test_stationarity(self, data, significance_level=0.1):
        """
        Test for stationarity using Augmented Dickey-Fuller test.
        
        The Dickey-Fuller test is used to determine non-stationarity based on the 
        autoregressive model: ΔXt = (b-1)Xt-1 + a + et = b*Xt-1 + a + et
        
        Hypotheses:
        - H0: b* = 0 (non-stationary asset)
        - H1: b* < 0 (stationary asset)
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data to test for stationarity
        significance_level : float
            Significance level for the test
            
        Returns:
        --------
        list : List of non-stationary asset symbols
        """
        print("\n=== 2) Cointegrated Asset Pair Identification ===")
        
        non_stationary_assets = []
        columns = data.columns
        
        for col in columns:
            try:
                X = data[col].values
                result = adfuller(X)
                
                # Check if asset is non-stationary
                if result[1] > significance_level and result[0] > list(result[4].values())[1]:
                    non_stationary_assets.append(col)
            except Exception as e:
                print(f"Error testing {col}: {e}")
        
        percentage_non_stationary = 100 * len(non_stationary_assets) / len(columns)
        print(f"1) Percentage of Non-Stationary Assets: {percentage_non_stationary:.2f}%")
        
        return non_stationary_assets
    
    def find_cointegrated_pairs_method1(self, non_stationary_assets, significance_level=0.02):
        """
        First approach to identify cointegrated pairs using Engle-Granger procedure.
        
        For each pair, we perform:
        1. Linear regression: Yt = α*Xt + ut
        2. Calculate residuals: ut = Yt - α*Xt  
        3. Test residuals for stationarity using Dickey-Fuller test
        
        Parameters:
        -----------
        non_stationary_assets : list
            List of non-stationary asset symbols
        significance_level : float
            P-value threshold for cointegration test
            
        Returns:
        --------
        list : List of cointegrated pairs (tuples)
        """
        print("\n=== 2.1) First Approach for Cointegrated Pair Identification ===")
        
        # Generate all possible pairs
        asset_combinations = list(combinations(non_stationary_assets, 2))
        print(f"2) Found {len(non_stationary_assets)} non-stationary assets from total of {len(self.data_is.columns)}")
        print(f"   Evaluating {len(asset_combinations)} pairs for cointegration")
        
        cointegrated_pairs_method1 = []
        
        for comb in tqdm(asset_combinations, desc="Testing cointegration (Method 1)"):
            try:
                # Perform OLS regression
                model = sm.OLS(endog=self.data_is[comb[1]], exog=self.data_is[comb[0]]).fit()
                
                # Test residuals for stationarity
                result = adfuller(model.resid)
                
                # If residuals are stationary, pairs are cointegrated
                if result[1] < significance_level:
                    cointegrated_pairs_method1.append(comb)
            except Exception as e:
                print(f"Error testing pair {comb}: {e}")
        
        percentage_cointegrated = round(100 * len(cointegrated_pairs_method1) / len(asset_combinations), 2)
        print(f"3) Found {len(cointegrated_pairs_method1)} cointegrated pairs from {len(asset_combinations)} total pairs")
        print(f"   Representing {percentage_cointegrated}% of the sample")
        
        return cointegrated_pairs_method1
    
    def find_cointegrated_pairs_method2(self, non_stationary_assets, significance_level=0.02):
        """
        Second approach using statsmodels coint function directly.
        
        This method uses the built-in cointegration test which is more robust
        and provides additional statistical measures.
        
        Parameters:
        -----------
        non_stationary_assets : list
            List of non-stationary asset symbols  
        significance_level : float
            P-value threshold for cointegration test
            
        Returns:
        --------
        tuple : (score_matrix, pvalue_matrix, cointegrated_pairs)
        """
        print("\n=== 2.2) Second Approach for Cointegrated Pair Identification ===")
        
        n_assets = len(non_stationary_assets)
        score_matrix = np.zeros((n_assets, n_assets))
        pvalue_matrix = np.ones((n_assets, n_assets))
        cointegrated_pairs_method2 = []
        
        for i in tqdm(range(n_assets), desc="Testing cointegration (Method 2)"):
            for j in range(i+1, n_assets):
                try:
                    X = self.data_is[non_stationary_assets[i]]
                    Y = self.data_is[non_stationary_assets[j]]
                    
                    # Perform cointegration test
                    result = coint(X, Y)
                    score = result[0]
                    pvalue = result[1]
                    
                    score_matrix[i, j] = score
                    pvalue_matrix[i, j] = pvalue
                    
                    if pvalue < significance_level:
                        cointegrated_pairs_method2.append((non_stationary_assets[i], non_stationary_assets[j]))
                        
                except Exception as e:
                    print(f"Error testing pair ({non_stationary_assets[i]}, {non_stationary_assets[j]}): {e}")
        
        total_possible_pairs = len(list(combinations(non_stationary_assets, 2)))
        percentage_cointegrated = round(100 * len(cointegrated_pairs_method2) / total_possible_pairs, 2)
        
        print(f"Found {len(cointegrated_pairs_method2)} cointegrated pairs from {total_possible_pairs} possible pairs")
        print(f"Representing {percentage_cointegrated}% of the sample")
        
        return score_matrix, pvalue_matrix, cointegrated_pairs_method2
    
    def identify_common_cointegrated_pairs(self, pairs_method1, pairs_method2):
        """
        Identify pairs that are cointegrated according to both methods.
        
        Parameters:
        -----------
        pairs_method1 : list
            Cointegrated pairs from first method
        pairs_method2 : list  
            Cointegrated pairs from second method
            
        Returns:
        --------
        list : Common cointegrated pairs
        """
        print("\n=== 2.3) Common Cointegrated Pairs Identification ===")
        
        common_pairs = list(set(pairs_method1) & set(pairs_method2))
        self.cointegrated_pairs = common_pairs
        
        print(f"Found {len(common_pairs)} pairs cointegrated by both methods")
        
        return common_pairs
    
    def rank_pairs_by_correlation(self, pairs):
        """
        Rank cointegrated pairs by their correlation coefficient.
        
        Higher correlation indicates stronger linear relationship between assets,
        which can improve the stability of the pairs trading strategy.
        
        Parameters:
        -----------
        pairs : list
            List of cointegrated pairs
            
        Returns:
        --------
        list : Pairs ranked by correlation (highest to lowest)
        """
        print("\n=== 2.5) Pair Ranking by Correlation Index ===")
        
        correlation_pairs = []
        
        for pair in pairs:
            X = self.data_is[pair[0]]
            Y = self.data_is[pair[1]]
            
            correlation_coeff = pearsonr(X, Y)[0]
            correlation_pairs.append([correlation_coeff, pair])
        
        # Sort by correlation (highest first)
        correlation_df = pd.DataFrame(correlation_pairs, columns=['Correlation', 'Pair'])
        correlation_df = correlation_df.sort_values(by='Correlation', ascending=False)
        
        self.correlation_ranked_pairs = correlation_df['Pair'].values
        
        print("Top 5 pairs by correlation:")
        for i, (corr, pair) in enumerate(correlation_df.head().values):
            print(f"{i+1}. {pair[0]}/{pair[1]}: {corr:.4f}")
        
        return self.correlation_ranked_pairs
    
    def visualize_residuals(self, pairs, max_pairs=None):
        """
        Visualize the residuals of cointegrated pairs.
        
        The residuals represent the spread between the two assets in each pair.
        Stationary residuals indicate mean-reverting behavior, which is the
        foundation of the pairs trading strategy.
        
        Parameters:
        -----------
        pairs : list
            List of cointegrated pairs to visualize
        max_pairs : int, optional
            Maximum number of pairs to plot
        """
        print("\n=== 2.6) Cointegrated Pair Residuals Visualization ===")
        
        if max_pairs is not None:
            pairs = pairs[:max_pairs]
        
        fig, axes = plt.subplots(len(pairs), 1, figsize=(20, 5*len(pairs)))
        if len(pairs) == 1:
            axes = [axes]
        
        for i, pair in enumerate(tqdm(pairs, desc="Plotting residuals")):
            X = self.data_is[pair[0]]
            Y = self.data_is[pair[1]]
            
            # Perform regression
            model = sm.OLS(endog=Y, exog=X).fit()
            residuals = model.resid
            
            # Calculate z-score of residuals
            p_value = adfuller(model.resid)[1]
            residuals_zscore = (residuals - residuals.mean()) / residuals.std()
            
            # Plot residuals
            axes[i].plot(residuals_zscore.index, residuals_zscore.values, 
                        label=f'P-value = {p_value:.5f}', color='blue')
            axes[i].axhline(y=residuals_zscore.mean(), color="black", linestyle='-', alpha=0.7)
            axes[i].axhline(y=1, color="red", linestyle='--', alpha=0.7)
            axes[i].axhline(y=-1, color="red", linestyle='--', alpha=0.7)
            
            axes[i].set_ylabel('Z-Score', fontsize=12)
            axes[i].set_xlabel('Days', fontsize=12)
            axes[i].set_title(f"Endogenous = {pair[1]}, Exogenous = {pair[0]}, Beta = {model.params.values[0]:.2f}", fontsize=12)
            axes[i].legend(loc="upper left", fontsize=10)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def execute_long_short_strategy(self, data, pairs, capital_per_leg=100000, deviation_threshold=1.0):
        """
        Execute Long & Short trading strategy.
        
        In L&S operations, market direction (trending or sideways) is irrelevant.
        The approach is arbitrage between securities seeking relative performance
        between long and short positions.
        
        Strategy Rules:
        - When residual falls below -1 std dev: Buy Y, Sell X  
        - When residual rises above +1 std dev: Sell Y, Buy X
        - Exit when residual crosses zero (mean reversion)
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data for strategy execution
        pairs : list
            List of cointegrated pairs to trade
        capital_per_leg : float
            Capital allocated to each leg of the trade
        deviation_threshold : float
            Standard deviation threshold for entry signals
            
        Returns:
        --------
        tuple : (down_profits, up_profits) - Trading results for both directions
        """
        print("\n=== 3) Long & Short Trading Results ===")
        
        down_profits = []  # Results for downward residual entries
        up_profits = []    # Results for upward residual entries
        
        for i, pair in enumerate(tqdm(pairs, desc="Executing strategy")):
            X = data[pair[0]]
            Y = data[pair[1]]
            
            # Calculate regression parameters
            model = sm.OLS(exog=X, endog=Y).fit()
            residuals = model.resid
            beta = model.params[0]
            
            # Create trading DataFrame
            trading_data = pd.DataFrame(index=data.index)
            trading_data["X"] = X
            trading_data["Y"] = Y
            trading_data["residual"] = (residuals - residuals.mean()) / residuals.std()
            
            # Initialize signals
            trading_data['signal_open_down'] = 0
            trading_data['signal_open_up'] = 0
            
            # Generate trading signals
            signal_open_down = []
            signal_open_up = []
            open_down = 0
            open_up = 0
            
            for day in range(1, len(trading_data)):
                # Downward entry signals (residual crosses below -deviation_threshold)
                open_down_1 = trading_data['residual'].iloc[day-1] > -deviation_threshold
                open_down_2 = trading_data['residual'].iloc[day] < -deviation_threshold
                close_down_1 = trading_data['residual'].iloc[day-1] < 0
                close_down_2 = trading_data['residual'].iloc[day] > 0
                
                if open_down_1 & open_down_2:
                    open_down = -1
                if (open_down == -1) & close_down_1 & close_down_2:
                    open_down = 0
                signal_open_down.append(open_down)
                
                # Upward entry signals (residual crosses above +deviation_threshold)
                open_up_1 = trading_data['residual'].iloc[day-1] < deviation_threshold
                open_up_2 = trading_data['residual'].iloc[day] > deviation_threshold
                close_up_1 = trading_data['residual'].iloc[day-1] > 0
                close_up_2 = trading_data['residual'].iloc[day] < 0
                
                if open_up_1 & open_up_2:
                    open_up = 1
                if (open_up == 1) & close_up_1 & close_up_2:
                    open_up = 0
                signal_open_up.append(open_up)
            
            # Assign signals to DataFrame
            trading_data.loc[1:, 'signal_open_down'] = signal_open_down
            trading_data.loc[1:, 'signal_open_up'] = signal_open_up
            
            # Calculate trade profits for downward entries
            pos_start_down = np.where(trading_data["signal_open_down"].diff() < 0)[0]
            pos_close_down = np.where(trading_data["signal_open_down"].diff() > 0)[0]
            len_down = min(len(pos_start_down), len(pos_close_down))
            
            if len_down > 0:
                if beta > 0:
                    Y_down_profit = +(trading_data["Y"].iloc[pos_close_down].values[:len_down] - 
                                    trading_data["Y"].iloc[pos_start_down].values[:len_down])
                    X_down_profit = -(trading_data["X"].iloc[pos_close_down].values[:len_down] - 
                                    trading_data["X"].iloc[pos_start_down].values[:len_down])
                else:
                    Y_down_profit = +(trading_data["Y"].iloc[pos_close_down].values[:len_down] - 
                                    trading_data["Y"].iloc[pos_start_down].values[:len_down])
                    X_down_profit = +(trading_data["X"].iloc[pos_close_down].values[:len_down] - 
                                    trading_data["X"].iloc[pos_start_down].values[:len_down])
            else:
                Y_down_profit = np.array([])
                X_down_profit = np.array([])
            
            # Store downward trade results
            down_profits.append([
                pair[0], pair[1], X_down_profit, Y_down_profit,
                pos_start_down, pos_close_down,
                trading_data["X"], trading_data["Y"], trading_data['residual'], beta
            ])
            
            # Calculate trade profits for upward entries
            pos_start_up = np.where(trading_data["signal_open_up"].diff() > 0)[0]
            pos_close_up = np.where(trading_data["signal_open_up"].diff() < 0)[0]
            len_up = min(len(pos_start_up), len(pos_close_up))
            
            if len_up > 0:
                if beta > 0:
                    Y_up_profit = -(trading_data["Y"].iloc[pos_close_up].values[:len_up] - 
                                  trading_data["Y"].iloc[pos_start_up].values[:len_up])
                    X_up_profit = +(trading_data["X"].iloc[pos_close_up].values[:len_up] - 
                                  trading_data["X"].iloc[pos_start_up].values[:len_up])
                else:
                    Y_up_profit = -(trading_data["Y"].iloc[pos_close_up].values[:len_up] - 
                                  trading_data["Y"].iloc[pos_start_up].values[:len_up])
                    X_up_profit = -(trading_data["X"].iloc[pos_close_up].values[:len_up] - 
                                  trading_data["X"].iloc[pos_start_up].values[:len_up])
            else:
                Y_up_profit = np.array([])
                X_up_profit = np.array([])
            
            # Store upward trade results
            up_profits.append([
                pair[0], pair[1], X_up_profit, Y_up_profit,
                pos_start_up, pos_close_up,
                trading_data["X"], trading_data["Y"], trading_data['residual'], beta
            ])
        
        return down_profits, up_profits
    
    def calculate_strategy_performance(self, down_profits, up_profits, sample_name=""):
        """
        Calculate and display strategy performance metrics.
        
        Parameters:
        -----------
        down_profits : list
            Trading results for downward entries
        up_profits : list
            Trading results for upward entries
        sample_name : str
            Name identifier for the sample period
            
        Returns:
        --------
        pd.DataFrame : Performance summary by pair
        """
        print(f"\n=== 3.1) Performance Analysis with Beta Adjustment {sample_name} ===")
        
        results = []
        
        for i in range(len(down_profits)):
            # Calculate profit/loss with beta adjustment
            down_pnl = np.sum(down_profits[i][3] + down_profits[i][-1] * down_profits[i][2])
            up_pnl = np.sum(up_profits[i][3] + up_profits[i][-1] * up_profits[i][2])
            
            pair_name = f"{down_profits[i][0]}/{down_profits[i][1]}"
            results.append([pair_name, down_pnl, up_pnl])
        
        results_df = pd.DataFrame(results, columns=["Pair", "Profit Down", "Profit Up"])
        results_df = results_df.set_index("Pair")
        
        # Display summary statistics
        total_down_profit = results_df["Profit Down"].sum()
        total_up_profit = results_df["Profit Up"].sum()
        total_profit = total_down_profit + total_up_profit
        
        print(f"Total Down Profits: ${total_down_profit:.2f}")
        print(f"Total Up Profits: ${total_up_profit:.2f}")
        print(f"Total Strategy Profit: ${total_profit:.2f}")
        print(f"Number of Pairs Traded: {len(results_df)}")
        
        # Visualize results
        plt.figure(figsize=(15, 6))
        results_df.plot(kind='bar', figsize=(15, 6))
        plt.xlabel("Cointegrated Pairs", fontsize=12)
        plt.ylabel("Profit ($)", fontsize=12)
        plt.title(f"Profit by Entry Direction {sample_name}", fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return results_df
    
    def visualize_trade_analysis(self, profits_data, trade_type="Down", max_pairs=None):
        """
        Visualize detailed trade analysis for each pair.
        
        Parameters:
        -----------
        profits_data : list
            Trading results data
        trade_type : str
            Type of trades ("Down" or "Up")
        max_pairs : int, optional
            Maximum number of pairs to visualize
        """
        if max_pairs is not None:
            profits_data = profits_data[:max_pairs]
        
        fig, axes = plt.subplots(len(profits_data), 2, figsize=(20, 6*len(profits_data)))
        if len(profits_data) == 1:
            axes = axes.reshape(1, -1)
        
        for i, profit_data in enumerate(profits_data):
            # Extract trade data
            asset_x, asset_y = profit_data[0], profit_data[1]
            x_profits, y_profits = profit_data[2], profit_data[3]
            start_positions, close_positions = profit_data[4], profit_data[5]
            price_x, price_y, residuals, beta = profit_data[6], profit_data[7], profit_data[8], profit_data[9]
            
            # Plot X asset (first leg)
            if len(x_profits) > 0:
                axes[i, 0].bar(price_x.iloc[close_positions].index[:len(x_profits)], 
                              x_profits, width=6, alpha=0.7)
            axes[i, 0].plot(residuals.index, residuals, "g:", alpha=0.8)
            axes[i, 0].axhline(y=-1.0, color='r', linestyle='-', alpha=0.5)
            axes[i, 0].axhline(y=1.0, color='b', linestyle='-', alpha=0.5)
            axes[i, 0].axhline(y=0.0, color='k', linestyle='-', alpha=0.5)
            axes[i, 0].set_ylabel("Residual and Profit ($)", fontsize=10)
            
            # Twin axis for prices
            ax2 = axes[i, 0].twinx()
            ax2.plot(price_x.index, price_x, color='blue', alpha=0.7)
            if len(start_positions) > 0 and len(close_positions) > 0:
                ax2.plot(price_x.iloc[start_positions].index, 
                        price_x.iloc[start_positions], 'v', markersize=8, color='red', label='Sell')
                ax2.plot(price_x.iloc[close_positions].index, 
                        price_x.iloc[close_positions], '^', markersize=8, color='green', label='Buy')
            ax2.set_ylabel("Price ($)", fontsize=10)
            ax2.legend()
            axes[i, 0].set_title(f"{trade_type}: Beta={beta:.2f}, {asset_x} [X-Asset], Sum=${np.sum(x_profits):.2f}", fontsize=10)
            
            # Plot Y asset (second leg)
            if len(y_profits) > 0:
                axes[i, 1].bar(price_y.iloc[close_positions].index[:len(y_profits)], 
                              y_profits, width=6, alpha=0.7)
            axes[i, 1].plot(residuals.index, residuals, "g:", alpha=0.8)
            axes[i, 1].axhline(y=-1.0, color='r', linestyle='-', alpha=0.5)
            axes[i, 1].axhline(y=1.0, color='b', linestyle='-', alpha=0.5)
            axes[i, 1].axhline(y=0.0, color='k', linestyle='-', alpha=0.5)
            axes[i, 1].set_ylabel("Residual and Profit ($)", fontsize=10)
            
            # Twin axis for prices  
            ax2 = axes[i, 1].twinx()
            ax2.plot(price_y.index, price_y, color='orange', alpha=0.7)
            if len(start_positions) > 0 and len(close_positions) > 0:
                ax2.plot(price_y.iloc[start_positions].index, 
                        price_y.iloc[start_positions], '^', markersize=8, color='green', label='Buy')
                ax2.plot(price_y.iloc[close_positions].index, 
                        price_y.iloc[close_positions], 'v', markersize=8, color='red', label='Sell')
            ax2.set_ylabel("Price ($)", fontsize=10)
            ax2.legend()
            axes[i, 1].set_title(f"{trade_type}: {asset_y} [Y-Asset], Sum=${np.sum(y_profits):.2f}", fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self):
        """
        Run the complete Long & Short cointegration analysis.
        
        This method executes the full pipeline:
        1. Data acquisition and preparation
        2. Cointegration analysis
        3. Strategy execution (in-sample and out-of-sample)
        4. Performance evaluation
        
        Returns:
        --------
        dict : Complete analysis results
        """
        print("Starting Complete Long & Short Cointegration Analysis")
        print("=" * 60)
        
        # Step 1: Data acquisition
        self.acquire_market_data()
        self.split_sample_data()
        
        # Step 2: Cointegration analysis
        non_stationary_assets = self.test_stationarity(self.data_is)
        pairs_method1 = self.find_cointegrated_pairs_method1(non_stationary_assets)
        score_matrix, pvalue_matrix, pairs_method2 = self.find_cointegrated_pairs_method2(non_stationary_assets)
        
        # Find common pairs and rank by correlation
        common_pairs = self.identify_common_cointegrated_pairs(pairs_method1, pairs_method2)
        if len(common_pairs) == 0:
            print("No common cointegrated pairs found. Using Method 2 results.")
            common_pairs = pairs_method2[:10]  # Take top 10 from method 2
        
        ranked_pairs = self.rank_pairs_by_correlation(common_pairs)
        
        # Visualize top pairs (limit to avoid clutter)
        self.visualize_residuals(ranked_pairs[:min(5, len(ranked_pairs))])
        
        # Step 3: Strategy execution - In-Sample
        print("\n" + "="*50)
        print("IN-SAMPLE ANALYSIS")
        print("="*50)
        
        down_profits_is, up_profits_is = self.execute_long_short_strategy(self.data_is, ranked_pairs)
        results_is = self.calculate_strategy_performance(down_profits_is, up_profits_is, "(In-Sample)")
        
        # Visualize trade analysis for top pairs
        print("\nDetailed Trade Analysis (In-Sample):")
        self.visualize_trade_analysis(down_profits_is[:min(3, len(down_profits_is))], "Down", max_pairs=3)
        
        # Step 4: Strategy execution - Out-of-Sample  
        print("\n" + "="*50)
        print("OUT-OF-SAMPLE ANALYSIS")
        print("="*50)
        
        down_profits_oos, up_profits_oos = self.execute_long_short_strategy(self.data_oos, ranked_pairs)
        results_oos = self.calculate_strategy_performance(down_profits_oos, up_profits_oos, "(Out-of-Sample)")
        
        # Final considerations
        print("\n" + "="*50)
        print("5) FINAL CONSIDERATIONS")
        print("="*50)
        
        print("""
        Key Points for Long & Short Operations:
        
        1. STOP LOSS: Significant losses can occur; stop loss mechanisms are essential
        
        2. COINTEGRATION QUALITY: While cointegration indicates mean reversion,
           reasonable distortions and adequate frequency are important
        
        3. SAMPLE VALIDATION: Both in-sample and out-of-sample evaluation are 
           fundamental, including Walk Forward Analysis (WFA)
        
        4. TRANSACTION COSTS: Correct calculation of fees and margins is 
           fundamental to sustain operations
        
        5. OPERATIONAL CONSIDERATIONS: Even in primitive form, L&S operations
           can show winning potential over time
        
        Note: This implementation has educational nature. Making the strategy 
        operational requires important complementary studies and risk management.
        """)
        
        return {
            'data_is': self.data_is,
            'data_oos': self.data_oos,
            'cointegrated_pairs': ranked_pairs,
            'results_in_sample': results_is,
            'results_out_sample': results_oos,
            'down_profits_is': down_profits_is,
            'up_profits_is': up_profits_is,
            'down_profits_oos': down_profits_oos,
            'up_profits_oos': up_profits_oos
        }

def main():
    """
    Main execution function for the Long & Short Cointegration Strategy.
    """
    # Initialize strategy
    strategy = LongShortCointegrationStrategy()
    
    # Run complete analysis
    results = strategy.run_complete_analysis()
    
    print("\nAnalysis completed successfully!")
    print("All results have been stored in the results dictionary.")
    
    return results

if __name__ == "__main__":
    # Set style for plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Run the analysis
    analysis_results = main()
