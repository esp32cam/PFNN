import os
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
from scipy import stats
import networkx as nx
from matplotlib.patches import FancyBboxPatch
# from mesa.batchrunner import batch_run # Not currently used here

try:
    from sp500_data_loader import get_sp500_tickers
    from mesa_market_model import StockMarketModel
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    raise

def load_real_market_data(sp500_csv_path, tickers, max_days=200):
    """Load real market data for specified tickers"""
    real_data = {}
    for ticker in tickers:
        csv_file = os.path.join(sp500_csv_path, f"{ticker}.csv")
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                if 'close' in df.columns and len(df) >= max_days:
                    # Take the most recent max_days data points
                    real_data[ticker] = df['close'].tail(max_days).values
            except Exception as e:
                print(f"Error loading real data for {ticker}: {e}")
    return real_data

def load_simulation_data(base_output_dir, tickers):
    """Load simulation data for specified tickers"""
    sim_data = {}
    for ticker in tickers:
        ticker_dir = os.path.join(base_output_dir, ticker)
        safe_ticker = ticker.replace('/', '_').replace('.', '_')
        sim_file = os.path.join(ticker_dir, f"{safe_ticker}_simulation_data.csv")
        
        if os.path.exists(sim_file):
            try:
                df = pd.read_csv(sim_file, index_col=0)
                if 'Price' in df.columns:
                    sim_data[ticker] = df['Price'].values
            except Exception as e:
                print(f"Error loading simulation data for {ticker}: {e}")
    return sim_data

def calculate_correlation_matrix(data_dict):
    """Calculate correlation matrix from price data dictionary"""
    if len(data_dict) < 2:
        return None
    
    # Create DataFrame with aligned data
    tickers = list(data_dict.keys())
    min_length = min(len(data_dict[ticker]) for ticker in tickers)
    
    aligned_data = {}
    for ticker in tickers:
        aligned_data[ticker] = data_dict[ticker][:min_length]
    
    df = pd.DataFrame(aligned_data)
    return df.corr()

def calculate_pcmi(real_corr, sim_corr):
    """Calculate PCMI (Pearson Correlation Matrix Improvement) metrics"""
    if real_corr is None or sim_corr is None:
        return None
    
    # Align matrices - use intersection of tickers
    common_tickers = real_corr.index.intersection(sim_corr.index)
    if len(common_tickers) < 2:
        return None
    
    real_aligned = real_corr.loc[common_tickers, common_tickers]
    sim_aligned = sim_corr.loc[common_tickers, common_tickers]
    
    # Extract upper triangular values (excluding diagonal)
    mask = np.triu(np.ones_like(real_aligned), k=1).astype(bool)
    real_values = real_aligned.values[mask]
    sim_values = sim_aligned.values[mask]
    
    # Calculate metrics
    correlation, p_value = pearsonr(real_values, sim_values)
    mae = np.mean(np.abs(real_values - sim_values))
    rmse = np.sqrt(np.mean((real_values - sim_values) ** 2))
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'mae': mae,
        'rmse': rmse,
        'common_tickers': list(common_tickers),
        'n_pairs': len(real_values)
    }

def create_correlation_network_plots(real_corr, sim_corr, pcmi_metrics, output_dir, correlation_threshold=0.3):
    """Create network visualization of stock correlations"""
    if real_corr is None or sim_corr is None or pcmi_metrics is None:
        return
    
    common_tickers = pcmi_metrics['common_tickers']
    real_aligned = real_corr.loc[common_tickers, common_tickers]
    sim_aligned = sim_corr.loc[common_tickers, common_tickers]
    
    # Create network graphs
    def create_network_from_corr(corr_matrix, threshold=correlation_threshold):
        G = nx.Graph()
        # Add nodes
        for ticker in corr_matrix.index:
            G.add_node(ticker)
        
        # Add edges for correlations above threshold
        for i, ticker1 in enumerate(corr_matrix.index):
            for j, ticker2 in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicates
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > threshold:
                        G.add_edge(ticker1, ticker2, weight=abs(corr_val), correlation=corr_val)
        return G
    
    # Create networks
    real_network = create_network_from_corr(real_aligned, correlation_threshold)
    sim_network = create_network_from_corr(sim_aligned, correlation_threshold)
    
    # Plot network comparison
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Real market network
    pos_real = nx.spring_layout(real_network, k=3, iterations=50, seed=42)
    edges_real = real_network.edges()
    weights_real = [real_network[u][v]['weight'] for u, v in edges_real]
    colors_real = [real_network[u][v]['correlation'] for u, v in edges_real]
    
    nx.draw_networkx_nodes(real_network, pos_real, ax=axes[0,0], 
                          node_color='lightblue', node_size=800, alpha=0.8)
    edges_plot = nx.draw_networkx_edges(real_network, pos_real, ax=axes[0,0],
                                       width=[w*3 for w in weights_real],
                                       edge_color=colors_real, edge_cmap=plt.cm.RdBu_r,
                                       edge_vmin=-1, edge_vmax=1, alpha=0.7)
    nx.draw_networkx_labels(real_network, pos_real, ax=axes[0,0], font_size=8)
    axes[0,0].set_title(f'Real Market Correlation Network\n(|correlation| > {correlation_threshold})')
    axes[0,0].axis('off')
    
    # Simulated network
    pos_sim = nx.spring_layout(sim_network, k=3, iterations=50, seed=42)
    edges_sim = sim_network.edges()
    weights_sim = [sim_network[u][v]['weight'] for u, v in edges_sim]
    colors_sim = [sim_network[u][v]['correlation'] for u, v in edges_sim]
    
    nx.draw_networkx_nodes(sim_network, pos_sim, ax=axes[0,1], 
                          node_color='lightcoral', node_size=800, alpha=0.8)
    if edges_sim:  # Only draw edges if they exist
        nx.draw_networkx_edges(sim_network, pos_sim, ax=axes[0,1],
                              width=[w*3 for w in weights_sim],
                              edge_color=colors_sim, edge_cmap=plt.cm.RdBu_r,
                              edge_vmin=-1, edge_vmax=1, alpha=0.7)
    nx.draw_networkx_labels(sim_network, pos_sim, ax=axes[0,1], font_size=8)
    axes[0,1].set_title(f'ABM Simulation Correlation Network\n(|correlation| > {correlation_threshold})')
    axes[0,1].axis('off')
    
    # Network statistics comparison
    real_stats = {
        'Nodes': real_network.number_of_nodes(),
        'Edges': real_network.number_of_edges(),
        'Density': nx.density(real_network),
        'Avg Clustering': nx.average_clustering(real_network) if real_network.number_of_edges() > 0 else 0
    }
    
    sim_stats = {
        'Nodes': sim_network.number_of_nodes(),
        'Edges': sim_network.number_of_edges(),
        'Density': nx.density(sim_network),
        'Avg Clustering': nx.average_clustering(sim_network) if sim_network.number_of_edges() > 0 else 0
    }
    
    # Plot network statistics
    stats_df = pd.DataFrame([real_stats, sim_stats], index=['Real Market', 'ABM Simulation'])
    stats_df[['Edges', 'Density', 'Avg Clustering']].plot(kind='bar', ax=axes[1,0])
    axes[1,0].set_title('Network Statistics Comparison')
    axes[1,0].set_ylabel('Value')
    axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Degree distribution comparison
    real_degrees = [d for n, d in real_network.degree()]
    sim_degrees = [d for n, d in sim_network.degree()]
    
    axes[1,1].hist(real_degrees, bins=max(10, len(real_degrees)//3), alpha=0.7, 
                   label='Real Market', density=True)
    axes[1,1].hist(sim_degrees, bins=max(10, len(sim_degrees)//3), alpha=0.7, 
                   label='ABM Simulation', density=True)
    axes[1,1].set_xlabel('Node Degree')
    axes[1,1].set_ylabel('Density')
    axes[1,1].set_title('Degree Distribution Comparison')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pcmi_network_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create detailed correlation strength analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Correlation strength distribution
    real_corr_values = []
    sim_corr_values = []
    
    for i, ticker1 in enumerate(real_aligned.index):
        for j, ticker2 in enumerate(real_aligned.columns):
            if i < j:
                real_corr_values.append(real_aligned.iloc[i, j])
                sim_corr_values.append(sim_aligned.iloc[i, j])
    
    # Binned scatter plot by correlation strength
    corr_bins = [-1, -0.5, -0.1, 0.1, 0.5, 1.0]
    colors = ['red', 'orange', 'gray', 'lightblue', 'blue']
    
    for i in range(len(corr_bins)-1):
        mask = (np.array(real_corr_values) >= corr_bins[i]) & (np.array(real_corr_values) < corr_bins[i+1])
        if np.any(mask):
            axes[0,0].scatter(np.array(real_corr_values)[mask], np.array(sim_corr_values)[mask], 
                            c=colors[i], label=f'{corr_bins[i]:.1f} to {corr_bins[i+1]:.1f}', 
                            alpha=0.6, s=30)
    
    # Add diagonal line
    axes[0,0].plot([-1, 1], [-1, 1], 'k--', linewidth=2, alpha=0.8, label='Perfect Agreement')
    axes[0,0].set_xlabel('Real Market Correlations')
    axes[0,0].set_ylabel('ABM Simulation Correlations')
    axes[0,0].set_title('Correlation Comparison by Strength')
    axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,0].grid(True, alpha=0.3)
    
    # Strong correlation analysis (|correlation| > 0.5)
    strong_real = [x for x in real_corr_values if abs(x) > 0.5]
    strong_sim = [sim_corr_values[i] for i, x in enumerate(real_corr_values) if abs(x) > 0.5]
    
    if strong_real:
        axes[0,1].scatter(strong_real, strong_sim, alpha=0.7, s=50, c='red')
        axes[0,1].plot([-1, 1], [-1, 1], 'k--', linewidth=2, alpha=0.8)
        axes[0,1].set_xlabel('Real Market Strong Correlations')
        axes[0,1].set_ylabel('ABM Simulation Correlations')
        axes[0,1].set_title(f'Strong Correlations Only (|r| > 0.5)\nn = {len(strong_real)} pairs')
        axes[0,1].grid(True, alpha=0.3)
    else:
        axes[0,1].text(0.5, 0.5, 'No strong correlations\nfound in real data', 
                      ha='center', va='center', transform=axes[0,1].transAxes)
        axes[0,1].set_title('Strong Correlations Analysis')
    
    # Sector clustering analysis (if we can infer sectors from ticker patterns)
    # Simple sector grouping based on common patterns
    tech_tickers = [t for t in common_tickers if t in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD', 'INTC', 'ADBE']]
    financial_tickers = [t for t in common_tickers if t in ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'COF', 'SCHW']]
    healthcare_tickers = [t for t in common_tickers if t in ['JNJ', 'PFE', 'UNH', 'MRK', 'ABBV', 'TMO', 'DHR', 'BMY', 'AMGN']]
    
    sector_data = {
        'Technology': tech_tickers,
        'Financial': financial_tickers,
        'Healthcare': healthcare_tickers
    }
    
    # Calculate average intra-sector correlations
    sector_results = {}
    for sector, tickers in sector_data.items():
        if len(tickers) > 1:
            sector_real_corrs = []
            sector_sim_corrs = []
            for i, t1 in enumerate(tickers):
                for j, t2 in enumerate(tickers):
                    if i < j and t1 in real_aligned.index and t2 in real_aligned.index:
                        sector_real_corrs.append(real_aligned.loc[t1, t2])
                        sector_sim_corrs.append(sim_aligned.loc[t1, t2])
            
            if sector_real_corrs:
                sector_results[sector] = {
                    'real_avg': np.mean(sector_real_corrs),
                    'sim_avg': np.mean(sector_sim_corrs),
                    'real_std': np.std(sector_real_corrs),
                    'sim_std': np.std(sector_sim_corrs)
                }
    
    if sector_results:
        sectors = list(sector_results.keys())
        real_avgs = [sector_results[s]['real_avg'] for s in sectors]
        sim_avgs = [sector_results[s]['sim_avg'] for s in sectors]
        
        x = np.arange(len(sectors))
        width = 0.35
        
        axes[1,0].bar(x - width/2, real_avgs, width, label='Real Market', alpha=0.8)
        axes[1,0].bar(x + width/2, sim_avgs, width, label='ABM Simulation', alpha=0.8)
        axes[1,0].set_xlabel('Sector')
        axes[1,0].set_ylabel('Average Intra-Sector Correlation')
        axes[1,0].set_title('Sector Correlation Analysis')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(sectors)
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    else:
        axes[1,0].text(0.5, 0.5, 'Insufficient sector data\nfor analysis', 
                      ha='center', va='center', transform=axes[1,0].transAxes)
        axes[1,0].set_title('Sector Analysis')
    
    # Error analysis by correlation magnitude
    corr_mags = np.abs(real_corr_values)
    errors = np.abs(np.array(real_corr_values) - np.array(sim_corr_values))
    
    # Bin by correlation magnitude
    mag_bins = np.linspace(0, 1, 11)
    binned_errors = []
    bin_centers = []
    
    for i in range(len(mag_bins)-1):
        mask = (corr_mags >= mag_bins[i]) & (corr_mags < mag_bins[i+1])
        if np.any(mask):
            binned_errors.append(errors[mask])
            bin_centers.append((mag_bins[i] + mag_bins[i+1]) / 2)
    
    if binned_errors:
        axes[1,1].boxplot(binned_errors, positions=bin_centers, widths=0.05)
        axes[1,1].set_xlabel('|Real Correlation|')
        axes[1,1].set_ylabel('Absolute Error')
        axes[1,1].set_title('Prediction Error by Correlation Strength')
        axes[1,1].grid(True, alpha=0.3)
    else:
        axes[1,1].text(0.5, 0.5, 'Error analysis\nnot available', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Error Analysis')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pcmi_detailed_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    return real_network, sim_network, sector_results

def create_pcmi_plots(real_corr, sim_corr, pcmi_metrics, output_dir):
    """Create PCMI comparison plots"""
    if real_corr is None or sim_corr is None or pcmi_metrics is None:
        return
    
    common_tickers = pcmi_metrics['common_tickers']
    real_aligned = real_corr.loc[common_tickers, common_tickers]
    sim_aligned = sim_corr.loc[common_tickers, common_tickers]
    
    # Plot 1: Side-by-side correlation matrices
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Real market correlation matrix
    sns.heatmap(real_aligned, annot=False, cmap='RdBu_r', center=0, 
                square=True, ax=axes[0], cbar_kws={'shrink': 0.8})
    axes[0].set_title('Real Market Correlation Matrix')
    axes[0].set_xlabel('Stocks')
    axes[0].set_ylabel('Stocks')
    
    # Simulated correlation matrix
    sns.heatmap(sim_aligned, annot=False, cmap='RdBu_r', center=0, 
                square=True, ax=axes[1], cbar_kws={'shrink': 0.8})
    axes[1].set_title('ABM Simulation Correlation Matrix')
    axes[1].set_xlabel('Stocks')
    axes[1].set_ylabel('Stocks')
    
    # Difference matrix
    diff_matrix = sim_aligned - real_aligned
    sns.heatmap(diff_matrix, annot=False, cmap='RdBu_r', center=0, 
                square=True, ax=axes[2], cbar_kws={'shrink': 0.8})
    axes[2].set_title('Difference (Sim - Real)')
    axes[2].set_xlabel('Stocks')
    axes[2].set_ylabel('Stocks')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pcmi_correlation_matrices.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Scatter plot of correlations
    mask = np.triu(np.ones_like(real_aligned), k=1).astype(bool)
    real_values = real_aligned.values[mask]
    sim_values = sim_aligned.values[mask]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(real_values, sim_values, alpha=0.6, s=50)
    
    # Add diagonal line
    min_val = min(real_values.min(), sim_values.min())
    max_val = max(real_values.max(), sim_values.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Agreement')
    
    # Add best fit line
    z = np.polyfit(real_values, sim_values, 1)
    p = np.poly1d(z)
    plt.plot(real_values, p(real_values), "b-", linewidth=2, alpha=0.8, label=f'Best Fit (slope={z[0]:.3f})')
    
    plt.xlabel('Real Market Correlations')
    plt.ylabel('ABM Simulation Correlations')
    plt.title(f'PCMI Analysis: Real vs Simulated Correlations\n'
              f'Pearson r = {pcmi_metrics["correlation"]:.3f}, '
              f'MAE = {pcmi_metrics["mae"]:.3f}, '
              f'RMSE = {pcmi_metrics["rmse"]:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pcmi_scatter_plot.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram comparison
    axes[0].hist(real_values, bins=30, alpha=0.7, label='Real Market', density=True)
    axes[0].hist(sim_values, bins=30, alpha=0.7, label='ABM Simulation', density=True)
    axes[0].set_xlabel('Correlation Values')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Distribution of Correlation Values')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot comparison
    box_data = [real_values, sim_values]
    box_labels = ['Real Market', 'ABM Simulation']
    axes[1].boxplot(box_data, labels=box_labels)
    axes[1].set_ylabel('Correlation Values')
    axes[1].set_title('Box Plot Comparison')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pcmi_distribution_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()

def perform_pcmi_analysis(base_output_dir, sp500_csv_path, max_stocks=50):
    """Perform PCMI analysis comparing simulation with real market data"""
    print("\n--- Starting PCMI Analysis ---")
    
    # Get list of processed tickers
    processed_tickers = [d for d in os.listdir(base_output_dir) 
                        if os.path.isdir(os.path.join(base_output_dir, d)) and d != '__pycache__']
    
    if len(processed_tickers) < 2:
        print("Need at least 2 stocks for correlation analysis.")
        return
    
    # Limit number of stocks for manageable computation
    if len(processed_tickers) > max_stocks:
        processed_tickers = processed_tickers[:max_stocks]
        print(f"Limited analysis to {max_stocks} stocks for computational efficiency.")
    
    print(f"Performing PCMI analysis on {len(processed_tickers)} stocks.")
    
    # Load data
    print("Loading simulation data...")
    sim_data = load_simulation_data(base_output_dir, processed_tickers)
    
    print("Loading real market data...")
    real_data = load_real_market_data(sp500_csv_path, processed_tickers)
    
    # Find common tickers
    common_tickers = set(sim_data.keys()).intersection(set(real_data.keys()))
    if len(common_tickers) < 2:
        print("Need at least 2 common tickers between simulation and real data.")
        return
    
    common_tickers = list(common_tickers)
    print(f"Found {len(common_tickers)} common tickers for analysis.")
    
    # Filter data to common tickers
    sim_data_filtered = {ticker: sim_data[ticker] for ticker in common_tickers}
    real_data_filtered = {ticker: real_data[ticker] for ticker in common_tickers}
    
    # Calculate correlation matrices
    print("Calculating correlation matrices...")
    real_corr = calculate_correlation_matrix(real_data_filtered)
    sim_corr = calculate_correlation_matrix(sim_data_filtered)
    
    # Calculate PCMI metrics
    print("Calculating PCMI metrics...")
    pcmi_metrics = calculate_pcmi(real_corr, sim_corr)
    
    if pcmi_metrics is None:
        print("Could not calculate PCMI metrics.")
        return
    
    # Print results
    print(f"\n--- PCMI Analysis Results ---")
    print(f"Number of stocks analyzed: {len(common_tickers)}")
    print(f"Number of correlation pairs: {pcmi_metrics['n_pairs']}")
    print(f"Pearson correlation between real and simulated correlations: {pcmi_metrics['correlation']:.4f}")
    print(f"P-value: {pcmi_metrics['p_value']:.4e}")
    print(f"Mean Absolute Error (MAE): {pcmi_metrics['mae']:.4f}")
    print(f"Root Mean Square Error (RMSE): {pcmi_metrics['rmse']:.4f}")
      # Create plots
    print("Creating PCMI plots...")
    create_pcmi_plots(real_corr, sim_corr, pcmi_metrics, base_output_dir)
    
    # Create network analysis plots
    print("Creating network analysis plots...")
    real_net, sim_net, sector_results = create_correlation_network_plots(
        real_corr, sim_corr, pcmi_metrics, base_output_dir, correlation_threshold=0.3
    )
      # Save detailed results
    results_file = os.path.join(base_output_dir, "pcmi_analysis_results.txt")
    with open(results_file, 'w') as f:
        f.write("PCMI Analysis Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now()}\n")
        f.write(f"Number of stocks analyzed: {len(common_tickers)}\n")
        f.write(f"Stocks included: {', '.join(sorted(common_tickers))}\n\n")
        
        f.write("Correlation Analysis:\n")
        f.write(f"Correlation pairs analyzed: {pcmi_metrics['n_pairs']}\n")
        f.write(f"Pearson correlation (real vs sim): {pcmi_metrics['correlation']:.6f}\n")
        f.write(f"P-value: {pcmi_metrics['p_value']:.6e}\n")
        f.write(f"Mean Absolute Error: {pcmi_metrics['mae']:.6f}\n")
        f.write(f"Root Mean Square Error: {pcmi_metrics['rmse']:.6f}\n\n")
        
        # Network statistics
        if 'real_net' in locals() and 'sim_net' in locals():
            f.write("Network Analysis (threshold = 0.3):\n")
            f.write(f"Real Market Network:\n")
            f.write(f"  - Nodes: {real_net.number_of_nodes()}\n")
            f.write(f"  - Edges: {real_net.number_of_edges()}\n")
            f.write(f"  - Density: {nx.density(real_net):.4f}\n")
            f.write(f"  - Avg Clustering: {nx.average_clustering(real_net):.4f}\n\n")
            
            f.write(f"ABM Simulation Network:\n")
            f.write(f"  - Nodes: {sim_net.number_of_nodes()}\n")
            f.write(f"  - Edges: {sim_net.number_of_edges()}\n")
            f.write(f"  - Density: {nx.density(sim_net):.4f}\n")
            f.write(f"  - Avg Clustering: {nx.average_clustering(sim_net):.4f}\n\n")
        
        # Sector analysis results
        if 'sector_results' in locals() and sector_results:
            f.write("Sector Analysis:\n")
            for sector, stats in sector_results.items():
                f.write(f"{sector}:\n")
                f.write(f"  - Real avg correlation: {stats['real_avg']:.4f} ± {stats['real_std']:.4f}\n")
                f.write(f"  - Sim avg correlation: {stats['sim_avg']:.4f} ± {stats['sim_std']:.4f}\n")
            f.write("\n")
        
        f.write("Interpretation:\n")
        if pcmi_metrics['correlation'] > 0.7:
            f.write("- Strong positive correlation between real and simulated correlation patterns\n")
        elif pcmi_metrics['correlation'] > 0.5:
            f.write("- Moderate positive correlation between real and simulated correlation patterns\n")
        elif pcmi_metrics['correlation'] > 0.3:
            f.write("- Weak positive correlation between real and simulated correlation patterns\n")
        else:
            f.write("- Poor correlation between real and simulated correlation patterns\n")
            
        if pcmi_metrics['p_value'] < 0.05:
            f.write("- Correlation is statistically significant (p < 0.05)\n")
        else:
            f.write("- Correlation is not statistically significant (p >= 0.05)\n")
            
        f.write("\nRecommendations:\n")
        if pcmi_metrics['correlation'] < 0.3:
            f.write("- Consider adjusting ABM parameters to better capture market correlations\n")
            f.write("- Review agent interaction mechanisms\n")
            f.write("- Investigate sector-specific behaviors\n")
        elif pcmi_metrics['mae'] > 0.5:
            f.write("- Correlation direction is reasonable but magnitude needs improvement\n")
            f.write("- Fine-tune influence strength and price impact factors\n")
        else:
            f.write("- Model shows reasonable correlation structure\n")
            f.write("- Consider testing on different time periods\n")
    
    print(f"PCMI analysis complete. Results saved to {results_file}")
    return pcmi_metrics

def perform_predictive_validation(real_data, sim_data, output_dir, prediction_horizon=10):
    """
    Analyze ABM's predictive capability by comparing ABM forecasts with actual future stock movements
    
    Args:
        real_data: Dictionary of real stock price data
        sim_data: Dictionary of simulated stock price data
        output_dir: Directory to save analysis results
        prediction_horizon: Number of days ahead to validate predictions
    """
    print("Performing predictive validation analysis...")
    
    results = {}
    validation_metrics = {}
    
    # Common tickers
    common_tickers = set(real_data.keys()).intersection(set(sim_data.keys()))
    if len(common_tickers) < 2:
        print("Not enough common tickers for predictive validation")
        return None
    
    print(f"Analyzing predictive capability for {len(common_tickers)} stocks")
    
    for ticker in common_tickers:
        real_prices = real_data[ticker]
        sim_prices = sim_data[ticker]
        
        # Ensure we have enough data for prediction validation
        min_length = min(len(real_prices), len(sim_prices))
        if min_length < prediction_horizon + 20:
            continue
            
        # Split data: use first part for "training", last part for validation
        split_point = min_length - prediction_horizon
        
        # Real data: historical and future
        real_historical = real_prices[:split_point]
        real_future = real_prices[split_point:split_point + prediction_horizon]
        
        # Simulated data: use as prediction for the future period
        sim_historical = sim_prices[:split_point]
        sim_prediction = sim_prices[split_point:split_point + prediction_horizon]
        
        # Calculate price change predictions vs reality
        real_changes = np.diff(real_future) / real_future[:-1] * 100  # Percentage changes
        sim_changes = np.diff(sim_prediction) / sim_prediction[:-1] * 100
        
        # Calculate direction accuracy (up/down prediction)
        real_directions = np.sign(real_changes)
        sim_directions = np.sign(sim_changes)
        direction_accuracy = np.mean(real_directions == sim_directions)
        
        # Calculate magnitude correlation
        magnitude_corr, mag_p_value = pearsonr(np.abs(real_changes), np.abs(sim_changes))
        
        # Calculate price level accuracy
        price_mae = np.mean(np.abs(real_future - sim_prediction))
        price_rmse = np.sqrt(np.mean((real_future - sim_prediction) ** 2))
        price_mape = np.mean(np.abs((real_future - sim_prediction) / real_future)) * 100
        
        # Store results
        results[ticker] = {
            'real_historical': real_historical,
            'real_future': real_future,
            'sim_historical': sim_historical,
            'sim_prediction': sim_prediction,
            'real_changes': real_changes,
            'sim_changes': sim_changes,
            'direction_accuracy': direction_accuracy,
            'magnitude_correlation': magnitude_corr,
            'magnitude_p_value': mag_p_value,
            'price_mae': price_mae,
            'price_rmse': price_rmse,
            'price_mape': price_mape
        }
    
    # Calculate overall validation metrics
    if results:
        direction_accuracies = [results[t]['direction_accuracy'] for t in results]
        magnitude_correlations = [results[t]['magnitude_correlation'] for t in results if not np.isnan(results[t]['magnitude_correlation'])]
        price_mapes = [results[t]['price_mape'] for t in results]
        
        validation_metrics = {
            'mean_direction_accuracy': np.mean(direction_accuracies),
            'std_direction_accuracy': np.std(direction_accuracies),
            'mean_magnitude_correlation': np.mean(magnitude_correlations) if magnitude_correlations else 0,
            'std_magnitude_correlation': np.std(magnitude_correlations) if magnitude_correlations else 0,
            'mean_price_mape': np.mean(price_mapes),
            'std_price_mape': np.std(price_mapes),
            'num_stocks_analyzed': len(results),
            'prediction_horizon': prediction_horizon
        }
    
    # Create visualization
    create_predictive_validation_plots(results, validation_metrics, output_dir)
    
    # Save detailed results
    save_predictive_validation_results(results, validation_metrics, output_dir)
    
    return validation_metrics

def create_predictive_validation_plots(results, validation_metrics, output_dir):
    """Create comprehensive plots for predictive validation analysis"""
    if not results:
        return
    
    # Create multiple subplots for different aspects of validation
    fig = plt.figure(figsize=(20, 24))
    
    # Select top 6 stocks for detailed plotting
    sorted_stocks = sorted(results.keys(), 
                          key=lambda x: results[x]['direction_accuracy'], 
                          reverse=True)[:6]
    
    # 1. Individual stock prediction plots (2x3 grid)
    for i, ticker in enumerate(sorted_stocks):
        ax = plt.subplot(6, 3, i + 1)
        data = results[ticker]
        
        # Plot historical and predicted prices
        hist_days = range(len(data['real_historical']))
        future_days = range(len(data['real_historical']), 
                          len(data['real_historical']) + len(data['real_future']))
        
        plt.plot(hist_days, data['real_historical'], 'b-', label='Real Historical', alpha=0.7)
        plt.plot(future_days, data['real_future'], 'b--', label='Real Future', linewidth=2)
        plt.plot(future_days, data['sim_prediction'], 'r--', label='ABM Prediction', linewidth=2)
        
        plt.axvline(x=len(data['real_historical'])-1, color='gray', linestyle=':', alpha=0.7)
        plt.title(f'{ticker}\nDir. Acc: {data["direction_accuracy"]:.2f}, MAPE: {data["price_mape"]:.1f}%')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
    
    # 2. Direction accuracy distribution
    ax = plt.subplot(6, 3, 7)
    direction_accuracies = [results[t]['direction_accuracy'] for t in results]
    plt.hist(direction_accuracies, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(direction_accuracies), color='red', linestyle='--', 
                label=f'Mean: {np.mean(direction_accuracies):.3f}')
    plt.axvline(0.5, color='gray', linestyle=':', label='Random (0.5)')
    plt.xlabel('Direction Accuracy')
    plt.ylabel('Number of Stocks')
    plt.title('Distribution of Direction Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Magnitude correlation distribution
    ax = plt.subplot(6, 3, 8)
    magnitude_corrs = [results[t]['magnitude_correlation'] for t in results 
                      if not np.isnan(results[t]['magnitude_correlation'])]
    if magnitude_corrs:
        plt.hist(magnitude_corrs, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.axvline(np.mean(magnitude_corrs), color='red', linestyle='--',
                    label=f'Mean: {np.mean(magnitude_corrs):.3f}')
        plt.axvline(0, color='gray', linestyle=':', label='No Correlation')
    plt.xlabel('Magnitude Correlation')
    plt.ylabel('Number of Stocks')
    plt.title('Distribution of Change Magnitude Correlation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. MAPE distribution
    ax = plt.subplot(6, 3, 9)
    mapes = [results[t]['price_mape'] for t in results]
    plt.hist(mapes, bins=15, alpha=0.7, color='orange', edgecolor='black')
    plt.axvline(np.mean(mapes), color='red', linestyle='--',
                label=f'Mean: {np.mean(mapes):.1f}%')
    plt.xlabel('MAPE (%)')
    plt.ylabel('Number of Stocks')
    plt.title('Distribution of Price Prediction Error (MAPE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Scatter plot: Direction accuracy vs Magnitude correlation
    ax = plt.subplot(6, 3, 10)
    dir_acc = [results[t]['direction_accuracy'] for t in results]
    mag_corr = [results[t]['magnitude_correlation'] for t in results 
               if not np.isnan(results[t]['magnitude_correlation'])]
    if len(dir_acc) == len(mag_corr):
        plt.scatter(dir_acc, mag_corr, alpha=0.7, s=50)
        plt.xlabel('Direction Accuracy')
        plt.ylabel('Magnitude Correlation')
        plt.title('Direction Accuracy vs Magnitude Correlation')
        plt.grid(True, alpha=0.3)
        # Add correlation coefficient
        if len(dir_acc) > 2:
            corr_coef, _ = pearsonr(dir_acc, mag_corr)
            plt.text(0.05, 0.95, f'r = {corr_coef:.3f}', transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 6. Performance ranking
    ax = plt.subplot(6, 3, 11)
    # Create composite score: direction accuracy + (1 - normalized MAPE)
    scores = {}
    max_mape = max(results[t]['price_mape'] for t in results)
    for ticker in results:
        norm_mape = results[ticker]['price_mape'] / max_mape if max_mape > 0 else 0
        scores[ticker] = results[ticker]['direction_accuracy'] + (1 - norm_mape)
    
    sorted_tickers = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:10]
    score_values = [scores[t] for t in sorted_tickers]
    
    bars = plt.barh(range(len(sorted_tickers)), score_values, alpha=0.7, color='purple')
    plt.yticks(range(len(sorted_tickers)), sorted_tickers)
    plt.xlabel('Composite Performance Score')
    plt.title('Top 10 Stocks by Prediction Performance')
    plt.grid(True, alpha=0.3)
    
    # 7. Summary statistics box
    ax = plt.subplot(6, 3, 12)
    ax.axis('off')
    
    summary_text = f"""
    PREDICTIVE VALIDATION SUMMARY
    
    Stocks Analyzed: {validation_metrics['num_stocks_analyzed']}
    Prediction Horizon: {validation_metrics['prediction_horizon']} days
    
    DIRECTION ACCURACY:
    Mean: {validation_metrics['mean_direction_accuracy']:.3f}
    Std: {validation_metrics['std_direction_accuracy']:.3f}
    
    MAGNITUDE CORRELATION:
    Mean: {validation_metrics['mean_magnitude_correlation']:.3f}
    Std: {validation_metrics['std_magnitude_correlation']:.3f}
    
    PRICE ERROR (MAPE):
    Mean: {validation_metrics['mean_price_mape']:.1f}%
    Std: {validation_metrics['std_price_mape']:.1f}%
    
    INTERPRETATION:
    Direction Accuracy > 0.5: {"Good" if validation_metrics['mean_direction_accuracy'] > 0.5 else "Poor"}
    Magnitude Correlation > 0.3: {"Good" if validation_metrics['mean_magnitude_correlation'] > 0.3 else "Poor"}
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictive_validation_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def save_predictive_validation_results(results, validation_metrics, output_dir):
    """Save detailed predictive validation results to text file"""
    output_file = os.path.join(output_dir, 'predictive_validation_results.txt')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("PREDICTIVE VALIDATION ANALYSIS RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("OVERVIEW:\n")
        f.write(f"Number of stocks analyzed: {validation_metrics['num_stocks_analyzed']}\n")
        f.write(f"Prediction horizon: {validation_metrics['prediction_horizon']} days\n\n")
        
        f.write("AGGREGATE METRICS:\n")
        f.write(f"Mean Direction Accuracy: {validation_metrics['mean_direction_accuracy']:.4f} ± {validation_metrics['std_direction_accuracy']:.4f}\n")
        f.write(f"Mean Magnitude Correlation: {validation_metrics['mean_magnitude_correlation']:.4f} ± {validation_metrics['std_magnitude_correlation']:.4f}\n")
        f.write(f"Mean Price Error (MAPE): {validation_metrics['mean_price_mape']:.2f}% ± {validation_metrics['std_price_mape']:.2f}%\n\n")
        
        f.write("INTERPRETATION:\n")
        f.write(f"Direction Prediction: {'GOOD' if validation_metrics['mean_direction_accuracy'] > 0.5 else 'POOR'} ")
        f.write(f"(>{0.5} is better than random)\n")
        f.write(f"Magnitude Correlation: {'GOOD' if validation_metrics['mean_magnitude_correlation'] > 0.3 else 'POOR'} ")
        f.write(f"(>0.3 indicates meaningful relationship)\n")
        f.write(f"Price Accuracy: {'GOOD' if validation_metrics['mean_price_mape'] < 5 else 'MODERATE' if validation_metrics['mean_price_mape'] < 15 else 'POOR'} ")
        f.write(f"(<5% excellent, <15% moderate, >15% poor)\n\n")
        
        f.write("DETAILED STOCK-BY-STOCK RESULTS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Ticker':<8} {'Dir.Acc':<8} {'Mag.Corr':<8} {'MAPE%':<8} {'MAE':<10} {'RMSE':<10}\n")
        f.write("-" * 80 + "\n")
        
        # Sort by direction accuracy
        sorted_stocks = sorted(results.keys(), 
                              key=lambda x: results[x]['direction_accuracy'], 
                              reverse=True)
        
        for ticker in sorted_stocks:
            data = results[ticker]
            f.write(f"{ticker:<8} {data['direction_accuracy']:<8.3f} ")
            f.write(f"{data['magnitude_correlation']:<8.3f} ")
            f.write(f"{data['price_mape']:<8.1f} ")
            f.write(f"{data['price_mae']:<10.2f} ")
            f.write(f"{data['price_rmse']:<10.2f}\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("RECOMMENDATIONS:\n\n")
        
        if validation_metrics['mean_direction_accuracy'] > 0.6:
            f.write("[+] STRONG directional prediction capability detected!\n")
        elif validation_metrics['mean_direction_accuracy'] > 0.5:
            f.write("[+] MODERATE directional prediction capability detected.\n")
        else:
            f.write("[-] POOR directional prediction capability. Consider:\n")
            f.write("  - Adjusting ABM parameters\n")
            f.write("  - Improving PFNN attractor selection\n")
            f.write("  - Adding more market factors to the model\n")
        
        if validation_metrics['mean_magnitude_correlation'] > 0.3:
            f.write("[+] GOOD correlation between predicted and actual change magnitudes.\n")
        else:
            f.write("[-] POOR magnitude correlation. The model may need:\n")
            f.write("  - Better volatility modeling\n")
            f.write("  - Improved agent behavior calibration\n")
        
        if validation_metrics['mean_price_mape'] < 10:
            f.write("[+] ACCEPTABLE price prediction accuracy.\n")
        else:
            f.write("[-] HIGH price prediction error. Consider:\n")
            f.write("  - Normalizing price predictions\n")
            f.write("  - Using relative changes instead of absolute prices\n")
            f.write("  - Improving model calibration\n")

def main():
    print("Starting Multi-Stock ABM Simulation Run...")

    abm_parameters = {
        "num_agents": 2000,
        "initial_price": 100.0,
        "price_impact_factor": 0.05,
        "influence_strength": 0.75,
        "attractor_stats_base_path": "stock_analysis_results"
    }
    max_simulation_steps = 200
    base_output_dir = "multistock_abm_outputs"
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"Output will be saved in: {base_output_dir}")

    all_tickers_full_list = get_sp500_tickers()
    all_tickers_filtered = [ticker for ticker in all_tickers_full_list if ticker and isinstance(ticker, str)]

    test_ticker_limit = 500
    if test_ticker_limit is not None:
        all_tickers = all_tickers_filtered[:test_ticker_limit]
        print(f"Processing a subset of {len(all_tickers)} tickers (limit was {test_ticker_limit}).")
    else:
        all_tickers = all_tickers_filtered
        print(f"Processing all {len(all_tickers)} fetched and filtered tickers.")

    if not all_tickers:
        print("No tickers to process. Exiting."); return

    print(f"\nABM parameters: {abm_parameters}")
    print(f"Max simulation steps per stock: {max_simulation_steps}")

    successful_runs, failed_runs, skipped_runs_no_attractor = 0, 0, 0

    for i, ticker in enumerate(all_tickers):
        print(f"\n--- Processing ticker {i+1}/{len(all_tickers)}: {ticker} ---")
        ticker_specific_output_dir = os.path.join(base_output_dir, ticker)
        os.makedirs(ticker_specific_output_dir, exist_ok=True)

        try:
            model_instance = StockMarketModel(
                num_agents=abm_parameters["num_agents"],
                initial_price=abm_parameters["initial_price"],
                price_impact_factor=abm_parameters["price_impact_factor"],
                influence_strength=abm_parameters["influence_strength"],
                ticker_for_attractor=ticker,
                attractor_stats_base_path=abm_parameters["attractor_stats_base_path"]
            )

            if model_instance.attractor_influence_vector is None:
                print(f"Attractor info not loaded for {ticker}. Skipping ABM run.")
                skipped_runs_no_attractor += 1; continue

            print(f"Attractor info loaded. Running simulation for {max_simulation_steps} steps...")
            for _ in range(max_simulation_steps): model_instance.step()
            print(f"Simulation complete for {ticker}.")

            model_data_df = model_instance.datacollector.get_model_vars_dataframe()
            if model_data_df is not None and not model_data_df.empty:
                safe_ticker_fname = ticker.replace('/', '_').replace('.', '_')
                csv_filename = os.path.join(ticker_specific_output_dir, f"{safe_ticker_fname}_simulation_data.csv")
                model_data_df.to_csv(csv_filename, index=True)
                print(f"Saved simulation data for {ticker} to {csv_filename}")

                try:
                    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                    fig.suptitle(f"ABM Simulation for {ticker} (Influenced)", fontsize=14)
                    axes[0].plot(model_data_df.index, model_data_df['Price'], label='Price')
                    axes[0].set_ylabel('Price'); axes[0].grid(True); axes[0].legend()
                    axes[1].bar(model_data_df.index, model_data_df['Volume'], label='Volume', color='orange')
                    axes[1].set_xlabel('Step'); axes[1].set_ylabel('Volume'); axes[1].grid(True); axes[1].legend()
                    plt.tight_layout(rect=[0, 0, 1, 0.96])
                    plot_filename = os.path.join(ticker_specific_output_dir, f"{safe_ticker_fname}_simulation_plot.png")
                    plt.savefig(plot_filename); plt.close(fig)
                    print(f"Saved simulation plot for {ticker} to {plot_filename}")
                except Exception as e_plot: print(f"Error generating plot for {ticker}: {e_plot}")
                successful_runs += 1
            else: print(f"No data collected for {ticker}."); failed_runs +=1
        except Exception as e_model_run:
            print(f"Error during ABM simulation for {ticker}: {e_model_run}")
            failed_runs += 1

    print(f"\n--- Multi-Stock ABM Simulation Summary ---")
    print(f"Total tickers targeted: {len(all_tickers)}")
    print(f"Successful ABM runs: {successful_runs}")
    print(f"Failed ABM runs: {failed_runs}")
    print(f"Skipped (no attractor data): {skipped_runs_no_attractor}")

    print("\n--- Starting Basic Cross-Stock Analysis ---")
    all_summary_stats = []
    processed_tickers_summary = [d for d in os.listdir(base_output_dir) if os.path.isdir(os.path.join(base_output_dir, d))]

    if not processed_tickers_summary: print("No processed ticker data for cross-stock analysis.")
    else:
        print(f"Found data for {len(processed_tickers_summary)} tickers for cross-stock analysis.")
        for ticker_dir_name in processed_tickers_summary:
            safe_ticker_part = ticker_dir_name.replace('/', '_').replace('.', '_') # Should match dir name
            sim_data_path = os.path.join(base_output_dir, ticker_dir_name, f"{safe_ticker_part}_simulation_data.csv")
            if os.path.exists(sim_data_path):
                try:
                    sim_df = pd.read_csv(sim_data_path, index_col=0)
                    if not sim_df.empty and 'Price' in sim_df.columns and 'Volume' in sim_df.columns:
                        all_summary_stats.append({
                            'Ticker': ticker_dir_name,
                            'MeanPrice': sim_df['Price'].mean(),
                            'FinalPrice': sim_df['Price'].iloc[-1] if not sim_df['Price'].empty else None,
                            'TotalVolume': sim_df['Volume'].sum(),
                            'PriceVolatility (StdDev)': sim_df['Price'].std(),
                            'AttractorFileUsed': sim_df['AttractorLoaded'].all() if 'AttractorLoaded' in sim_df.columns else False
                        })
                except Exception as e_stat: print(f"Error processing sim data for {ticker_dir_name}: {e_stat}")

        if all_summary_stats:
            summary_df = pd.DataFrame(all_summary_stats)
            summary_csv_path = os.path.join(base_output_dir, "multistock_abm_summary_stats.csv")
            try:
                summary_df.to_csv(summary_csv_path, index=False)
                print(f"\nSaved cross-stock summary to {summary_csv_path}")
                print("Summary Statistics (first 5 rows):\n", summary_df.head())
                if not summary_df.empty:
                    plt.figure(figsize=(12, 7))
                    plot_df_s = summary_df.sort_values(by='FinalPrice', ascending=False).head(min(20, len(summary_df)))
                    sns.barplot(data=plot_df_s, x='Ticker', y='FinalPrice', palette='viridis')
                    plt.title(f'Final Prices from ABM (Top {len(plot_df_s)} by Price)')
                    plt.xlabel('Ticker'); plt.ylabel('Final Price'); plt.xticks(rotation=45, ha='right'); plt.tight_layout()
                    plt.savefig(os.path.join(base_output_dir, "multistock_final_prices_barplot.png")); plt.close()
                    print(f"Saved summary plot of final prices.")
            except Exception as e_save_s: print(f"Error saving summary stats/plot: {e_save_s}")
        else: print("No summary statistics calculated.")
      # Perform PCMI analysis
    sp500_csv_path = r"C:\Oepnfilelearningpath\Thesis\PFNN\sp500_csv_data"
    perform_pcmi_analysis(base_output_dir, sp500_csv_path, max_stocks=50)
      # Perform Predictive Validation Analysis
    print("\n" + "="*60)
    print("PERFORMING PREDICTIVE VALIDATION ANALYSIS")
    print("="*60)
    
    # Load real and simulated data for predictive validation
    real_data = load_real_market_data(sp500_csv_path, all_tickers[:50], max_days=300)
    sim_data = load_simulation_data(base_output_dir, all_tickers[:50])
    
    if real_data and sim_data:
        validation_metrics = perform_predictive_validation(
            real_data, sim_data, base_output_dir, prediction_horizon=10
        )
        
        if validation_metrics:
            print(f"\nPREDICTIVE VALIDATION RESULTS:")
            print(f"Direction Accuracy: {validation_metrics['mean_direction_accuracy']:.3f} ± {validation_metrics['std_direction_accuracy']:.3f}")
            print(f"Magnitude Correlation: {validation_metrics['mean_magnitude_correlation']:.3f} ± {validation_metrics['std_magnitude_correlation']:.3f}")
            print(f"Price Error (MAPE): {validation_metrics['mean_price_mape']:.1f}% ± {validation_metrics['std_price_mape']:.1f}%")
            print(f"Stocks Analyzed: {validation_metrics['num_stocks_analyzed']}")
              # Interpretation
            if validation_metrics['mean_direction_accuracy'] > 0.5:
                print("[+] ABM shows predictive capability (better than random)")
            else:
                print("[-] ABM shows poor predictive capability (worse than random)")
                
            print(f"\nDetailed results saved in: {base_output_dir}/predictive_validation_results.txt")
            print(f"Visualization saved in: {base_output_dir}/predictive_validation_analysis.png")
        else:
            print("Could not perform predictive validation - insufficient data")
    else:
        print("Could not load data for predictive validation")
    
    # Perform Temporal Analysis
    print("\n" + "="*60)
    print("PERFORMING TEMPORAL PATTERN ANALYSIS")
    print("="*60)
    
    if real_data and sim_data:
        temporal_results = perform_temporal_analysis(
            real_data, sim_data, base_output_dir, window_sizes=[5, 10, 20]
        )
        
        if temporal_results:
            # Calculate summary metrics
            vol_ratios = [temporal_results[t]['volatility_ratio'] for t in temporal_results]
            trend_similarities = [temporal_results[t]['trend_similarity'] for t in temporal_results]
            
            print(f"\nTEMPORAL ANALYSIS RESULTS:")
            print(f"Volatility Ratio: {np.mean(vol_ratios):.3f} ± {np.std(vol_ratios):.3f}")
            print(f"Trend Similarity: {np.mean(trend_similarities):.3f} ± {np.std(trend_similarities):.3f}")
            print(f"Stocks Analyzed: {len(temporal_results)}")
              # Interpretation
            avg_vol_ratio = np.mean(vol_ratios)
            if 0.8 <= avg_vol_ratio <= 1.2:
                print("[+] Good volatility matching")
            else:
                                print("[-] Poor volatility matching - model needs calibration")
                
            print(f"\nDetailed results saved in: {base_output_dir}/temporal_analysis_results.txt")
            print(f"Visualization saved in: {base_output_dir}/temporal_analysis.png")
        else:
            print("Could not perform temporal analysis - insufficient data")
    
    print("\nMulti-Stock ABM Simulation script finished.")

def perform_temporal_analysis(real_data, sim_data, output_dir, window_sizes=[5, 10, 20]):
    """
    Analyze temporal patterns and rolling correlation between ABM and real market data
    
    Args:
        real_data: Dictionary of real stock price data
        sim_data: Dictionary of simulated stock price data
        output_dir: Directory to save analysis results
        window_sizes: List of window sizes for rolling correlation analysis
    """
    print("Performing temporal pattern analysis...")
    
    temporal_results = {}
    
    # Common tickers
    common_tickers = set(real_data.keys()).intersection(set(sim_data.keys()))
    if len(common_tickers) < 2:
        print("Not enough common tickers for temporal analysis")
        return None
    
    for ticker in list(common_tickers)[:10]:  # Analyze top 10 stocks
        real_prices = real_data[ticker]
        sim_prices = sim_data[ticker]
        
        min_length = min(len(real_prices), len(sim_prices))
        if min_length < max(window_sizes) + 10:
            continue
            
        # Align data
        real_aligned = real_prices[:min_length]
        sim_aligned = sim_prices[:min_length]
        
        # Calculate returns
        real_returns = np.diff(real_aligned) / real_aligned[:-1]
        sim_returns = np.diff(sim_aligned) / sim_aligned[:-1]
        
        # Rolling correlations for different window sizes
        rolling_corrs = {}
        for window in window_sizes:
            if len(real_returns) >= window:
                corrs = []
                for i in range(window, len(real_returns)):
                    real_window = real_returns[i-window:i]
                    sim_window = sim_returns[i-window:i]
                    if np.std(real_window) > 0 and np.std(sim_window) > 0:
                        corr, _ = pearsonr(real_window, sim_window)
                        corrs.append(corr)
                    else:
                        corrs.append(0)
                rolling_corrs[window] = corrs
        
        # Volatility analysis
        real_volatility = np.std(real_returns) * np.sqrt(252)  # Annualized
        sim_volatility = np.std(sim_returns) * np.sqrt(252)
        volatility_ratio = sim_volatility / real_volatility if real_volatility > 0 else 0
          # Trend analysis (using linear regression)
        days = np.arange(len(real_aligned))
        real_trend, _, real_r_value, real_p_value, _ = stats.linregress(days, real_aligned)
        sim_trend, _, sim_r_value, sim_p_value, _ = stats.linregress(days, sim_aligned)
        
        temporal_results[ticker] = {
            'real_returns': real_returns,
            'sim_returns': sim_returns,
            'rolling_correlations': rolling_corrs,
            'real_volatility': real_volatility,
            'sim_volatility': sim_volatility,
            'volatility_ratio': volatility_ratio,
            'real_trend': real_trend,
            'sim_trend': sim_trend,
            'real_trend_r2': real_r_value**2,
            'sim_trend_r2': sim_r_value**2,
            'trend_similarity': abs(real_trend - sim_trend) / (abs(real_trend) + 1e-6)
        }
    
    # Create temporal analysis plots
    create_temporal_analysis_plots(temporal_results, output_dir, window_sizes)
    
    # Save temporal analysis results
    save_temporal_analysis_results(temporal_results, output_dir, window_sizes)
    
    return temporal_results

def create_temporal_analysis_plots(temporal_results, output_dir, window_sizes):
    """Create comprehensive plots for temporal analysis"""
    if not temporal_results:
        return
    
    fig = plt.figure(figsize=(20, 16))
    
    # Select top 4 stocks for detailed plotting
    sorted_stocks = list(temporal_results.keys())[:4]
    
    # 1. Rolling correlations for each stock
    for i, ticker in enumerate(sorted_stocks):
        ax = plt.subplot(4, 4, i + 1)
        data = temporal_results[ticker]
        
        for window in window_sizes:
            if window in data['rolling_correlations']:
                corrs = data['rolling_correlations'][window]
                if corrs:
                    plt.plot(corrs, label=f'{window}-day window', alpha=0.7)
        
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.title(f'{ticker} - Rolling Correlations')
        plt.xlabel('Time Period')
        plt.ylabel('Correlation')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 2. Volatility comparison
    ax = plt.subplot(4, 4, 5)
    tickers = list(temporal_results.keys())
    real_vols = [temporal_results[t]['real_volatility'] for t in tickers]
    sim_vols = [temporal_results[t]['sim_volatility'] for t in tickers]
    
    plt.scatter(real_vols, sim_vols, alpha=0.7, s=50)
    plt.plot([min(real_vols), max(real_vols)], [min(real_vols), max(real_vols)], 
             'r--', alpha=0.7, label='Perfect Match')
    plt.xlabel('Real Market Volatility')
    plt.ylabel('ABM Volatility')
    plt.title('Volatility Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Volatility ratio distribution
    ax = plt.subplot(4, 4, 6)
    vol_ratios = [temporal_results[t]['volatility_ratio'] for t in tickers]
    plt.hist(vol_ratios, bins=15, alpha=0.7, color='orange', edgecolor='black')
    plt.axvline(1.0, color='red', linestyle='--', label='Perfect Match')
    plt.axvline(np.mean(vol_ratios), color='blue', linestyle='--', 
                label=f'Mean: {np.mean(vol_ratios):.2f}')
    plt.xlabel('ABM/Real Volatility Ratio')
    plt.ylabel('Number of Stocks')
    plt.title('Distribution of Volatility Ratios')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Trend comparison
    ax = plt.subplot(4, 4, 7)
    real_trends = [temporal_results[t]['real_trend'] for t in tickers]
    sim_trends = [temporal_results[t]['sim_trend'] for t in tickers]
    
    plt.scatter(real_trends, sim_trends, alpha=0.7, s=50)
    trend_range = [min(min(real_trends), min(sim_trends)), 
                   max(max(real_trends), max(sim_trends))]
    plt.plot(trend_range, trend_range, 'r--', alpha=0.7, label='Perfect Match')
    plt.xlabel('Real Market Trend')
    plt.ylabel('ABM Trend')
    plt.title('Trend Direction Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Return distribution comparison for first stock
    if sorted_stocks:
        ticker = sorted_stocks[0]
        ax = plt.subplot(4, 4, 8)
        data = temporal_results[ticker]
        
        plt.hist(data['real_returns'], bins=30, alpha=0.5, label='Real Market', 
                density=True, color='blue')
        plt.hist(data['sim_returns'], bins=30, alpha=0.5, label='ABM Simulation', 
                density=True, color='red')
        plt.xlabel('Daily Returns')
        plt.ylabel('Density')
        plt.title(f'{ticker} - Return Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 6. Average rolling correlation by window size
    ax = plt.subplot(4, 4, 9)
    avg_corrs_by_window = {}
    for window in window_sizes:
        all_corrs = []
        for ticker in temporal_results:
            if window in temporal_results[ticker]['rolling_correlations']:
                corrs = temporal_results[ticker]['rolling_correlations'][window]
                all_corrs.extend([c for c in corrs if not np.isnan(c)])
        if all_corrs:
            avg_corrs_by_window[window] = np.mean(all_corrs)
    
    if avg_corrs_by_window:
        windows = list(avg_corrs_by_window.keys())
        avg_corrs = list(avg_corrs_by_window.values())
        plt.bar(windows, avg_corrs, alpha=0.7, color='green')
        plt.xlabel('Window Size (days)')
        plt.ylabel('Average Rolling Correlation')
        plt.title('Average Correlation by Time Window')
        plt.grid(True, alpha=0.3)
    
    # 7. Trend R² comparison
    ax = plt.subplot(4, 4, 10)
    real_r2s = [temporal_results[t]['real_trend_r2'] for t in tickers]
    sim_r2s = [temporal_results[t]['sim_trend_r2'] for t in tickers]
    
    plt.scatter(real_r2s, sim_r2s, alpha=0.7, s=50)
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Perfect Match')
    plt.xlabel('Real Market Trend R²')
    plt.ylabel('ABM Trend R²')
    plt.title('Trend Strength Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Summary statistics
    ax = plt.subplot(4, 4, 11)
    ax.axis('off')
    
    # Calculate summary statistics
    avg_vol_ratio = np.mean([temporal_results[t]['volatility_ratio'] for t in tickers])
    avg_trend_sim = np.mean([temporal_results[t]['trend_similarity'] for t in tickers])
    
    summary_text = f"""
    TEMPORAL ANALYSIS SUMMARY
    
    Stocks Analyzed: {len(temporal_results)}
    Window Sizes: {window_sizes} days
    
    VOLATILITY ANALYSIS:
    Avg Vol Ratio: {avg_vol_ratio:.2f}
    (1.0 = perfect match)
    
    TREND ANALYSIS:
    Avg Trend Similarity: {avg_trend_sim:.2f}
    (lower = more similar)
    
    CORRELATION WINDOWS:
    """
    
    for window in window_sizes:
        if window in avg_corrs_by_window:
            summary_text += f"{window}-day: {avg_corrs_by_window[window]:.3f}\n    "
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def save_temporal_analysis_results(temporal_results, output_dir, window_sizes):
    """Save detailed temporal analysis results to text file"""
    output_file = os.path.join(output_dir, 'temporal_analysis_results.txt')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("TEMPORAL PATTERN ANALYSIS RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("OVERVIEW:\n")
        f.write(f"Number of stocks analyzed: {len(temporal_results)}\n")
        f.write(f"Rolling correlation windows: {window_sizes} days\n\n")
        
        # Calculate aggregate metrics
        vol_ratios = [temporal_results[t]['volatility_ratio'] for t in temporal_results]
        trend_similarities = [temporal_results[t]['trend_similarity'] for t in temporal_results]
        
        f.write("AGGREGATE VOLATILITY METRICS:\n")
        f.write(f"Mean volatility ratio (ABM/Real): {np.mean(vol_ratios):.3f} ± {np.std(vol_ratios):.3f}\n")
        f.write(f"Median volatility ratio: {np.median(vol_ratios):.3f}\n")
        f.write(f"Volatility ratios in range [0.8, 1.2]: {np.sum((np.array(vol_ratios) >= 0.8) & (np.array(vol_ratios) <= 1.2))}/{len(vol_ratios)}\n\n")
        
        f.write("AGGREGATE TREND METRICS:\n")
        f.write(f"Mean trend similarity: {np.mean(trend_similarities):.3f} ± {np.std(trend_similarities):.3f}\n")
        f.write(f"(Lower values indicate better similarity)\n\n")
          # Rolling correlation summary
        f.write("ROLLING CORRELATION SUMMARY:\n")
        for window in window_sizes:
            all_corrs = []
            for ticker in temporal_results:
                if window in temporal_results[ticker]['rolling_correlations']:
                    corrs = temporal_results[ticker]['rolling_correlations'][window]
                    all_corrs.extend([c for c in corrs if not np.isnan(c)])
            
            if all_corrs:
                f.write(f"{window}-day window: {np.mean(all_corrs):.3f} ± {np.std(all_corrs):.3f} ")
                f.write(f"(n={len(all_corrs)} observations)\n")
        
        f.write("\nDETAILED STOCK-BY-STOCK RESULTS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Ticker':<8} {'VolRatio':<9} {'TrendSim':<9} {'RealVol%':<9} {'SimVol%':<9}\n")
        f.write("-" * 80 + "\n")
        for ticker in sorted(temporal_results.keys()):
            data = temporal_results[ticker]
            f.write(f"{ticker:<8} {data['volatility_ratio']:<9.3f} ")
            f.write(f"{data['trend_similarity']:<9.3f} ")
            f.write(f"{data['real_volatility']*100:<9.1f} ")
            f.write(f"{data['sim_volatility']*100:<9.1f}\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("TEMPORAL ANALYSIS INTERPRETATION:\n\n")
        
        avg_vol_ratio = np.mean(vol_ratios)
        if 0.8 <= avg_vol_ratio <= 1.2:
            f.write("[+] GOOD volatility matching between ABM and real market.\n")
        elif 0.5 <= avg_vol_ratio <= 2.0:
            f.write("[~] MODERATE volatility matching. Consider parameter adjustment.\n")
        else:
            f.write("[-] POOR volatility matching. Significant model recalibration needed.\n")
        
        avg_trend_sim = np.mean(trend_similarities)
        if avg_trend_sim < 0.5:
            f.write("[+] GOOD trend direction similarity.\n")
        elif avg_trend_sim < 1.0:
            f.write("[~] MODERATE trend direction similarity.\n")
        else:
            f.write("[-] POOR trend direction similarity.\n")

if __name__ == '__main__':
    main()