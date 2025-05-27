import os
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
# from mesa.batchrunner import batch_run # Not currently used here

try:
    from sp500_data_loader import get_sp500_tickers
    from mesa_market_model import StockMarketModel
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    raise

def main():
    print("Starting Multi-Stock ABM Simulation Run...")

    abm_parameters = {
        "num_agents": 20,
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

    test_ticker_limit = 10
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
    print("\nMulti-Stock ABM Simulation script finished.")

if __name__ == '__main__':
    main()