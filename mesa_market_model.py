import mesa
import matplotlib.pyplot as plt
import pandas as pd # Ensure pandas is imported for DataFrame conversion
import numpy as np # Ensure numpy is imported
import os # Ensure os is imported
import seaborn as sns # Ensure seaborn is imported

class TraderAgent(mesa.Agent):
    """
    A simple trader agent.
    """
    def __init__(self, unique_id, model, starting_cash=1000, starting_shares=10):
        # Mesa Agent requires these attributes
        self.unique_id = unique_id
        self.model = model
        self.cash = starting_cash
        self.shares = starting_shares
        self.action_intent = "HOLD"  # Can be 'BUY', 'SELL', 'HOLD'
        self.order_quantity = 0

    def step(self):
        """
        Agent's step: decide action (buy/sell/hold).
        Incorporates influence from the model's attractor_influence_vector.
        """
        default_prob_buy = 0.35
        default_prob_sell = 0.35
        # default_prob_hold = 0.30 (implicitly)

        prob_buy = default_prob_buy
        prob_sell = default_prob_sell

        # Access influence_strength from the model
        current_influence_strength = self.model.influence_strength

        if self.model.attractor_influence_vector is not None and current_influence_strength > 0:
            if len(self.model.attractor_influence_vector) > 0:
                sentiment_indicator = self.model.attractor_influence_vector[0]
                
                positive_sentiment_threshold = 0.1
                negative_sentiment_threshold = -0.1

                # Define the maximum change in probability due to sentiment
                max_prob_shift = 0.15 # e.g., makes buy_prob 0.50 (0.35+0.15) and sell_prob 0.20 (0.35-0.15)

                if sentiment_indicator > positive_sentiment_threshold:
                    prob_buy = default_prob_buy + (max_prob_shift * current_influence_strength)
                    prob_sell = default_prob_sell - (max_prob_shift * current_influence_strength)
                elif sentiment_indicator < negative_sentiment_threshold:
                    prob_buy = default_prob_buy - (max_prob_shift * current_influence_strength)
                    prob_sell = default_prob_sell + (max_prob_shift * current_influence_strength)
                
                prob_buy = max(0.0, min(1.0, prob_buy))
                prob_sell = max(0.0, min(1.0, prob_sell))
                
                if prob_buy + prob_sell > 1.0: # Safeguard
                    current_sum = prob_buy + prob_sell
                    prob_buy = prob_buy / current_sum
                    prob_sell = prob_sell / current_sum
        
        rand_action = self.model.random.random()
        if rand_action < prob_buy:
            self.action_intent = "BUY"
            self.order_quantity = 1
        elif rand_action < (prob_buy + prob_sell):
            self.action_intent = "SELL"
            self.order_quantity = 1
        else:
            self.action_intent = "HOLD"
            self.order_quantity = 0

class StockMarketModel(mesa.Model):
    """
    A simple agent-based stock market model.
    """
    def __init__(self, num_agents, initial_price=100.0, price_impact_factor=0.05,
                 ticker_for_attractor=None, attractor_stats_base_path="stock_analysis_results",
                 influence_strength=1.0):
        super().__init__() 
        self.num_agents = num_agents
        self.current_price = initial_price
        self.influence_strength = influence_strength
        self.current_volume = 0
        self.price_impact_factor = price_impact_factor

        self.schedule = mesa.time.RandomActivation(self)
        self.agents_list = []
        for i in range(self.num_agents):
            agent = TraderAgent(unique_id=i, model=self, starting_cash=1000, starting_shares=10)
            self.schedule.add(agent)
            self.agents_list.append(agent)
        
        self.attractor_influence_vector = None
        self.ticker_for_attractor = ticker_for_attractor
        if self.ticker_for_attractor:
            stats_filename = f"{self.ticker_for_attractor}_pfnn_simple_attractor_stats.npy"
            attractor_file_path = os.path.join(
                attractor_stats_base_path, 
                self.ticker_for_attractor, 
                "pfnn_simple", 
                stats_filename
            )
            if os.path.exists(attractor_file_path):
                try:
                    self.attractor_influence_vector = np.load(attractor_file_path)
                    print(f"Successfully loaded attractor stats for {self.ticker_for_attractor} from {attractor_file_path}")
                except Exception as e:
                    print(f"Error loading attractor stats from {attractor_file_path}: {e}")
            else:
                print(f"Attractor stats file not found: {attractor_file_path}. No attractor influence.")
        else:
            print("No ticker for attractor. No attractor influence.")

        model_reporters_dict = {
            "Price": "current_price",
            "Volume": "current_volume",
            "AttractorLoaded": lambda m: True if m.attractor_influence_vector is not None else False
        }
        self.datacollector = mesa.DataCollector(model_reporters=model_reporters_dict)
        
        self.buy_orders_total = 0
        self.sell_orders_total = 0

    def collect_orders(self):
        self.buy_orders_total = 0
        self.sell_orders_total = 0
        for agent in self.schedule.agents:
            if agent.action_intent == "BUY":
                self.buy_orders_total += agent.order_quantity
            elif agent.action_intent == "SELL":
                self.sell_orders_total += agent.order_quantity

    def apply_market_mechanism(self):
        self.current_volume = min(self.buy_orders_total, self.sell_orders_total)
        net_imbalance = self.buy_orders_total - self.sell_orders_total
        price_change = net_imbalance * self.price_impact_factor
        self.current_price += price_change
        self.current_price = max(0.01, self.current_price) # Price floor

        for agent in self.schedule.agents:
            agent.action_intent = "HOLD"
            agent.order_quantity = 0

    def step(self):
        self.current_volume = 0 
        self.schedule.step()
        self.collect_orders()
        self.apply_market_mechanism()
        self.datacollector.collect(self)
        
        if hasattr(self.schedule, 'steps') and self.schedule.steps % 10 == 0:
             print(f"Model Step: {self.schedule.steps}, Price: {self.current_price:.2f}, Volume: {self.current_volume}")

if __name__ == '__main__':
    params = {
        "num_agents": [10, 20, 30],
        "influence_strength": [0.0, 0.5, 1.0],
        "initial_price": 100.0,
        "price_impact_factor": 0.05,
        "ticker_for_attractor": "AAPL", # Ensure this file exists for testing influenced runs
        "attractor_stats_base_path": "stock_analysis_results"
    }

    iterations_per_combination = 5
    max_simulation_steps = 100

    print("Starting batch run...")
    batch_run_results = mesa.batch_run(
        StockMarketModel,
        parameters=params,
        iterations=iterations_per_combination,
        max_steps=max_simulation_steps,
        number_processes=1,
        data_collection_period=-1,
        display_progress=True
    )
    print("\nBatch run complete.")

    if batch_run_results:
        results_df = pd.DataFrame(batch_run_results)
        batch_results_filename = "abm_batch_run_results.csv"
        try:
            results_df.to_csv(batch_results_filename, index=False)
            print(f"Saved batch run results to {batch_results_filename}")
            print(f"DataFrame head:\n{results_df.head()}")
        except Exception as e:
            print(f"Error saving batch run results to CSV: {e}")

        print("\n--- Starting Quantitative Analysis of Batch Run Results ---")
        try:
            full_results_df = pd.read_csv(batch_results_filename)
            print(f"Loaded batch results from {batch_results_filename}. Shape: {full_results_df.shape}")
            grouped_analysis = full_results_df.groupby(['num_agents', 'influence_strength'])[['Price', 'Volume']].mean().reset_index()
            print("\nAggregated Results (Mean Final Price and Volume by parameters):")
            print(grouped_analysis)

            plt.figure(figsize=(10, 6))
            sns.lineplot(data=grouped_analysis, x='influence_strength', y='Price', hue='num_agents', marker='o', palette='viridis')
            plt.title('Mean Final Price vs. Attractor Influence Strength')
            plt.xlabel('Attractor Influence Strength'); plt.ylabel('Mean Final Price'); plt.grid(True); plt.legend(title='Num Agents')
            plt.savefig("batch_analysis_price_vs_influence.png")
            print(f"Saved price analysis plot.")
            plt.close()

            plt.figure(figsize=(10, 6))
            sns.lineplot(data=grouped_analysis, x='influence_strength', y='Volume', hue='num_agents', marker='o', palette='viridis')
            plt.title('Mean Final Volume vs. Attractor Influence Strength')
            plt.xlabel('Attractor Influence Strength'); plt.ylabel('Mean Final Volume'); plt.grid(True); plt.legend(title='Num Agents')
            plt.savefig("batch_analysis_volume_vs_influence.png")
            print(f"Saved volume analysis plot.")
            plt.close()

            if 'AttractorLoaded' in full_results_df.columns and 'ticker_for_attractor' in full_results_df.columns:
                fixed_ticker = params["ticker_for_attractor"] 
                attractor_loaded_summary = full_results_df[
                    full_results_df['ticker_for_attractor'] == fixed_ticker
                ].groupby(['influence_strength'])['AttractorLoaded'].mean().reset_index()
                print(f"\nProportion of runs where attractor for '{fixed_ticker}' was loaded:")
                print(attractor_loaded_summary)
        except FileNotFoundError:
            print(f"Error: Batch results file {batch_results_filename} not found.")
        except Exception as e:
            print(f"An error occurred during analysis: {e}")
        print("\nAnalysis script section finished.")
    else:
        print("Batch run produced no results.")
    print("\nScript finished.")