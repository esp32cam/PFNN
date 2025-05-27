import mesa
import numpy as np
import pandas as pd # For DataCollector output display

class NoiseTrader(mesa.agent.Agent):
    """A trader that makes random buy/sell/hold decisions."""
    def __init__(self, unique_id, model):
        super().__init__(model=model) # unique_id is handled by Agent base class
        self.custom_id = unique_id # For easier identification if needed
        self.action = "hold" # Default action

    def step(self):
        # Access model attributes (though not strictly needed for random choice here)
        # current_price = self.model.market_price
        # attractor_info = self.model.attractor_signals

        # Randomly choose to buy, sell, or hold
        choice = self.model.random.choice(["buy", "sell", "hold"])
        self.action = choice
        # print(f"NoiseTrader {self.custom_id} (Internal ID: {self.unique_id}): Action = {self.action}, Price = {current_price:.2f}")

class MomentumTrader(mesa.agent.Agent):
    """A trader that decides based on recent price trends."""
    def __init__(self, unique_id, model):
        super().__init__(model=model)
        self.custom_id = unique_id
        self.action = "hold"

    def step(self):
        # Access model attributes
        current_price = self.model.market_price
        price_history = self.model.price_history
        # attractor_info = self.model.attractor_signals

        if len(price_history) < 1:
            self.action = "hold" # Not enough history to make a decision
        else:
            previous_price = price_history[-1] # Get the most recent previous price
            if current_price > previous_price:
                self.action = "buy"
            elif current_price < previous_price:
                self.action = "sell"
            else:
                self.action = "hold"
        # print(f"MomentumTrader {self.custom_id} (Internal ID: {self.unique_id}): Action = {self.action}, Price = {current_price:.2f}, PrevPrice = {price_history[-1] if price_history else 'N/A'}")


class StockMarketModel(mesa.Model):
    """A simple agent-based model of a stock market."""
    def __init__(self, num_noise_traders, num_momentum_traders, initial_price=100.0, price_history_length=2):
        super().__init__()
        self.num_noise_traders = num_noise_traders
        self.num_momentum_traders = num_momentum_traders
        self.market_price = initial_price
        # Use a deque for price_history for efficient fixed-length storage
        from collections import deque
        self.price_history = deque(maxlen=price_history_length) 
        self.price_history.append(initial_price) # Start with initial price in history

        self.random = np.random.default_rng() # For Mesa's AgentSet and agent decisions

        # Placeholder for attractor signals
        self.attractor_signals = {"lyapunov_exp": 0.1, "fractal_dim": 1.5}

        # Agent creation
        all_agents_list = []
        agent_id_counter = 0

        for i in range(self.num_noise_traders):
            nt = NoiseTrader(unique_id=f"NT_{i}", model=self)
            all_agents_list.append(nt)
            agent_id_counter += 1
        
        for i in range(self.num_momentum_traders):
            mt = MomentumTrader(unique_id=f"MT_{i}", model=self)
            all_agents_list.append(mt)
            agent_id_counter += 1

        self.schedule = mesa.agent.AgentSet(all_agents_list, random=self.random)
        
        # Model-level time tracking (AgentSet does not manage this)
        self.steps = 0
        self.time = 0.0 # Can represent simulation time more granularly if needed

        # DataCollector setup
        # Agent reporters: collect action for each agent type
        # We need a way to distinguish agent types for reporting if collecting individual actions.
        # Simpler: collect aggregate counts of actions.
        # For now, let's collect market price and total buys/sells.
        
        model_reporters = {
            "MarketPrice": "market_price",
            "TotalBuys": lambda m: sum(1 for agent in m.schedule if agent.action == "buy"),
            "TotalSells": lambda m: sum(1 for agent in m.schedule if agent.action == "sell"),
            "TotalHolds": lambda m: sum(1 for agent in m.schedule if agent.action == "hold"),
            # Example of how to access attractor signals
            "LyapunovExp": lambda m: m.attractor_signals["lyapunov_exp"],
            "FractalDim": lambda m: m.attractor_signals["fractal_dim"]
        }
        # Agent reporters can be tricky with AgentSet if not careful with agent IDs or types
        # For simplicity, we'll focus on model reporters.
        self.datacollector = mesa.DataCollector(
            model_reporters=model_reporters
        )
        self.datacollector.collect(self) # Collect initial state

    def step(self):
        """Advance the model by one step."""
        # Store current market price in history *before* it's updated
        # but *after* momentum traders might have used the previous step's price.
        # The timing here is subtle. For momentum traders to react to price at t-1 vs t-2,
        # the price history should be updated at the end of the previous step or start of current.
        # Current setup: price_history has price from t-1. Agents use current_price (from t-1) and price_history[-1] (from t-2).
        # Then, current_price (from t-1) is added to history. Then market_price is updated to t.
        
        # Let's refine: Momentum traders should compare current price (end of t-1) with price at end of t-2.
        # So, price_history should contain [price_t-2, price_t-1] when agents make decisions for step t.
        # The self.market_price is effectively price_t-1 at the start of the step method.
        
        if self.steps > 0: # Add previous step's market price to history
             self.price_history.append(self.market_price)

        # Agents make decisions based on current market state
        self.schedule.do("step") # Agents update their 'self.action'

        # Aggregate actions
        total_buys = 0
        total_sells = 0
        for agent in self.schedule: # Iterate through AgentSet
            if agent.action == "buy":
                total_buys += 1
            elif agent.action == "sell":
                total_sells += 1
        
        # Update market price based on net demand
        # Simple mechanism: price moves proportionally to net order imbalance
        price_change = (total_buys - total_sells) * 0.1 
        self.market_price += price_change
        
        # Ensure price doesn't go negative (or apply other constraints)
        if self.market_price < 0.01: # Arbitrary floor
            self.market_price = 0.01

        # Collect data after all updates for the current step
        self.datacollector.collect(self)

        # Increment model time and step count
        self.steps += 1
        self.time += 1.0 # Assuming each step is one unit of time

        # print(f"Model Step {self.steps}: Market Price = {self.market_price:.2f}, Buys = {total_buys}, Sells = {total_sells}")


# Main execution block
if __name__ == '__main__':
    num_noise_traders = 20
    num_momentum_traders = 10
    num_steps = 50 # Increased steps for more data

    # Reset agent ID counter for this model type for cleaner multiple runs if needed in other contexts
    # Note: AgentSet uses weakrefs, and Agent base class manages unique_id per model instance.
    # Explicitly clearing _ids is more for scenarios where you create multiple model *classes* dynamically
    # or have very specific needs for ID resetting across identical model instantiations in one script.
    # For this basic ABM, direct instantiation of MyModel should be fine.
    # If issues arise with IDs in complex setups, one might do:
    # if StockMarketModel in mesa.agent.Agent._ids:
    #     del mesa.agent.Agent._ids[StockMarketModel]
    # However, this is usually not needed for typical Mesa usage.

    print("Initializing Stock Market ABM...")
    model = StockMarketModel(num_noise_traders, num_momentum_traders, initial_price=100.0)
    
    print(f"Running model for {num_steps} steps...")
    for i in range(num_steps):
        model.step()
    print("Model run complete.")

    # Get collected data
    model_data = model.datacollector.get_model_vars_dataframe()
    
    print("\nMarket Price Over Time:")
    print(model_data["MarketPrice"])

    print("\nAggregate Actions Over Time:")
    print(model_data[["TotalBuys", "TotalSells", "TotalHolds"]])

    # Example of how to access a specific agent's final action (if needed, though not collected by default DataCollector)
    # Can be useful for debugging or specific analysis if agents had more complex state.
    # print("\nExample: Final action of first noise trader:")
    # first_noise_trader = [agent for agent in model.schedule if isinstance(agent, NoiseTrader)][0]
    # print(f"NoiseTrader {first_noise_trader.custom_id} final action: {first_noise_trader.action}")

    # Plotting (optional, requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:red'
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Market Price', color=color)
        ax1.plot(model_data.index, model_data["MarketPrice"], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_title('Stock Market ABM Simulation')

        ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
        color_buy = 'tab:green'
        color_sell = 'tab:blue'
        ax2.set_ylabel('Number of Orders') 
        ax2.plot(model_data.index, model_data["TotalBuys"], color=color_buy, linestyle='--', label='Total Buys')
        ax2.plot(model_data.index, model_data["TotalSells"], color=color_sell, linestyle=':', label='Total Sells')
        ax2.tick_params(axis='y')
        
        fig.tight_layout() # otherwise the right y-label is slightly clipped
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.savefig("stock_market_abm_output.png")
        print("\nSaved plot to stock_market_abm_output.png")
        # plt.show() # Uncomment to display plot if running in suitable environment
    except ImportError:
        print("\nMatplotlib not installed. Skipping plot generation.")
    except Exception as e:
        print(f"\nError during plotting: {e}. Skipping plot generation.")
