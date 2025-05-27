import mesa
import numpy as np

class SimpleAgent(mesa.Agent):
    """An agent with a unique ID."""
    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        self.model = model

    def step(self):
        print(f"Agent {self.unique_id} is stepping.")

class SimpleModel(mesa.Model):
    """A model with some number of agents."""
    def __init__(self, N):
        self.num_agents = N
        self.random = np.random.default_rng() # Add random attribute
        self.schedule = mesa.time.BaseScheduler(self) # Using BaseScheduler

        # Create agents and add them to the scheduler
        for i in range(self.num_agents):
            a = SimpleAgent(i, self)
            self.schedule.add(a)

    def step(self):
        """Advance the model by one step."""
        print(f"Model step. Number of agents in scheduler: {len(self.schedule.agents)}")
        # Manual step to ensure agents are processed if schedule.step() is problematic
        if not self.schedule.agents:
            print("No agents in scheduler to step.")
            return

        print("Manually stepping agents:")
        for agent in self.schedule.agents:
            agent.step()
        # self.schedule.step() # We'll rely on manual stepping for now
        print("Model step finished.")

# Run the model
num_agents = 3
model = SimpleModel(num_agents)
for i in range(3): # Run for 3 steps
    print(f"--- Iteration {i+1} ---")
    model.step()
