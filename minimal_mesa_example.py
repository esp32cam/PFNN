import mesa
import numpy as np
from mesa.agent import Agent, AgentSet # Correctly import AgentSet

class MyAgent(Agent):
    """A simple agent that prints its ID during its step."""
    def __init__(self, unique_id, model):
        # Pass model to superclass, unique_id is handled by Agent class
        super().__init__(model=model)
        # We can store unique_id if needed for other purposes, but super() handles internal ID
        self.my_id = unique_id

    def step(self):
        print(f"Agent {self.my_id} (Mesa internal ID: {self.unique_id}) is stepping.")
        # Example of agent updating an internal attribute
        self.model.step_count_for_agent += 1


class MyModel(mesa.Model):
    """A simple model that uses AgentSet for scheduling."""
    def __init__(self, num_agents):
        super().__init__() # Initialize base Model class
        self.num_agents = num_agents
        self.random = np.random.default_rng() # Mesa components expect this
        self.rng = self.random # Alias for older examples if any component uses rng

        # For tracking changes by agents
        self.step_count_for_agent = 0

        # Create agents
        agents = []
        for i in range(self.num_agents):
            # Pass the model instance to the agent's constructor
            a = MyAgent(unique_id=i, model=self)
            agents.append(a)

        # Create an AgentSet from the list of agents and the model's random generator
        # This AgentSet will manage our agents.
        self.schedule = AgentSet(agents, random=self.random) # Renamed to self.schedule

        # Model-level time and step counting
        self.running = True # To control the simulation run
        self.steps = 0
        self.time = 0.0

        print(f"Model initialized with {len(self.schedule)} agents.") # Corrected to len(self.schedule)
        # The _ids defaultdict in the Agent class is global per model type if not reset.
        # For a truly clean run each time, you might need to clear Agent._ids[self.__class__]
        # or ensure unique model instances if that becomes an issue in complex scenarios.
        # For this minimal example, it's fine.

    def step(self):
        """Advance the model by one step."""
        print(f"\n--- Model Step {self.steps + 1} (Time: {self.time}) ---")
        # Use AgentSet.do() to call the 'step' method on all agents.
        # AgentSet.shuffle_do() could be used if random execution order is needed each step.
        self.schedule.do("step")
        self.steps += 1 # Advance step count for the model
        self.time += 1.0 # Advance time for the model


# Main execution
if __name__ == "__main__":
    num_agents = 3
    num_steps = 2

    # Reset agent ID counter for this model type if running multiple model instances in same script
    # For a single run like this, it's not strictly necessary but good practice for re-runs.
    if MyModel in Agent._ids:
        del Agent._ids[MyModel]

    model = MyModel(num_agents)

    for i in range(num_steps):
        model.step()

    print(f"\nTotal steps counted by agents: {model.step_count_for_agent}")
