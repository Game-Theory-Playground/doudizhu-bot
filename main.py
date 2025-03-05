import rlcard
from rlcard.agents import RandomAgent

env = rlcard.make('blackjack')
env.set_agents([RandomAgent(num_actions=env.num_actions)])

print("Actions: ", env.num_actions)
print("Players: ", env.num_players)
print("State Shape: ", env.state_shape)
print("Action Shape: ", env.action_shape)

trajectories, payoffs = env.run()

print("Trajectories: ", trajectories)
print("Payoffs: ", payoffs)