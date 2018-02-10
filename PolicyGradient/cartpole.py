import gym
import numpy as np
from open_ai.PL_Brain import PolicyGradient

pl_brain = PolicyGradient(
    action_dim=2,
    state_dim=4,
    gamma=0.99,
    learning_rate=0.02
)

#run a episode
def make_episode():
    state_list, action_list, reward_list = [], [], []
    env = gym.make("CartPole-v0")

    state = env.reset()

    reward_sum = 0
    while True:
        # env.render()
        state_list.append(state)

        action = pl_brain.choose_action(state)
        action_list.append(action)

        state, reward, done, info = env.step(action)
        reward_list.append(reward)

        reward_sum += reward

        if done:
            break
    return state_list, action_list, reward_list, reward_sum


if __name__ == '__main__':

    for i in range(0, 500):
        state_list, action_list, reward_list, reward_sum = make_episode()
        print("Episode {}, Reward Sum {}.".format(i, reward_sum))
        pl_brain.train(state_list, action_list, reward_list)
