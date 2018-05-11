from agent import DQNAgent
from preprocess import preprocess

import gym
env = gym.make('Breakout-v0')
agent = DQNAgent(state_shape=(-1, 105, 80, 1),
                 action_dim=4,
                 checkpoint_directory="./models_checkpoints/rl/",
                 batch_size=32,
                 initial_epsilon=1.0,
                 final_epsilon=0.05,
                 exploration_steps=300000,
                 observation_steps=50000,
                 device_name="gpu:0")
# agent.load_last_checkpoint()
total_reward = 0.0
for i_episode in range(100000):
    observation = env.reset()
    for t in range(10000000):
#         env.render()
        now_state= preprocess(observation)
        action = agent.get_action(now_state, training=True).numpy()
        observation, reward, done, info = env.step(action)
        if(done):
            done = 1
        else:
            done = 0
        next_state= preprocess(observation)
        agent.step(now_state, action, reward, next_state, done)
        total_reward += reward
        if done:
            if agent.step_count > agent.observation_steps:
                agent.copy_base_to_target()

            if i_episode % 50 == 0:
                print("#############################")
                print("Episode {} finished after {} timesteps".format(i_episode,t+1))
                print("reward[%d - %d]: %3f" % (i_episode-49, i_episode, total_reward/50))
                print("epsilon: %s"% agent.epsilon)
                print("#############################")
                if agent.step_count > agent.observation_steps:
                    agent.save(i_episode)

                total_reward = 0.0

            break