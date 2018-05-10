from agent import DQNAgent
from preprocess import preprocess



import gym
env = gym.make('Breakout-v0')
agent = DQNAgent(state_shape=(-1, 105, 80, 1), action_dim=4,
                 checkpoint_directory="./models_checkpoints/rl/", batch_size=32,
                 device_name="gpu:0")
for i_episode in range(10000):
    observation = env.reset()
    total_reward = 0
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
            if agent.step_count > 5000:
                agent.copy_base_to_target()

            if i_episode % 50 == 0:
                print("Episode {} finished after {} timesteps".format(i_episode,t+1))
                print("reward: %d" % total_reward)
                print("epsilon: %s"% agent.epsilon)
                if agent.step_count > 50000:
                    agent.save(i_episode)
            break