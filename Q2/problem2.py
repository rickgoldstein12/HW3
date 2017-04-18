from deeprl_hw3 import imitation
from keras.optimizers import Adam
import gym

expert = imitation.load_model('CartPole-v0_config.yaml',
                              'CartPole-v0_weights.h5f')
env = gym.make('CartPole-v0')
hard_env = gym.make('CartPole-v0')
hard_env = imitation.wrap_cartpole(hard_env)

# imitation.generate_expert_training_data(expert, env)



print('Expert')
imitation.test_cloned_policy(env, expert, render=False)
imitation.test_cloned_policy(hard_env, expert, render=False)
for num_expert_episodes in [1,10,50,100]:
# for num_expert_episodes in [1]:
    print('==================')
    print('Clone with ' + str(num_expert_episodes) + ' episode samples')
    s, a = imitation.generate_expert_training_data(expert, env,
                                                   num_episodes=num_expert_episodes,
                                                   render=False)
    clone = imitation.load_model('CartPole-v0_config.yaml')
    adam = Adam(lr=0.00025)
    clone.compile(adam, loss='binary_crossentropy', metrics=['accuracy'])
    hc = clone.fit(s,a, batch_size=32, epochs=50, verbose=0)

    print('Final Train Loss: ' + str(hc.history['loss'][-1]))
    print('Final Train Accuracy: ' + str(hc.history['acc'][-1]))    
    imitation.test_cloned_policy(env, clone, render=False)
    imitation.test_cloned_policy(hard_env, clone, render=False)    
    

