#fix rewards not 1
# fix gamma / gt
# fix which to use
#op
# from __future__ import division, absolute_import
# from __future__ import print_function, unicode_literals

import gym
import numpy as np
import tensorflow as tf
from keras.models import model_from_yaml
import time

# calc total reward
def get_total_reward(env, model,printTime):


    minn = 1000
    maxx = 0
    total = 0

    # run 100 episodes
    for i in range(100):
        newstate=env.reset()

        stepsInEpisode = 0

        isTerm = False
        # finish episode
        while (not isTerm):
            action = choose_best_action(model, newstate)
            newstate, reward, isTerm, junk = env.step(action)
            stepsInEpisode = stepsInEpisode + 1

        #save episode info
        if stepsInEpisode > maxx:
            maxx = stepsInEpisode
        if stepsInEpisode < minn:
            minn = stepsInEpisode
        total = total + stepsInEpisode

    #calc average
    total = total/100.0
    #return
    print (printTime,total,minn,maxx)
    
    return total,minn,maxx

    """compute total reward

    Parameters
    ----------
    env: gym.core.Env
      The environment. 
    model: (your action model, which can be anything)

    Returns
    -------
    total_reward: float, min, max
    """


# choose action function, according to weights
def choose_action(model, observation):
    """choose the action 

    Parameters
    ----------
    model: (your action model, which can be anything)
    observation: given observation

    Returns
    -------
    action: one_hot vector
        
    """
    p = model.predict(observation[np.newaxis,:])[0]
    one_hot_action = np.random.multinomial(1, p)
    return one_hot_action

def choose_best_action(model, observation):
    """chooses and returns the best action
    """
    p = model.predict(observation[np.newaxis,:])[0]
    return np.argmax(p)

#heart of reinforce algorithm
def reinforce(env):

    # CONSTANTS
    MAXITERS = 1000  #should converge in 1000 iters
    #alpha = 1e-1
    gamma = .99
    
    # init optimizer for model
    # optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-08)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-4)
    
    # load network
    with tf.Session() as sess:
        model = load_model('CartPole-v0_config.yaml')
        model.compile(optimizer=optimizer,loss='binary_crossentropy')
        # repeat forever

        # init session and load in variables
        
        # setup structures for derivs
        action_chosen = tf.placeholder(tf.float32, shape=(2), name="a")
        rewardx = tf.placeholder(tf.float32, name = "r")
    
        loss = tf.log(model.output[0])*action_chosen*rewardx
    
        opt_grad = optimizer.compute_gradients(loss)
        update_op = optimizer.apply_gradients(opt_grad)
        
    
        
        sess.run(tf.global_variables_initializer())    
        #episode = []
        # loop thru training iterations
        for i in range(MAXITERS):
            # print(sess.run(model.trainable_weights))
            
            # print every 100 or 25
            if i % 100 == 0:
                get_total_reward(env,model,i)
        
            # set up memory structures
            stepsInEpisode = 0
            stateMem = []
            actionMem = []
            rewardMem = []

            # reset env
            newstate = env.reset()
        
            isTerm = False
            while(not isTerm): 
                #save state, choose action, save action, apply (step)
                stateMem.append(newstate)
                action = choose_action(model, newstate)
                actionMem.append(action)
                newstate, reward, isTerm, junk = env.step(np.argmax(action))
                #print (reward)
                rewardMem.append(reward)
                stepsInEpisode = stepsInEpisode + 1

        
            #UPDATE NEURAL NETWORK AFTER EPISODE ENDS
            for j in range(stepsInEpisode):
                # reward is number of steps including and after self (reward is always 1)
                G_t = (stepsInEpisode-j) * gamma**j 

                cur_state = np.array(stateMem[j])
                feed_dict={action_chosen:     actionMem[j],
                           rewardx:           -1*(G_t),
                           model.inputs[0]:   cur_state[np.newaxis,:]}

                sess.run(update_op, feed_dict=feed_dict)


        



def load_model(model_config_path, model_weights_path=None):
    """Load a saved model.

    Parameters
    ----------
    model_config_path: str
      The path to the model configuration yaml file. We have provided
      you this file for problems 2 and 3.
    model_weights_path: str, optional
      If specified, will load keras weights from hdf5 file.

    Returns
    -------
    keras.models.Model
    """
    with open(model_config_path, 'r') as f:
        model = model_from_yaml(f.read())

    if model_weights_path is not None:
        model.load_weights(model_weights_path)

    model.summary()

    return model


def main():
    # my code here
    env = gym.make('CartPole-v0')


    finalWeights = reinforce(env)

    #env.reset()
    #env.render()
    #env.step(0)

    #qq = 1
    #qq = qq+1

if __name__ == "__main__":
    main()
