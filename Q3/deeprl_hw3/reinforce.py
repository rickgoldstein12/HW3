#fix rewards not 1
# fix gamma / gt
# fix which to use
#op
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import gym
import numpy as np
#import random
import tensorflow as tf
#import math
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import model_from_yaml
#import time

# calc total reward
def get_total_reward(env, model,printTime):


    minn = 1000
    maxx = 0
    total = 0

    # run 100 episodes
    for i in range(100):
        env.reset()

        stepsInEpisode = 0

        #burn in first state
        pof1, action = choose_action(model, env.env.state)
        newstate, reward, isTerm, junk = env.step(action)
        stepsInEpisode = stepsInEpisode + 1

        # finish episode
        while (not isTerm):
            pof1, action = choose_action(model, env.env.state)
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
    p: float 
        probability of action 1
    action: int
        the action you choose
    """
    #manual reshape
    obstemp = np.zeros((1,4))
    obstemp[0][0] = observation[0]
    obstemp[0][1] = observation[1]
    obstemp[0][2] = observation[2]
    obstemp[0][3] = observation[3]

    # get p1 only from model
    p1 = model.predict(obstemp)[0][1]
    
    #draw random to decide on action
    if np.random.random() < p1:
        return p1,1
    else:
        return p1,0


# deterministic choose action function
def choose_actionDET(model, observation):
    '''choose the action 

    Parameters
    ----------
    model: (your action model, which can be anything)
    observation: given observation

    Returns
    -------
    p: float 
        probability of action 1
    action: int
        the action you choose
    '''

    obstemp = np.zeros((1,4))
    obstemp[0][0] = observation[0]
    obstemp[0][1] = observation[1]
    obstemp[0][2] = observation[2]
    obstemp[0][3] = observation[3]

    p1 = model.predict(obstemp)[0][1]
    # return best action
    if (p1>=.5):
        return p1,1
    else:
        return p1,0


#def test():
#    env.reset()
#    obstemp = np.zeros((1,4))
#    obstemp[0][0] = x[0]
#    obstemp[0][1] = x[1]
#    obstemp[0][2] = x[2]
#    obstemp[0][3] = x[3]

#    obstemp2 = np.zeros((1,4))
#    obstemp2[0][0] = tf.placeholder(tf.float32,[1])
#    obstemp2[0][1] = x[1]
#    obstemp2[0][2] = x[2]
#    obstemp2[0][3] = x[3]

#    x2 = tf.placeholder(tf.float32, [1, 4])
#    R = model.predict(x2)
#    tf.gradients(model.weights,tf.log(R))

    #var = tf.Variable()              # Must be a tf.float32 or tf.float64 variable.
    #loss = some_function_of(var, data)
    #data = tf.placeholder(tf.float32,shape = [None,4])
#    loss = tf.log(model.output)
#    var_grad = tf.gradients(loss, model.weights)
    
#    sess = tf.InteractiveSession()
    
#    sess.run(tf.global_variables_initializer())
#    sess.run(loss,{model.input:[[1,2,3,4]],})
#    sess.run(var_grad,{model.input:[env.env.state],})

#heart of reinforce algorithm
def reinforce(env):

    # CONSTANTS
    MAXITERS = 1000  #should converge in 1000 iters
    #alpha = 1e-1
    gamma = .99
    
    # init optimizer for model
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-08)

    
    # load network
    model = load_model('CartPole-v0_config.yaml')
    #model.compile(optimizer=optimizer,loss='mse')
    # repeat forever

    # init session and load in variables
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())  
 
    # loop thru training iterations
    for i in range(MAXITERS):
        # print every 100 or 25
        if i % 25 == 0:
        #if i % 100 == 0:
            get_total_reward(env,model,i)
        
        # set up memory structures
        stepsInEpisode = 0
        stateMem = []
        actionMem = []

        # reset env
        env.reset()

        # burnin first step, not in loop
        #save init state
        stateMem.append(env.env.state)
        
        #env.render()
        #decide first action and save
        pof1, action = choose_action(model,env.env.state)
        actionMem.append(action)
        
        # apply action
        newstate,reward,isTerm, junk = env.env.step(action)
        stepsInEpisode = stepsInEpisode+1

        #finish episode
        while(not isTerm): 
            #save state, choose action, save action, apply (step)
            stateMem.append(env.env.state)
            #env.render()
            pof1, action = choose_action(model, env.env.state)
            actionMem.append(action)
            newstate, reward, isTerm, junk = env.step(action)
            stepsInEpisode = stepsInEpisode + 1

        #UPDATE NEURAL NETWORK AFTER EPISODE ENDS

        # setup structures for derivs
        actionchosen = tf.placeholder(tf.int32)
        
        loss = tf.log(model.output[0][actionchosen])
        
        var_grad = tf.gradients(loss, model.weights)
        update_op = optimizer.apply_gradients(zip(var_grad, model.trainable_weights))
        
        sess.run(tf.global_variables_initializer())  
 
        #sess.run(loss,{model.input:[[1,2,3,4]],})
        #sess.run(tf.global_variables_initializer())
        
        #delta adds together all gradient updates in batch
        delta = -1
                
        for j in range(stepsInEpisode):
            # reward is number of steps including and after self (reward is always 1)
            G_t = 0
            for k in range(stepsInEpisode-j):
                G_t = G_t*gamma+1

            #print sess.run(var_grad,{model.input:[stateMem[j]]})
            
            #calc gradient
            gradd = sess.run(var_grad,{model.input:[stateMem[j]],actionchosen:actionMem[j]})
            
            #get multiplier f or this gradient
            multt = -1*G_t
            #multiply one by one to elements in this list
            for qq in range(np.size(gradd)):
                gradd[qq] = gradd[qq]*multt
                     
            #add these scaled gradients to delta
            if delta == -1:
                delta = gradd
            else:
                delta = delta+gradd
        #print (delta)
        sess.run(update_op, feed_dict={g: s for g, s in zip(var_grad, delta)})        
                
        #desiredWeights = model.get_weights()+ delta           
        #model.set_weights(desiredWeights)
        
        #if alpha > .1:
        #   alpha = alpha - .005
            
        #sess.close()
            #tf.log(a)
            #model = model + alpha*gamma^j*???

    """Policy gradient algorithm

    Parameters
    ----------
    env: your environment

    Returns
    -------
    total_reward: float
    """
    return model.get_weights()


'''
def getNetwork():
    model = Sequential([
        Dense(32, input_shape=(4,)),
        Activation('relu'),
        Dense(2),
        Activation('softmax'),
    ])
    return model

def getNetwork2():
    model = Sequential([
        Dense(32, input_shape=(4,)),
        Activation('relu'),
        Dense(32),
        Activation('relu'),
        Dense(32),
        Activation('relu'),
        Dense(2),
        Activation('softmax'),
    ])
    return model
'''

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
