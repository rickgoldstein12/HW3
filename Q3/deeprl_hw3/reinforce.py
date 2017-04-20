import gym
import numpy as np
import random
import tensorflow as tf
import math

from keras.models import Sequential
from keras.layers import Dense, Activation


def get_total_reward(env, model,printTime):


    minn = 1000000
    maxx = 0
    total = 0

    # run 100 episodes
    for i in range(100):
        env.reset()

        stepsInEpisode = 0

        pof1, action = choose_action(model, env.env.state)
        newstate, reward, isTerm, junk = env.step(action)
        stepsInEpisode = stepsInEpisode + 1

        # finish episode
        while (not isTerm):
            pof1, action = choose_action(model, env.env.state)
            newstate, reward, isTerm, junk = env.step(action)
            stepsInEpisode = stepsInEpisode + 1

        if stepsInEpisode > maxx:
            maxx = stepsInEpisode
        if stepsInEpisode < minn:
            minn = stepsInEpisode
        total = total + stepsInEpisode


    total = total/100.0
    print printTime,total,minn,maxx
    
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

"""
def choose_actionDET(model, observation):
    choose the action 

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
    

    obstemp = np.zeros((1,4))
    obstemp[0][0] = observation[0]
    obstemp[0][1] = observation[1]
    obstemp[0][2] = observation[2]
    obstemp[0][3] = observation[3]

    # get p1 only from model
    p1 = model.predict(obstemp)[0][1]
    if (p1>=.5):
        return p1,1
    else:
        return p1,0
"""

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

    obstemp = np.zeros((1,4))
    obstemp[0][0] = observation[0]
    obstemp[0][1] = observation[1]
    obstemp[0][2] = observation[2]
    obstemp[0][3] = observation[3]

    # get p1 only from model
    p1 = model.predict(obstemp)[0][1]
    if random.random() < p1:
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
    
def reinforce(env):

    MAXITERS = 10000
    #MAXEPISODELENGTH = 1000
    alpha = .1
    gamma = .9

    # make policy network (modeL)
    model = getNetwork2()
    #model.compile(optimizer='rmsprop',
                  #loss='mse')
    # repeat forever
    for i in range(MAXITERS):
        if i % 100 == 0:
            get_total_reward(env,model,i)
        
        stepsInEpisode = 0
        stateMem = []

        # memory to keep track of states
        #stateMemory = np.zeros((MAXEPISODELENGTH,4))
        #savep1 = np.zeros((MAXEPISODELENGTH))
        # reset env
        env.reset()

        # burnin first step, not in loop
        #stateMemory[0,:] = env.env.state
        stateMem.append(env.env.state)
        #env.render()
        pof1, action = choose_action(model,env.env.state)
        #savep1[0] = pof1
        newstate,reward,isTerm, junk = env.env.step(action)
        stepsInEpisode = stepsInEpisode+1

        #finish episode
        while(not isTerm): # and stepsInEpisode < MAXEPISODELENGTH):
            #stateMemory[stepsInEpisode, :] = env.env.state
            stateMem.append(env.env.state)
            #env.render()
            pof1, action = choose_action(model, env.env.state)
            #savep1[stepsInEpisode] = pof1
            newstate, reward, isTerm, junk = env.step(action)
            stepsInEpisode = stepsInEpisode + 1

        # setup structures for derivs
        loss = tf.log(model.output)
        var_grad = tf.gradients(loss, model.weights)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        #sess.run(loss,{model.input:[[1,2,3,4]],})
        
        delta = -1
        for j in range(stepsInEpisode):
            # reward is number of steps including and after self (reward is always 1)
            G_t = stepsInEpisode - j
            #a = np.empty((2))
            #a[0] = 1-savep1[j]
            #a[1] = savep1[j]
            #print j
            #print stateMem
            #print sess.run(var_grad,{model.input:[stateMem[j]]})
            gradd = sess.run(var_grad,{model.input:[stateMem[j]]})
            
            #get multiplier
            multt = alpha*math.pow(gamma,j)*G_t
            #multiply one by one
            for qq in range(np.size(gradd)):
                gradd[qq] = gradd[qq]*multt
                     
            if delta == -1:
                delta = gradd
            else:
                delta = delta+gradd
                          
        desiredWeights = model.get_weights()+ delta           
        model.set_weights(desiredWeights)
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
