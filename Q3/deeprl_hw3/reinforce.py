import gym
import numpy as np
import random
import tensorflow as tf

def get_total_reward(env, model):


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
            pof1, action = choose_action(model, env.state)
            newstate, reward, isTerm, junk = env.step(action)
            stepsInEpisode = stepsInEpisode + 1

        if stepsInEpisode > maxx:
            maxx = stepsInEpisode
        if stepsInEpisode < minn:
            minn = stepsInEpisode
        total = total + stepsInEpisode

    total = total/100.0

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


def reinforce(env):

    MAXITERS = 100000
    MAXEPISODELENGTH = 100000
    alpha = .1
    gamma = .9

    # make policy network (modeL)
    model = getNetwork()
    #model.compile(optimizer='rmsprop', loss='mse')
    # repeat forever
    for i in range(MAXITERS):
        if i % 100 == 0:
            get_total_reward(env,model)

        stepsInEpisode = 0

        # memory to keep track of states
        stateMemory = np.zeros((MAXEPISODELENGTH,4))
        savep1 = np.zeros((MAXEPISODELENGTH))
        # reset env
        env.reset()

        # burnin first step, not in loop
        stateMemory[0,:] = env.state
        # env.render()
        pof1, action = choose_action(model,env.state)
        savep1[0] = pof1
        newstate,reward,isTerm, junk = env.step(action)
        stepsInEpisode = stepsInEpisode+1

        #finish episode
        while(not isTerm and stepsInEpisode < MAXEPISODELENGTH):
            stateMemory[stepsInEpisode, :] = env.state
            # env.render()
            pof1, action = choose_action(model, env.state)
            savep1[stepsInEpisode] = pof1
            newstate, reward, isTerm, junk = env.step(action)
            stepsInEpisode = stepsInEpisode + 1

        for j in range(stepsInEpisode):
            # reward is number of steps including and after self (reward is always 1)
            G_t = stepsInEpisode - j
            a = np.empty((2))
            a[0] = 1-savep1[j]
            a[1] = savep1[j]
            tf.log(a)
            #model = model + alpha*gamma^j*???

    """Policy gradient algorithm

    Parameters
    ----------
    env: your environment

    Returns
    -------
    total_reward: float
    """
    return 0

from keras.models import Sequential
from keras.layers import Dense, Activation



def getNetwork():
    model = Sequential([
        Dense(32, input_shape=(4,)),
        Activation('relu'),
        Dense(2),
        Activation('softmax'),
    ])
    return model




def main():
    # my code here
    env = gym.make('CartPole-v0')

    reinforce(env)

    #env.reset()
    #env.render()
    #env.step(0)

    #qq = 1
    #qq = qq+1

if __name__ == "__main__":
    main()