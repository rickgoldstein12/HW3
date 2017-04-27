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
        newstate=env.reset()

        stepsInEpisode = 0

        isTerm = False
        # finish episode
        while (not isTerm):
            pof1, action = choose_action(model, newstate)
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
    #model = getNetwork()
    model.compile(optimizer=optimizer,loss='binary_crossentropy')
    # repeat forever

    # init session and load in variables
    sess = tf.Session()
    
    # setup structures for derivs
    actionchosen = tf.placeholder(tf.int32, name="a")
    rewardx = tf.placeholder(tf.float32, name = "r")
    
    #gradtemp = list(zip(*model.optimizer.optimizer.compute_gradients(tf.log(model.output[0][actionchosen]), model.trainable_weights))[0])
    #scaled_grads = [(tf.multiply(vg,rewardx)) for vg in grad]
    #update_op = model.optimizer.optimizer.apply_gradients(zip(scaled_grads, model.trainable_weights))
    
    
    loss = tf.log(model.output[0][actionchosen])
    
    var_grad = tf.gradients(loss, model.weights)
    
    scaled_grads = [(tf.multiply(vg,rewardx)) for vg in var_grad]
    
    update_op = optimizer.apply_gradients(zip(scaled_grads, model.trainable_weights))
    
    #grad = list(zip(*model.optimizer.optimizer.compute_gradients(tf.log(model.output[0][actionchosen]), model.trainable_weights))[0])
    #update_op2 = optimizer.apply_gradients(zip(grad, model.trainable_weights))
    #foo = model.optimizer.optimizer.apply_gradients(zip(grad, model.trainable_weights))
    
    sess.run(tf.global_variables_initializer())    
    #episode = []
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
        rewardMem = []

        # reset env
        newstate = env.reset()
        
        isTerm = False
        while(not isTerm): 
           #save state, choose action, save action, apply (step)
           stateMem.append(newstate)
           pof1, action = choose_action(model, newstate)
           actionMem.append(action)
           newstate, reward, isTerm, junk = env.step(action)
           #print (reward)
           rewardMem.append(reward)
           stepsInEpisode = stepsInEpisode + 1       
        
        # burnin first step, not in loop
        #save init state

        #env.render()
        #decide first action and save
        

        #finish episode


        #print (actionMem)
        #print (stateMem)

        #UPDATE NEURAL NETWORK AFTER EPISODE ENDS
        for j in range(stepsInEpisode):
            # reward is number of steps including and after self (reward is always 1)
            G_t = 0
            #for k in range(stepsInEpisode-j):
            #   G_t = G_t*gamma+1
            gamult = 1
            for k in range(stepsInEpisode-j):
                G_t = G_t+rewardMem[k]
                gamult = gamult*gamma
            multt = -1.0*G_t*gamult
            
            cur_state = np.array(stateMem[j]).reshape((1,4))
         
            #gradient =sess.run(var_grad, feed_dict={actionchosen:actionMem[j],rewardx:multt,model.inputs[0]:cur_state})   
            
            #for a in range(0, len(grad)):
             #   gradient[a] = multt*gradient[a]
            #q1= sess.run(var_grad, feed_dict={actionchosen:actionMem[j],rewardx:multt,model.inputs[0]:cur_state})  
            #q2= sess.run(scaled_grads, feed_dict={actionchosen:actionMem[j],rewardx:multt,model.inputs[0]:cur_state})
            #q3=sess.run(foo, feed_dict={key: value for key, value in zip(grad,gradient)})   
            q3= sess.run(update_op, feed_dict={actionchosen:actionMem[j],rewardx:multt,model.inputs[0]:cur_state})
        
        
'''
        for xxx in range(0, 1000):
    		   if xxx % 25 == 0:
    		        get_total_reward(env,model,i)  
    	       #print xxx
    		   done = False
    		   state = env.reset()
    		   episode[:] = []
    		   while(done == False):
    			   probability, action = choose_action(model, state)
    			   observation, reward, done, info = env.step(action)
    			   episode.append([state, reward, action])
    			   state = observation
    		   discount = 1
    		   G = 0.0
    		   for ii in range(0, len(episode)):
    			   frame = episode[ii]
    			   G = 0.0
    			   for l in range(ii, len(episode)):
    				   G = G + episode[l][1]
    			   state = frame[0]
    			   action = frame[2]
    			#sess.run(tf.global_variables_initializer())
    			   #gradient = sess2.run(grad, feed_dict={model.input:np.reshape(state, (1,4)), index:action})
    			#print "-------------------------------->" , len(gradient)
    			   #for a in range(0, len(gradient)):
    				   #gradient[a] = -discount*G*gradient[a]
    			#sess.run(tf.global_variables_initializer())
    			   G_t = 0
                #for k in range(stepsInEpisode-j):
                #   G_t = G_t*gamma+1
    			   gamult = 1
    			   for k in range(stepsInEpisode-ii):
    			       G_t = G_t+rewardMem[k]
    			       gamult = gamult*gamma
    			   multt = -1.0*G_t*gamult
    			   q3= sess.run(update_op, feed_dict={actionchosen:episode[l][2],rewardx:multt,model.inputs[0]:episode[l][0].reshape((1,4))})
    			   #q3= sess.run(update_op, feed_dict={actionchosen:e,rewardx:gradient,model.inputs[0]:)})
    			   #bleh = sess.run(foo, feed_dict={key: value for key, value in zip(grad, gradient)})
    			   discount = discount*gamma
    return model.get_weights()
'''
        #sess.run(loss,{model.input:[[1,2,3,4]],})
        #sess.run(tf.global_variables_initializer())
       
        #delta adds together all gradient updates in batch
        #delta = -1
                

            #print sess.run(var_grad,{model.input:[stateMem[j]]})
            
            #calc gradient
            
            #get multiplier f or this gradient
            
            #print(G_t)
            #multiply one by one to elements in this list
            #for qq in range(np.size(gradd)):
            #    gradd[qq] = gradd[qq]*multt
            
            #gradd = sess.run(var_grad,{model.input:[stateMem[j]],actionchosen:actionMem[j]})
            
            #print (multt)
            #print (type(multt))
            #print (actionMem[j])
            #print (type(actionMem[j]))

            #sess.run(test_op, feed_dict={actionchosen:[actionMem[j]]})
            
            #print (cur_state)
            #print (type(cur_state))
                     
                     
            #add these scaled gradients to delta
            #if delta == -1:
            #    delta = gradd
            #else:
            #    delta = delta+gradd
        #print (delta)
             
                
        #desiredWeights = model.get_weights()+ delta           
        #model.set_weights(desiredWeights)
        
        #if alpha > .1:
        #   alpha = alpha - .005
            
        #sess.close()
            #tf.log(a)
            #model = model + alpha*gamma^j*???


    



def getNetwork():
    model = Sequential([
        Dense(32, input_shape=(4,)),
        Activation('relu'),
        Dense(2),
        Activation('softmax'),
    ])
    return model
'''
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
