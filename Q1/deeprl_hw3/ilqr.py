"""LQR, iLQR and MPC."""

from deeprl_hw3.controllers import approximate_A, approximate_B
import numpy as np
import scipy.linalg
import sys

def simulate_dynamics_next(env, x, u):
    """Step simulator to see how state changes.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.
    u: np.array
      The command to test. When approximating B you will need to
      perturb this.

    Returns
    -------
    next_x: np.array
    """
    env.state = np.copy(x)
    x_nxt, reward, is_done, info = env._step(u.copy(),env.dt)
    return x_nxt.copy()
    #return (x_nxt - x)

    #return np.zeros(x.shape)

def cost_inter(env, x, u):
    """intermediate cost function

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.
    u: np.array
      The command to test. When approximating B you will need to
      perturb this.

    Returns
    -------
    l, l_x, l_xx, l_u, l_uu, l_ux. The first term is the loss, where the remaining terms are derivatives respect to the
    corresponding variables, ex: (1) l_x is the first order derivative d l/d x (2) l_xx is the second order derivative
    d^2 l/d x^2
    """
    #loss
    #l = np.linalg.norm(u)#np.sum(u**2)
    l = np.sum(u**2)
    #the loss function doesn't depend on x, so both are zeros
    l_x = np.zeros(x.shape[0])
    l_xx = np.zeros((x.shape[0], x.shape[0]))
    #first derivative
    l_u = 2 * u.copy()
    l_uu = 2 * np.eye(env.DOF)
    l_ux = np.zeros((env.DOF, x.shape[0]))
    # print(l)
    # print(l_x)
    # print(l_xx)
    # print(l_u)
    # print(l_ux)

    # using LQR's cost function
    # diff_x = x - env.goal
    # #print(diff_x)
    # Q = env.Q
    # R = env.R
    # l = np.sum(np.dot(diff_x.T,np.dot(Q,diff_x)) + np.dot(u.T,np.dot(R,u)))
    # l_x = 2 * np.dot(diff_x,Q)
    # l_xx = 2 * Q
    # l_u = 2 * np.dot(u,R)
    # l_uu = 2 * R 
    # l_ux = np.zeros((env.DOF, x.shape[0]))

    #print(l_x)
    #print(l_xx)

    return l*env.dt,l_x*env.dt,l_xx*env.dt,l_u*env.dt,l_uu*env.dt,l_ux*env.dt
    #return l, l_x, l_xx, l_u, l_uu, l_ux



def cost_final(env, x):
    """cost function of the last step

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.

    Returns
    -------
    l, l_x, l_xx The first term is the loss, where the remaining terms are derivatives respect to the
    corresponding variables
    """
    diff_x = x - env.goal
    weights = 1e4
    #l = np.linalg.norm(diff_x) * weights
    l = np.sum(diff_x**2) * weights
    l_x = 2 * diff_x * weights
    l_xx = 2 * np.eye(x.shape[0]) * weights
    #print(l_xx_old)
    #return l*env.dt,l_x*env.dt,l_xx*env.dt
    return l, l_x, l_xx

def simulate(env, x0, U):

    x_arr = np.zeros((x0.shape[0],U.shape[1]))
    env.state = x0.copy()
    next_state = x0.copy()
    total_cost = 0
    for i in range(U.shape[1]):

      x_arr[:,i] = next_state.copy()
      #calculate the cost
      cost, _, _, _, _, _ = cost_inter(env, x_arr[:,i].copy(), U[:,i].copy())
      total_cost += cost
      env.state = x_arr[:,i].copy()
      next_state, _, _, _ = env.step(U[:,i].copy())
    #calculate the end cost
    cost, _, _ = cost_final(env, next_state.copy())
    total_cost += cost
    return x_arr.copy(), total_cost, next_state.copy()    

# def line_search_alpha(env,new_u,old_u,k_arr,K_arr,diff_x,x_0,t):

#   last_alpha = 0.9999999999
#   step_size = 0.05
#   last_cost = np.inf
#   #loop until the cost decrease
#   while(last_alpha > 0):

#     alpha = last_alpha - step_size
#     #print(alpha)
#     u_test = new_u.copy()
#     u_test[:,t] = old_u[:,t].copy() + alpha * k_arr[t].copy() + np.dot(K_arr[t].copy(),diff_x.copy())
#     #roll out to test the cost
#     _, cost, _  = simulate(env,x_0,u_test)
#     print(cost)
#     if(cost > last_cost):
#       return last_alpha
#     else:
#       last_cost = cost
#       last_alpha = alpha
#   return last_alpha if last_alpha > 0 else 0.0000001



def forward_pass(env, old_x_arr,old_u,k_arr,K_arr):
  tN = old_u.shape[1]
  new_x_arr = np.zeros(np.shape(old_x_arr))
  new_u = np.zeros(old_u.shape)
  next_x =  old_x_arr[:,0].copy() #the initial position is the same
  for t in range(tN):
    new_x_arr[:,t] = next_x.copy()

    #diff_x = (new_x_arr[:,t] - env.goal) - (old_x_arr[:,t] - env.goal)
    diff_x = new_x_arr[:,t].copy() - old_x_arr[:,t].copy()


    #best_alpha = line_search_alpha(env,new_u,old_u,k_arr,K_arr,diff_x,old_x_arr[:,0],t)
    best_alpha = 1
    new_u[:,t] = old_u[:,t].copy() + best_alpha * k_arr[t].copy() + np.dot(K_arr[t].copy(),diff_x.copy())
    #new_u[:,t] = old_u[:,t].copy() + k_arr[t].copy() + np.dot(K_arr[t].copy(),diff_x.copy())

    env.state = next_x.copy()
    next_x,_,_,_ = env.step(new_u[:,t].copy())

  return new_x_arr, new_u, next_x

def backward_pass(env, x_arr, u_arr, x_final):
  """
  Calculate the K and k control values 
  """
  tN = x_arr.shape[1]
  #calculate the end cost
  l,l_x,l_xx = cost_final(env, x_final)
  #the value function at the final state is just the cost
  V_x = l_x
  V_xx = l_xx
  #declare k arrays
  k_arr = np.zeros((tN, env.DOF))
  K_arr = np.zeros((tN, env.DOF, x_final.shape[0]))

  for t in range(tN-1,-1,-1):
    #calculate the intermediate cost
    l, l_x, l_xx, l_u, l_uu, l_ux = cost_inter(env,x_arr[:,t].copy(), u_arr[:,t].copy())
    #calculate the dynamics
    f_x = np.eye(x_final.shape[0]) + approximate_A(env,x_arr[:,t].copy(), u_arr[:,t].copy()) * env.dt
    f_u = approximate_B(env,x_arr[:,t].copy(), u_arr[:,t].copy()) * env.dt
    #now we can get all the Q values
    Q_x = l_x + np.dot(f_x.T, V_x)
    Q_u = l_u + np.dot(f_u.T,V_x)
    Q_xx = l_xx   + np.dot(f_x.T, V_xx).dot(f_x)
    Q_ux = l_ux   + np.dot(f_u.T, V_xx).dot(f_x)
    Q_xu = l_ux.T + np.dot(f_x.T, V_xx).dot(f_u)
    Q_uu = l_uu   + np.dot(f_u.T, V_xx).dot(f_u)

    #regularize the matrix and inverse it 
    u,s,v = np.linalg.svd(Q_uu)
    s_inv = np.diag(1.0/(s+1))
    Q_uu_inv = np.dot(u, np.dot(s_inv, v.T))

    k_arr[t] = -np.dot(Q_uu_inv, Q_u)
    K_arr[t] = -np.dot(Q_uu_inv, Q_ux)
    #the next value function
    V_x = Q_x - np.dot(Q_u,Q_uu_inv).dot(Q_ux)
    V_xx = Q_xx - np.dot(Q_xu,Q_uu_inv).dot(Q_ux)

  return K_arr, k_arr


def calc_ilqr_input(env, sim_env, tN=50, max_iter=1e6, U=None):
    """Calculate the optimal control input for the given state.


    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use this environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.
    tN: number of control steps you are going to execute
    max_itr: max iterations for optmization

    Returns
    -------
    U: np.array
      The SEQUENCE of commands to execute. The size should be (tN, #parameters)
    """

    #start state
    x_0 = env.state.copy()

    #we start with an empty control
    #U = U[:tN,:].T

    U = np.zeros((env.DOF,tN)) if U is None else U
    cost_list = []
    #run a simulation of the controls to get a list of states
    x_arr, last_cost, final_x = simulate(sim_env,x_0,U)
    #max_iter = 10
    for i in range(0,max_iter):
      #do a backward pass on the U and X_arr
      K_arr, k_arr = backward_pass(sim_env, x_arr.copy(), U.copy(), final_x.copy())
      #now we do a forward pass
      x_arr, U, final_x = forward_pass(sim_env, x_arr.copy(),U.copy(),k_arr.copy(),K_arr.copy())
      #run simulate again to get the cost
      _, new_cost, _ = simulate(sim_env,x_0.copy(),U.copy())
      #calculate the differences in cost
      cost_diff = np.abs(last_cost - new_cost)
      last_cost = new_cost

      cost_list.append(new_cost)

      sys.stdout.write("\rstep:{} cost:{} diff:{}".format(i,new_cost,cost_diff))
      sys.stdout.flush()
      #break if the difference between iteration is less than certain threshold
      if(cost_diff < 1e-6):
        break

    return U, cost_list
