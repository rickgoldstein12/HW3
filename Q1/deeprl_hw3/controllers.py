"""LQR, iLQR and MPC."""

import numpy as np
import scipy.linalg
import copy


def simulate_dynamics(env, x, u, dt=1e-5):
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
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    xdot: np.array
      This is the **CHANGE** in x. i.e. (x[1] - x[0]) / dt
      If you return x you will need to solve a different equation in
      your LQR controller.
    """
    env.state = copy.deepcopy(x)
    x_nxt, reward, is_done, info = env._step(u,dt)
    return (x_nxt - x)/dt
    #return np.zeros(x.shape)


def approximate_A(env, x, u, delta=1e-5, dt=1e-5):
    """Approximate A matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. You will need to perturb this.
    u: np.array
      The command to test.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    A: np.array
      The A matrix for the dynamics at state x and command u.
    """

    #simulate one step with current state and controls
    #f = Ax + Bu
    #df/dx = A
    A = np.zeros((x.shape[0], x.shape[0]))

    for d in range(0, np.size(x,0)):
      x_copy1 = copy.deepcopy(x)
      x_copy1[d] += delta
      x1_dot = simulate_dynamics(env, x_copy1, u, dt)
      x_copy = copy.deepcopy(x)
      x_copy[d] -= delta
      x2_dot = simulate_dynamics(env, x_copy, u, dt)
      A[:,d] = (x1_dot  - x2_dot)/(delta*2)
    return A
    #return 


def approximate_B(env, x, u, delta=1e-5, dt=1e-5):
    """Approximate B matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test.
    u: np.array
      The command to test. You will ned to perturb this.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    B: np.array
      The B matrix for the dynamics at state x and command u.
    """

    #simulate one step with current state and controls
    #f = Ax + Bu
    #df/dx = A
    B = np.zeros((x.shape[0], u.shape[0]))
    for d in range(0, np.size(u,0)):
      u_copy = copy.deepcopy(u)
      u_copy[d] += delta
      u1_dot = simulate_dynamics(env, x, u_copy, dt)
      u_copy = copy.deepcopy(u)
      u_copy[d] -= delta
      u2_dot = simulate_dynamics(env, x, u_copy, dt)
      B[:,d] = (u1_dot - u2_dot)/(delta*2)

    return B
    #return np.zeros((x.shape[0], u.shape[0]))



def calc_lqr_input(env, sim_env):
    """Calculate the optimal control input for the given state.

    If you are following the API and simulate dynamics is returning
    xdot, then you should use the scipy.linalg.solve_continuous_are
    function to solve the Ricatti equations.

    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use this environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.

    Returns
    -------
    u: np.array
      The command to execute at this point.
    """
    #based heavilty of wikipedia:https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator
    
    #get the current state from env
    x = env.state
    Q = env.Q
    R = env.R
    u = np.zeros(env.DOF)

    delta = 1e-5
    A = approximate_A(sim_env, x, u, delta)
    B = approximate_B(sim_env, x, u, delta)

    #get approximate A and B
    P = scipy.linalg.solve_continuous_are(A,B,Q,R)
    K = np.dot(np.linalg.inv(R),np.dot(B.T,P))
    #print(x)

    #get the optimal control to the goal
    optimal_control = -np.dot(K,env.goal)

    return optimal_control
