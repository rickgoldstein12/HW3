import gym
from deeprl_hw3.controllers import calc_lqr_input
from deeprl_hw3.ilqr import calc_ilqr_input
import matplotlib.pyplot as plt
import time
import numpy as np

def main():

	# env_name = "TwoLinkArm-v0"
	# #env_name = "TwoLinkArm-limited-torque-v0"
	env_name = "TwoLinkArm-v1"
	# env_name = "TwoLinkArm-limited-torque-v1"

	env = gym.make(env_name)
	sim_env = gym.make(env_name)

	look_step = 100
	max_itr = 100
	state = env.reset()
	_ = sim_env.reset()
	total_reward = 0
	q_states = []
	q_dot_states = []

	cost_list = []
	control_list = []
	#controls = None
	#calculate controls
	
	t = 100
	is_terminal = False
	while(not is_terminal):
		q_states.append(state[:env.DOF])
		q_dot_states.append(state[env.DOF:])

		if(t >= look_step):
			t = 0
			controls, c_list = calc_ilqr_input(env, sim_env, look_step, max_itr)
			cost_list = cost_list + c_list

		next_state,reward,is_terminal,_ = env.step(controls[:,t])
		control_list.append(controls[:,t])

		total_reward += reward
		state = next_state
		t += 1

	print("{},total reward:{}".format(env_name,total_reward))

	f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

	ax1.set_title("Control Inputs over time")
	ax1.set_ylabel('control input')
	#ax1.plot([0,look_step],[0,0],'--',c='r')
	ax1.plot(control_list)

	ax2.set_title("q over time")
	ax2.set_ylabel('q')
	ax2.plot([0,len(q_states)],[env._goal_q[0],env._goal_q[0]],'--',c='r')
	ax2.plot([0,len(q_states)],[env._goal_q[1],env._goal_q[1]],'--',c='r')
	ax2.plot(q_states)

	ax3.set_title("q-dot over time")
	ax3.set_ylabel('q-dot')
	ax3.plot(q_dot_states)
	ax3.plot([0,len(q_states)],[env.goal_dq[0],env.goal_dq[0]],'--',c='r')
	ax3.plot([0,len(q_states)],[env.goal_dq[1],env.goal_dq[1]],'--',c='r')
	ax3.plot
	f.suptitle("Plots with iLQR for {}".format(env_name))

	plt.show()
	print(len(cost_list))
	plt.title("cost over iterations")
	plt.plot([look_step,look_step],[np.max(cost_list),np.max(cost_list)],'--',c='r')
	plt.plot(cost_list)
	plt.show()

if __name__ == '__main__':
	main()