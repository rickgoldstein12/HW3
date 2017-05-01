import gym
from deeprl_hw3.controllers import calc_lqr_input
#from deeprl_hw3.ilqr import calc_ilqr_input
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

def main():

	# env_name = "TwoLinkArm-v0"
	# env_name = "TwoLinkArm-limited-torque-v0"
	# env_name = "TwoLinkArm-v1"
	env_name = "TwoLinkArm-limited-torque-v1"

	env = gym.make(env_name)
	sim_env = gym.make(env_name)

	is_terminal = False
	state = env.reset()
	sim_env.reset()
	last_u = np.zeros(env.DOF)

	control_inputs = []
	clipped_inputs = []
	q_states = []
	q_dot_states = []
	total_reward = 0
	step_num = 0
	while(not is_terminal):
		q_states.append(state[:env.DOF])
		q_dot_states.append(state[env.DOF:])
		#get control
		controls = calc_lqr_input(env, sim_env,last_u)
		nxtstate, reward, is_terminal, info = env.step(controls)
		control_inputs.append(controls)
		clipped_inputs.append(np.clip(controls,env.action_space.low,env.action_space.high))
		#control_inputs.append(np.clip(controls,env.action_space.low,env.action_space.high))
		#env.render()
		#input()
		state = nxtstate
		step_num += 1
		total_reward += reward
	print("{},total reward:{}, number of steps:{}".format(env_name,total_reward,step_num))

	f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

	ax1.set_title("Control Inputs over time")
	ax1.set_ylabel('control input')
	#ax1.plot([0,len(control_inputs)],[0,0],'--',c='r')
	ax1.plot(control_inputs)

	orange_patch = mpatches.Patch(color='orange', label='joint 2 controls')
	blue_patch = mpatches.Patch(color='blue', label='joint 1 controls')
	#red_line = mlines.Line2D(color='red', label='The red data')
	ax1.legend(handles=[orange_patch,blue_patch])


	ax2.set_title("q over time")
	ax2.set_ylabel('q')
	ax2.plot([0,len(q_states)],[env._goal_q[0],env._goal_q[0]],'--',c='r')
	ax2.plot([0,len(q_states)],[env._goal_q[1],env._goal_q[1]],'--',c='r')
	ax2.plot(q_states)

	orange_patch = mpatches.Patch(color='orange', label='joint 2 position')
	blue_patch = mpatches.Patch(color='blue', label='joint 1 position')
	red_line = mlines.Line2D([],[],color='red', label='target positions',linestyle='--')
	ax2.legend(handles=[orange_patch,blue_patch,red_line])


	ax3.set_title("q-dot over time")
	ax3.set_ylabel('q-dot')
	ax3.plot(q_dot_states)
	ax3.plot([0,len(q_states)],[env.goal_dq[0],env.goal_dq[0]],'--',c='r')
	ax3.plot([0,len(q_states)],[env.goal_dq[1],env.goal_dq[1]],'--',c='r')

	orange_patch = mpatches.Patch(color='orange', label='joint 2 velocity')
	blue_patch = mpatches.Patch(color='blue', label='joint 1 velocity')
	red_line = mlines.Line2D([],[],color='red', label='target velocities',linestyle='--')
	ax3.legend(handles=[orange_patch,blue_patch,red_line])

	f.suptitle("Plots with LQR for {}".format(env_name))
	plt.show()

	plt.title('Clipped Control Inputs over time for {}'.format(env_name))
	plt.ylabel('control input')
	#ax1.plot([0,len(control_inputs)],[0,0],'--',c='r')
	plt.plot(clipped_inputs)

	orange_patch = mpatches.Patch(color='orange', label='joint 2 controls')
	blue_patch = mpatches.Patch(color='blue', label='joint 1 controls')
	#red_line = mlines.Line2D(color='red', label='The red data')
	plt.legend(handles=[orange_patch,blue_patch])
	plt.show()

if __name__ == '__main__':
	main()