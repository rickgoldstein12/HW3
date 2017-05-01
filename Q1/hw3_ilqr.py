import gym
from deeprl_hw3.controllers import calc_lqr_input
from deeprl_hw3.ilqr import calc_ilqr_input
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
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
	controls = np.zeros((env.DOF,look_step))
	#calculate controls
	
	step_sizes = look_step
	steppers = 0
	num_steps = 0

	is_terminal = False
	while(not is_terminal):

		nxt_controls = np.zeros((env.DOF,look_step))
		nxt_controls[:,:look_step-step_sizes] = controls[:,step_sizes:]

		controls, c_list = calc_ilqr_input(env, sim_env, look_step, max_itr,nxt_controls)

		steppers += 1
		cost_list = cost_list + c_list
		for t in range(step_sizes):
			q_states.append(state[:env.DOF])
			q_dot_states.append(state[env.DOF:])

			next_state,reward,is_terminal,_ = env.step(controls[:,t])
			env.render()
			control_list.append(controls[:,t])
			total_reward += reward
			state = next_state
			num_steps += 1
			if(is_terminal):
				break

	print("\n{},total reward:{}, num steps:{}".format(env_name,total_reward,num_steps))

	f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

	ax1.set_title("Control Inputs over time")
	ax1.set_ylabel('control input')
	#ax1.plot([0,len(control_inputs)],[0,0],'--',c='r')
	ax1.plot(control_list)

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

	f.suptitle("Plots with iLQR for {}".format(env_name))

	plt.show()
	plt.title("iLQR cost over iterations for {}".format(env_name))
	#plt.plot([look_step,look_step],[np.max(cost_list),np.max(cost_list)],'--',c='r')
	for s in range(steppers):

		plt.plot([step_sizes * (1+s),step_sizes * (1+s)],[0,np.max(cost_list)],'--',c='r')
	plt.plot(cost_list)
	blue_patch = mpatches.Patch(color='blue', label='total cost')
	red_line = mlines.Line2D([],[],color='red', label='algorithm restart points',linestyle='--')
	#red_line = mlines.Line2D(color='red', label='The red data')
	plt.legend(handles=[blue_patch, red_line])
	plt.show()

if __name__ == '__main__':
	main()