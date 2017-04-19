import gym
from deeprl_hw3.controllers import calc_lqr_input
import time

def main():
	#env = gym.make("TwoLinkArm-limited-torque-v0")
	env = gym.make("TwoLinkArm-v0")
	#sim_env = gym.make("TwoLinkArm-limited-torque-v0")
	sim_env = gym.make("TwoLinkArm-v0")

	is_terminal = False
	state = env.reset()
	sim_env.reset()
	while(not is_terminal):
		
		#get control
		controls = calc_lqr_input(env, sim_env)
		print(controls)
		nxtstate, reward, is_terminal, info = env.step(controls)
		print(nxtstate)
		env.render()
		input()


if __name__ == '__main__':
	main()