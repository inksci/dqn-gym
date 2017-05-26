import numpy as np

class move_gym():
	def __init__(self,):
		pass
	def reset(self):
		self.target_x=0
		self.target_y=0
		self.move_x=20
		self.move_y=20
		# theta=np.random.rand()*2*np.pi
		# self.move_x=5*np.cos(theta)
		# self.move_y=5*np.sin(theta)

		# theta=np.random.rand()*2*np.pi
		# self.target_x=5*np.cos(theta)
		# self.target_y=5*np.sin(theta)

		state=np.array([self.target_x, self.target_y, self.move_x, self.move_y])
		return state
		
	def step(self, action):
		# return: next_state,reward,done,_
		if action==0:
			self.move_y=self.move_y+1
		if action==1:
			self.move_y=self.move_y-1
		if action==2:
			self.move_x=self.move_x+1
		if action==3:
			self.move_x=self.move_x-1

		next_state=np.array([self.target_x, self.target_y, self.move_x, self.move_y])

		if ((self.move_x-self.target_x)**2+(self.move_y-self.target_y)**2)<=0:
			reward=10*2
			done=1
		else:
			reward=-1
			done=0

		return next_state,reward,done, 17
		
