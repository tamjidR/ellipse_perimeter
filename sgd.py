import torch
import numpy as np
import tqdm


class SGD:

	print("Running SGD...")
	print("Torch version ",torch.__version__)
	print("Numpy version ",np.__version__)

	if torch.cuda.is_available():
		print("CUDA available", torch.cuda.get_device_name())
		DEVICE = "cuda:0"
	else:
		print("CUDA not available, using CPU")
		DEVICE = "cpu"

		

	# SGD Parameters
	n = 100
	learning_rate = 5e-5

	# model parameters
	model = None
	params = None

	opt = torch.optim.SGD([params], lr=learning_rate)


	# learning parameters
	X = None
	Y = None

	# learning history
	loss_history = []
	param_history = []

	def set_to_device():
		inputs = []

	def train(epochs=10000):





