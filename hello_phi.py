# SGD for y = x^phi model
import torch
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt
from ellipse import ellipse_perimeter
from tqdm import tqdm

# print("Running SGD...")
print("Torch version ",torch.__version__)
print("Numpy version ",np.__version__)

if torch.cuda.is_available():
	print("CUDA available", torch.cuda.get_device_name())
	DEVICE = "cuda:0"
else:
	print("CUDA not available, using CPU")
	DEVICE = "cpu"

def rmse(y, y_hat):
	# computes rmse error
	return torch.sqrt(torch.mean((y-y_hat).pow(2).sum()))

def actual_perimeter(x):
	ret = []
	# print(x)
	for r in x:
		a, b = r[0].item(), r[1].item()
		ret.append(ellipse_perimeter(a,b))
	ret = Variable(torch.FloatTensor(ret).reshape((x.size()[0],1)))
	return ret
	# raise Exception()

def learned_perimeter(x, w, b):
	# ret = 0
	return torch.matmul(x,w)

def learned_perimeter_2(x, w):
	for r in x:
		a, b = r[0].item(), r[1].item()
		c = a-b
		d = a+b
		print(a,b)
		raise Exception


# number of examples
n = 20000
learning_rate = 5e-4

# define model
x = torch.reshape(torch.rand(n*2, requires_grad=False),(n,2)).cuda()


# Model Parameters
w = torch.cuda.FloatTensor([np.pi,np.pi])

w_hat = torch.cuda.FloatTensor([1,0])
b_hat = torch.cuda.FloatTensor([0])
w_hat.requires_grad_()
b_hat.requires_grad_()
# print(exp_hat)
	
# Get Actual Targets
y = actual_perimeter(x).cuda()

# Optimizer
opt = torch.optim.SGD([w_hat,b_hat], lr=learning_rate)
# print(exp_hat)
# holds parameter and loss history
loss_history = []
k_history = []

def symm_loss(y, y_hat, w_hat, b_hat):
	target_loss = rmse(y,y_hat)
	symm_loss = torch.abs(torch.matmul(w_hat, torch.cuda.FloatTensor([1,-1])))
	b_loss = 100*b_hat**2
	return target_loss+symm_loss+b_loss


# training loop
for i in tqdm(range(0,10000)):
	# print("Iteration {}".format(i))
	opt.zero_grad()
	y_hat = learned_perimeter(x, w_hat, b_hat)

	loss = rmse(y,y_hat)
	# loss = symm_loss(y, y_hat,w_hat, b_hat)

	loss_history.append(loss.item())
	k_history.append((w_hat[0].item(),w_hat[1].item(),b_hat[0].item()))

	# compute gradient

	loss.backward()
	# Update parameters
	opt.step()
	# print(exp_hat)

	# print("loss = {}".format(loss.item()))
	# print("exp = {}".format(exp_hat.data[0]))

	# exp_hat.data -= learning_rate * exp_hat.grad.data
	# exp_hat.grad.data.zero_()
# print(k_history)
print(w_hat.data, b_hat.data)

with torch.no_grad():
	stupid_guess = learned_perimeter(x,w,0)

	print(rmse(y,stupid_guess))
	print(rmse(y,y_hat))
	stupid_guess = stupid_guess.cpu().numpy()
# plt.plot(k_history, loss_history)
	
	y_hat = y_hat.cpu().numpy()
	y = y.cpu().numpy()
	print(type(y), type(y_hat), type(stupid_guess))
	plt.plot(y,y_hat, 'go')
	plt.plot(y,stupid_guess, 'ro')


	plt.plot(np.linspace(0,2*np.pi,100),np.linspace(0,2*np.pi,100))
	# plt.scatter(x, y_hat.detach().numpy())
	# plt.scatter(x, y)
	# plt.plot(k_history)
	# plt.plot(k_history)
	plt.show()




