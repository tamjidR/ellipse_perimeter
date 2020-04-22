from ellipse import ellipse_perimeter
import numpy as np
import matplotlib.pyplot as plt

def get_ab(x):
	a=(1+x)/2
	b=(1-x)/2
	return a, b

x = np.linspace(0,1,1000)
y_actual = [ellipse_perimeter(get_ab(xi)) for xi in x]
y_guessed = [2.8]



plt.plot(x,y_actual)
plt.show()