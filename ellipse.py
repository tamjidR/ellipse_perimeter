import math
def ellipse_perimeter(*args):
	if len(args)==2:
		a = args[0]
		b = args[1]
	else:
		a,b = args[0]
	h = (a-b)**2/(a+b)**2

	ret = (a+b)
	mult = 1+h/4+h**2/64+h**3/256+25*h**4/16384+49*h**5/65536+441*h**6/1048576
	return math.pi*ret*mult

# print(ellipse_perimeter(10, 0))