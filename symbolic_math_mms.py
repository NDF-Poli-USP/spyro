from sympy import *

# Define symbolic variables
x, y, z, t, c, pi = symbols("x y z t c pi")

# Define the velocity model
c = 1 #+ sin(pi*x)*sin(pi*y)
# u= x*(x-1)*y*(y-1)*t
u = sin(pi*x)*sin(pi*y)*t**2

dudt = diff(u, t)
du2dt2 = diff(dudt, t)

dudx = diff(u, x)
du2dx2 = diff(dudx, x)
dudy = diff(u, y)
du2dy2 = diff(dudy, y) 

f = (1/c**2) *du2dt2 - (du2dx2 + du2dy2)

print(simplify(f))

# a = f-((sin(3*t))*(-(9*x*y*(x - 1)*(y - 1) + 2*(x*(x - 1) + y*(y - 1))*(sin(pi*x)*sin(pi*y) + 1)**2)/(sin(pi*x)*sin(pi*y) + 1)**2 ))
# print(simplify(a))