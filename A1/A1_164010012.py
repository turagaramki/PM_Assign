import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline
# Sorce sink pair in freestream
LAMBDA = 1.0

u_free = 1.0
v_free = 0.0  # Free stream velocity components

V = complex(u_free, v_free)  # Free stream velocity function in complex plane

xsource, ysource = -1.0, 0.0  # Location of source
zsource = complex(xsource, ysource)
xsink, ysink = 1.0, 0.0  # Location of sink
zsink = complex(xsink, ysink)
# descrtize the computational domain of interest
x, y = np.linspace(-4.0, 4.0, 100), np.linspace(-2.0, 2.0, 100)
X, Y = np.meshgrid(x, y)

# Computational domain in complex plane
Z = X+1j*Y

def dWdz(z):
    return (V+LAMBDA/(2.0*np.pi*(z-zsource))-LAMBDA/(2.0*np.pi*(z-zsink)))

def Eu_int(fprime,dt,z):
    dW=fprime(z)
    xold,yold=z.real,z.imag
    u,v=dW.real,-dW.imag
    xnew=xold+dt*u
    ynew=yold+dt*v
    znew=xnew+1j*ynew
    return znew

def RK2_int(fprime,dt,z):
    xold,yold=z.real,z.imag
    dW=fprime(z)
    u,v=dW.real,-dW.imag
    xk1=dt*u
    yk1=dt*v
    zk1=(xold+xk1)+1j*(yold+yk1)
    dW=fprime(zk1)
    u,v=dW.real,-dW.imag
    xk2=dt*u
    yk2=dt*v
    xnew=xold+0.5*(xk1+xk2)
    ynew=yold+0.5*(yk1+yk2)
    znew=xnew+1j*ynew
    return znew

def fW(z):
    return (V*Z+LAMBDA/(2.0*np.pi)*np.log(Z-zsource)+LAMBDA/(2.0*np.pi)*np.log(Z-zsink)*-1)
    
W=fW(Z)

dW=dWdz(Z)#V+LAMBDA/(2.0*np.pi*(Z-zsource))-LAMBDA/(2.0*np.pi*(Z-zsink))
u=dW.real
v=-dW.imag
plt.figure(figsize=(15,4.0/8.0*15))
plt.streamplot(X, Y, u, v, density=2.0, linewidth=1, arrowsize=2, arrowstyle='->',color='b')
plt.scatter([xsource,xsink],[ysource,ysink],color='r',marker='o',s=80)
plt.contour(X, Y, W.imag,levels=[0,],linewidth=2, linestyle='solid')
# plt.axis('square')

t=0.0
tf=10.0
dt=0.1

x_t=-2.0
y_t=np.linspace(-2.0,2.0,21)

Ze=x_t+1j*y_t
Zrk2=x_t+1j*y_t

Eu_trace=[Ze]
Rk2_trace=[Zrk2]

while t<tf:
    Ze=Eu_int(dWdz,dt,Ze)
    Zrk2=RK2_int(dWdz,dt,Zrk2)
    Eu_trace.append(Ze)
    Rk2_trace.append(Zrk2)
    t+=dt
    
Eu_trace=np.array(Eu_trace)
Rk2_trace=np.array(Rk2_trace)

# plt.figure(2)
plt.figure(figsize=(15,4.0/8.0*15))
for i in range(len(Ze)):
    plt.plot(Eu_trace[:,i].real,Eu_trace[:,i].imag)
    
plt.title('Tracer plot with Euler integration')
plt.scatter([xsource,xsink],[ysource,ysink],color='r',marker='o',s=40)
plt.xlim(-2,2)
plt.ylim(-2,2)

# plt.figure(3)
plt.figure(figsize=(15,4.0/8.0*15))

for i in range(len(Ze)):
    plt.plot(Rk2_trace[:,i].real,Rk2_trace[:,i].imag)
    
plt.title('Tracer plot with RK2 integration')
plt.scatter([xsource,xsink],[ysource,ysink],color='r',marker='o',s=40)
plt.xlim(-2,2)
plt.ylim(-2,2)

plt.show()