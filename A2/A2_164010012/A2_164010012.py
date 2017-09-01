import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from numba import jit

# %matplotlib inline
plt.close('all')

@jit
def vortex_velocity(z, z_vor, gamma, method, delta):

    k_z = -1j/(2*np.pi*(z - z_vor))
    r = abs(z-z_vor)
    vel = method(gamma, k_z, r, delta)
    return vel


def point_vortex(gamma, k_z, r, delta):
    # print('point')
    return (gamma*k_z).conjugate()


def krasny_blob(gamma, k_z, r, delta):
    k_delta = r**2/(r**2 + delta**2)
    # print('krasny')
    return (gamma*k_z*k_delta).conjugate()


def beale2nd_blob(gamma, k_z, r, delta):
    # print('beale2nd')
    k_delta = 1-np.exp(-r**2/delta**2)
    return (gamma*k_z*k_delta).conjugate()


def chorin_blob(gamma, k_z, r, delta):
    # print('chorin')
    if r >= delta:
        k_delta = 1.0
    else:
        k_delta = r/delta
    return (gamma*k_z*k_delta).conjugate()


class Vortices:

    def __init__(self, gamma, pos, delta, blob_method):
        self.gamma = gamma
        self.pos = pos
        self.delta = delta
        self.blob_method = blob_method

    def set_state(self, state):
        self.pos = state
        return state

    def get_state(self):
        return self.pos

    def get_velocity(self):
        vel = np.zeros_like(self.pos)
        for i, z_i in enumerate(self.pos):
            for j, z_j in enumerate(self.pos):
                if i != j:
                    vel[i] += vortex_velocity(z_i, z_j,
                                              self.gamma[j], self.blob_method,
                                              self.delta[j])
        return vel


def integrate(integrable, take_step, dt, tf):
    result = [integrable.get_state()]
    t = 0.0
    while t < tf:
        state = take_step(integrable, dt, t)
        result.append(state.copy())
        t += dt
#         print((tf-t)/t*100)

    return np.asarray(result)


def euler_step(integrable, dt, t):
    vel = integrable.get_velocity()
    pos = integrable.get_state()
    return integrable.set_state(pos + vel*dt)


def rk2_step(vortices, dt, t):
    orig_pos = vortices.get_state().copy()
    vel = vortices.get_velocity()
    vortices.set_state(orig_pos + vel*dt*0.5)
    vel = vortices.get_velocity()
    return vortices.set_state(orig_pos + vel*dt)

# def rk4_step(vortices, dt, t):
#     orig_pos = vortices.get_state().copy()
#     vel = vortices.get_velocity()
#     vortices.set_state(orig_pos + vel*dt*0.5)
#     vel = vortices.get_velocity()
#     return vortices.set_state(orig_pos + vel*dt)


def strength_per_len(x):
    gamma_0 = 1.0
    b = 1.0
    return gamma_0*(4*x/b**2)/np.sqrt(1 - 4*x**2/b**2)


def discretise_point_vortex(N):
    b = 1.0

# for  uniformly spaced points
#     p = -b/2
#     q = b/2
#     dx = ((q-p)/N)*np.ones(N)
#     x = np.linspace(p + dx[0]/2, q - dx[0]/2, N)

# #for sine-spaced points
    c = (b/2)*np.sin(np.linspace(-np.pi/2+1e-1, np.pi/2-1e-1, N+1))
    x = (c[1:] + c[:-1])/2.0
    dx = c[1:] - c[:-1]

    y = np.zeros_like(x)
    z = x + 1j*y

    gamma = dx*strength_per_len(x)

    return z, gamma, dx


def vor_sheet_roll_up(N, tf, blb):

    vor_z, gamma, dx = discretise_point_vortex(N)
    delta = dx*15.0
    # print(blob)

    dt = 0.01

    vortices2 = Vortices(gamma, vor_z, delta, blb)
    results = integrate(vortices2, rk2_step, dt, tf)

    return results

# matplotlib.use('Agg')
def a1():
    pwd = os.getcwd()
    N = 200
    T = np.linspace(0.25, 1.0, 4)
    T=np.array([1.0])
    blob = [krasny_blob]#, beale2nd_blob, chorin_blob]
    j = 1
    for blb in blob:
        plt.figure(j, figsize=(10, 8))
        j += 1
        k = 411
        print(blb.__name__)
        for tf in T:
            # print('Simulating Timestep %f sec',tf)
            result = vor_sheet_roll_up(N, tf, blb)
            plt.subplot(k)
            plt.plot(result.real[-1], result.imag[-1])
            plt.axis('equal')
            title = 'After '+str(tf) + ' Sec'
            plt.title(title)
            k += 1
        suptitle = 'Vortex sheet Roll-up with '+blb.__name__
        plt.suptitle(suptitle)
        plt.subplots_adjust(hspace=0.75, wspace=0.3)
        figname = blb.__name__ + '.png'
        plt.savefig(os.path.join(pwd, figname), dpi=600)

    plt.show()
