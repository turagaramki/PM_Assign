import numpy as np
import matplotlib.pyplot as plt


class Circle:

    """docstring for Circle"self,radius,origin,num_points """

    def __init__(self, radius, origin, num_points):
        self.radius = radius
        self.origin = origin
        self.num_points = num_points

        self.angles = np.linspace(0, 2*np.pi, num_points+1)
        self.points = self.origin+radius*(np.cos(self.angles) +
                                          1j*np.sin(self.angles))
        self.end = self.points[1:]
        self.start = self.points[:-1]

    def mid_points(self):
        midpoints = 0.5*(self.start+self.end)
        return midpoints

    def draw(self):
        plt.close('all')
        plt.plot(self.points.real, self.points.imag, '--+')
        plt.plot(self.mid_points().real, self.mid_points().imag, '*')
        plt.show()

    def panel_lengths(self):
        plenghts = abs(self.end-self.start)
        return plenghts


C = Circle(1.0, complex(0, 0), 50)

# print(C.draw())

# print(C.panel_lengths())

def vel_constant_strength(z,l):
    vel_z_prime = (1j/(2*np.pi))*np.log((z - l)/z)
    return vel_z_prime
    
def get_vel(z,z_0,l,rot_angle):
    z_prime = (z - z_0)*np.exp(-1j*rot_angle)
    v_z_prime = vel_constant_strength(z_prime,l).conjugate()
    v = (v_z_prime*np.exp(1j*rot_angle))
    return v


#assembling the "b" matrix
def assemble_b(circ,v_inf):
    b = np.zeros(circ.N+1)
    b[:circ.N] = v_inf.real*(circ.unit_normals()).real + \
    v_inf.imag*(circ.unit_normals()).imag
    return b
#To consider a moving body replace "v_inf" by "vinf + v_body"
  
#Assembling the "A" matrix
def assemble_A(circ):
    A = np.empty((circ.N+1,circ.N))
    
    for i in range(circ.N):
        v = get_vel(circ.control_points()[i],circ.a,circ.lengths(),\
        circ.rotation_angle())
        A[i] = -(v.real*circ.unit_normals()[i].real + \
        v.imag*circ.unit_normals()[i].imag)
        
    A[circ.N] = np.ones(circ.N)
    return A
    
#Solving for gamma
#gamma = np.linalg.lstsq(A,b)
def vel_at_r(r,gamma,circ1,circ2,theta,v_inf):
    
    B = np.empty((circ2.N,circ1.N)) + 1j*np.empty((circ2.N,circ1.N))
    
    for i in range(circ2.N):
        B[i] = get_vel(circ2.b[i],circ1.a,circ1.lengths(),\
        circ1.rotation_angle())
        
    vel = np.dot(B,gamma[0])
    tot_vel = vel + v_inf
    vel_mag = np.abs(tot_vel)
    
    vel_r_exact = (1.0 - (circ1.r/circ2.r)**2)*(abs(v_inf)*np.cos(theta))
    vel_theta_exact = -(1.0 + (circ1.r/circ2.r)**2)*(abs(v_inf)*np.sin(theta))
    vel_exact = np.sqrt(vel_r_exact**2 + vel_theta_exact**2)
    return vel_mag,vel_exact