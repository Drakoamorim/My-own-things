import numpy as np
from matplotlib import pyplot as plt
from numpy import sin, cos
from numpy.linalg import inv
from math import pi as pi
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

def StateDot(State):  # Calculates the rate of change of the State Vector {Velocity, Position}
    #  Differential Equations Matrices
    A = np.array([[1, 0],
                  [0, 1]])
    B = np.array([[mi, 0],
                  [-1, 0]])
    F = np.array([-(g / L) * sin(State[1]), 0])

    # State = np.array([omega, theta])
    return inv(A).dot(F - B.dot(State))


def RK(State):  # Calculates the State vector in a step dt based in the previous State
    k1 = StateDot(State)
    k2 = StateDot(State + k1 * dt / 2)
    k3 = StateDot(State + k2 * dt / 2)
    k4 = StateDot(State + k3 * dt)
    StatePlus = State + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6  # State plus is the State vector in a t+dt time
    return StatePlus


# --- Parameters
L = 1  # Rigid bar length
g = 9.7  # Gravity acceleration
mi = .3  # Friction coefficient

# --- Initial conditions
theta = pi / 4
omega = 15

# --- Time span of the simulation
time = np.linspace(0, 60, 6001)     # starts in 0 seconds, finishes in 60 seconds, with a dt of 0.01
dt = time[1] - time[0]

#   Lists for storing the system solution
Theta = []
Omega = []
Alpha = []

state = [omega, theta]  # Initial state

for t in time:  # For each t, the State is calculated based on the previous t
    statedot = StateDot(state)  # = [alpha, omega]
    Theta.append(state[1])
    Omega.append(state[0])
    Alpha.append(statedot[0])
    state = RK(state)

#   Mechanical Energy of the System

E = 1/2 * (L*np.array(Omega))**2 + g * L * cos(np.array(Theta)) - g * L



# ------- Animation ------- #

fig = plt.figure(figsize=(13, 13), dpi=100)  # Creates the figure that will be updated in the future for each frame

ax_1 = fig.add_subplot(211)     # creates the space to be plotted the energy diagram
ax_1.set_xlim(time[0]-5, 30)
ax_1.set_ylim(-25, 120)
ax_1.set_xlabel('t(s)')
ax_1.set_ylabel('E(J/kg)')

Eline, = plt.plot([], [], linewidth=2, color='#BD0A5D')     # Eline is the Line object that will draw our Energy diagram

ax_2 = fig.add_subplot(212)  # create an axes object, here the pendulum will be plotted
ax_2.axis('scaled')
ax_2.set_xlim((-1.1 * L, 1.1 * L))
ax_2.set_ylim((-1.1 * L, 1.1 * L))

origin = (0, 0)

line, = plt.plot([], [], linewidth=1, color='#B9B9B9')  # creates a line object, which will be the rigid rod
point, = plt.plot([], [], marker='o', linewidth=100, color='#3E3E3E')  # create a line object as a single point
time_template = 'time = %.1fs'
time_text = ax_2.text(0.05, 0.9, '', transform=ax_2.transAxes)  # create a text object for show the time


def update(frame):
    # uptate function for animation frames
    Eline.set_data((time[0:frame]), (E[0:frame]))
    
    line.set_data((origin[0], L * sin(Theta[frame])), (origin[1], -L * cos(Theta[frame])))
    point.set_data((L * sin(Theta[frame]), (-L * cos(Theta[frame]))))
    time_text.set_text(time_template % (frame*dt))

    return Eline, line, point, time_text


plt.ioff()
ani = FuncAnimation(fig, update, frames=range(len(time)), interval=1000 * dt, repeat=False)  # frames = how many frames will have the animation, interval = how many milliseconds pass between each frame
ani.save('pendulum.mp4', writer='ffmpeg')   # you need ffmpeg to save your video
