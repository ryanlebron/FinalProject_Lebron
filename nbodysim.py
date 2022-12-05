import numpy as np
import matplotlib.pyplot as plt


#function to obtain acceleration of particles
def acceleration(position,mass,gravity,softening):

    N = position.shape[0]
    a = np.zeros((N,3))

    for i in range(N):
        for j in range(N):
            #change in positions per particle
            dx = position[j,0] - position[i,0]
            dy = position[j,1] - position[i,1]
            dz = position[j,2] - position[i,2]

            #accelerations
            a[i,0] += G * mass[j] * dx * (dx**2 + dy**2 + dz**2 + softening**2)**(-1.5)
            a[i,1] += G * mass[j] * dy * (dx**2 + dy**2 + dz**2 + softening**2)**(-1.5)
            a[i,2] += G * mass[j] * dz * (dx**2 + dy**2 + dz**2 + softening**2)**(-1.5)
    return a