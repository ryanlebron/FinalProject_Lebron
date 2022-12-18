import numpy as np
import matplotlib.pyplot as plt


#function to obtain acceleration of particles
def acceleration(position, mass, G, softening):

    N = position.shape[0]
    acceleration = np.zeros((N,3))

    for i in range(N):
        for j in range(N):
            #change in positions per particle
            dx = position[j,0] - position[i,0]
            dy = position[j,1] - position[i,1]
            dz = position[j,2] - position[i,2]

            #accelerations
            inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)**(-1.5)
            acceleration[i,0] += G * (dx *inv_r3) * mass[j]
            acceleration[i,1] += G * (dy *inv_r3) * mass[j]
            acceleration[i,2] += G * (dz *inv_r3) * mass[j]

    return acceleration

def nbodysim():
	
	# parameters for simulating
	N            = 50 #total number of particles
	t            = 0  #current time of the simulation
	tEnd         = 20.0 #time at which simulation ends
	dt           = 0.1 #timestep
	softening    = 0.1 #softening parameter
	G            = 1.0 #gravitational constant

	# initial conditions
	mass = 20*np.ones((N,1))/N #total mass of particles is 20
	pos  = np.random.randn(N,3) #randomly selected positions and velocities
	vel  = np.random.randn(N,3)

	# calculate initial accelerations
	acc = acceleration( pos, mass, G, softening )
	
	# number of timesteps
	Nt = int(tEnd/dt)
	
	# save particle positions for plotting trails
	pos_save = np.zeros((N,3,Nt+1))
	pos_save[:,:,0] = pos

	t_all = np.arange(Nt+1)*dt
	
	# initialize figure stuff
	fig = plt.figure(figsize=(4,5), dpi=80)
	grid = plt.GridSpec(2, 1, wspace=0.0, hspace=0.3)
	ax1 = plt.subplot(grid[0:2,0])

	# main loop, including leapfrog 
	for i in range(Nt):
		# half step kick
		vel += acc * dt/2.0
		
		# full step drift
		pos += vel * dt
		
		# update accelerations
		acc = acceleration( pos, mass, G, softening )
		
		# update time
		t += dt
		
		# save positions for plotting trail
		pos_save[:,:,i+1] = pos
		
		# plot in real time
		if True or (i == Nt-1):
			plt.sca(ax1)
			plt.cla()
			xx = pos_save[:,0,max(i-50,0):i+1]
			yy = pos_save[:,1,max(i-50,0):i+1]
			plt.scatter(xx,yy,s=1,color=[.7,.7,1])
			plt.scatter(pos[:,0],pos[:,1],s=10,color='blue')
			ax1.set(xlim=(-5, 5), ylim=(-5, 5))
			ax1.set_aspect('equal', 'box')
			ax1.set_xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
			ax1.set_yticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
			
			plt.pause(0.001)
	plt.show()
	
nbodysim()