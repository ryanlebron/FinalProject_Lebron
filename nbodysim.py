import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

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

#function to obtain kinetic and potential energies of particles
def energy (pos , vel, mass, G):

	N = pos.shape[0]
	KE = 0.5 * np.sum(np.sum(mass*vel**2))
	PE = np.zeros((N,3))

	for i in range(N):
		for j in range(N):
			dx = pos[j,0] - pos[i,0]
			dy = pos[j,1] - pos[i,1]
			dz = pos[j,2] - pos[i,2]
			inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
			if inv_r > 0:
				PE[i,0] += (G * mass[j] * mass[i]) / (inv_r)
				PE[i,1] += (G * mass[j] * mass[i]) / (inv_r)
				PE[i,2] += (G * mass[j] * mass[i]) / (inv_r)
			

	PE = -np.sqrt(PE[i,0]**2 + PE[i,1]**2 + PE[i,2]**2)

	return KE,PE

#main simulation function
def nbodysim():
	
	# initial parameters
	N         = 20  # total number of particles
	t         = 0      # current time of the simulation
	tEnd      = 10.0   # time at which simulation ends
	dt        = 0.01   # timestep
	softening = 0.1    # softening length
	G         = 1.0    # gravity
	
    #initial conditions
	mass = 20.0*np.ones((N,1))/N  # total mass of particles is 20
	pos  = np.random.randn(N,3)   # randomly selected positions and velocities
	vel  = np.random.randn(N,3)
	
	# initial acceleration
	acc = acceleration( pos, mass, G, softening )
	
	# initial energy
	KE, PE  = energy( pos, vel, mass, G)

	# number of timesteps
	Nt = int(tEnd/dt)
	
	# save energies and positions for live plotting trails
	pos_save = np.zeros((N,3,Nt+1))
	pos_save[:,:,0] = pos
	KE_save = np.zeros(Nt+1)
	KE_save[0] = KE
	PE_save = np.zeros(Nt+1)
	PE_save[0] = PE
	t_all = np.arange(Nt+1)*dt
	
	# figure stuff
	fig = plt.figure(figsize=(4,5), dpi=80)
	grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
	ax1 = plt.subplot(grid[0:2,0])
	ax2 = plt.subplot(grid[2,0])
	
	# main loop, with leapfrog
	for i in range(Nt):

		# half step kick
		vel += acc * dt/2.0
		
		# full step drift
		pos += vel * dt
		
		# update accelerations
		acc = acceleration( pos, mass, G, softening )
		
		# update time
		t += dt
		
		# get energy of system
		KE, PE  = energy( pos, vel, mass, G)
		
		# save energies, positions 
		pos_save[:,:,i+1] = pos
		KE_save[i+1] = KE
		PE_save[i+1] = PE
		
		# plot in real time
		if True or (i == Nt-1):
			plt.sca(ax1)
			plt.cla()
			xx = pos_save[:,0,max(i-50,0):i+1]
			yy = pos_save[:,1,max(i-50,0):i+1]
			plt.scatter(xx,yy,s=1,color=[.7,.7,1])
			plt.scatter(pos[:,0],pos[:,1],s=10,color='blue')
			ax1.set(xlim=(-2, 2), ylim=(-2, 2))
			ax1.set_aspect('equal', 'box')
			ax1.set_xticks([-2,-1,0,1,2])
			ax1.set_yticks([-2,-1,0,1,2])
			
			plt.sca(ax2)
			plt.cla()
			plt.scatter(t_all,KE_save,color='orange',s=1,label='KE')
			plt.scatter(t_all,PE_save,color='blue',s=1,label='PE')
			plt.scatter(t_all,KE_save+PE_save,color='black',s=1,label='Etot')
			ax2.set(xlim=(0, tEnd), ylim=(-100, 100))
			
			
			plt.pause(0.001)
	    
	plt.sca(ax2)
	plt.xlabel('time')
	plt.ylabel('energy')
	ax2.legend(loc='upper right')
	plt.show()

nbodysim()