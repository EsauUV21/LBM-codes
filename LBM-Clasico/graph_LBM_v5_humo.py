import matplotlib.pyplot as plt
import matplotlib.animation as animation
import jax.numpy as jnp
import pickle
import time

def get_1(u):
    return u

def mag_vel(u):
    return jnp.linalg.norm(u, axis=-1, ord=2)

fig, axs = plt.subplots(1, 2, figsize = (10, 10))

file1 = open('vel_humo', 'rb')
vels = mag_vel(pickle.load(file1))

file2 = open('vor_humo', 'rb')
vor = get_1(pickle.load(file2))

#file3 = open('conc_humo', 'rb')
#concs = get_1(pickle.load(file3))


grafico1 = axs[0].imshow(vels, cmap='viridis', animated = True, origin='lower', vmin = 0.005, vmax = 0.1)
axs[0].set_title('Velocidad')
plt.colorbar(grafico1, ax=axs[0])

grafico2 = axs[1].imshow(vels, cmap='plasma', animated = True, origin='lower', vmin = 0.0, vmax = 0.006)
axs[1].set_title('Vorticidad')
plt.colorbar(grafico2, ax=axs[1])

#grafico3 = axs[1, 0].imshow(concs, cmap='plasma', animated = True, origin='lower', vmin = 0.0, vmax = 1.0)
#axs[1, 0].set_title('Concentración')
#plt.colorbar(grafico3, ax=axs[1, 0])


def update_fig(*args):

    try:
        vels = mag_vel(pickle.load(file1))
        grafico1.set_data(vels)
        vors = get_1(pickle.load(file2))
        grafico2.set_data(vors)
        #concs = get_1(pickle.load(file3))
        #grafico3.set_data(concs)

        
        return [grafico1, grafico2]
    except EOFError:
        file1.seek(0)
        file2.seek(2)
        #file3.seek(2)

    vels = mag_vel(pickle.load(file1))
    grafico1.set_data(vels)
    vors = get_1(pickle.load(file2))
    grafico2.set_data(vors)
    #concs = get_1(pickle.load(file3))
    #grafico3.set_data(concs)
 
    return [grafico1, grafico2]

ani = animation.FuncAnimation(fig, update_fig, interval = 50, blit = False, frames=500)
plt.show()

file1.close();
