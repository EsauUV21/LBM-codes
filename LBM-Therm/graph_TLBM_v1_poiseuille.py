import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation
import jax.numpy as jnp
import pickle
import time

plt.rcParams['text.usetex'] = True

def get_temp(u):
    return u

def mag_vel(u):
    return jnp.linalg.norm(u, axis=-1, ord=2)

fig, axs = plt.subplots(1, 2, figsize = (10, 5))

file1 = open('vel_poise', 'rb')
vels = mag_vel(pickle.load(file1))

file2 = open('temp_poise', 'rb')
vor = get_temp(pickle.load(file2))

grafico1 = axs[0].imshow(vels, cmap='viridis', animated = True, origin='lower', vmin = 0.0, vmax = 0.8)
#axs[0].set_title('Velocidad $u$')
cax = make_axes_locatable(axs[0]).append_axes("bottom", "5%", pad=0.25);
plt.colorbar(grafico1, cax = cax, orientation="horizontal");

grafico2 = axs[1].imshow(vels, cmap='plasma', animated = True, origin='lower', vmin = 0.0, vmax = 5.0)
#plt.colorbar(shrink=0.8, aspect=20);
#axs[1].set_title('$\\Theta=\\frac{T-T_0}{T_h-T_c}$')
cax = make_axes_locatable(axs[1]).append_axes("bottom", "5%", pad=0.25);
plt.colorbar(grafico2, cax = cax, orientation="horizontal");

def update_fig(*args):

    try:
        vels = mag_vel(pickle.load(file1))
        grafico1.set_data(vels)
        vors = get_temp(pickle.load(file2))
        grafico2.set_data(vors)
        
        return [grafico1, grafico2]
    except EOFError:
        file1.seek(0)
        file2.seek(2)

    vels = mag_vel(pickle.load(file1))
    grafico1.set_data(vels)
    vors = get_temp(pickle.load(file2))
    grafico2.set_data(vors)
  
    return [grafico1, grafico2]

ani = animation.FuncAnimation(fig, update_fig, interval = 50, blit = False, frames=500)
plt.show()

file1.close();
