import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.animation as animation
import jax.numpy as jnp
import pickle
import time

plt.rcParams['text.usetex'] = True

def velocidad(u):
    return jnp.sqrt(jnp.sum(u * u, axis = 0));

fig, axs = plt.subplots(1, 2, figsize = (10, 5))

file1 = open('choques-v1-u.dat', 'rb')
#file2 = open('choques-v1-T.dat', 'rb')
file2 = open('choques-v1-rho.dat', 'rb')
u = velocidad(pickle.load(file1));
rho = pickle.load(file2);
        
grafico1 = axs[0].imshow(u, cmap='plasma', animated = True, origin='lower', vmin = 0.0, vmax = 0.6)
#axs[0].set_title('Velocidad 1')
cax = make_axes_locatable(axs[0]).append_axes("bottom", "5%", pad=0.25);
plt.colorbar(grafico1, cax = cax, orientation="horizontal");

grafico2 = axs[1].imshow(rho, cmap='viridis', animated = True, origin='lower', vmin = 0.0, vmax = 2.3)
#axs[1].set_title('Densidad 2')
cax = make_axes_locatable(axs[1]).append_axes("bottom", "5%", pad=0.25);
plt.colorbar(grafico2, cax = cax, orientation="horizontal");

#grafico2 = axs[1].imshow(rho, cmap='plasma', animated = True, origin='lower', vmin = 0.0, vmax = 1000000.0)
#axs[1].set_title('Temperatura')
#plt.colorbar(grafico2, ax=axs[1])

def update_fig(*args):

    try:
        u = velocidad(pickle.load(file1));
        rho = pickle.load(file2);
        
        grafico1.set_data(u);
        grafico2.set_data(rho);
        
        return [grafico1, grafico2]
    except EOFError:
        file1.seek(0)
        file2.seek(0)

    u = velocidad(pickle.load(file1));
    rho = pickle.load(file2);
    
    grafico1.set_data(u);
    grafico2.set_data(rho);
  
    return [grafico1, grafico2]

ani = animation.FuncAnimation(fig, update_fig, interval = 1, blit = True, frames=500)
plt.show()

file1.close();
