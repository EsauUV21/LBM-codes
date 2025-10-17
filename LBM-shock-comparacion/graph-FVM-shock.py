import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.animation as animation
import jax.numpy as jnp
import pickle
import time

plt.rcParams['text.usetex'] = True

def pressure(u):
    return u[4];

def densidad(u):
    return u[0];
    #return jnp.linalg.norm(u, axis=-1, ord=2)

cx = 1;
cy = 2;
cz = 3;
coords2D = jnp.array([cx, cy])

def velocidad(prim):
    _u = prim[coords2D];
       
    return jnp.sqrt(jnp.sum(_u * _u, axis = 0));

fig, axs = plt.subplots(1, 2, figsize = (10, 5))

file1 = open('data-v1.dat', 'rb')
prim = pickle.load(file1)
rho = densidad(prim)
p = pressure(prim)
u = velocidad(prim)

#grafico1 = axs[1].imshow(rho, cmap='viridis', animated = True, origin='lower', vmin = 0.0, vmax = 2.5)
#axs[1].set_title('Densidad')
#plt.colorbar(grafico1, ax=axs[1])

grafico1 = axs[0].imshow(p, cmap='plasma', animated = True, origin='lower', vmin = 0.0, vmax = 0.6)
cax = make_axes_locatable(axs[0]).append_axes("bottom", "5%", pad=0.25);
plt.colorbar(grafico1, cax = cax, orientation="horizontal");

grafico2 = axs[1].imshow(rho, cmap='viridis', animated = True, origin='lower', vmin = 0.0, vmax = 2.3)
cax = make_axes_locatable(axs[1]).append_axes("bottom", "5%", pad=0.25);
plt.colorbar(grafico2, cax = cax, orientation="horizontal");

def update_fig(*args):

    try:
        prim = pickle.load(file1)
        rho = densidad(prim)
        p = pressure(prim)
        u = velocidad(prim)
        grafico1.set_data(u)
        grafico2.set_data(rho)
        
        return [grafico1, grafico2]
    except EOFError:
        file1.seek(0)

    prim = pickle.load(file1)
    rho = densidad(prim)
    p = pressure(prim)
    u = velocidad(prim)
    grafico1.set_data(u)
    grafico2.set_data(rho)
  
    return [grafico1, grafico2]

ani = animation.FuncAnimation(fig, update_fig, interval = 1, blit = True, frames=500)
plt.show()

file1.close();
