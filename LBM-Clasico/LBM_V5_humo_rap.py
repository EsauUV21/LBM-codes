from jax import lax
import jax.numpy as jnp
import jax
import pickle
import os

archivos = ["vel_humo", "vor_humo", "conc_humo"]
X_NODES = 100
Y_NODES = 100

tau = 1 / 1.9704433497536948

vels = jnp.array([
    [0,  1,  0, -1,  0,  1, -1, -1,  1],
    [0,  0,  1,  0, -1,  1,  1, -1, -1]
])

vel_der = jnp.array([1, 5, 8]);
vel_izq = jnp.array([3, 6, 7]);
vel_arr = jnp.array([2, 5, 6]);
vel_abj = jnp.array([4, 7, 8]);
vel_ver = jnp.array([0, 2, 4]);
vel_hor = jnp.array([0, 1, 3]);

w = jnp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

cs = 1 / 3 ** 0.5

pasos = 20000

max_x_vel = 0.25

cilindro_x = X_NODES // 5
cilindro_y = Y_NODES // 2
radio_cilindro = Y_NODES // 9

screen_shot = 100
iteraciones_saltar = 0

#Nueve velocidades discretas
Q = 9

normal = jnp.arange(9)
opuesto = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

def c_densidad(f):
    return jnp.sum(f, axis = -1)

def c_macro_vel(f, vels, rho):
    return jnp.einsum('NMq,Dq->NMD', f, vels) / rho[..., jnp.newaxis]


def equilibrio(rho, u):
    prod1 = jnp.einsum('NMD,Dq->NMq', u, vels)
    u_mag = jnp.sum(u * u, axis = -1)[..., jnp.newaxis]
    
    return w[jnp.newaxis, jnp.newaxis, :] * rho[..., jnp.newaxis] \
           * (1 + 3 * prod1 + 9 / 2 * prod1 ** 2 \
              - 3 / 2 * u_mag)

def vorticidad(u):
    dux_dx, dux_dy = jnp.gradient(u[..., 0])
    duy_dx, duy_dy = jnp.gradient(u[..., 1])

    return duy_dx - dux_dy

jax.config.update("jax_enable_x64", True)

#Definiendo el obstáculo circular
x = jnp.arange(X_NODES)
y = jnp.arange(Y_NODES)
X, Y = jnp.meshgrid(x, y)

obstaculo = X < 0
obstaculo.at[0,:].set(True)
#obstaculo.at[-1,:].set(True)
#obstaculo.at[:,0].set(True)
#obstaculo.at[:,-1].set(True)

perfil_vel = jnp.zeros((Y_NODES, X_NODES, 2))#.at[0, 50, 1].set(max_x_vel)

f = equilibrio(jnp.ones((Y_NODES, X_NODES)), perfil_vel)
g = equilibrio(jnp.zeros((Y_NODES, X_NODES)).at[5, 50].set(1), perfil_vel)
inic = f.copy()

@jax.jit
def optim(f, g):
    #El flujo que sale
    #f = f.at[-1, 1:-1, vel_abj].set(f[-2, 1:-1, vel_abj])
    f = f.at[-1, 1:-1].set(f[-2, 1:-1])
    f = f.at[1:-1, 0].set(f[1:-1, 1])
    f = f.at[1:-1, -1].set(f[1:-1, -2])
    
    #Momentos
    rho = c_densidad(f)
    #conc = c_densidad(g)
    u = c_macro_vel(f, vels, rho)

    #Flujo de entrada
    u = u.at[5, 50, :].set([0, max_x_vel])
    #conc = conc.at[5, 50].set(1.0)
    
    #rho = rho.at[5, 50].set(
    #    (jnp.sum(f[5, 50, vel_hor]) + \
    #    2 * jnp.sum(f[5,, 50, vel_abj])) / \
    #    (1 - u[5, 50, 1])
    #)

    #Nueva distribución de equilibrio
    feq = equilibrio(rho, u)
    #g = equilibrio(conc, u)

    #Condición de frontera
    #f = f.at[1, 50, vel_arr].set(feq[1, 50, vel_arr])
    f = f.at[5, 50].set(feq[5, 50])

    #Cálculo de la colisión
    f_nuev = (1 - 1 / tau) * f + (1 / tau) * feq


    #Propagación
    for i in range(Q):
        
        f_nuev = f_nuev.at[:, :, i].set(            
            jnp.roll(
                jnp.roll(
                    f_nuev[:, :, i],
                    shift = vels[1, i],
                    axis = 0),
            shift = vels[0, i],
            axis = 1)
        )

    f_nuev = f_nuev.at[0, :, vel_arr].set(inic[0, :, vel_arr])
    f_nuev = f_nuev.at[:, -1, vel_izq].set(inic[:, -1, vel_izq])
    f_nuev = f_nuev.at[:, 0, vel_der].set(inic[:, 0, vel_der])

    #CBB
    for i in range(Q):
        f_nuev = f_nuev.at[obstaculo, i].set(f[obstaculo, opuesto[i]])


    return (f_nuev, u, rho, vorticidad(u))

for arch in archivos:
    try:
        os.remove(arch)
    except FileNotFoundError:
        print("No existe el archivo.")

with open(archivos[0], 'ab') as f_vels, open(archivos[1], 'ab') as f_vors, open(archivos[2], 'ab') as f_conc:
    for paso in range(pasos):
        f_nuev, u, rho, vor = optim(f, g)

        if paso % screen_shot == 0:
            print(paso)

            if paso >= iteraciones_saltar:
                if paso == iteraciones_saltar:
                    print("Listo!")
                
                pickle.dump(vor, f_vors)
                pickle.dump(u, f_vels)
                #pickle.dump(conc, f_conc)


        f = f_nuev
