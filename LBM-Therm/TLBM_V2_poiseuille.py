from jax import lax
import jax.numpy as jnp
import jax
import pickle
import os

archivos = ["vel_poise", "temp_poise"]
X_NODES = 100
Y_NODES = 100

R = 1
T0 = 200
Tc = T0 - 10
Th = T0 + 10
cv = 3 / 2 * R
cp = 5 / 2 * R
u0 = 0.0

tau_f = 0.75
tau_h = 1.056
_1_tau_f = 1 / tau_f
_1_tau_h = 1 / tau_h
_1_tau_fh = _1_tau_f + _1_tau_h

vels = jnp.array([
    [0,  1,  0, -1,  0,  1, -1, -1,  1],
    [0,  0,  1,  0, -1,  1,  1, -1, -1]
])

a = jnp.array([[0.00005, 0.0]]) * jnp.ones((Y_NODES, X_NODES))[..., jnp.newaxis]

vel_der = jnp.array([1, 5, 8]);
vel_izq = jnp.array([3, 6, 7]);
vel_arr = jnp.array([2, 5, 6]);
vel_abj = jnp.array([4, 7, 8]);
vel_ver = jnp.array([0, 2, 4]);
vel_hor = jnp.array([0, 1, 3]);

w = jnp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

#cs = 1 / 3 ** 0.5

pasos = 30000

screen_shot = 100
iteraciones_saltar = 1

#Nueve velocidades discretas
Q = 9

normal = jnp.arange(9)
opuesto = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

def c_macro_vel(f, vels, rho):
    return jnp.einsum('NMq,Dq->NMD', f, vels) / rho[..., jnp.newaxis]


def equilibrio(rho, u, E):
    prod1 = jnp.einsum('NMD,Dq->NMq', u, vels)
    prod1_2 = prod1 ** 2
    u_mag = jnp.sum(u * u, axis = -1)[..., jnp.newaxis]
    vels_mag = jnp.sum(vels * vels, axis = 0)[jnp.newaxis, jnp.newaxis,:]
    p0 = R * T0 * rho
    coef = 1 / (R * T0)
    coef_2 = coef ** 2
    w_rho = w[jnp.newaxis, jnp.newaxis, :] * rho[..., jnp.newaxis]
    
    
    f = w_rho \
           * (1 + 3 * prod1 + 9 * 0.5 * prod1_2 - 3 * 0.5 * u_mag)

    h = w[jnp.newaxis, jnp.newaxis, :] * p0[..., jnp.newaxis] \
           * (3 * prod1 + 9 * prod1_2 - 3 * u_mag + 0.5 * (3 * vels_mag - 2)) \
           + E[..., jnp.newaxis] * f

    vels_a = jnp.einsum('NMD,Dq->NMq', a, vels)
    a_u = jnp.sum(a * u, axis = -1)[..., jnp.newaxis]
    F = w_rho * (3 * vels_a + 9 * vels_a * prod1 - 3 * a_u)

    q = 3 * R * T0 * (3 * E[..., jnp.newaxis] * w_rho * vels_a + f * vels_a)

    Z = (3 * R * T0) ** 0.5 * (prod1 - 0.5 * u_mag)

    return (f, h, F, q, Z)

def temps(T):
    return (T -Tc) / (Th - Tc);

def perfil_inicial(u0):
    column = jnp.arange(Y_NODES)[:, None]
    height = Y_NODES - 1
    _1_height_2 = 1 / (height ** 2)
    A = jnp.tile(4 * u0 * _1_height_2 * column * (height - column), (1, X_NODES))

    return A[..., jnp.newaxis] * jnp.array([[[1.0, 0.0]]])

def propaga(df):
    df_nuev = df
    
    for i in range(Q):
        df_nuev = df_nuev.at[:, :, i].set(            
            jnp.roll(
                jnp.roll(
                    df_nuev[:, :, i],
                    shift = vels[1, i],
                    axis = 0),
            shift = vels[0, i],
            axis = 1)
        )

    return df_nuev
    

jax.config.update("jax_enable_x64", True)

perfil_vel = perfil_inicial(u0)

T_ini = jnp.full((Y_NODES, X_NODES), T0).at[0, :].set(Tc).at[-1, :].set(Th)
E_ini = cv * T_ini
f, h, F, q, Z = equilibrio(jnp.ones((Y_NODES, X_NODES)), perfil_vel, E_ini)
inic = f.copy()

@jax.jit
def optim(f, h):   
    #Momentos
    rho = jnp.sum(f, axis = -1)
    u = c_macro_vel(f, vels, rho)
    E = jnp.sum(h, axis = -1) / rho

    #Nueva distribución de equilibrio
    feq, heq, F, q, Z = equilibrio(rho, u, E)

    #Cálculo de la colisión
    f_nuev = (1 - _1_tau_f) * f + _1_tau_f * feq + F
    h_nuev = (1 - _1_tau_h) * h + _1_tau_h * heq + _1_tau_fh * Z * (f - feq) + q


    #Propagación
    f_nuev = propaga(f_nuev)
    h_nuev = propaga(h_nuev)
        
    f_nuev = f_nuev.at[0, :, :].set(feq[0, :, :] + f[1, :, :] - feq[1, :, :])
    f_nuev = f_nuev.at[-1, :, :].set(feq[-1, :, :] + f[-2, :, :] - feq[-2, :, :])
    h_nuev = h_nuev.at[0, :, :].set(heq[0, :, :] + h[1, :, :] - heq[1, :, :])
    h_nuev = h_nuev.at[-1, :, :].set(heq[-1, :, :] + h[-2, :, :] - heq[-2, :, :])

    u_mag = jnp.sum(u * u, axis = -1);
    T = 1 / cv * (E - 0.5 * u_mag);

    mach = jnp.sqrt(u_mag / (T * R));
    max_mach = jnp.max(mach);
    
    
    return (f_nuev, h_nuev, u, rho, temps(T), max_mach)

for arch in archivos:
    try:
        os.remove(arch)
    except FileNotFoundError:
        print("No existe el archivo.")

with open(archivos[0], 'ab') as f_vels, open(archivos[1], 'ab') as f_temps:
    for paso in range(pasos):
        f, h, u, rho, temp, max_mach = optim(f, h)

        if paso % screen_shot == 0:
            #print(h[32, :, 5])
            print(paso, " max_mach: ", max_mach);

            if paso >= iteraciones_saltar:
                if paso == iteraciones_saltar:
                    print("Listo!")
                
                pickle.dump(temp, f_temps)
                pickle.dump(u, f_vels)
