import jax
import jax.numpy as jnp
import pickle
import os
from jax import lax

#Constantes
N = 100;
gamma = 1.4;
const1 = 1 / (gamma - 1);
const2 = (gamma - 1);
rho = 0;
cx = 1;
cy = 2;
cz = 3;
coords = jnp.array([cx, cy, cz]);
coords2D = jnp.array([cx, cy])
rho_flux = jnp.array([0, 1, 2]);
momentum_flux = jnp.arange(3, 12);
energy_flux = jnp.array([12, 13, 14]);
din = 4;
c_min = 0.0;
c_max = 1.0;
tam = (c_max - c_min) / N;
archivos = ['data-v1.dat']

#Calculando las variables conservativas
def calc_conserv(prim):
    cons = prim.at[rho].set(prim[rho]);

    cons = prim.at[coords].set(prim[rho] * prim[coords]);
 
    kinetic = 0.5 * prim[rho] * jnp.sum(prim[coords] * prim[coords], axis = 0);
    intern = const1 * prim[din];
        
    return cons.at[din].set(kinetic + intern);

#Calculando las variables conservativas
def calc_prim(cons):
    #Densidad a prim
    prim = cons.at[rho].set(cons[rho]);

    #Valocidades a densidad
    prim = cons.at[coords].set(prim[coords] / cons[rho]);
 
    #Cálculo de presión a partir de la energía
    intern = cons[din] - 0.5 * prim[rho] * jnp.sum(prim[coords] * prim[coords], axis = 0);
    pressure = const2 * intern;
        
    return prim.at[din].set(pressure);
 
#Flujos a través de las celdas
def flujos(prim, cons):
    #flujos anteriores
    fprimx = jnp.zeros_like(prim);
    fprimy = jnp.zeros_like(prim);
    #fprimz = jnp.zeros_like(prim);

    #Flujos de masa
    fprimx = fprimx.at[rho].set(prim[rho] * prim[cx]);
    fprimy = fprimy.at[rho].set(prim[rho] * prim[cy]);
    #fprimz = fprimz.at[rho].set(prim[rho] * prim[cz]);
    
    #Flujos de momento
    fprimx = fprimx.at[coords].set(cons[cx][jnp.newaxis, ...] * prim[coords]);
    fprimx = fprimx.at[cx].set(fprimx[cx] + prim[din]);
    
    fprimy = fprimy.at[coords].set(cons[cy][jnp.newaxis, ...] * prim[coords]);
    fprimy = fprimy.at[cy].set(fprimy[cy] + prim[din]);

    #fprimz = fprimz.at[coords].set(cons[cz][jnp.newaxis, ...] * prim[coords]);
    #fprimz = fprimz.at[cz].set(fprimy[cz] + prim[din]);
                
    #Flujos de energía
    fprimx = fprimx.at[din].set((prim[din] + cons[din]) * prim[cx]);
    fprimy = fprimy.at[din].set((prim[din] + cons[din]) * prim[cy]);
    #fprimz = fprimz.at[din].set((prim[din] + cons[din]) * prim[cz]);

    #return (fprimx, fprimy, fprimz);
    return (fprimx, fprimy);
    
def sound_speed(prim):
    return (gamma * prim[din] / prim[rho]) ** 0.5;
    
def min_max(prim):
    sp = sound_speed(prim);
    sp_min_hor = prim[cx] - sp
    sp_max_hor = prim[cx] + sp
    sp_min_ver = prim[cy] - sp
    sp_max_ver = prim[cy] + sp
    
    sp_min_left = sp_min_hor[:, 0:(N-1)];
    sp_max_left = sp_max_hor[:, 0:(N-1)];

    sp_min_right = sp_min_hor[:, 1:N];
    sp_max_right = sp_max_hor[:, 1:N];

    sp_min_bottom = sp_min_ver[0:(N-1), :];
    sp_max_bottom = sp_max_ver[0:(N-1), :];

    sp_min_top = sp_min_ver[1:N, :];
    sp_max_top = sp_max_ver[1:N, :];
    
    sR = jnp.maximum(jnp.maximum(sp_max_left, sp_max_right), 0.0);
    sL = jnp.minimum(jnp.minimum(sp_min_left, sp_min_right), 0.0);
    sT = jnp.maximum(jnp.maximum(sp_max_top, sp_max_bottom), 0.0);
    sB = jnp.minimum(jnp.minimum(sp_min_top, sp_min_bottom), 0.0);
  
    return (sR, sL, sB, sT);
    
def fluxes_HLL_method(prim, cons):
    fprimx, fprimy = flujos(prim, cons);
    fprim_left = fprimx[:, :, 0:(N-1)];
    fprim_right = fprimx[:, :, 1:N];
    fprim_bottom = fprimy[:,0:(N-1), :];
    fprim_top = fprimy[:, 1:N, :];
    
    _sR, _sL, _sB, _sT = min_max(prim);
    sR = _sR[jnp.newaxis, ...];
    sL = _sL[jnp.newaxis, ...];
    sT = _sT[jnp.newaxis, ...];
    sB = _sB[jnp.newaxis, ...];
    
    left_cons = cons[:, :, 0:(N-1)];
    right_cons = cons[:, :, 1:N];
    bottom_cons = cons[:, 0:(N-1), :];
    top_cons = cons[:, 1:N, :];

    f_hor = (sR * fprim_left - sL * fprim_right
             + (sR * sL) * (right_cons - left_cons)
             ) / (sR - sL);
             
    f_ver = (sT * fprim_bottom - sB * fprim_top
             + (sT * sB) * (top_cons - bottom_cons)
              ) / (sT - sB);
             
    return (f_hor, f_ver)

def paso(carry):
    tiempo, prim, tiempo_max = carry;

    cons = calc_conserv(prim);

    prim_crop = prim[:, 1:N-1, 1:N-1];
    sp = sound_speed(prim_crop);
    sp_max = prim_crop[coords2D] + sp[jnp.newaxis, ...];
    #dt = jnp.min(tam / sp_max);
    #dt = dt * 0.9;
    dt = jnp.array(1.83e-8);

    tiempo += dt;
    
    #Condiciones de borde
    prim = prim.at[:, :, 0].set(prim[:, :, 1]);
    prim = prim.at[:, :, -1].set(prim[:, :, -2]);
    prim = prim.at[:, 0, :].set(prim[:, 1, :]);
    prim = prim.at[:, -1, :].set(prim[:, -2, :]);
    
    _fh, _fv = fluxes_HLL_method(prim, cons);
    fh = _fh[:, 1:(N-1), :]
    fv = _fv[:, :, 1:(N-1)]
    
    cons = cons.at[:, 1:(N-1), 1:(N-1)].set(cons[:, 1:(N-1), 1:(N-1)] - (0.05) * (fh[:, :, 1:(N - 1)] - fh[:, :, 0:(N - 2)] + fv[:, 1:(N - 1), :] - fv[:, 0:(N - 2), :]));
    prim = calc_prim(cons)
    
    return (tiempo, prim, tiempo_max);
    
def cond(carry):
    tiempo, _, tiempo_max = carry;

    return tiempo < tiempo_max;

def prueba():
    prim = jnp.zeros((5, N, N));
    cilindro_x = 50;
    cilindro_y = 50;
    radio_cilindro = 3;
    
    #Definiendo el obstáculo circular
    x = jnp.arange(N);
    y = jnp.arange(N);
    X, Y = jnp.meshgrid(x, y);

    #alta = jnp.logical_or(jnp.sqrt((X - cilindro_x) ** 2 + (Y - cilindro_y) ** 2) < radio_cilindro, jnp.sqrt((X - 150) ** 2 + (Y - 150) ** 2) < radio_cilindro)
    alta = jnp.sqrt((X - cilindro_x) ** 2 + (Y - cilindro_y) ** 2) < radio_cilindro;
    baja = jnp.logical_not(alta);

    prim = prim.at[rho, alta].set(1.0);
    prim = prim.at[din, alta].set(10.0);
    
    prim = prim.at[rho, baja].set(1.0);
    prim = prim.at[din, baja].set(0.00001);

    return prim;

@jax.jit
def blucle_interno(tiempo, prim, tiempo_max):
    return lax.while_loop(cond, paso, (tiempo, prim, tiempo_max));

tiempo_final = jnp.array(1.83e-4/5);
tiempo = jnp.array(0.0);
rangos = jnp.arange(0.0, tiempo_final, 1.83e-6/4)
prim = prueba();

cs = sound_speed(prim)[0,0].item();

for arch in archivos:
    try:
        os.remove(arch)
    except FileNotFoundError:
        print("No existe el archivo.")

with open(archivos[0], 'ab') as f_data:
    mach = 0.0;
    
    for t in rangos:
        tiempo, prim, _ = blucle_interno(tiempo, prim, t);

        
        vls = jnp.sum(prim[coords] * prim[coords], axis = 0);
        new_mach = jnp.max(vls).item()/cs;

        if new_mach > mach:
            mach = new_mach;
        
        pickle.dump(prim, f_data);
    
    print("Max mach = ", mach);
    
    
