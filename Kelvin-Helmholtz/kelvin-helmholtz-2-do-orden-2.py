import jax
import jax.numpy as jnp
import pickle
import os
from jax import lax
from jax import debug;

#Constantes
N_x = 300;
N_y = 300;
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
tam = (c_max - c_min) / N_x;
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

#Flujos a través de las celdas
def flujos2(prim, u, c1):
    #flujos anteriores
    fprim = jnp.zeros_like(prim);
    #fprimz = jnp.zeros_like(prim);

    #Flujos de masa
    fprim = fprim.at[rho].set(prim[rho] * prim[c1]);
    #fprimz = fprimz.at[rho].set(prim[rho] * prim[cz]);
    
    #Flujos de momento
    fprim = fprim.at[coords].set(prim[rho] * prim[c1][jnp.newaxis, ...] * prim[coords]);
    fprim = fprim.at[c1].set(fprim[c1] + prim[din]);
                
    #Flujos de energía
    #eth = p(eq_pr)/(gam-1d0)
    eth = const1 * prim[din];
    fprim = fprim.at[din].set((prim[din] + eth + 0.5 * prim[rho] * u) * prim[c1]);
    #fprimz = fprimz.at[din].set((prim[din] + cons[din]) * prim[cz]);

    #return (fprimx, fprimy, fprimz);
    return fprim;
    
def sound_speed(prim):
    return (gamma * prim[din] / prim[rho]) ** 0.5;

def minmod2(a, b):
    c = a * b;
    sng = jnp.sign(c);
    s_a = jnp.sign(a);
    a = jnp.abs(a);
    b = jnp.abs(b);

    return s_a * jnp.where(sng > 0, jnp.minimum(a, b), 0.0);
    
def limiter2(pm2, pm1, pm, pM1):
    dl = pm1 - pm2;
    d0 = pm - pm1;
    dr = pM1 - pm;

    pm_l = pm1 + 0.5 * minmod2(d0, dl);
    pm_r = pm - 0.5 * minmod2(dr, d0);

    return (pm_l, pm_r);

def flujos2orden(prim):
    pll = jnp.roll(prim, shift=2, axis=2);
    pl = jnp.roll(prim, shift=1, axis=2);
    pr = prim;
    prr = jnp.roll(prim, shift=-1, axis=2);

    pbb = jnp.roll(prim, shift=2, axis=1);
    pb = jnp.roll(prim, shift=1, axis=1);
    pt = prim;
    ptt = jnp.roll(prim, shift=-1, axis=1);

    p_l, p_r = limiter2(pll, pl, pr, prr);
    p_b, p_t = limiter2(pbb, pb, pt, ptt);
    
    return fluxes_HLLC_method(pl, pr, pb, pt);

def fluxes_HLLC_method(pp_l, pp_r, pp_b, pp_t):
    p_r = pp_r[din]; p_l = pp_l[din];
    p_t = pp_t[din]; p_b = pp_b[din];

    v_r = pp_r[coords2D]; v_l = pp_l[coords2D];
    v_t = pp_t[coords2D]; v_b = pp_b[coords2D];
    #debug.print("v shape={}", v_r.shape);

    u_l = jnp.sum(v_l * v_l, axis = 0);
    u_r = jnp.sum(v_r * v_r, axis = 0);
    u_b = jnp.sum(v_b * v_b, axis = 0);
    u_t = jnp.sum(v_t * v_t, axis = 0);
    #debug.print("v shape={}", u_l.shape);

    rho_r = pp_r[rho]; rho_l = pp_l[rho];
    rho_t = pp_t[rho]; rho_b = pp_b[rho];

    p_r = jnp.maximum(p_r, 0); p_l = jnp.maximum(p_l, 0);
    p_t = jnp.maximum(p_t, 0); p_b = jnp.maximum(p_b, 0);

    cf_r = jnp.sqrt(gamma * p_r / rho_r); cf_l = jnp.sqrt(gamma * p_l / rho_l);
    cf_t = jnp.sqrt(gamma * p_t / rho_t); cf_b = jnp.sqrt(gamma * p_b / rho_b);

    #Auxiliar para cx y cy
    _cx = 0; _cy = 1;

    vs_min_h = jnp.minimum(v_l[_cx] - cf_l, v_r[_cx] - cf_r);
    vs_max_h = jnp.maximum(v_l[_cx] + cf_l, v_r[_cx] + cf_r);
    vs_min_v = jnp.minimum(v_b[_cy] - cf_b, v_t[_cy] - cf_t);
    vs_max_v = jnp.maximum(v_b[_cy] + cf_b, v_t[_cy] + cf_t);

    fprimx_r = flujos2(pp_r, u_r, cx);
    fprimy_t = flujos2(pp_t, u_t, cy);
    fprimx_l = flujos2(pp_l, u_l, cx);
    fprimy_b = flujos2(pp_b, u_b, cy);

    dx_min_h = vs_min_h - v_l[_cx];
    dx_max_h = vs_max_h - v_r[_cx];
    
    dx_min_v = vs_min_v - v_b[_cy];
    dx_max_v = vs_max_v - v_t[_cy];

    dxde_min_h = dx_min_h * rho_l;
    dxde_max_h = dx_max_h * rho_r;

    dxde_min_v = dx_min_v * rho_b;
    dxde_max_v = dx_max_v * rho_t;

    sm_h = (dxde_max_h * v_r[_cx] + p_l - dxde_min_h * v_l[_cx] - p_r) / (dxde_max_h - dxde_min_h);
    sm_v = (dxde_max_v * v_t[_cy] + p_b - dxde_min_v * v_b[_cy] - p_t) / (dxde_max_v - dxde_min_v);

    pta_h = (dxde_max_h * p_l - dxde_min_h * p_r + dxde_min_h * dxde_max_h * (v_r[_cx] - v_l[_cx])) / (dxde_max_h - dxde_min_h);
    pta_v = (dxde_max_v * p_b - dxde_min_v * p_t + dxde_min_v * dxde_max_v * (v_t[_cy] - v_b[_cy])) / (dxde_max_v - dxde_min_v);

    cond_h = sm_h > 0;
    da_h = jnp.where(cond_h, dxde_min_h / (vs_min_h - sm_h), dxde_max_h / (vs_max_h - sm_h));
   
    en_h = jnp.where(cond_h, const1 * p_l + 0.5 * rho_l * u_l, const1 * p_r + 0.5 * rho_r * u_r);

    cond_v = sm_v > 0;
    da_v = jnp.where(cond_v, dxde_min_v / (vs_min_v - sm_v), dxde_max_v / (vs_max_v - sm_v));
    
    en_v = jnp.where(cond_v, const1 * p_b + 0.5 * rho_b * u_b, const1 * p_t + 0.5 * rho_t * u_t);

    #flujos en x
    fmx = jnp.zeros_like(prim);

    aux_fmx_01 = fprimx_l[din] + vs_min_h * (sm_h * (en_h + pta_h) - v_l[_cx] * (en_h + p_l)) / (vs_min_h - sm_h);
    aux_fmx_02 = fprimx_r[din] + vs_max_h * (sm_h * (en_h + pta_h) - v_r[_cx] * (en_h + p_r)) / (vs_max_h - sm_h);
    fmx = fmx.at[din].set(jnp.where(cond_h, aux_fmx_01, aux_fmx_02));

    aux_fmx_11 = fprimx_l[cx] + vs_min_h * (sm_h * da_h - v_l[_cx] * rho_l);
    aux_fmx_12 = fprimx_r[cx] + vs_max_h * (sm_h * da_h - v_r[_cx] * rho_r);
    fmx = fmx.at[cx].set(jnp.where(cond_h, aux_fmx_11, aux_fmx_12));

    aux_fmx_31 = vs_min_h * (da_h - rho_l);
    aux_fmx_32 = vs_max_h * (da_h - rho_r);   
    fmx = fmx.at[rho].set(jnp.where(cond_h, fprimx_l[rho] + aux_fmx_31, fprimx_r[rho] + aux_fmx_32));

    aux_fmx_21 = fprimx_l[cy] + aux_fmx_31 * v_l[_cy];
    aux_fmx_22 = fprimx_r[cy] + aux_fmx_32 * v_r[_cy];   
    fmx = fmx.at[cy].set(jnp.where(cond_h, aux_fmx_21, aux_fmx_22));

    #flujos en y
    fmy = jnp.zeros_like(prim);
    
    aux_fmy_01 = fprimy_b[din] + vs_min_v * (sm_v * (en_v + pta_v) - v_b[_cy] * (en_v + p_b)) / (vs_min_v - sm_v);
    aux_fmy_02 = fprimy_t[din] + vs_max_v * (sm_v * (en_v + pta_v) - v_t[_cy] * (en_v + p_t)) / (vs_max_v - sm_v);
    fmy = fmy.at[din].set(jnp.where(cond_v, aux_fmy_01, aux_fmy_02));

    aux_fmy_11 = fprimy_b[cy] + vs_min_v * (sm_v * da_v - v_b[_cy] * rho_b);
    aux_fmy_12 = fprimy_t[cy] + vs_max_v * (sm_v * da_v - v_t[_cy] * rho_t);
    fmy = fmy.at[cy].set(jnp.where(cond_v, aux_fmy_11, aux_fmy_12));

    aux_fmy_31 = vs_min_v * (da_v - rho_b);
    aux_fmy_32 = vs_max_v * (da_v - rho_t);
    fmy = fmy.at[rho].set(jnp.where(cond_v, fprimy_b[rho] + aux_fmy_31, fprimy_t[rho] + aux_fmy_32));

    aux_fmy_21 = fprimy_b[cx] + aux_fmy_31 * v_b[_cx];
    aux_fmy_22 = fprimy_t[cx] + aux_fmy_32 * v_t[_cx];
    fmy = fmy.at[cx].set(jnp.where(cond_v, aux_fmy_21, aux_fmy_22));

    f_hor = jnp.where(vs_min_h > 0.0, fprimx_l, jnp.where(vs_max_h < 0.0, fprimx_r, fmx));
    f_ver = jnp.where(vs_min_v > 0.0, fprimy_b, jnp.where(vs_max_v < 0.0, fprimy_t, fmy));
             
    return (f_hor, f_ver);

def paso(carry):
    tiempo, prim, tiempo_max = carry;

    cons = calc_conserv(prim);

    prim_crop = prim;#[:, 1:N-1, 1:N-1];
    #prim_crop = prim[:, 1:N_y-1, :];
    
    sp = sound_speed(prim_crop);
    sp_max = prim_crop[coords2D] + sp[jnp.newaxis, ...];
    #dt = jnp.min(tam / sp_max);
    #dt = dt * 0.9;
    #dt = jnp.array(1.83e-8);
    dt = jnp.array(1.0e-4);

    tiempo += dt;
    
    #Condiciones de borde
    #prim = prim.at[:, 0, :].set(prim[:, 1, :]);
    #prim = prim.at[:, -1, :].set(prim[:, -2, :]);
    #prim = prim.at[:, 0, :].set(prim[:, 1, :]);
    #prim = prim.at[:, -1, :].set(prim[:, -2, :]);
    
    #fh, fv = fluxes_HLL_method(prim, cons);
    _fh, _fv = flujos2orden(prim);
    #fh = _fh[:, 1:(N-1), :]
    #fv = _fv[:, :, 1:(N-1)]
    
    #fh = _fh[:, 1:(N-1), 1:N];
    #fv = _fv[:, 1:N, 1:(N-1)];

    #fh = _fh[:, 1:(N_y-1), :];
    #fv = _fv[:, 1:N_y, :];

    fh = _fh;#[:, 1:(N_y-1), :];
    fv = _fv;#[:, 1:N_y, :];
    
    #cons = cons.at[:, 1:(N-1), 1:(N-1)].set(cons[:, 1:(N-1), 1:(N-1)] - (0.05) * (fh[:, :, 1:(N - 1)] - fh[:, :, 0:(N - 2)] + fv[:, 1:(N - 1), :] - fv[:, 0:(N - 2), :]));
    cons = cons - (dt / tam) * (jnp.roll(fv, shift=-1, axis=1) - fv + jnp.roll(fh, shift=-1, axis=2) - fh);
    #debug.print("cons={}", cons[rho, 100, 95:115]);
    #cons = cons - (0.05) * (fh - jnp.roll(fh, shift=1, axis=2) + fv - jnp.roll(fv, shift=1, axis=1));
    prim = calc_prim(cons);
    
    return (tiempo, prim, tiempo_max);
    
def cond(carry):
    tiempo, _, tiempo_max = carry;

    return tiempo < tiempo_max;

def prueba():
    prim = jnp.zeros((5, N_y, N_x));
    #cilindro_x = 50;
    #cilindro_y = 50;
    #radio_cilindro = 7;
    
    #Definiendo el obstáculo circular
    x = jnp.arange(N_x);
    y = jnp.arange(N_y);
    X, Y = jnp.meshgrid(x, y);

    #alta = jnp.logical_or(jnp.sqrt((X - cilindro_x) ** 2 + (Y - cilindro_y) ** 2) < radio_cilindro, jnp.sqrt((X - 150) ** 2 + (Y - 150) ** 2) < radio_cilindro)
    alta = jnp.logical_and(N_y / 3 < Y, Y < 2 * N_y / 3);
    baja = jnp.logical_not(alta);

    prim = prim.at[rho, alta].set(2.0);
    prim = prim.at[din, alta].set(1.0);
    prim = prim.at[cx, alta].set(0.5);
    
    prim = prim.at[rho, baja].set(1.0);
    prim = prim.at[din, baja].set(1.0);
    prim = prim.at[cx, baja].set(-0.5);

    prim = prim.at[cy].set(0.02 * jnp.sin(4 * jnp.pi * X / N_x));

    return prim;

@jax.jit
def blucle_interno(tiempo, prim, tiempo_max):
    return lax.while_loop(cond, paso, (tiempo, prim, tiempo_max));

tiempo_final = jnp.array(3.0);
#tiempo_final = jnp.array(1.83e-6);
tiempo = jnp.array(0.0);
rangos = jnp.arange(0.0, tiempo_final, 0.04)
prim = prueba();

cs = sound_speed(prim)[0, N_x-1].item();

cron = False;
target_time = 0.1;

for arch in archivos:
    try:
        os.remove(arch)
    except FileNotFoundError:
        print("No existe el archivo.")

with open(archivos[0], 'ab') as f_data:
    mach = 0.0;
    print("Cs = ", cs);
    
    for t in rangos:
        tiempo, prim, _ = blucle_interno(tiempo, prim, t);

        
        vls = jnp.sum(prim[coords] * prim[coords], axis = 0);
        new_mach = jnp.max(vls).item()/cs;

        if new_mach > mach:
            mach = new_mach;

        if not cron or tiempo >= target_time:
            pickle.dump(prim, f_data);

            if (cron):
                print("Tiempo = ", tiempo);
                break;
    
    print("Max mach = ", mach);
    
    
