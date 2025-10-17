import jax;
jax.config.update("jax_enable_x64", True);
#jax.config.update("jax_log_compiles", True);

import jax.numpy as jnp;
from jax import lax;
from jax import debug;
import pickle;
import os;

archivos = ['choques-v1-u.dat', 'choques-v1-rho.dat', 'choques-v1-T.dat'];

cs = 0;

N = 100;
gamma = 1.4;
b = 2 / (gamma - 1);
R = 287;
Pr = 0.71;
mu0 = 1.86e-5;

cp = (b + 2) * R / 2;

_2_bR = 2 / (b * R);

T0 = 303;
rho0 = 1.165;
p0 = rho0 * R * T0;
D = 2;
Q = 12;
dt = 0.01;
Mach = 10;
#Tc = 100 * T0 * (1 + (gamma - 1) * Mach ** 2 / 2);
Tc = 1200 * T0;
_1_Tc = 1 / Tc;
L0 = 2.0;
_1_dx = N / L0;


r = 6 ** 0.5;
s = ((9 - 3 * 5 ** 0.5) / 4) ** 0.5;
t = ((9 + 3 * 5 ** 0.5) / 4) ** 0.5;

c_char = (R * Tc) ** 0.5;
_1_c_char = 1 / c_char;

vels = c_char * jnp.array([
        [r,  0, -r,  0,  s, -s, -s,  s,  t, -t, -t,  t],
        [0,  r,  0, -r,  s,  s, -s, -s,  t,  t, -t, -t]
    ]);
 
val1 = 1/36;
val2 = (5 + 2 * 5 ** 0.5) / 45;
val3 = (5 - 2 * 5 ** 0.5) / 45;
 
w = jnp.array([
        val1, val1, val1, val1,
        val2, val2, val2, val2,
        val3, val3, val3, val3
    ]);

vels_norm = _1_c_char * vels;
vels_2_D = (jnp.sum(vels_norm * vels_norm, axis = 0) - D)[..., jnp.newaxis, jnp.newaxis];
w_com = w[..., jnp.newaxis, jnp.newaxis];
def equilibrio(rho, u, E, T, p):
    f = jnp.zeros((Q, N, N));
    u_vels = jnp.einsum('dNM,dP->PNM', u, vels_norm);
    u_vels_2 = u_vels * u_vels;
    
    u_2 = jnp.sum(u * u, axis = 0);
    
    term1 = 1 + u_vels + 0.5 * (u_vels_2 - u_2 + (T - 1) * vels_2_D);

    feq = w_com * rho * (term1 +
        (1 / 6) * u_vels * (u_vels_2 - 3 * u_2 + 3 * (T - 1) * (vels_2_D - 2)));
        
    heq = E * w_com * rho * term1 + w_com * p * (u_vels + u_vels_2 - u_2 + 0.5 * vels_2_D);
       
    return (feq, heq);
    
def minmod(c1, c2):
    signo = jnp.where(c1 * c2 <= 0, 0, jnp.where(c1 > 0, 1, -1));
    abs_c1 = jnp.abs(c1);
    abs_c2 = jnp.abs(c2);
    
    return 0.5 * signo * jnp.minimum(abs_c1, abs_c2);
    
def calculaF(f, sig, axis, flip):
    Fpm = 0.5 * ((vels + sig * jnp.abs(vels))[2 - axis])[..., jnp.newaxis, jnp.newaxis] * f;
    
    deltaFp_I05_J = jnp.roll(Fpm, shift = -1, axis = axis) - Fpm;
    
    return jnp.roll(Fpm, shift = flip, axis = axis) + sig * 0.5 * minmod(deltaFp_I05_J, jnp.roll(deltaFp_I05_J, shift = sig, axis = axis));
    
def calc_deriv(f, axis):
    F_L_I05J = calculaF(f, 1, axis, 0);
    F_R_I05J = calculaF(f, -1, axis, -1);
    F_I05J = F_L_I05J + F_R_I05J;
    F_Im05J = jnp.roll(F_I05J, shift = 1, axis = axis);

    return _1_dx * (F_I05J - F_Im05J);
    
def paso(rho, u, E, T, p, w_f, w_h, f, h):
    mu = dt * R * rho * T / w_f;
    lamb = dt * cp * p / w_h;

    feq, heq = equilibrio(rho, _1_c_char * u, E, _1_Tc * T, p);
    u_vels = jnp.einsum('dNM,dP->PNM', u, vels);
    _05_u_2 = 0.5 * jnp.sum(u * u, axis = 0);
    
    f_1 = (f + w_f * feq) / (1 + w_f);   
    fact_aux = (u_vels - _05_u_2) * (f_1 - feq);
    
    h_1 = (h + w_f * fact_aux + w_h * heq) / (1 + w_h);
    w_hf = w_h - w_f;

    point_grad_f = calc_deriv(f_1, 2) + calc_deriv(f_1, 1);
    point_grad_h = calc_deriv(h_1, 2) + calc_deriv(h_1, 1);
          
    f_new = f - dt * point_grad_f + w_f * (feq - f_1);
    h_new = h - dt * point_grad_h + w_h * (heq - h_1) + w_hf * fact_aux;

    rho = jnp.sum(f_new, axis = 0);
    
    u = jnp.einsum('dQ,QNM->dNM', vels, f_new) / rho;
    E = jnp.sum(h_new, axis = 0) / rho;
    T = _2_bR * (E - _05_u_2);
    p = R * rho * T;
    
    w_f = dt * p / mu;
    w_h = dt * cp * p / lamb;

    #debug.print("tau_f={}, tau_h={}", dt / w_f[50,50], dt / w_h[50,50]);
    
    return (rho, u, E, T, p, w_f, w_h, f_new, h_new);

@jax.jit
def aux_intern(_, carry):
    rho, u, E, T, p, w_f, w_h, f, h = carry;
    
    tupla = paso(rho, u, E, T, p, w_f, w_h, f, h);
    
    return tupla;  
    
def prueba():
    u = jnp.zeros((2, N, N));
    rho = jnp.zeros((N, N));
    p = jnp.zeros((N, N));
    
    #Definiendo el dominio:
    cilindro_x = 50;
    cilindro_y = 50;
    radio_cilindro = 7;
    
    #Definiendo el obstáculo circular
    x = jnp.arange(N);
    y = jnp.arange(N);
    X, Y = jnp.meshgrid(x, y);

    alta = jnp.sqrt((X - cilindro_x) ** 2 + (Y - cilindro_y) ** 2) < radio_cilindro
    baja = jnp.logical_not(alta);

    rho = rho.at[alta].set(1.3 * rho0);
    rho = rho.at[baja].set(1.3 * rho0);
    #p = p.at[alta].set(p0 * 0.025);
    p = p.at[alta].set(p0 * 10000);
    p = p.at[baja].set(p0 * 0.0005);
    T = (1 / R) * p / rho;
    _05_u_2 = 0.5 * jnp.sum(u * u, axis = 0);
    E = 0.5 * b * R * T + _05_u_2;

    lamb0 = mu0 * cp / Pr;

    tau_f0 = mu0 / p0;
    tau_h0 = lamb0 / (p0 * cp);

    global cs;
    cs = jnp.sqrt(gamma * R * T[0,0]).item();
    debug.print("cs={}", cs);

    global dt;
    dt = 100 * tau_f0;
    debug.print("dt={}", dt);

    w_f = (dt / tau_f0) * jnp.ones((N, N));
    w_h = (dt / tau_h0) * jnp.ones((N, N));

    feq, heq = equilibrio(rho, _1_c_char * u, E, _1_Tc * T, p);

    return (rho, u, E, T, p, w_f, w_h, feq, heq);
    
pasos = 10000;
pasos_int = 200;
rho, u, E, T, p, w_f, w_h, f, h = prueba();

for arch in archivos:
    try:
        os.remove(arch)
    except FileNotFoundError:
        print("No existe el archivo.");
        
with open(archivos[0], 'ab') as f_data1, open(archivos[1], 'ab') as f_data2, open(archivos[2], 'ab') as f_data3:
    mach = 0;
    
    for t in range(int(pasos / pasos_int)):
        rho, u, E, T, p, w_f, w_h, f, h = lax.fori_loop(0, pasos_int, aux_intern, (rho, u, E, T, p, w_f, w_h, f, h));

        ve = jnp.sqrt(jnp.max(jnp.sum(u * u, axis = 0)));
        #debug.print("max-vel={}", ve);

        new_mach = ve.item() / cs;
        if (new_mach > mach):
            mach = new_mach;
        
        pickle.dump(u, f_data1);
        pickle.dump(rho, f_data2);
        pickle.dump(T, f_data3);
    
    print("Mach=", mach);
