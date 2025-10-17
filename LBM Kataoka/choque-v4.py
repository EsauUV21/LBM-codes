import jax
jax.config.update("jax_enable_x64", True);

import jax.numpy as jnp
from jax import lax
from jax import debug
import pickle
import os

N = 100;
gamma = 1.4;
b = 2 / (gamma - 1);
R = 1;
T0 = 1.0;
A = 1 / (R * T0) ** 0.5;
c1 = 2 ** 0.5;
c2 = 3 / c1;
Q = 16;
#mu0 = 0.075;
mu0 = 0.55;
dt = 0.01
archivos = ['data-v4-u.dat', 'data-v4-rho.dat'];

vels = A * jnp.array([
    [1,  0, -1,  0,  6,  0, -6,  0, c1, -c1, -c1,  c1, c2, -c2, -c2,  c2],
    [0,  1,  0, -1,  0,  6,  0, -6, c1,  c1, -c1, -c1, c2,  c2, -c2, -c2]
]);

vels_norm = A * vels;

tabla = jnp.array(
    [[       0,             (b - 2) / 25,              0,     -36 / 115,           (b + 4) / 115,     1 / 115,       0,        2 * (b - 2) / 25,         0,     72 / 115, -2 * (b + 4) / 115,  -2 / 115,          0],
     [  1 / 96, (-121 * b - 408) / 86400, (b + 2) / 1728, -799 / 397440, (19 * b + 306) / 397440, 19 / 397440,       0,   (-2 * b + 29) / 32440, -1 / 2592, -29 / 298080,    (b + 4) / 74520, 1 / 74520,  1 / 46656],
     [81 / 160,    (-229 * b + 8) / 3200,  (b + 2) / 320,      -117/640,      (9 * b + 38) / 640,     9 / 640,  9 / 40,     (-14 * b + 3) / 400,    1 / 80,      9 / 160,     -(b + 4) / 160,  -1 / 160,   -3 / 320],
     [ -4 / 15,    (89 * b + 222) / 2700, -(b + 2) / 270,      13 / 135,      -(2 * b + 9) / 270,    -1 / 135, -2 / 45, 2 * (7 * b + 11) / 2025,  -7 / 810,     -2 / 405,      (b + 4) / 810,   1 / 810,   8 / 3645]]
);
   
def aux_multi(i, carry):
    f, T, T_2, u_c, u_c_2, u_c_T, u_c_u_2, u_c_2_T, u_2_u_c_2, u_c_3, u_2, u_2_2, u_2_T, prev_index, calc = carry;
    index = jnp.trunc(i / 4).astype(jnp.int32);
    a = tabla[index, :];

    x = lax.cond(index == prev_index,
                 lambda _: calc,
                 lambda _: a[0] + a[1] * T + a[2] * T_2 + a[3] * u_2 + a[4] * u_2_T + a[5] * u_2_2,
                 0);
    
    f = f.at[i].set(x + a[6] * u_c[i] + a[7] * u_c_T[i]
    + a[8] * u_c_u_2[i] + a[9] * u_c_2[i]
    + a[10] * u_c_2_T[i] + a[11] * u_2_u_c_2[i]
    + a[12] * u_c_3[i]);
                
    return (f, T, T_2, u_c, u_c_2, u_c_T, u_c_u_2, u_c_2_T, u_2_u_c_2, u_c_3, u_2, u_2_2, u_2_T, index, x)

o = jnp.zeros((N, N));
def equilibrio(rho, u, T):
    f = jnp.zeros((Q, N, N));
    T_2 = T * T;
    u_c = jnp.einsum('DNM,Dq->qNM', u, vels_norm);
    u_c_2 = u_c * u_c;
    u_c_T = u_c * T;
    u_c_2_T = T * u_c_2;
    u_c_3 = u_c ** 3;
    u_2 = jnp.sum(u * u, axis = 0);
    u_c_u_2 = u_c * u_2;
    u_2_u_c_2 = u_2 * u_c_2;
    u_2_2 = u_2 ** 2;
    u_2_T = T * u_2;
    
    #Creando el array con los coeficientes para multiplicar por la tabla
    tupla = lax.fori_loop(0, 
                  Q,
                  aux_multi, 
                  (f, T, T_2, u_c, u_c_2, u_c_T, u_c_u_2, u_c_2_T, u_2_u_c_2, u_c_3, u_2, u_2_2, u_2_T, -1, o)
                  );

    return rho * tupla[0];
                  
def shifted(f):
    #c > 0
    f_y_p1 = jnp.roll(f, shift = 1, axis = 1);
    f_y_p2 = jnp.roll(f, shift = 2, axis = 1);
    #c < 0
    f_y_m1 = jnp.roll(f, shift = -1, axis = 1);
    f_y_m2 = jnp.roll(f, shift = -2, axis = 1);
    #c > 0
    f_x_p1 = jnp.roll(f, shift = 1, axis = 2);
    f_x_p2 = jnp.roll(f, shift = 2, axis = 2);
    #c < 0
    f_x_m1 = jnp.roll(f, shift = -1, axis = 2);
    f_x_m2 = jnp.roll(f, shift = -2, axis = 2);

    #cx > 0 cy > 0
    #f_xy_pp1 = jnp.roll(jnp.roll(f, shift = 1, axis = 1), shift = 1, axis = 2);
    #f_xy_pp2 = jnp.roll(jnp.roll(f, shift = 2, axis = 1), shift = 2, axis = 2);
    #cx < 0 cy > 0
    #f_xy_mp1 = jnp.roll(jnp.roll(f, shift = 1, axis = 1), shift = -1, axis = 2);
    #f_xy_mp2 = jnp.roll(jnp.roll(f, shift = 2, axis = 1), shift = -2, axis = 2);
    #cx < 0 cy < 0
    #f_xy_mm1 = jnp.roll(jnp.roll(f, shift = -1, axis = 1), shift = -1, axis = 2);
    #f_xy_mm2 = jnp.roll(jnp.roll(f, shift = -2, axis = 1), shift = -2, axis = 2);
    #cx > 0 cy < 0
    #f_xy_pm1 = jnp.roll(jnp.roll(f, shift = -1, axis = 1), shift = 1, axis = 2);
    #f_xy_pm2 = jnp.roll(jnp.roll(f, shift = -2, axis = 1), shift = 2, axis = 2);
    
    return 0.5 * jnp.array([
        3 * f - 4 * f_y_p1 + f_y_p2,
        -3 * f + 4 * f_y_m1 - f_y_m2,
        3 * f - 4 * f_x_p1 + f_x_p2,
        -3 * f + 4 * f_x_m1 - f_x_m2#,
        
     #   3 * f - 4 * f_xy_pp1 + f_xy_pp2,
     #   -3 * f + 4 * f_xy_mp1 - f_xy_mp2,
     #   3 * f - 4 * f_xy_mm1 + f_xy_mm2,
     #   -3 * f + 4 * f_xy_pm1 - f_xy_pm2
    ]);
             
def phi(rho, T):
    return mu0 / (rho * (T0 * T) ** 0.5);

@jax.jit  
def deriv_f(f, feq, rho, u, T):
    df = shifted(f);
    
    #Casos del for
    cases = [
        lambda x: x[2],
        lambda x: x[0],
        lambda x: -x[3],
        lambda x: -x[1],
        lambda x: 6 * x[2],
        lambda x: 6 * x[0],
        lambda x: -6 * x[3],
        lambda x: -6 * x[1],
        lambda x: c1 * (x[0] + x[2]),
        lambda x: c1 * (-x[3] + x[0]),
        lambda x: -c1 * (x[1] + x[3]),
        lambda x: c1 * (x[2] - x[1]),
        lambda x: c2 * (x[0] + x[2]),
        lambda x: c2 * (-x[3] + x[0]),
        lambda x: -c2 * (x[1] + x[3]),
        lambda x: c2 * (x[2] - x[1])
    ];
    
    #debug.print("i={}", phi(rho, T)[45:55, 45:55]);
    
    f_nuev = jnp.zeros_like(f);
    def aux_loop(i, carry):
        f_nuev, f, feq = carry;
        der = lax.switch(i, cases, df);

        f_nuev = f_nuev.at[i].set(f[i] +dt * (-der[i] + (feq[i] - f[i]) / phi(rho, T)));
        
        return (f_nuev, f, feq);
    
    f_nuev, _, _ = lax.fori_loop(0, Q, aux_loop, (f_nuev, f, feq));
    
    return f_nuev;

@jax.jit
def c_densidad(f):
    return jnp.sum(f, axis = 0)

@jax.jit
def c_macro_vel(f, rho):
    return jnp.einsum('qNM,Dq->DNM', f, vels) / rho;

@jax.jit
def c_temp(f, rho, u):
    vels_2 = vels * vels;
    suma = jnp.sum(jnp.einsum('qNM,Dq->DNM', f, vels_2), axis=0) / rho;
    u_2 = jnp.sum(u * u, axis = 0);
    
    return (suma - u_2) / (2 * R);

def prueba():
    u = jnp.zeros((2, N, N));
    rho = jnp.zeros((N, N));
    T = jnp.zeros((N, N));
    
    #Definiendo el dominio:
    cilindro_x = 50;
    cilindro_y = 50;
    radio_cilindro = 3;
    
    #Definiendo el obstáculo circular
    x = jnp.arange(N);
    y = jnp.arange(N);
    X, Y = jnp.meshgrid(x, y);

    alta = jnp.sqrt((X - cilindro_x) ** 2 + (Y - cilindro_y) ** 2) < radio_cilindro
    baja = jnp.logical_not(alta);

    rho = rho.at[alta].set(1.0);
    rho = rho.at[baja].set(0.5);
    T = T.at[alta].set(1.0);
    T = T.at[baja].set(0.5);

    feq = equilibrio(rho, jnp.zeros((2, N, N)), (1 / T0) * T);

    return (feq, feq, rho, u, T);
  
pasos = 5000;
pasos_int = 10;
f, feq, rho, u, T = prueba();

for arch in archivos:
    try:
        os.remove(arch)
    except FileNotFoundError:
        print("No existe el archivo.")

@jax.jit
def aux_intern(i, carry):
    f, feq, rho, u, T = carry;

    f_nuev = deriv_f(f, feq, rho, u, T);
    rho = c_densidad(f_nuev);
    u = c_macro_vel(f_nuev, rho);
    T = c_temp(f_nuev, rho, u);
    
    feq = equilibrio(rho, A * u, (1 / T0) * T);

    return (f_nuev, feq, rho, u, T);
    

with open(archivos[0], 'ab') as f_data1, open(archivos[1], 'ab') as f_data2:
    for t in range(int(pasos / pasos_int)):
        f, feq, rho, u, T = lax.fori_loop(0, pasos_int, aux_intern, (f, feq, rho, u, T));
        
        pickle.dump(u, f_data1);
        pickle.dump(rho, f_data2);
    
print("Termino!");
