import jax
import jax.numpy as jnp
from jax import jit, random
from functools import partial
import os
import numpy as np 
import scipy.interpolate as spi 

npx, npy, npz = 256, 256, 256   
num_dev = 4                       
npz_local = npz // num_dev         

Lx, Ly, Lz = 1.0, 1.0, 1.0      
dxi = 1.0 / (npx - 1)
deta = 1.0 / (npy - 1)
dzeta = 1.0 / (npz - 1)

dt = 2e-4
tf = 1500
Ma = 0.1
Pr = 0.7
Ra = 1e7
Ma2_inv = 1.0 / (Ma**2)


CONTROL_MODE = os.environ.get("RB_CONTROL_MODE", "reference").strip().lower()
CONTROL_RESTART = "RESTART0001499.dat.npz"

NU0_REFERENCE = 16.268831

CONTROL_ACTION_TIME = 0.005
CONTROL_ACTION_STEPS = int(round(CONTROL_ACTION_TIME / dt))
CONTROL_BETA_REFERENCE = 20.0
CONTROL_AMPLITUDE_REFERENCE = 1.0

CONTROL_TRANSIENT_TIME = 50.0
CONTROL_AVERAGING_TIME = 100.0
CONTROL_LOG_TIME = 0.05
CONTROL_SNAPSHOT_TIME = 10.0

DIFF_WINDOW_TIME = 5.0
DIFF_REWARD_START_TIME = 2.5
DIFF_N_BLOCKS = int(round(DIFF_WINDOW_TIME / CONTROL_ACTION_TIME))
DIFF_REWARD_START_BLOCK = int(round(DIFF_REWARD_START_TIME / CONTROL_ACTION_TIME))

GRAD_CHECK_WINDOW_TIME = 0.05
GRAD_CHECK_REWARD_START_TIME = 0.025
GRAD_CHECK_N_BLOCKS = int(round(GRAD_CHECK_WINDOW_TIME / CONTROL_ACTION_TIME))
GRAD_CHECK_REWARD_START_BLOCK = int(
    round(GRAD_CHECK_REWARD_START_TIME / CONTROL_ACTION_TIME)
)

OPT_INITIAL_PARAMETERS = (0.80, 10.0)
OPT_ITERATIONS = 8
OPT_LEARNING_RATE = (0.03, 0.50)
FD_EPSILON_SWEEP = (
    (3.0e-3, 1.0e-3, 3.0e-4),
    (3.0e-2, 1.0e-2, 3.0e-3),
)


xi_1d = np.linspace(0, 1, npx)
eta_1d = np.linspace(0, 1, npy)
zeta_1d = np.linspace(0, 1, npz)


gamma_xy = 1.2
gamma_z = 1.2

x_1d = Lx * 0.5 * (1.0 + np.tanh(gamma_xy * (2.0 * xi_1d - 1.0)) / np.tanh(gamma_xy))
dx_dxi = Lx * (gamma_xy / np.tanh(gamma_xy)) * (1.0 - np.tanh(gamma_xy * (2.0 * xi_1d - 1.0))**2)
d2x_dxi2 = Lx * (gamma_xy / np.tanh(gamma_xy)) * (-4.0 * gamma_xy) * (1.0 - np.tanh(gamma_xy * (2.0 * xi_1d - 1.0))**2) * np.tanh(gamma_xy * (2.0 * xi_1d - 1.0))
Jx = jnp.array(dx_dxi).reshape(npx, 1, 1, 1)
Jxx = jnp.array(d2x_dxi2).reshape(npx, 1, 1, 1)

y_1d = Ly * 0.5 * (1.0 + np.tanh(gamma_xy * (2.0 * eta_1d - 1.0)) / np.tanh(gamma_xy))
dy_deta = Ly * (gamma_xy / np.tanh(gamma_xy)) * (1.0 - np.tanh(gamma_xy * (2.0 * eta_1d - 1.0))**2)
d2y_deta2 = Ly * (gamma_xy / np.tanh(gamma_xy)) * (-4.0 * gamma_xy) * (1.0 - np.tanh(gamma_xy * (2.0 * eta_1d - 1.0))**2) * np.tanh(gamma_xy * (2.0 * eta_1d - 1.0))
Jy = jnp.array(dy_deta).reshape(1, npy, 1, 1)
Jyy = jnp.array(d2y_deta2).reshape(1, npy, 1, 1)

z_1d = Lz * 0.5 * (1.0 + np.tanh(gamma_z * (2.0 * zeta_1d - 1.0)) / np.tanh(gamma_z))
dz_dzeta = Lz * (gamma_z / np.tanh(gamma_z)) * (1.0 - np.tanh(gamma_z * (2.0 * zeta_1d - 1.0))**2)
d2z_dzeta2 = Lz * (gamma_z / np.tanh(gamma_z)) * (-4.0 * gamma_z) * (1.0 - np.tanh(gamma_z * (2.0 * zeta_1d - 1.0))**2) * np.tanh(gamma_z * (2.0 * zeta_1d - 1.0))
Jz = jnp.array(dz_dzeta).reshape(1, 1, npz, 1)
Jzz = jnp.array(d2z_dzeta2).reshape(1, 1, npz, 1)

Jz_split = np.moveaxis(Jz.reshape(1, 1, num_dev, npz_local, 1), 2, 0)
Jzz_split = np.moveaxis(Jzz.reshape(1, 1, num_dev, npz_local, 1), 2, 0)



c0, c1, c2, c3, c4, c5, c6 = -1/140, 5/84, -101/420, 319/420, 107/210, -19/210, 1/105

@jit
def compute_flux_x(fp, fm):
    N = fp.shape[0] - 7 
    flux_p = c0*fp[0:N] + c1*fp[1:N+1] + c2*fp[2:N+2] + c3*fp[3:N+3] + c4*fp[4:N+4] + c5*fp[5:N+5] + c6*fp[6:N+6]
    flux_m = c0*fm[7:N+7] + c1*fm[6:N+6] + c2*fm[5:N+5] + c3*fm[4:N+4] + c4*fm[3:N+3] + c5*fm[2:N+2] + c6*fm[1:N+1]
    return flux_p + flux_m

@jit
def compute_flux_y(fp, fm):
    N = fp.shape[1] - 7 
    flux_p = c0*fp[:, 0:N] + c1*fp[:, 1:N+1] + c2*fp[:, 2:N+2] + c3*fp[:, 3:N+3] + c4*fp[:, 4:N+4] + c5*fp[:, 5:N+5] + c6*fp[:, 6:N+6]
    flux_m = c0*fm[:, 7:N+7] + c1*fm[:, 6:N+6] + c2*fm[:, 5:N+5] + c3*fm[:, 4:N+4] + c4*fm[:, 3:N+3] + c5*fm[:, 2:N+2] + c6*fm[:, 1:N+1]
    return flux_p + flux_m

@jit
def compute_flux_z(fp, fm):
    N = fp.shape[2] - 7 
    flux_p = c0*fp[:, :, 0:N] + c1*fp[:, :, 1:N+1] + c2*fp[:, :, 2:N+2] + c3*fp[:, :, 3:N+3] + c4*fp[:, :, 4:N+4] + c5*fp[:, :, 5:N+5] + c6*fp[:, :, 6:N+6]
    flux_m = c0*fm[:, :, 7:N+7] + c1*fm[:, :, 6:N+6] + c2*fm[:, :, 5:N+5] + c3*fm[:, :, 4:N+4] + c4*fm[:, :, 3:N+3] + c5*fm[:, :, 2:N+2] + c6*fm[:, :, 1:N+1]
    return flux_p + flux_m

@jit
def extrapolate_ghosts_3rd(f0, f1, f2, f3):
    fm1 = 4*f0 - 6*f1 + 4*f2 - f3
    fm2 = 4*fm1 - 6*f0 + 4*f1 - f2
    fm3 = 4*fm2 - 6*fm1 + 4*f0 - f1
    fm4 = 4*fm3 - 6*fm2 + 4*fm1 - f0
    return fm1, fm2, fm3, fm4
@jit
def extrapolate_neumann_3rd(f0, f1, f2, f3):
    fm1 = -(10.0/3.0)*f0 + 6.0*f1 - 2.0*f2 + (1.0/3.0)*f3
    fm2 = -(80.0/3.0)*f0 + 40.0*f1 - 15.0*f2 + (8.0/3.0)*f3
    fm3 = -90.0*f0 + 135.0*f1 - 54.0*f2 + 10.0*f3
    fm4 = -(665.0/3.0)*f0 + 336.0*f1 - 140.0*f2 + (80.0/3.0)*f3
    return fm1, fm2, fm3, fm4

@jit
def apply_boundaries_and_halos(local_ub, pid):
    ubs = jnp.pad(local_ub, ((4,4), (4,4), (4,4), (0,0)), mode='constant')


    f0, f1, f2, f3 = ubs[4, 4:-4, 4:-4, :], ubs[5, 4:-4, 4:-4, :], ubs[6, 4:-4, 4:-4, :], ubs[7, 4:-4, 4:-4, :]
    fm1_D, fm2_D, fm3_D, fm4_D = extrapolate_ghosts_3rd(f0[..., 1:4], f1[..., 1:4], f2[..., 1:4], f3[..., 1:4])
    fm1_P, fm2_P, fm3_P, fm4_P = extrapolate_neumann_3rd(f0[..., 0], f1[..., 0], f2[..., 0], f3[..., 0])
    fm1_T, fm2_T, fm3_T, fm4_T = extrapolate_neumann_3rd(f0[..., 4], f1[..., 4], f2[..., 4], f3[..., 4])
    
    ubs = ubs.at[3, 4:-4, 4:-4, 1:4].set(fm1_D); ubs = ubs.at[3, 4:-4, 4:-4, 0].set(fm1_P); ubs = ubs.at[3, 4:-4, 4:-4, 4].set(fm1_T)
    ubs = ubs.at[2, 4:-4, 4:-4, 1:4].set(fm2_D); ubs = ubs.at[2, 4:-4, 4:-4, 0].set(fm2_P); ubs = ubs.at[2, 4:-4, 4:-4, 4].set(fm2_T)
    ubs = ubs.at[1, 4:-4, 4:-4, 1:4].set(fm3_D); ubs = ubs.at[1, 4:-4, 4:-4, 0].set(fm3_P); ubs = ubs.at[1, 4:-4, 4:-4, 4].set(fm3_T)
    ubs = ubs.at[0, 4:-4, 4:-4, 1:4].set(fm4_D); ubs = ubs.at[0, 4:-4, 4:-4, 0].set(fm4_P); ubs = ubs.at[0, 4:-4, 4:-4, 4].set(fm4_T)

    f0, f1, f2, f3 = ubs[-5, 4:-4, 4:-4, :], ubs[-6, 4:-4, 4:-4, :], ubs[-7, 4:-4, 4:-4, :], ubs[-8, 4:-4, 4:-4, :]
    fm1_D, fm2_D, fm3_D, fm4_D = extrapolate_ghosts_3rd(f0[..., 1:4], f1[..., 1:4], f2[..., 1:4], f3[..., 1:4])
    fm1_P, fm2_P, fm3_P, fm4_P = extrapolate_neumann_3rd(f0[..., 0], f1[..., 0], f2[..., 0], f3[..., 0])
    fm1_T, fm2_T, fm3_T, fm4_T = extrapolate_neumann_3rd(f0[..., 4], f1[..., 4], f2[..., 4], f3[..., 4])
    
    ubs = ubs.at[-4, 4:-4, 4:-4, 1:4].set(fm1_D); ubs = ubs.at[-4, 4:-4, 4:-4, 0].set(fm1_P); ubs = ubs.at[-4, 4:-4, 4:-4, 4].set(fm1_T)
    ubs = ubs.at[-3, 4:-4, 4:-4, 1:4].set(fm2_D); ubs = ubs.at[-3, 4:-4, 4:-4, 0].set(fm2_P); ubs = ubs.at[-3, 4:-4, 4:-4, 4].set(fm2_T)
    ubs = ubs.at[-2, 4:-4, 4:-4, 1:4].set(fm3_D); ubs = ubs.at[-2, 4:-4, 4:-4, 0].set(fm3_P); ubs = ubs.at[-2, 4:-4, 4:-4, 4].set(fm3_T)
    ubs = ubs.at[-1, 4:-4, 4:-4, 1:4].set(fm4_D); ubs = ubs.at[-1, 4:-4, 4:-4, 0].set(fm4_P); ubs = ubs.at[-1, 4:-4, 4:-4, 4].set(fm4_T)


    f0, f1, f2, f3 = ubs[4:-4, 4, 4:-4, :], ubs[4:-4, 5, 4:-4, :], ubs[4:-4, 6, 4:-4, :], ubs[4:-4, 7, 4:-4, :]
    fm1_D, fm2_D, fm3_D, fm4_D = extrapolate_ghosts_3rd(f0[..., 1:4], f1[..., 1:4], f2[..., 1:4], f3[..., 1:4])
    fm1_P, fm2_P, fm3_P, fm4_P = extrapolate_neumann_3rd(f0[..., 0], f1[..., 0], f2[..., 0], f3[..., 0])
    fm1_T, fm2_T, fm3_T, fm4_T = extrapolate_neumann_3rd(f0[..., 4], f1[..., 4], f2[..., 4], f3[..., 4])
    
    ubs = ubs.at[4:-4, 3, 4:-4, 1:4].set(fm1_D); ubs = ubs.at[4:-4, 3, 4:-4, 0].set(fm1_P); ubs = ubs.at[4:-4, 3, 4:-4, 4].set(fm1_T)
    ubs = ubs.at[4:-4, 2, 4:-4, 1:4].set(fm2_D); ubs = ubs.at[4:-4, 2, 4:-4, 0].set(fm2_P); ubs = ubs.at[4:-4, 2, 4:-4, 4].set(fm2_T)
    ubs = ubs.at[4:-4, 1, 4:-4, 1:4].set(fm3_D); ubs = ubs.at[4:-4, 1, 4:-4, 0].set(fm3_P); ubs = ubs.at[4:-4, 1, 4:-4, 4].set(fm3_T)
    ubs = ubs.at[4:-4, 0, 4:-4, 1:4].set(fm4_D); ubs = ubs.at[4:-4, 0, 4:-4, 0].set(fm4_P); ubs = ubs.at[4:-4, 0, 4:-4, 4].set(fm4_T)

    f0, f1, f2, f3 = ubs[4:-4, -5, 4:-4, :], ubs[4:-4, -6, 4:-4, :], ubs[4:-4, -7, 4:-4, :], ubs[4:-4, -8, 4:-4, :]
    fm1_D, fm2_D, fm3_D, fm4_D = extrapolate_ghosts_3rd(f0[..., 1:4], f1[..., 1:4], f2[..., 1:4], f3[..., 1:4])
    fm1_P, fm2_P, fm3_P, fm4_P = extrapolate_neumann_3rd(f0[..., 0], f1[..., 0], f2[..., 0], f3[..., 0])
    fm1_T, fm2_T, fm3_T, fm4_T = extrapolate_neumann_3rd(f0[..., 4], f1[..., 4], f2[..., 4], f3[..., 4])
    
    ubs = ubs.at[4:-4, -4, 4:-4, 1:4].set(fm1_D); ubs = ubs.at[4:-4, -4, 4:-4, 0].set(fm1_P); ubs = ubs.at[4:-4, -4, 4:-4, 4].set(fm1_T)
    ubs = ubs.at[4:-4, -3, 4:-4, 1:4].set(fm2_D); ubs = ubs.at[4:-4, -3, 4:-4, 0].set(fm2_P); ubs = ubs.at[4:-4, -3, 4:-4, 4].set(fm2_T)
    ubs = ubs.at[4:-4, -2, 4:-4, 1:4].set(fm3_D); ubs = ubs.at[4:-4, -2, 4:-4, 0].set(fm3_P); ubs = ubs.at[4:-4, -2, 4:-4, 4].set(fm3_T)
    ubs = ubs.at[4:-4, -1, 4:-4, 1:4].set(fm4_D); ubs = ubs.at[4:-4, -1, 4:-4, 0].set(fm4_P); ubs = ubs.at[4:-4, -1, 4:-4, 4].set(fm4_T)

    send_top, send_bot = ubs[:, :, -8:-4, :], ubs[:, :, 4:8, :]
    recv_bot = jax.lax.ppermute(send_top, axis_name='z', perm=[(i, (i+1)%num_dev) for i in range(num_dev)])
    recv_top = jax.lax.ppermute(send_bot, axis_name='z', perm=[(i, (i-1)%num_dev) for i in range(num_dev)])
    ubs = ubs.at[:, :, 0:4, :].set(recv_bot)
    ubs = ubs.at[:, :, -4:, :].set(recv_top)

    def apply_z_bot(u):
        f0, f1, f2, f3 = u[4:-4, 4:-4, 4, :], u[4:-4, 4:-4, 5, :], u[4:-4, 4:-4, 6, :], u[4:-4, 4:-4, 7, :]
        fm1_D, fm2_D, fm3_D, fm4_D = extrapolate_ghosts_3rd(f0[..., 1:5], f1[..., 1:5], f2[..., 1:5], f3[..., 1:5])
        fm1_P, fm2_P, fm3_P, fm4_P = extrapolate_neumann_3rd(f0[..., 0], f1[..., 0], f2[..., 0], f3[..., 0])
        
        u = u.at[4:-4, 4:-4, 3, 1:5].set(fm1_D); u = u.at[4:-4, 4:-4, 3, 0].set(fm1_P)
        u = u.at[4:-4, 4:-4, 2, 1:5].set(fm2_D); u = u.at[4:-4, 4:-4, 2, 0].set(fm2_P)
        u = u.at[4:-4, 4:-4, 1, 1:5].set(fm3_D); u = u.at[4:-4, 4:-4, 1, 0].set(fm3_P)
        u = u.at[4:-4, 4:-4, 0, 1:5].set(fm4_D); u = u.at[4:-4, 4:-4, 0, 0].set(fm4_P)
        return u

    def apply_z_top(u):
        f0, f1, f2, f3 = u[4:-4, 4:-4, -5, :], u[4:-4, 4:-4, -6, :], u[4:-4, 4:-4, -7, :], u[4:-4, 4:-4, -8, :]
        fm1_D, fm2_D, fm3_D, fm4_D = extrapolate_ghosts_3rd(f0[..., 1:5], f1[..., 1:5], f2[..., 1:5], f3[..., 1:5])
        fm1_P, fm2_P, fm3_P, fm4_P = extrapolate_neumann_3rd(f0[..., 0], f1[..., 0], f2[..., 0], f3[..., 0])
        
        u = u.at[4:-4, 4:-4, -4, 1:5].set(fm1_D); u = u.at[4:-4, 4:-4, -4, 0].set(fm1_P)
        u = u.at[4:-4, 4:-4, -3, 1:5].set(fm2_D); u = u.at[4:-4, 4:-4, -3, 0].set(fm2_P)
        u = u.at[4:-4, 4:-4, -2, 1:5].set(fm3_D); u = u.at[4:-4, 4:-4, -2, 0].set(fm3_P)
        u = u.at[4:-4, 4:-4, -1, 1:5].set(fm4_D); u = u.at[4:-4, 4:-4, -1, 0].set(fm4_P)
        return u

    ubs = jax.lax.cond(pid == 0, apply_z_bot, lambda u: u, ubs)
    ubs = jax.lax.cond(pid == num_dev - 1, apply_z_top, lambda u: u, ubs)
    return ubs

@jit
def calc_local_resid(local_ub, pid, Jx, Jxx, Jy, Jyy, Jz_loc, Jzz_loc):
    ubs = apply_boundaries_and_halos(local_ub, pid)
    u_c0, u_c1, u_c2, u_c3, u_c4 = ubs[...,0], ubs[...,1], ubs[...,2], ubs[...,3], ubs[...,4]

    flux_x = jnp.stack([u_c1*(u_c0+Ma2_inv), u_c0+u_c1**2, u_c1*u_c2, u_c1*u_c3, u_c1*u_c4], axis=-1)
    lam_x = jnp.expand_dims(jnp.abs(1.5*u_c1) + jnp.sqrt(0.25*u_c1**2 + u_c0 + Ma2_inv), -1)
    num_flux_x = compute_flux_x(0.5*(flux_x + lam_x*ubs)[:,4:-4,4:-4,:], 0.5*(flux_x - lam_x*ubs)[:,4:-4,4:-4,:])
    resid = -(num_flux_x[1:] - num_flux_x[:-1]) / (dxi * Jx)
    
    flux_y = jnp.stack([u_c2*(u_c0+Ma2_inv), u_c2*u_c1, u_c0+u_c2**2, u_c2*u_c3, u_c2*u_c4], axis=-1)
    lam_y = jnp.expand_dims(jnp.abs(1.5*u_c2) + jnp.sqrt(0.25*u_c2**2 + u_c0 + Ma2_inv), -1)
    num_flux_y = compute_flux_y(0.5*(flux_y + lam_y*ubs)[4:-4,:,4:-4,:], 0.5*(flux_y - lam_y*ubs)[4:-4,:,4:-4,:])
    resid += -(num_flux_y[:,1:] - num_flux_y[:,:-1]) / (deta * Jy)

    flux_z = jnp.stack([u_c3*(u_c0+Ma2_inv), u_c3*u_c1, u_c3*u_c2, u_c0+u_c3**2, u_c3*u_c4], axis=-1)
    lam_z = jnp.expand_dims(jnp.abs(1.5*u_c3) + jnp.sqrt(0.25*u_c3**2 + u_c0 + Ma2_inv), -1)
    num_flux_z = compute_flux_z(0.5*(flux_z + lam_z*ubs)[4:-4,4:-4,:,:], 0.5*(flux_z - lam_z*ubs)[4:-4,4:-4,:,:])
    resid += -(num_flux_z[:,:,1:] - num_flux_z[:,:,:-1]) / (dzeta * Jz_loc)

    u_x, u_y, u_z = ubs[:,4:-4,4:-4,:], ubs[4:-4,:,4:-4,:], ubs[4:-4,4:-4,:,:]
    
    d2u_dxi2 = (-9*u_x[0:npx] + 128*u_x[1:npx+1] - 1008*u_x[2:npx+2] + 8064*u_x[3:npx+3] - 14350*u_x[4:npx+4] + 8064*u_x[5:npx+5] - 1008*u_x[6:npx+6] + 128*u_x[7:npx+7] - 9*u_x[8:npx+8]) / (5040.0*dxi**2)
    du_dxi = (3*u_x[0:npx] - 32*u_x[1:npx+1] + 168*u_x[2:npx+2] - 672*u_x[3:npx+3] + 672*u_x[5:npx+5] - 168*u_x[6:npx+6] + 32*u_x[7:npx+7] - 3*u_x[8:npx+8]) / (840.0*dxi)
    lap_x = d2u_dxi2/(Jx**2) - du_dxi*Jxx/(Jx**3)

    d2u_deta2 = (-9*u_y[:,0:npy] + 128*u_y[:,1:npy+1] - 1008*u_y[:,2:npy+2] + 8064*u_y[:,3:npy+3] - 14350*u_y[:,4:npy+4] + 8064*u_y[:,5:npy+5] - 1008*u_y[:,6:npy+6] + 128*u_y[:,7:npy+7] - 9*u_y[:,8:npy+8]) / (5040.0*deta**2)
    du_deta = (3*u_y[:,0:npy] - 32*u_y[:,1:npy+1] + 168*u_y[:,2:npy+2] - 672*u_y[:,3:npy+3] + 672*u_y[:,5:npy+5] - 168*u_y[:,6:npy+6] + 32*u_y[:,7:npy+7] - 3*u_y[:,8:npy+8]) / (840.0*deta)
    lap_y = d2u_deta2/(Jy**2) - du_deta*Jyy/(Jy**3)

    d2u_dzeta2 = (-9*u_z[:,:,0:npz_local] + 128*u_z[:,:,1:npz_local+1] - 1008*u_z[:,:,2:npz_local+2] + 8064*u_z[:,:,3:npz_local+3] - 14350*u_z[:,:,4:npz_local+4] + 8064*u_z[:,:,5:npz_local+5] - 1008*u_z[:,:,6:npz_local+6] + 128*u_z[:,:,7:npz_local+7] - 9*u_z[:,:,8:npz_local+8]) / (5040.0*dzeta**2)
    du_dzeta = (3*u_z[:,:,0:npz_local] - 32*u_z[:,:,1:npz_local+1] + 168*u_z[:,:,2:npz_local+2] - 672*u_z[:,:,3:npz_local+3] + 672*u_z[:,:,5:npz_local+5] - 168*u_z[:,:,6:npz_local+6] + 32*u_z[:,:,7:npz_local+7] - 3*u_z[:,:,8:npz_local+8]) / (840.0*dzeta)
    lap_z = d2u_dzeta2/(Jz_loc**2) - du_dzeta*Jzz_loc/(Jz_loc**3)

    resid = resid.at[..., 0:4].add((Pr/Ra)**0.5 * (lap_x+lap_y+lap_z)[..., 0:4])
    resid = resid.at[..., 4].add((Ra * Pr)**(-0.5) * (lap_x+lap_y+lap_z)[..., 4])
    resid = resid.at[..., 3].add(local_ub[..., 4])


    resid = resid.at[0, :, :, 1:4].set(0.0)    
    resid = resid.at[-1, :, :, 1:4].set(0.0)   
    resid = resid.at[:, 0, :, 1:4].set(0.0)   
    resid = resid.at[:, -1, :, 1:4].set(0.0)   
    
    resid = jax.lax.cond(pid == 0, lambda r: r.at[:, :, 0, 1:5].set(0.0), lambda r: r, resid)
    resid = jax.lax.cond(pid == num_dev - 1, lambda r: r.at[:, :, -1, 1:5].set(0.0), lambda r: r, resid)

    return resid


@partial(jax.pmap, axis_name='z', static_broadcasted_argnums=(9,), in_axes=(0, 0, 0, None, None, None, None, 0, 0, None), donate_argnums=(0, 1))
def pmap_advance_fused(loc_ub, loc_tec, pid, Jx, Jxx, Jy, Jyy, Jz_loc, Jzz_loc, n_steps):
    
    def body_fn(i, val):
        u, u_tec = val
        a30, a32, a40, a43, a52, a54 = 0.355909775, 0.644090224, 0.367933791, 0.632066208, 0.237593836, 0.762406163
        b10, b21, b32, b43, b54 = 0.377268915, 0.377268915, 0.242995220, 0.238458932, 0.287632146
        
        u1 = u
        u = u + b10 * dt * calc_local_resid(u, pid, Jx, Jxx, Jy, Jyy, Jz_loc, Jzz_loc)
        u = u + b21 * dt * calc_local_resid(u, pid, Jx, Jxx, Jy, Jyy, Jz_loc, Jzz_loc)
        u2 = u
        u = a30*u1 + a32*u + b32 * dt * calc_local_resid(u, pid, Jx, Jxx, Jy, Jyy, Jz_loc, Jzz_loc)
        u = a40*u1 + a43*u + b43 * dt * calc_local_resid(u, pid, Jx, Jxx, Jy, Jyy, Jz_loc, Jzz_loc)
        r_final = calc_local_resid(u, pid, Jx, Jxx, Jy, Jyy, Jz_loc, Jzz_loc)
        u = a52*u2 + a54*u + b54 * dt * r_final
        
        return u, u_tec + u

    final_ub, final_tec = jax.lax.fori_loop(0, n_steps, body_fn, (loc_ub, loc_tec))
    
    resid_final = calc_local_resid(final_ub, pid, Jx, Jxx, Jy, Jyy, Jz_loc, Jzz_loc)
    local_sq_sum = jnp.sum(resid_final**2)
    global_sq_sum = jax.lax.psum(local_sq_sum, axis_name='z')
    global_norm = jnp.sqrt(global_sq_sum) / (npx * npy * npz)
    

    theta = final_ub[..., 4]
    dt_dz = (-25*theta[:,:,0] + 48*theta[:,:,1] - 36*theta[:,:,2] + 16*theta[:,:,3] - 3*theta[:,:,4]) / (12*dzeta)
    nu_local_bot = - dt_dz / Jz_loc[0,0,0,0]
    w = Jx[..., 0, 0] * Jy[..., 0, 0]
    local_nu_bot = jax.lax.cond(pid == 0, lambda _: jnp.sum(nu_local_bot * w) / jnp.sum(w), lambda _: 0.0, operand=None)
    global_nu_bot = jax.lax.psum(local_nu_bot, axis_name='z')
    dt_dz_top = (25*theta[:,:,-1] - 48*theta[:,:,-2] + 36*theta[:,:,-3] - 16*theta[:,:,-4] + 3*theta[:,:,-5]) / (12*dzeta)
    nu_local_top = -dt_dz_top / Jz_loc[0,0,-1,0]
    local_nu_top = jax.lax.cond(pid == num_dev-1,            
                                lambda _: jnp.sum(nu_local_top * w) / jnp.sum(w),
                                lambda _: 0.0, operand=None)
    global_nu_top = jax.lax.psum(local_nu_top, axis_name='z')
    global_nu_wall = 0.5 * (global_nu_bot + global_nu_top)    
    u_z = final_ub[..., 3]
    W_x = jnp.ones(npx).at[0].set(0.5).at[-1].set(0.5)
    W_y = jnp.ones(npy).at[0].set(0.5).at[-1].set(0.5)
    W_z = jnp.ones(npz_local)
    W_z = jax.lax.cond(pid == 0, lambda w: w.at[0].set(0.5), lambda w: w, W_z)
    W_z = jax.lax.cond(pid == num_dev - 1, lambda w: w.at[-1].set(0.5), lambda w: w, W_z)
    W_3D = W_x.reshape(npx, 1, 1) * W_y.reshape(1, npy, 1) * W_z.reshape(1, 1, npz_local)

    dV_exact = Jx[..., 0] * Jy[..., 0] * Jz_loc[..., 0] * dxi * deta * dzeta * W_3D
    local_uzT = jnp.sum(u_z * theta * dV_exact)
    local_vol = jnp.sum(dV_exact)
    
    global_uzT = jax.lax.psum(local_uzT, axis_name='z')
    global_vol = jax.lax.psum(local_vol, axis_name='z')
    
    global_nu_vol = 1.0 + jnp.sqrt(Ra * Pr) * (global_uzT / global_vol)
    
    ubs_pad = apply_boundaries_and_halos(final_ub, pid)
    u_p, v_p, w_p, T_p = ubs_pad[..., 1], ubs_pad[..., 2], ubs_pad[..., 3], ubs_pad[..., 4]

    u_phys = ubs_pad[4:-4, 4:-4, 4:-4, 1]
    v_phys = ubs_pad[4:-4, 4:-4, 4:-4, 2]
    w_phys = ubs_pad[4:-4, 4:-4, 4:-4, 3]
    T_phys = ubs_pad[4:-4, 4:-4, 4:-4, 4]


    def get_grads(phi_phys):
        dphi_dxi = jnp.gradient(phi_phys, dxi, axis=0)
        dphi_deta = jnp.gradient(phi_phys, deta, axis=1)
        dphi_dzeta = jnp.gradient(phi_phys, dzeta, axis=2)
        dx = dphi_dxi / Jx[..., 0]
        dy = dphi_deta / Jy[..., 0]
        dz = dphi_dzeta / Jz_loc[..., 0]
        return dx, dy, dz

    local_u2_int = jnp.sum((u_phys**2 + v_phys**2 + w_phys**2) * dV_exact)
    global_u2_int = jax.lax.psum(local_u2_int, axis_name='z')
    U_rms = jnp.sqrt(global_u2_int / global_vol)
    global_Re_rms = jnp.sqrt(Ra / Pr) * U_rms
    u_x, u_y, u_z = get_grads(u_phys)
    v_x, v_y, v_z = get_grads(v_phys)
    w_x, w_y, w_z = get_grads(w_phys)
    T_x, T_y, T_z = get_grads(T_phys)

    omega_x = w_y - v_z
    omega_y = u_z - w_x
    omega_z = v_x - u_y
    enstrophy = omega_x**2 + omega_y**2 + omega_z**2

    grad_u2 = 2.0 * (u_x**2 + v_y**2 + w_z**2) + \
              (u_y + v_x)**2 + \
              (u_z + w_x)**2 + \
              (v_z + w_y)**2
    grad_T2 = T_x**2 + T_y**2 + T_z**2

    eps_u = jnp.sqrt(Pr / Ra) * grad_u2
    eps_T = (1.0 / jnp.sqrt(Ra * Pr)) * grad_T2

    local_eps_moms = jnp.array([
        jnp.sum(eps_u * dV_exact), jnp.sum((eps_u**2) * dV_exact),
        jnp.sum((eps_u**3) * dV_exact), jnp.sum((eps_u**4) * dV_exact)
    ])
    local_ens_moms = jnp.array([
        jnp.sum(enstrophy * dV_exact), jnp.sum((enstrophy**2) * dV_exact),
        jnp.sum((enstrophy**3) * dV_exact), jnp.sum((enstrophy**4) * dV_exact)
    ])
    local_eps_T_int = jnp.sum(eps_T * dV_exact)

    global_eps_moms = jax.lax.psum(local_eps_moms, axis_name='z')
    global_ens_moms = jax.lax.psum(local_ens_moms, axis_name='z')
    global_eps_T_int = jax.lax.psum(local_eps_T_int, axis_name='z')

    avg_eps_u = global_eps_moms[0] / global_vol
    avg_eps_T = global_eps_T_int / global_vol
    global_nu_kin = 1.0 + jnp.sqrt(Ra * Pr) * avg_eps_u
    global_nu_th = jnp.sqrt(Ra * Pr) * avg_eps_T

    return final_ub, final_tec, global_norm, global_nu_bot, global_nu_top, global_nu_vol, global_nu_kin, global_nu_th, global_nu_wall, global_vol, global_eps_moms, global_ens_moms, global_Re_rms


def write_nusselt_log(script_dir, iter, ctime,
                      i_bot, a_bot,
                      i_top, a_top,
                      i_vol, a_vol,
                      i_kin, a_kin,
                      i_th,  a_th,
                      i_wall, a_wall,
                      i_re,  a_re): 
    log_filename = os.path.join(script_dir, "Nusselt_Quad_Log.dat")
    with open(log_filename, "a" if os.path.exists(log_filename) else "w") as f:
        if not os.path.exists(log_filename):
            f.write('Variables="Iter", "Time", '
                    '"Bot_Nu_Inst", "Bot_Nu_Avg", '
                    '"Top_Nu_Inst", "Top_Nu_Avg", '
                    '"Vol_Nu_Inst", "Vol_Nu_Avg", '
                    '"Kin_Nu_Inst", "Kin_Nu_Avg", '
                    '"Th_Nu_Inst", "Th_Nu_Avg", '
                    '"Wall_Nu_Inst", "Wall_Nu_Avg", '
                    '"Re_rms_Inst", "Re_rms_Avg"\n')
        f.write(f"{iter:<10d}\t{ctime:.4f}\t"
                f"{i_bot:.6f}\t{a_bot:.6f}\t"
                f"{i_top:.6f}\t{a_top:.6f}\t"
                f"{i_vol:.6f}\t{a_vol:.6f}\t"
                f"{i_kin:.6f}\t{a_kin:.6f}\t"
                f"{i_th:.6f}\t{a_th:.6f}\t"
                f"{i_wall:.6f}\t{a_wall:.6f}\t"
                f"{i_re:.6f}\t{a_re:.6f}\n") 



def write_solu_tecplot(script_dir, state):
    ctime = state['ctime']
    decimal = f"{ctime - int(ctime):.4f}"[1:]
    
    base_name = os.path.join(script_dir, f"TEC{int(ctime):07d}{decimal}")
    grid_file = base_name + ".xyz"
    sol_file = base_name + ".q"

    with open(grid_file, "wb") as fg:
        fg.write(np.array([npx, npy, npz], dtype='<i4').tobytes())
        
        fg.write(np.tile(x_1d.astype('<f4'), npy * npz).tobytes())
        fg.write(np.tile(np.repeat(y_1d.astype('<f4'), npx), npz).tobytes())
        fg.write(np.repeat(z_1d.astype('<f4'), npx * npy).tobytes())

    with open(sol_file, "wb") as fq:
        fq.write(np.array([npx, npy, npz], dtype='<i4').tobytes())
        fq.write(np.array([Ma, 0.0, Ra, ctime], dtype='<f4').tobytes())
        
        ub_device = state['ub_split']
        
        for v in range(5):
            for pid in range(num_dev):
                local_var = np.asarray(ub_device[pid][..., v]).astype('<f4')
                
                fq.write(local_var.flatten('F').tobytes())
                del local_var

def read_and_interpolate_data(namerestart, x_new, y_new, z_new):
    data = np.load(namerestart)
    ub_old = data['ub']
    

    npx_old, npy_old, npz_old = 128, 128, 128
    gamma_xy_old, gamma_z_old = 1.2, 1.2
    Lx_old, Ly_old, Lz_old = 1.0, 1.0, 1.0
    
    xi_old = np.linspace(0, 1, npx_old)
    eta_old = np.linspace(0, 1, npy_old)
    zeta_old = np.linspace(0, 1, npz_old)
    

    x_old = Lx_old * 0.5 * (1.0 + np.tanh(gamma_xy_old * (2.0 * xi_old - 1.0)) / np.tanh(gamma_xy_old))
    y_old = Ly_old * 0.5 * (1.0 + np.tanh(gamma_xy_old * (2.0 * eta_old - 1.0)) / np.tanh(gamma_xy_old))
    z_old = Lz_old * 0.5 * (1.0 + np.tanh(gamma_z_old * (2.0 * zeta_old - 1.0)) / np.tanh(gamma_z_old))
    

    npx_new, npy_new, npz_new = len(x_new), len(y_new), len(z_new)
    ub_new = np.zeros((npx_new, npy_new, npz_new, 5), dtype=np.float32)
    
    X_new, Y_new, Z_new = np.meshgrid(x_new, y_new, z_new, indexing='ij')
    
    
    for v in range(5):
        interp_func = spi.RegularGridInterpolator((x_old, y_old, z_old), ub_old[..., v], 
                                                  bounds_error=False, fill_value=None)
        ub_new[..., v] = interp_func((X_new, Y_new, Z_new))

    ub_split = np.stack(np.split(ub_new, num_dev, axis=2))
    return {'ctime': data['ctime'].item(), 'iter': data['iter'].item(), 'ub_split': ub_split}

def write_data(script_dir, state):
    ctime = state['ctime']
    filename = os.path.join(script_dir, f"RESTART{int(ctime):07d}.dat.npz")
    np.savez(filename, ctime=state['ctime'], iter=state['iter'], ub=np.concatenate(state['ub_split'], axis=2))

def read_data(namerestart):
    data = np.load(namerestart)
    ub_full = data['ub']
    ub_split = np.stack(np.split(ub_full, num_dev, axis=2))
    return {'ctime': data['ctime'].item(), 'iter': data['iter'].item(), 'ub_split': ub_split}


def compute_1d_spectrum_cpu(mid_line_data, x_1d_stretched, Lx_val, npx_val):
    u_mid = mid_line_data[:, 1]
    v_mid = mid_line_data[:, 2]
    w_mid = mid_line_data[:, 3]  
    t_mid = mid_line_data[:, 4]

    x_uniform = np.linspace(0, Lx_val, npx_val)

    u_uni = np.interp(x_uniform, x_1d_stretched, u_mid)
    v_uni = np.interp(x_uniform, x_1d_stretched, v_mid)
    w_uni = np.interp(x_uniform, x_1d_stretched, w_mid)
    t_uni = np.interp(x_uniform, x_1d_stretched, t_mid)

    u_fluc = u_uni - np.mean(u_uni)
    v_fluc = v_uni - np.mean(v_uni)
    w_fluc = w_uni - np.mean(w_uni)
    t_fluc = t_uni - np.mean(t_uni)

    window = np.hanning(npx_val)
    u_win = u_fluc * window
    v_win = v_fluc * window
    w_win = w_fluc * window
    t_win = t_fluc * window

    u_fft = np.fft.fft(u_win)
    v_fft = np.fft.fft(v_win)
    w_fft = np.fft.fft(w_win)
    t_fft = np.fft.fft(t_win)

    half_N = npx_val // 2
    Eu = 0.5 * (np.abs(u_fft[:half_N])**2 + np.abs(v_fft[:half_N])**2 + np.abs(w_fft[:half_N])**2) / npx_val
    ET = np.abs(t_fft[:half_N])**2 / npx_val

    return Eu, ET

def write_spectrum_log(script_dir, ctime, k_axis, Eu_avg, ET_avg):
    filename = os.path.join(script_dir, "Energy_Spectrum_TimeAvg.dat")
    with open(filename, "w") as f:
        f.write(f'# 3D Centerline Time-averaged Spectrum at t={ctime:.2f}\n')
        f.write('VARIABLES = "Wavenumber_k", "Eu_Kinetic", "ET_Thermal"\n')
        for i in range(1, len(k_axis)):
            f.write(f"{k_axis[i]:.8e}\t{Eu_avg[i]:.8e}\t{ET_avg[i]:.8e}\n")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    restart = 1
    namerestart = os.path.join(script_dir, 'RESTART0000999.dat.npz')

    sm_list = []
    am_list = []
    for pid in range(num_dev):
        sm_temp = np.ones((npx+8, npy+8, npz_local+8, 5), dtype=np.float32)
        am_temp = np.zeros((npx+8, npy+8, npz_local+8, 5), dtype=np.float32)
        
        sm_temp[0:4, :, :, 1:4] = -1.0
        sm_temp[-4:, :, :, 1:4] = -1.0
        sm_temp[:, 0:4, :, 1:4] = -1.0
        sm_temp[:, -4:, :, 1:4] = -1.0
        if pid == 0:
            sm_temp[:, :, 0:4, 1:4] = -1.0
            sm_temp[:, :, 0:4, 4] = -1.0
            am_temp[:, :, 0:4, 4] = 2.0
        if pid == num_dev - 1:
            sm_temp[:, :, -4:, 1:4] = -1.0
            sm_temp[:, :, -4:, 4] = -1.0
            
        sm_list.append(jax.device_put(sm_temp, device=jax.devices()[pid]))
        am_list.append(jax.device_put(am_temp, device=jax.devices()[pid]))
        del sm_temp, am_temp

    sm_split = jax.device_put_sharded(sm_list, jax.devices())
    am_split = jax.device_put_sharded(am_list, jax.devices())
    del sm_list, am_list 

    pid_array = np.arange(num_dev, dtype=np.int32)

    ub_list = []
    ub_tec_list = []
    if restart == 0:
        for pid in range(num_dev):
            z_start = pid * npz_local
            z_end = (pid + 1) * npz_local
            z_local_coords = z_1d[z_start:z_end]
            
            ub_temp = np.zeros((npx, npy, npz_local, 5), dtype=np.float32)
            
            base_temp = (1.0 - z_local_coords / Lz).reshape(1, 1, npz_local)
            key = jax.random.PRNGKey(42 + pid) 

            noise = jax.random.uniform(key, shape=(npx, npy, npz_local), minval=-0.05, maxval=0.05)
            

            ub_temp[..., 4] = base_temp + noise

            if pid == 0:
                ub_temp[:, :, 0, 1:4] = 0.0    
                ub_temp[:, :, 0, 4] = 1.0      
            if pid == num_dev - 1:
                ub_temp[:, :, -1, 1:4] = 0.0   
                ub_temp[:, :, -1, 4] = 0.0    
                

            ub_temp[0, :, :, 1:4] = 0.0; ub_temp[-1, :, :, 1:4] = 0.0
            ub_temp[:, 0, :, 1:4] = 0.0; ub_temp[:, -1, :, 1:4] = 0.0
                      
            ub_list.append(jax.device_put(ub_temp, device=jax.devices()[pid]))
            ub_tec_list.append(jax.device_put(np.zeros_like(ub_temp), device=jax.devices()[pid]))
            del ub_temp
            
        state = {'ctime': 0.0, 'iter': 0}
        state['ub_split'] = jax.device_put_sharded(ub_list, jax.devices())
        ub_tec_split = jax.device_put_sharded(ub_tec_list, jax.devices())
        del ub_list, ub_tec_list
        
    else:
        state = read_and_interpolate_data(namerestart, x_1d, y_1d, z_1d)
        # state = read_data(namerestart)
        state['ub_split'] = jax.device_put(state['ub_split']) 
        ub_tec_split = jax.device_put(np.zeros_like(state['ub_split'])) 
        import gc
        gc.collect() 

    N_STEPS = 100 

    t_start_avg = 1100.0  
    t_end_avg = 1500.0    
    steps_accumulated = 0
    avg_field_dumped = False 
    
    nu_bot_sum  = 0.0
    nu_top_sum  = 0.0
    nu_vol_sum  = 0.0
    nu_kin_sum  = 0.0
    nu_th_sum   = 0.0
    nu_wall_sum = 0.0
    nu_count    = 0
    nu_count = 0
    re_rms_sum  = 0.0
    eps_moms_sum = np.zeros(4)
    ens_moms_sum = np.zeros(4)


    state['ub_split'], ub_tec_split, _, _, _, _, _, _, _, _, _, _, _ = pmap_advance_fused(
        state['ub_split'], ub_tec_split, pid_array, 
        Jx, Jxx, Jy, Jyy, Jz_split, Jzz_split, 2
    )
    print("Compilation complete. Firing on all cylinders!")

    maxiter = int(tf / dt)

    dx_uni = Lx / (npx - 1)
    half_N = npx // 2
    k_axis = np.fft.fftfreq(npx, d=dx_uni)[:half_N] * 2 * np.pi 
    
    Eu_sum = np.zeros(half_N)
    ET_sum = np.zeros(half_N)
    spec_count = 0
    t_spec_start = 1100.0 
    
    ipy_mid = npy // 2
    ipz_mid = npz // 2
    pid_mid = ipz_mid // npz_local      
    z_loc_mid = ipz_mid % npz_local    

    while state['ctime'] <= tf:
        old_ctime = state['ctime']

        ub_new_split, tec_new_split, r_norm, nu_bot, nu_top, nu_vol, nu_kin, nu_th, nu_wall, g_vol, g_eps_moms, g_ens_moms, g_re_rms = pmap_advance_fused(
            state['ub_split'], ub_tec_split, pid_array, 
            Jx, Jxx, Jy, Jyy, Jz_split, Jzz_split, N_STEPS
        )
        
        state['ctime'] += N_STEPS * dt
        state['iter'] += N_STEPS
        state['ub_split'] = ub_new_split
        
        if t_start_avg <= old_ctime < t_end_avg:
            ub_tec_split = tec_new_split
            steps_accumulated += N_STEPS
        else:
            ub_tec_split = jnp.zeros_like(state['ub_split'])


        res_v = r_norm[0].item()
        i_bot  = nu_bot[0].item()
        i_top  = nu_top[0].item()
        i_vol  = nu_vol[0].item()
        i_kin  = nu_kin[0].item()
        i_th   = nu_th[0].item()
        i_wall = nu_wall[0].item()
        i_re   = g_re_rms[0].item()         

        if state['ctime'] >= t_start_avg:
            nu_bot_sum  += i_bot
            nu_top_sum  += i_top
            nu_vol_sum  += i_vol
            nu_kin_sum  += i_kin
            nu_th_sum   += i_th
            nu_wall_sum += i_wall
            nu_count    += 1
            re_rms_sum  += i_re  
            a_bot  = nu_bot_sum  / nu_count
            a_top  = nu_top_sum  / nu_count
            a_vol  = nu_vol_sum  / nu_count
            a_kin  = nu_kin_sum  / nu_count
            a_th   = nu_th_sum   / nu_count
            a_wall = nu_wall_sum / nu_count
            a_re   = re_rms_sum / nu_count 
        else:
            a_bot  = 0.0; a_top  = 0.0
            a_vol  = 0.0; a_kin  = 0.0
            a_th   = 0.0; a_wall = 0.0
            a_re   = 0.0   
        print(f"iter={state['iter']}, t={state['ctime']:.3f}, res={res_v:.3e} | "
              f"Vol Nu: {i_vol:.4f} | Kin Nu: {i_kin:.4f} | Re_rms: {i_re:.1f}")
        
        write_nusselt_log(script_dir, state['iter'], state['ctime'],
                        i_bot, a_bot, i_top, a_top,
                        i_vol, a_vol, i_kin, a_kin,
                        i_th,  a_th,  i_wall, a_wall,
                        i_re,  a_re)  

        vol_total = g_vol[0].item()
        eps_integrals = np.asarray(g_eps_moms[0])
        ens_integrals = np.asarray(g_ens_moms[0])

        avg_eps_inst = eps_integrals[0] / vol_total
        avg_ens_inst = ens_integrals[0] / vol_total


        eps_inst = np.array([
            avg_eps_inst,
            (eps_integrals[1] / vol_total) / (avg_eps_inst**2 + 1e-16),
            (eps_integrals[2] / vol_total) / (avg_eps_inst**3 + 1e-16),
            (eps_integrals[3] / vol_total) / (avg_eps_inst**4 + 1e-16)
        ])
        
        ens_inst = np.array([
            avg_ens_inst,
            (ens_integrals[1] / vol_total) / (avg_ens_inst**2 + 1e-16),
            (ens_integrals[2] / vol_total) / (avg_ens_inst**3 + 1e-16),
            (ens_integrals[3] / vol_total) / (avg_ens_inst**4 + 1e-16)
        ])


        if state['ctime'] >= t_start_avg:
            eps_moms_sum += eps_inst
            ens_moms_sum += ens_inst
            eps_avg = eps_moms_sum / nu_count
            ens_avg = ens_moms_sum / nu_count
        else:
            eps_avg = np.zeros(4)
            ens_avg = np.zeros(4)


        if (state['iter'] % 1000000 == 0) or (state['iter'] >= maxiter):
            write_data(script_dir, state) 
            write_solu_tecplot(script_dir, state)
            if state['iter'] >= maxiter:
                break

        if state['iter'] % 100 == 0 and state['ctime'] >= t_spec_start:
            mid_line_data = np.asarray(state['ub_split'][pid_mid][:, ipy_mid, z_loc_mid, :])
            
            Eu_inst, ET_inst = compute_1d_spectrum_cpu(mid_line_data, x_1d, Lx, npx)
            
            Eu_sum += Eu_inst
            ET_sum += ET_inst
            spec_count += 1

        if (state['iter'] % 200000 == 0) and (spec_count > 0):
            Eu_avg = Eu_sum / spec_count
            ET_avg = ET_sum / spec_count
            write_spectrum_log(script_dir, state['ctime'], k_axis, Eu_avg, ET_avg)
            print(f"   >>> 3D Centerline Energy spectrum updated and saved.")


CONTROL_SENSOR_Z = 1.0 / (2.0 * NU0_REFERENCE)
CONTROL_SENSOR_K1 = int(np.searchsorted(z_1d, CONTROL_SENSOR_Z))
CONTROL_SENSOR_K0 = CONTROL_SENSOR_K1 - 1
CONTROL_SENSOR_ALPHA = float(
    (CONTROL_SENSOR_Z - z_1d[CONTROL_SENSOR_K0])
    / (z_1d[CONTROL_SENSOR_K1] - z_1d[CONTROL_SENSOR_K0])
)
CONTROL_SENSOR_PID = CONTROL_SENSOR_K0 // npz_local
CONTROL_SENSOR_LOCAL_K0 = CONTROL_SENSOR_K0 % npz_local
CONTROL_SENSOR_LOCAL_K1 = CONTROL_SENSOR_K1 % npz_local
CONTROL_LOG_BLOCKS = int(round(CONTROL_LOG_TIME / CONTROL_ACTION_TIME))


@jit
def enforce_controlled_walls(local_ub, pid, wall_temperature):
    """Impose the controlled lower temperature and fixed upper temperature."""

    def lower_wall(u):
        u = u.at[:, :, 0, 1:4].set(0.0)
        return u.at[:, :, 0, 4].set(wall_temperature)

    def upper_wall(u):
        u = u.at[:, :, -1, 1:4].set(0.0)
        return u.at[:, :, -1, 4].set(0.0)

    local_ub = jax.lax.cond(pid == 0, lower_wall, lambda u: u, local_ub)
    return jax.lax.cond(pid == num_dev - 1, upper_wall, lambda u: u, local_ub)


@jit
def solve_zero_mean_threshold(signal, beta, weights):
    """Solve area_mean(tanh(beta * (signal - threshold))) = 0.

    The bracketed primal solve is wrapped in custom_root so automatic
    differentiation uses the implicit derivative of the converged constraint.
    """

    weight_sum = jnp.sum(weights)

    def equation(threshold):
        response = jnp.tanh(beta * (signal - threshold))
        return jnp.sum(weights * response) / weight_sum

    def solve(equation_fn, initial_guess):
        del initial_guess
        lower = jnp.min(signal) - 2.0
        upper = jnp.max(signal) + 2.0

        def bisect(_, bracket):
            lo, hi = bracket
            mid = 0.5 * (lo + hi)
            value = equation_fn(mid)
            lo = jnp.where(value > 0.0, mid, lo)
            hi = jnp.where(value > 0.0, hi, mid)
            return lo, hi

        lower, upper = jax.lax.fori_loop(0, 48, bisect, (lower, upper))
        return 0.5 * (lower + upper)

    def tangent_solve(linear_fn, rhs):
        slope = linear_fn(jnp.ones_like(rhs))
        safe_slope = jnp.where(
            jnp.abs(slope) > 1.0e-12,
            slope,
            jnp.where(slope < 0.0, -1.0e-12, 1.0e-12),
        )
        return rhs / safe_slope

    return jax.lax.custom_root(
        equation, jnp.asarray(0.0, dtype=signal.dtype), solve, tangent_solve
    )


def build_literature_control(local_ub, pid, Jx_metric, Jy_metric, parameters):
    """Construct the Zhou-Zhu simplified feedback control on every device."""
    amplitude = jnp.clip(parameters[0], 0.0, 1.0)
    beta = jnp.clip(parameters[1], 1.0, 50.0)

    def read_sensor_plane(u):
        theta0 = u[:, :, CONTROL_SENSOR_LOCAL_K0, 4]
        theta1 = u[:, :, CONTROL_SENSOR_LOCAL_K1, 4]
        return (1.0 - CONTROL_SENSOR_ALPHA) * theta0 + CONTROL_SENSOR_ALPHA * theta1

    local_plane = jax.lax.cond(
        pid == CONTROL_SENSOR_PID,
        read_sensor_plane,
        lambda u: jnp.zeros((npx, npy), dtype=u.dtype),
        local_ub,
    )
    sensor_plane = jax.lax.psum(local_plane, axis_name="z")

    weights = (
        Jx_metric[:, 0, 0, 0].reshape(npx, 1)
        * Jy_metric[0, :, 0, 0].reshape(1, npy)
    )
    weight_sum = jnp.sum(weights)
    sensor_mean = jnp.sum(weights * sensor_plane) / weight_sum
    sensor_fluctuation = sensor_plane - sensor_mean

    threshold = solve_zero_mean_threshold(sensor_fluctuation, beta, weights)
    wall_action = amplitude * jnp.tanh(beta * (sensor_fluctuation - threshold))
    action_mean = jnp.sum(weights * wall_action) / weight_sum
    action_rms = jnp.sqrt(jnp.sum(weights * wall_action**2) / weight_sum)
    wall_temperature = 1.0 + wall_action

    diagnostics = (
        threshold,
        action_mean,
        jnp.min(wall_action),
        jnp.max(wall_action),
        action_rms,
    )
    return wall_temperature, sensor_fluctuation, wall_action, diagnostics


def advance_one_control_action(
    local_ub, pid, wall_temperature, Jx_metric, Jxx_metric,
    Jy_metric, Jyy_metric, Jz_local, Jzz_local
):
    a30, a32 = 0.355909775, 0.644090224
    a40, a43 = 0.367933791, 0.632066208
    a52, a54 = 0.237593836, 0.762406163
    b10, b21 = 0.377268915, 0.377268915
    b32, b43, b54 = 0.242995220, 0.238458932, 0.287632146

    def residual(u):
        u = enforce_controlled_walls(u, pid, wall_temperature)
        return calc_local_resid(
            u, pid, Jx_metric, Jxx_metric, Jy_metric, Jyy_metric,
            Jz_local, Jzz_local
        )

    def one_step(_, u):
        u = enforce_controlled_walls(u, pid, wall_temperature)
        u1 = u
        u = enforce_controlled_walls(u + b10 * dt * residual(u), pid, wall_temperature)
        u = enforce_controlled_walls(u + b21 * dt * residual(u), pid, wall_temperature)
        u2 = u
        u = enforce_controlled_walls(
            a30 * u1 + a32 * u + b32 * dt * residual(u), pid, wall_temperature
        )
        u = enforce_controlled_walls(
            a40 * u1 + a43 * u + b43 * dt * residual(u), pid, wall_temperature
        )
        u = a52 * u2 + a54 * u + b54 * dt * residual(u)
        return enforce_controlled_walls(u, pid, wall_temperature)

    return jax.lax.fori_loop(0, CONTROL_ACTION_STEPS, one_step, local_ub)


def controlled_wall_nusselt(local_ub, pid, Jx_metric, Jy_metric, Jz_local):
    theta = local_ub[..., 4]
    weights = (
        Jx_metric[:, 0, 0, 0].reshape(npx, 1)
        * Jy_metric[0, :, 0, 0].reshape(1, npy)
    )
    weight_sum = jnp.sum(weights)

    grad_bottom = (
        -25.0 * theta[:, :, 0]
        + 48.0 * theta[:, :, 1]
        - 36.0 * theta[:, :, 2]
        + 16.0 * theta[:, :, 3]
        - 3.0 * theta[:, :, 4]
    ) / (12.0 * dzeta * Jz_local[0, 0, 0, 0])
    local_bottom = jax.lax.cond(
        pid == 0,
        lambda _: jnp.sum(weights * (-grad_bottom)) / weight_sum,
        lambda _: jnp.asarray(0.0, dtype=theta.dtype),
        operand=None,
    )

    grad_top = (
        25.0 * theta[:, :, -1]
        - 48.0 * theta[:, :, -2]
        + 36.0 * theta[:, :, -3]
        - 16.0 * theta[:, :, -4]
        + 3.0 * theta[:, :, -5]
    ) / (12.0 * dzeta * Jz_local[0, 0, -1, 0])
    local_top = jax.lax.cond(
        pid == num_dev - 1,
        lambda _: jnp.sum(weights * (-grad_top)) / weight_sum,
        lambda _: jnp.asarray(0.0, dtype=theta.dtype),
        operand=None,
    )
    return (
        jax.lax.psum(local_bottom, axis_name="z"),
        jax.lax.psum(local_top, axis_name="z"),
    )


@partial(
    jax.pmap,
    axis_name="z",
    static_broadcasted_argnums=(9, 10),
    in_axes=(0, 0, None, None, None, None, 0, 0, None, None, None),
)
def pmap_control_window(
    local_ub, pid, Jx_metric, Jxx_metric, Jy_metric, Jyy_metric,
    Jz_local, Jzz_local, parameters, n_blocks, reward_start_block
):
    """Run a closed-loop control window entirely on the devices."""
    zero = jnp.asarray(0.0, dtype=local_ub.dtype)
    initial_plane = jnp.zeros((npx, npy), dtype=local_ub.dtype)
    initial_carry = (
        local_ub, zero, zero, zero,
        zero, zero, zero, zero, zero,
        initial_plane, initial_plane,
    )

    def one_block(block, carry):
        (
            u, nu_bottom_sum, nu_top_sum, sample_count,
            _, _, _, _, _, _, _,
        ) = carry
        wall_temperature, sensor_fluctuation, wall_action, diagnostics = (
            build_literature_control(u, pid, Jx_metric, Jy_metric, parameters)
        )
        u = advance_one_control_action(
            u, pid, wall_temperature, Jx_metric, Jxx_metric,
            Jy_metric, Jyy_metric, Jz_local, Jzz_local
        )
        nu_bottom, nu_top = controlled_wall_nusselt(
            u, pid, Jx_metric, Jy_metric, Jz_local
        )
        include = block >= reward_start_block
        nu_bottom_sum = nu_bottom_sum + jnp.where(include, nu_bottom, 0.0)
        nu_top_sum = nu_top_sum + jnp.where(include, nu_top, 0.0)
        sample_count = sample_count + jnp.where(include, 1.0, 0.0)
        threshold, action_mean, action_min, action_max, action_rms = diagnostics
        return (
            u, nu_bottom_sum, nu_top_sum, sample_count,
            threshold, action_mean, action_min, action_max, action_rms,
            sensor_fluctuation, wall_action,
        )

    carry = jax.lax.fori_loop(0, n_blocks, one_block, initial_carry)
    (
        final_ub, nu_bottom_sum, nu_top_sum, sample_count,
        threshold, action_mean, action_min, action_max, action_rms,
        sensor_fluctuation, wall_action,
    ) = carry
    sample_count = jnp.maximum(sample_count, 1.0)
    mean_nu_bottom = nu_bottom_sum / sample_count
    mean_nu_top = nu_top_sum / sample_count
    enhancement = (mean_nu_bottom - NU0_REFERENCE) / NU0_REFERENCE
    objective = -enhancement
    return (
        final_ub, objective, mean_nu_bottom, mean_nu_top, enhancement,
        threshold, action_mean, action_min, action_max, action_rms,
        sensor_fluctuation, wall_action,
    )


def load_control_restart(script_dir):
    devices = jax.local_devices()
    if len(devices) < num_dev:
        raise RuntimeError(f"This calculation requires {num_dev} local devices, found {len(devices)}")

    restart_path = os.path.join(script_dir, CONTROL_RESTART)
    state = read_data(restart_path)
    host_shards = state["ub_split"]
    device_shards = [
        jax.device_put(host_shards[i], device=devices[i]) for i in range(num_dev)
    ]
    state["ub_split"] = jax.device_put_sharded(device_shards, devices[:num_dev])
    return state


def control_objective(
    parameters, initial_ub, pid_array,
    n_blocks=DIFF_N_BLOCKS,
    reward_start_block=DIFF_REWARD_START_BLOCK,
):
    results = pmap_control_window(
        initial_ub, pid_array, Jx, Jxx, Jy, Jyy, Jz_split, Jzz_split,
        parameters, n_blocks, reward_start_block
    )
    return results[1][0]


def write_control_history_header(filename):
    with open(filename, "w", encoding="utf-8") as stream:
        stream.write(
            "time\tNu_bottom\tNu_top\tNu_bottom_avg\tNu_top_avg\t"
            "enhancement_avg\tT0\taction_mean\taction_min\taction_max\t"
            "action_rms\tamplitude\tbeta\n"
        )


def run_forward_control(parameters, output_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, output_name)
    os.makedirs(output_dir, exist_ok=True)
    state = load_control_restart(script_dir)
    pid_array = jnp.arange(num_dev, dtype=jnp.int32)
    parameters = jnp.asarray(parameters, dtype=jnp.float32)

    start_time = float(state["ctime"])
    averaging_start = start_time + CONTROL_TRANSIENT_TIME
    end_time = averaging_start + CONTROL_AVERAGING_TIME
    next_snapshot = start_time + CONTROL_SNAPSHOT_TIME
    history_file = os.path.join(output_dir, "control_history.dat")
    write_control_history_header(history_file)

    bottom_sum = 0.0
    top_sum = 0.0
    count = 0
    print(
        f"Control sensor z={CONTROL_SENSOR_Z:.8f}, grid indices "
        f"({CONTROL_SENSOR_K0}, {CONTROL_SENSOR_K1}), alpha={CONTROL_SENSOR_ALPHA:.8f}"
    )
    print(
        f"Action update: {CONTROL_ACTION_TIME:g} t0 = {CONTROL_ACTION_STEPS} DNS steps; "
        f"A={float(parameters[0]):.6f}, beta={float(parameters[1]):.6f}"
    )

    while state["ctime"] < end_time - 0.5 * CONTROL_ACTION_TIME:
        results = pmap_control_window(
            state["ub_split"], pid_array, Jx, Jxx, Jy, Jyy,
            Jz_split, Jzz_split, parameters,
            CONTROL_LOG_BLOCKS, 0
        )
        state["ub_split"] = results[0]
        state["ctime"] += CONTROL_LOG_BLOCKS * CONTROL_ACTION_TIME
        state["iter"] += CONTROL_LOG_BLOCKS * CONTROL_ACTION_STEPS

        nu_bottom = float(results[2][0])
        nu_top = float(results[3][0])
        threshold = float(results[5][0])
        action_mean = float(results[6][0])
        action_min = float(results[7][0])
        action_max = float(results[8][0])
        action_rms = float(results[9][0])

        if state["ctime"] >= averaging_start:
            bottom_sum += nu_bottom
            top_sum += nu_top
            count += 1
        bottom_avg = bottom_sum / count if count else np.nan
        top_avg = top_sum / count if count else np.nan
        enhancement_avg = (
            (bottom_avg - NU0_REFERENCE) / NU0_REFERENCE if count else np.nan
        )

        with open(history_file, "a", encoding="utf-8") as stream:
            stream.write(
                f"{state['ctime']:.8f}\t{nu_bottom:.8f}\t{nu_top:.8f}\t"
                f"{bottom_avg:.8f}\t{top_avg:.8f}\t{enhancement_avg:.8e}\t"
                f"{threshold:.8e}\t{action_mean:.8e}\t{action_min:.8e}\t"
                f"{action_max:.8e}\t{action_rms:.8e}\t"
                f"{float(parameters[0]):.8f}\t{float(parameters[1]):.8f}\n"
            )

        if state["ctime"] >= next_snapshot - 0.5 * CONTROL_LOG_TIME:
            np.savez_compressed(
                os.path.join(output_dir, f"control_snapshot_{state['ctime']:.3f}.npz"),
                time=state["ctime"],
                sensor_fluctuation=np.asarray(results[10][0]),
                wall_action=np.asarray(results[11][0]),
                threshold=threshold,
                amplitude=float(parameters[0]),
                beta=float(parameters[1]),
            )
            next_snapshot += CONTROL_SNAPSHOT_TIME

        print(
            f"t={state['ctime']:.3f} Nu_b={nu_bottom:.4f} Nu_t={nu_top:.4f} "
            f"avg_b={bottom_avg:.4f} eta={100.0 * enhancement_avg:.2f}% "
            f"mean(action)={action_mean:.2e}"
        )

    write_data(output_dir, state)
    np.savez(
        os.path.join(output_dir, "control_summary.npz"),
        Nu0=NU0_REFERENCE,
        Nu_bottom_avg=bottom_avg,
        Nu_top_avg=top_avg,
        enhancement=enhancement_avg,
        parameters=np.asarray(parameters),
        sensor_z=CONTROL_SENSOR_Z,
    )
    print(f"Control results written to: {output_dir}")


def forward_mode_gradient(objective_fn, parameters):
    gradients = []
    value = None
    for index in range(parameters.shape[0]):
        direction = jnp.zeros_like(parameters).at[index].set(1.0)
        primal, tangent = jax.jvp(objective_fn, (parameters,), (direction,))
        primal.block_until_ready()
        if value is None:
            value = primal
        gradients.append(tangent)
    return value, jnp.stack(gradients)


def run_gradient_check():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "control_gradient_check")
    os.makedirs(output_dir, exist_ok=True)
    state = load_control_restart(script_dir)
    pid_array = jnp.arange(num_dev, dtype=jnp.int32)
    parameters = jnp.asarray(OPT_INITIAL_PARAMETERS, dtype=jnp.float32)
    objective_fn = lambda p: control_objective(
        p, state["ub_split"], pid_array,
        GRAD_CHECK_N_BLOCKS, GRAD_CHECK_REWARD_START_BLOCK
    )

    value, ad_gradient = forward_mode_gradient(objective_fn, parameters)
    names = ("amplitude", "beta")
    filename = os.path.join(output_dir, "gradient_check.dat")
    with open(filename, "w", encoding="utf-8") as stream:
        stream.write(
            "parameter\tepsilon\tAD_gradient\tFD_gradient\trelative_error\n"
        )
        for index, name in enumerate(names):
            ad_value = float(ad_gradient[index])
            for epsilon in FD_EPSILON_SWEEP[index]:
                direction = jnp.zeros_like(parameters).at[index].set(float(epsilon))
                plus = objective_fn(parameters + direction)
                minus = objective_fn(parameters - direction)
                plus.block_until_ready()
                minus.block_until_ready()
                fd_value = float((plus - minus) / (2.0 * float(epsilon)))
                scale = max(abs(ad_value), abs(fd_value), 1.0e-12)
                error = abs(ad_value - fd_value) / scale
                stream.write(
                    f"{name}\t{epsilon:.12e}\t{ad_value:.12e}\t"
                    f"{fd_value:.12e}\t{error:.12e}\n"
                )
                print(
                    f"{name}, eps={epsilon:.1e}: AD={ad_value:.6e}, "
                    f"FD={fd_value:.6e}, rel.err={error:.3e}"
                )
    print(f"Objective={float(value):.8e}; gradient check written to {filename}")


def run_parameter_optimization():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "control_optimization")
    os.makedirs(output_dir, exist_ok=True)
    state = load_control_restart(script_dir)
    pid_array = jnp.arange(num_dev, dtype=jnp.int32)
    parameters = jnp.asarray(OPT_INITIAL_PARAMETERS, dtype=jnp.float32)
    learning_rate = jnp.asarray(OPT_LEARNING_RATE, dtype=jnp.float32)
    objective_fn = lambda p: control_objective(p, state["ub_split"], pid_array)

    first_moment = jnp.zeros_like(parameters)
    second_moment = jnp.zeros_like(parameters)
    beta1, beta2 = 0.9, 0.999
    history_file = os.path.join(output_dir, "optimization_history.dat")
    with open(history_file, "w", encoding="utf-8") as stream:
        stream.write(
            "iteration\tobjective\tenhancement\tamplitude\tbeta\t"
            "gradient_amplitude\tgradient_beta\n"
        )

    for iteration in range(1, OPT_ITERATIONS + 1):
        value, gradient = forward_mode_gradient(objective_fn, parameters)
        with open(history_file, "a", encoding="utf-8") as stream:
            stream.write(
                f"{iteration}\t{float(value):.12e}\t{-float(value):.12e}\t"
                f"{float(parameters[0]):.8f}\t{float(parameters[1]):.8f}\t"
                f"{float(gradient[0]):.12e}\t{float(gradient[1]):.12e}\n"
            )
        print(
            f"iteration={iteration} J={float(value):.6e} "
            f"eta={-100.0 * float(value):.3f}% A={float(parameters[0]):.5f} "
            f"beta={float(parameters[1]):.5f} grad={np.asarray(gradient)}"
        )

        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * gradient**2
        first_hat = first_moment / (1.0 - beta1**iteration)
        second_hat = second_moment / (1.0 - beta2**iteration)
        parameters = parameters - learning_rate * first_hat / (jnp.sqrt(second_hat) + 1.0e-8)
        parameters = parameters.at[0].set(jnp.clip(parameters[0], 0.05, 1.0))
        parameters = parameters.at[1].set(jnp.clip(parameters[1], 1.0, 50.0))

    final_value = objective_fn(parameters)
    final_value.block_until_ready()
    np.savez(
        os.path.join(output_dir, "optimized_control_parameters.npz"),
        amplitude=float(parameters[0]),
        beta=float(parameters[1]),
        objective=float(final_value),
        enhancement=-float(final_value),
    )
    print(
        f"Final J={float(final_value):.8e}, enhancement={-100.0 * float(final_value):.3f}%"
    )
    print(f"Optimization results written to: {output_dir}")


def main_literature_control():
    if not np.isclose(CONTROL_ACTION_STEPS * dt, CONTROL_ACTION_TIME):
        raise ValueError("CONTROL_ACTION_TIME must be an integer multiple of dt")
    if not (0 <= DIFF_REWARD_START_BLOCK < DIFF_N_BLOCKS):
        raise ValueError("Invalid differentiable reward window")
    if not (0 <= GRAD_CHECK_REWARD_START_BLOCK < GRAD_CHECK_N_BLOCKS):
        raise ValueError("Invalid gradient-check reward window")
    if CONTROL_SENSOR_K1 // npz_local != CONTROL_SENSOR_PID:
        raise ValueError("The two sensor interpolation planes cross a device boundary")

    print(f"RB_CONTROL_MODE={CONTROL_MODE}")
    if CONTROL_MODE == "reference":
        run_forward_control(
            (CONTROL_AMPLITUDE_REFERENCE, CONTROL_BETA_REFERENCE),
            "control_reference",
        )
    elif CONTROL_MODE == "gradient_check":
        run_gradient_check()
    elif CONTROL_MODE == "optimize":
        run_parameter_optimization()
    elif CONTROL_MODE == "optimized_validation":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parameter_file = os.path.join(
            script_dir, "control_optimization", "optimized_control_parameters.npz"
        )
        data = np.load(parameter_file)
        run_forward_control(
            (float(data["amplitude"]), float(data["beta"])),
            "control_optimized_validation",
        )
    else:
        raise ValueError(
            "RB_CONTROL_MODE must be reference, gradient_check, optimize, "
            "or optimized_validation"
        )


if __name__ == "__main__":
    main_literature_control()
