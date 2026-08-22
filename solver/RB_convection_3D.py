import jax
import jax.numpy as jnp
from jax import jit, random
from functools import partial
import os
import numpy as np 
import scipy.interpolate as spi 
jax.config.update('jax_default_matmul_precision', 'float32')

npx, npy, npz = 456 ,456, 456  
num_dev = 4                
Lx, Ly, Lz = 1.0, 1.0, 1.0  
gamma_xy = 1.1             
gamma_z = 1.1           
dt = 1e-4               
tf = 2500               
Ma = 0.1
Pr = 0.7
Ra = 1e8
FLUX_ORDER = 7
VISC_ORDER = 8            
DIR_BND_ORDER = 4           
NEU_BND_ORDER = 4          
MAX_BND_ORDER = max(DIR_BND_ORDER, NEU_BND_ORDER)

restart = 1           
change = 0             
t_spec_start = 2000.0    
t_start_avg = 2000.0    
t_end_avg = 2500.0      

script_dir = os.path.dirname(os.path.abspath(__file__))
namerestart = os.path.join(script_dir, 'RESTART0001800.dat.npz')

npx_old, npy_old, npz_old = 384, 384, 384   
gamma_xy_old, gamma_z_old = 1.2, 1.2        
Lx_old, Ly_old, Lz_old = 1.0, 1.0, 1.0      

Ma2_inv = 1.0 / (Ma**2)
npz_local = npz // num_dev    
dxi = 1.0 / (npx - 1)
deta = 1.0 / (npy - 1)
dzeta = 1.0 / (npz - 1)
xi_1d = np.linspace(0, 1, npx)
eta_1d = np.linspace(0, 1, npy)
zeta_1d = np.linspace(0, 1, npz)
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

if FLUX_ORDER == 3:
    w0, w1, w2 = -1.0/6.0, 5.0/6.0, 2.0/6.0
    @jit
    def compute_flux_x(fp, fm):
        N = fp.shape[0] - 7 
        flux_p = w0*fp[2:N+2] + w1*fp[3:N+3] + w2*fp[4:N+4]
        flux_m = w2*fm[3:N+3] + w1*fm[4:N+4] + w0*fm[5:N+5]
        return flux_p + flux_m

    @jit
    def compute_flux_y(fp, fm):
        N = fp.shape[1] - 7 
        flux_p = w0*fp[:, 2:N+2] + w1*fp[:, 3:N+3] + w2*fp[:, 4:N+4]
        flux_m = w2*fm[:, 3:N+3] + w1*fm[:, 4:N+4] + w0*fm[:, 5:N+5]
        return flux_p + flux_m

    @jit
    def compute_flux_z(fp, fm):
        N = fp.shape[2] - 7 
        flux_p = w0*fp[:, :, 2:N+2] + w1*fp[:, :, 3:N+3] + w2*fp[:, :, 4:N+4]
        flux_m = w2*fm[:, :, 3:N+3] + w1*fm[:, :, 4:N+4] + w0*fm[:, :, 5:N+5]
        return flux_p + flux_m

elif FLUX_ORDER == 5:
    u5_c1, u5_c2, u5_c3, u5_c4, u5_c5 = 2.0/60.0, -13.0/60.0, 47.0/60.0, 27.0/60.0, -3.0/60.0
    @jit
    def compute_flux_x(fp, fm):
        N = fp.shape[0] - 7 
        flux_p = u5_c1*fp[1:N+1] + u5_c2*fp[2:N+2] + u5_c3*fp[3:N+3] + u5_c4*fp[4:N+4] + u5_c5*fp[5:N+5]
        flux_m = u5_c5*fm[2:N+2] + u5_c4*fm[3:N+3] + u5_c3*fm[4:N+4] + u5_c2*fm[5:N+5] + u5_c1*fm[6:N+6]
        return flux_p + flux_m

    @jit
    def compute_flux_y(fp, fm):
        N = fp.shape[1] - 7 
        flux_p = u5_c1*fp[:, 1:N+1] + u5_c2*fp[:, 2:N+2] + u5_c3*fp[:, 3:N+3] + u5_c4*fp[:, 4:N+4] + u5_c5*fp[:, 5:N+5]
        flux_m = u5_c5*fm[:, 2:N+2] + u5_c4*fm[:, 3:N+3] + u5_c3*fm[:, 4:N+4] + u5_c2*fm[:, 5:N+5] + u5_c1*fm[:, 6:N+6]
        return flux_p + flux_m

    @jit
    def compute_flux_z(fp, fm):
        N = fp.shape[2] - 7 
        flux_p = u5_c1*fp[:, :, 1:N+1] + u5_c2*fp[:, :, 2:N+2] + u5_c3*fp[:, :, 3:N+3] + u5_c4*fp[:, :, 4:N+4] + u5_c5*fp[:, :, 5:N+5]
        flux_m = u5_c5*fm[:, :, 2:N+2] + u5_c4*fm[:, :, 3:N+3] + u5_c3*fm[:, :, 4:N+4] + u5_c2*fm[:, :, 5:N+5] + u5_c1*fm[:, :, 6:N+6]
        return flux_p + flux_m

elif FLUX_ORDER == 7:
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

def get_dirichlet_weights(pts, max_pts):
    W = np.zeros((4, max_pts))
    powers = list(range(pts))
    A = np.array([[float(i**p) for p in powers] for i in range(pts)])
    for k in range(1, 5):
        v = np.array([float((-k)**p) for p in powers])
        W[k-1, :pts] = np.linalg.solve(A.T, v)
    return W

def get_neumann_weights(pts, max_pts):
    W = np.zeros((4, max_pts))
    powers = [0] + list(range(2, pts+1))
    A = np.array([[float(i**p) for p in powers] for i in range(pts)])
    for k in range(1, 5):
        v = np.array([float((-k)**p) for p in powers])
        W[k-1, :pts] = np.linalg.solve(A.T, v)
    return W

W_D_jax = jnp.array(get_dirichlet_weights(DIR_BND_ORDER, MAX_BND_ORDER), dtype=jnp.float32)
W_N_jax = jnp.array(get_neumann_weights(NEU_BND_ORDER, MAX_BND_ORDER), dtype=jnp.float32)

@jit
def extrapolate_x(f_in, W):
    return jnp.tensordot(W, f_in, axes=(1, 0))
@jit
def extrapolate_y(f_in, W):
    return jnp.moveaxis(jnp.tensordot(W, f_in, axes=(1, 1)), 0, 1)
@jit
def extrapolate_z(f_in, W):
    return jnp.moveaxis(jnp.tensordot(W, f_in, axes=(1, 2)), 0, 2)

@jit
def apply_boundaries_and_halos(local_ub, pid):
    ubs = jnp.pad(local_ub, ((4,4), (4,4), (4,4), (0,0)), mode='constant')
    f_in = ubs[4:4+MAX_BND_ORDER, 4:-4, 4:-4, :]
    fm_D = extrapolate_x(f_in[..., 1:4], W_D_jax)
    fm_P = extrapolate_x(f_in[..., 0], W_N_jax)
    fm_T = extrapolate_x(f_in[..., 4], W_N_jax)
    ubs = ubs.at[0:4, 4:-4, 4:-4, 1:4].set(fm_D[::-1])
    ubs = ubs.at[0:4, 4:-4, 4:-4, 0].set(fm_P[::-1])
    ubs = ubs.at[0:4, 4:-4, 4:-4, 4].set(fm_T[::-1])
    f_in = ubs[-5:-5-MAX_BND_ORDER:-1, 4:-4, 4:-4, :]
    fm_D = extrapolate_x(f_in[..., 1:4], W_D_jax)
    fm_P = extrapolate_x(f_in[..., 0], W_N_jax)
    fm_T = extrapolate_x(f_in[..., 4], W_N_jax)
    ubs = ubs.at[-4:, 4:-4, 4:-4, 1:4].set(fm_D)
    ubs = ubs.at[-4:, 4:-4, 4:-4, 0].set(fm_P)
    ubs = ubs.at[-4:, 4:-4, 4:-4, 4].set(fm_T)

    f_in = ubs[4:-4, 4:4+MAX_BND_ORDER, 4:-4, :]
    fm_D = extrapolate_y(f_in[..., 1:4], W_D_jax)
    fm_P = extrapolate_y(f_in[..., 0], W_N_jax)
    fm_T = extrapolate_y(f_in[..., 4], W_N_jax)
    ubs = ubs.at[4:-4, 0:4, 4:-4, 1:4].set(fm_D[:, ::-1, ...])
    ubs = ubs.at[4:-4, 0:4, 4:-4, 0].set(fm_P[:, ::-1, ...])
    ubs = ubs.at[4:-4, 0:4, 4:-4, 4].set(fm_T[:, ::-1, ...])
    f_in = ubs[4:-4, -5:-5-MAX_BND_ORDER:-1, 4:-4, :]
    fm_D = extrapolate_y(f_in[..., 1:4], W_D_jax)
    fm_P = extrapolate_y(f_in[..., 0], W_N_jax)
    fm_T = extrapolate_y(f_in[..., 4], W_N_jax)
    ubs = ubs.at[4:-4, -4:, 4:-4, 1:4].set(fm_D)
    ubs = ubs.at[4:-4, -4:, 4:-4, 0].set(fm_P)
    ubs = ubs.at[4:-4, -4:, 4:-4, 4].set(fm_T)
    send_top, send_bot = ubs[:, :, -8:-4, :], ubs[:, :, 4:8, :]
    recv_bot = jax.lax.ppermute(send_top, axis_name='z', perm=[(i, (i+1)%num_dev) for i in range(num_dev)])
    recv_top = jax.lax.ppermute(send_bot, axis_name='z', perm=[(i, (i-1)%num_dev) for i in range(num_dev)])
    ubs = ubs.at[:, :, 0:4, :].set(recv_bot)
    ubs = ubs.at[:, :, -4:, :].set(recv_top)

    def apply_z_bot(u):
        f_in = u[4:-4, 4:-4, 4:4+MAX_BND_ORDER, :]
        fm_D = extrapolate_z(f_in[..., 1:5], W_D_jax) # T (4) is Dirichlet here
        fm_P = extrapolate_z(f_in[..., 0], W_N_jax)
        u = u.at[4:-4, 4:-4, 0:4, 1:5].set(fm_D[..., ::-1, :])
        u = u.at[4:-4, 4:-4, 0:4, 0].set(fm_P[..., ::-1])
        return u

    def apply_z_top(u):
        f_in = u[4:-4, 4:-4, -5:-5-MAX_BND_ORDER:-1, :]
        fm_D = extrapolate_z(f_in[..., 1:5], W_D_jax)
        fm_P = extrapolate_z(f_in[..., 0], W_N_jax)
        u = u.at[4:-4, 4:-4, -4:, 1:5].set(fm_D)
        u = u.at[4:-4, 4:-4, -4:, 0].set(fm_P)
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
    
    if VISC_ORDER == 4:
        d2u_dxi2 = (-u_x[2:npx+2] + 16*u_x[3:npx+3] - 30*u_x[4:npx+4] + 16*u_x[5:npx+5] - u_x[6:npx+6]) / (12.0*dxi**2)
        du_dxi = (u_x[2:npx+2] - 8*u_x[3:npx+3] + 8*u_x[5:npx+5] - u_x[6:npx+6]) / (12.0*dxi)
        lap_x = d2u_dxi2/(Jx**2) - du_dxi*Jxx/(Jx**3)

        d2u_deta2 = (-u_y[:,2:npy+2] + 16*u_y[:,3:npy+3] - 30*u_y[:,4:npy+4] + 16*u_y[:,5:npy+5] - u_y[:,6:npy+6]) / (12.0*deta**2)
        du_deta = (u_y[:,2:npy+2] - 8*u_y[:,3:npy+3] + 8*u_y[:,5:npy+5] - u_y[:,6:npy+6]) / (12.0*deta)
        lap_y = d2u_deta2/(Jy**2) - du_deta*Jyy/(Jy**3)

        d2u_dzeta2 = (-u_z[:,:,2:npz_local+2] + 16*u_z[:,:,3:npz_local+3] - 30*u_z[:,:,4:npz_local+4] + 16*u_z[:,:,5:npz_local+5] - u_z[:,:,6:npz_local+6]) / (12.0*dzeta**2)
        du_dzeta = (u_z[:,:,2:npz_local+2] - 8*u_z[:,:,3:npz_local+3] + 8*u_z[:,:,5:npz_local+5] - u_z[:,:,6:npz_local+6]) / (12.0*dzeta)
        lap_z = d2u_dzeta2/(Jz_loc**2) - du_dzeta*Jzz_loc/(Jz_loc**3)

    elif VISC_ORDER == 6:
        d2u_dxi2 = (2*u_x[1:npx+1] - 27*u_x[2:npx+2] + 270*u_x[3:npx+3] - 490*u_x[4:npx+4] + 270*u_x[5:npx+5] - 27*u_x[6:npx+6] + 2*u_x[7:npx+7]) / (180.0*dxi**2)
        du_dxi = (-1*u_x[1:npx+1] + 9*u_x[2:npx+2] - 45*u_x[3:npx+3] + 45*u_x[5:npx+5] - 9*u_x[6:npx+6] + 1*u_x[7:npx+7]) / (60.0*dxi)
        lap_x = d2u_dxi2/(Jx**2) - du_dxi*Jxx/(Jx**3)

        d2u_deta2 = (2*u_y[:,1:npy+1] - 27*u_y[:,2:npy+2] + 270*u_y[:,3:npy+3] - 490*u_y[:,4:npy+4] + 270*u_y[:,5:npy+5] - 27*u_y[:,6:npy+6] + 2*u_y[:,7:npy+7]) / (180.0*deta**2)
        du_deta = (-1*u_y[:,1:npy+1] + 9*u_y[:,2:npy+2] - 45*u_y[:,3:npy+3] + 45*u_y[:,5:npy+5] - 9*u_y[:,6:npy+6] + 1*u_y[:,7:npy+7]) / (60.0*deta)
        lap_y = d2u_deta2/(Jy**2) - du_deta*Jyy/(Jy**3)

        d2u_dzeta2 = (2*u_z[:,:,1:npz_local+1] - 27*u_z[:,:,2:npz_local+2] + 270*u_z[:,:,3:npz_local+3] - 490*u_z[:,:,4:npz_local+4] + 270*u_z[:,:,5:npz_local+5] - 27*u_z[:,:,6:npz_local+6] + 2*u_z[:,:,7:npz_local+7]) / (180.0*dzeta**2)
        du_dzeta = (-1*u_z[:,:,1:npz_local+1] + 9*u_z[:,:,2:npz_local+2] - 45*u_z[:,:,3:npz_local+3] + 45*u_z[:,:,5:npz_local+5] - 9*u_z[:,:,6:npz_local+6] + 1*u_z[:,:,7:npz_local+7]) / (60.0*dzeta)
        lap_z = d2u_dzeta2/(Jz_loc**2) - du_dzeta*Jzz_loc/(Jz_loc**3)

    elif VISC_ORDER == 8:
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

@partial(jax.pmap, axis_name='z', static_broadcasted_argnums=(10,), in_axes=(0, 0, 0, None, None, None, None, 0, 0, 0, None), donate_argnums=(0, 1))
def pmap_advance_fused(loc_ub, loc_tec, pid, Jx, Jxx, Jy, Jyy, Jz_loc, Jzz_loc, bulk_loc, n_steps):
    
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
    local_nu_top = jax.lax.cond(pid == num_dev-1, lambda _: jnp.sum(nu_local_top * w) / jnp.sum(w), lambda _: 0.0, operand=None)
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
    u_phys = ubs_pad[4:-4, 4:-4, 4:-4, 1]
    v_phys = ubs_pad[4:-4, 4:-4, 4:-4, 2]
    w_phys = ubs_pad[4:-4, 4:-4, 4:-4, 3]
    T_phys = ubs_pad[4:-4, 4:-4, 4:-4, 4]

    def get_grads(phi_pad):
        dphi_dxi = jnp.gradient(phi_pad, dxi, axis=0)
        dphi_deta = jnp.gradient(phi_pad, deta, axis=1)
        dphi_dzeta = jnp.gradient(phi_pad, dzeta, axis=2)
        dx = dphi_dxi[4:-4, 4:-4, 4:-4] / Jx[..., 0]
        dy = dphi_deta[4:-4, 4:-4, 4:-4] / Jy[..., 0]
        dz = dphi_dzeta[4:-4, 4:-4, 4:-4] / Jz_loc[..., 0]
        return dx, dy, dz
        
    local_u2_int = jnp.sum((u_phys**2 + v_phys**2 + w_phys**2) * dV_exact)
    global_u2_int = jax.lax.psum(local_u2_int, axis_name='z')
    U_rms = jnp.sqrt(global_u2_int / global_vol)
    global_Re_rms = jnp.sqrt(Ra / Pr) * U_rms
    
    u_x, u_y, u_z = get_grads(ubs_pad[..., 1])
    v_x, v_y, v_z = get_grads(ubs_pad[..., 2])
    w_x, w_y, w_z = get_grads(ubs_pad[..., 3])
    T_x, T_y, T_z = get_grads(ubs_pad[..., 4])

    omega_x = w_y - v_z
    omega_y = u_z - w_x
    omega_z = v_x - u_y
    enstrophy = jnp.sqrt(Pr / Ra) * (omega_x**2 + omega_y**2 + omega_z**2)

    grad_u2 = 2.0*(u_x**2 + v_y**2 + w_z**2) + (u_y + v_x)**2 + (u_z + w_x)**2 + (v_z + w_y)**2
    grad_T2 = T_x**2 + T_y**2 + T_z**2
    eps_u = jnp.sqrt(Pr / Ra) * grad_u2
    eps_T = (1.0 / jnp.sqrt(Ra * Pr)) * grad_T2

    local_eps_moms = jnp.array([jnp.sum(eps_u * dV_exact), jnp.sum((eps_u**2) * dV_exact), jnp.sum((eps_u**3) * dV_exact), jnp.sum((eps_u**4) * dV_exact)])
    local_ens_moms = jnp.array([jnp.sum(enstrophy * dV_exact), jnp.sum((enstrophy**2) * dV_exact), jnp.sum((enstrophy**3) * dV_exact), jnp.sum((enstrophy**4) * dV_exact)])
    local_eps_T_int = jnp.sum(eps_T * dV_exact)

    global_eps_moms = jax.lax.psum(local_eps_moms, axis_name='z')
    global_ens_moms = jax.lax.psum(local_ens_moms, axis_name='z')
    global_eps_T_int = jax.lax.psum(local_eps_T_int, axis_name='z')
    
    avg_eps_u = global_eps_moms[0] / global_vol
    avg_eps_T = global_eps_T_int / global_vol
    global_nu_kin = 1.0 + jnp.sqrt(Ra * Pr) * avg_eps_u
    global_nu_th = jnp.sqrt(Ra * Pr) * avg_eps_T

    dV_bulk = dV_exact * bulk_loc
    local_vol_bulk = jnp.sum(dV_bulk)
    local_eps_moms_bulk = jnp.array([jnp.sum(eps_u * dV_bulk), jnp.sum((eps_u**2) * dV_bulk), jnp.sum((eps_u**3) * dV_bulk), jnp.sum((eps_u**4) * dV_bulk)])
    local_ens_moms_bulk = jnp.array([jnp.sum(enstrophy * dV_bulk), jnp.sum((enstrophy**2) * dV_bulk), jnp.sum((enstrophy**3) * dV_bulk), jnp.sum((enstrophy**4) * dV_bulk)])
    
    global_vol_bulk = jax.lax.psum(local_vol_bulk, axis_name='z')
    global_eps_moms_bulk = jax.lax.psum(local_eps_moms_bulk, axis_name='z')
    global_ens_moms_bulk = jax.lax.psum(local_ens_moms_bulk, axis_name='z')

    return (final_ub, final_tec, global_norm, global_nu_bot, global_nu_top, global_nu_vol, 
            global_nu_kin, global_nu_th, global_nu_wall, global_vol, global_eps_moms, global_ens_moms, 
            global_Re_rms, global_vol_bulk, global_eps_moms_bulk, global_ens_moms_bulk)

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
        
def write_moments_log(script_dir, iter, ctime, eps_i, eps_a, ens_i, ens_a, prefix="Global"):
    log_filename = os.path.join(script_dir, f"{prefix}_HighOrder_Moments_Log.dat")
    write_header = not os.path.exists(log_filename)
    with open(log_filename, "a") as f:
        if write_header:
            f.write('Variables="Iter", "Time", '
                    '"Eps_Mean_Inst", "Eps_Mean_Avg", "Eps_Mom2_Inst", "Eps_Mom2_Avg", '
                    '"Eps_Mom3_Inst", "Eps_Mom3_Avg", "Eps_Mom4_Inst", "Eps_Mom4_Avg", '
                    '"Ens_Mean_Inst", "Ens_Mean_Avg", "Ens_Mom2_Inst", "Ens_Mom2_Avg", '
                    '"Ens_Mom3_Inst", "Ens_Mom3_Avg", "Ens_Mom4_Inst", "Ens_Mom4_Avg"\n')
        f.write(f"{iter:<10d}\t{ctime:.4f}\t"
                f"{eps_i[0]:.6e}\t{eps_a[0]:.6e}\t{eps_i[1]:.6f}\t{eps_a[1]:.6f}\t"
                f"{eps_i[2]:.6f}\t{eps_a[2]:.6f}\t{eps_i[3]:.6f}\t{eps_a[3]:.6f}\t"
                f"{ens_i[0]:.6e}\t{ens_a[0]:.6e}\t{ens_i[1]:.6f}\t{ens_a[1]:.6f}\t"
                f"{ens_i[2]:.6f}\t{ens_a[2]:.6f}\t{ens_i[3]:.6f}\t{ens_a[3]:.6f}\n")

def write_spectrum_log(script_dir, ctime, k_axis, Eu_avg, ET_avg, prefix="Global"):
    filename = os.path.join(script_dir, f"{prefix}_Energy_Spectrum_TimeAvg.dat")
    with open(filename, "w") as f:
        f.write(f'# 3D {prefix} Time-averaged Spectrum at t={ctime:.2f}\n')
        f.write('VARIABLES = "Wavenumber_k", "Eu_Kinetic", "ET_Thermal"\n')
        for i in range(1, len(k_axis)):
            f.write(f"{k_axis[i]:.8e}\t{Eu_avg[i]:.8e}\t{ET_avg[i]:.8e}\n")

def compute_plane_averaged_spectrum_cpu(plane_data, x_1d_stretched, Lx_val, npx_val, z_1d, Lz_val):
    x_uniform = np.linspace(0, Lx_val, npx_val)
    f_interp = spi.interp1d(x_1d_stretched, plane_data, axis=0, kind='cubic')
    plane_uni = f_interp(x_uniform)  
    u_uni, v_uni, w_uni, t_uni = plane_uni[..., 1], plane_uni[..., 2], plane_uni[..., 3], plane_uni[..., 4]
    u_fluc = u_uni - np.mean(u_uni, axis=0)
    v_fluc = v_uni - np.mean(v_uni, axis=0)
    w_fluc = w_uni - np.mean(w_uni, axis=0)
    t_fluc = t_uni - np.mean(t_uni, axis=0)
    window = np.hanning(npx_val).reshape(npx_val, 1)
    u_win, v_win = u_fluc * window, v_fluc * window
    w_win, t_win = w_fluc * window, t_fluc * window
    u_fft = np.fft.fft(u_win, axis=0)
    v_fft = np.fft.fft(v_win, axis=0)
    w_fft = np.fft.fft(w_win, axis=0)
    t_fft = np.fft.fft(t_win, axis=0)
    half_N = npx_val // 2
    Eu_2d = 0.5 * (np.abs(u_fft[:half_N])**2 + np.abs(v_fft[:half_N])**2 + np.abs(w_fft[:half_N])**2) / npx_val
    ET_2d = np.abs(t_fft[:half_N])**2 / npx_val
    Eu_global = np.mean(Eu_2d, axis=1)
    ET_global = np.mean(ET_2d, axis=1)
    bulk_z_idx = np.where((z_1d >= 0.1 * Lz_val) & (z_1d <= 0.9 * Lz_val))[0]
    Eu_main = np.mean(Eu_2d[:, bulk_z_idx], axis=1)
    ET_main = np.mean(ET_2d[:, bulk_z_idx], axis=1)
    return Eu_global, ET_global, Eu_main, ET_main

def write_averaged_tecplot(script_dir, ub_tec_split, steps_accumulated):
    base_name = os.path.join(script_dir, "TEC_AVG_300_to_507")
    grid_file = base_name + ".xyz"
    sol_file = base_name + ".q"
    with open(grid_file, "wb") as fg:
        fg.write(np.array([npx, npy, npz], dtype='<i4').tobytes())
        fg.write(np.tile(x_1d.astype('<f4'), npy * npz).tobytes())
        fg.write(np.tile(np.repeat(y_1d.astype('<f4'), npx), npz).tobytes())
        fg.write(np.repeat(z_1d.astype('<f4'), npx * npy).tobytes())
    with open(sol_file, "wb") as fq:
        fq.write(np.array([npx, npy, npz], dtype='<i4').tobytes())
        fq.write(np.array([Ma, 0.0, Ra, 507.0], dtype='<f4').tobytes())
        for v in range(5):
            for pid in range(num_dev):
                local_var = np.asarray(ub_tec_split[pid][..., v]).astype('<f4')
                local_var = local_var / float(steps_accumulated)
                fq.write(local_var.flatten('F').tobytes())
                del local_var

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
        interp_func = spi.RegularGridInterpolator((x_old, y_old, z_old), ub_old[..., v], bounds_error=False, fill_value=None)
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

def main():
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
        if change == 1:
            state = read_and_interpolate_data(namerestart, x_1d, y_1d, z_1d)
        else:
            state = read_data(namerestart)
        state['ub_split'] = jax.device_put(state['ub_split']) 
        ub_tec_split = jax.device_put(np.zeros_like(state['ub_split'])) 
        import gc
        gc.collect() 
    bulk_mask = ((x_1d >= 0.1 * Lx) & (x_1d <= 0.9 * Lx)).reshape(npx, 1, 1) * \
                ((y_1d >= 0.1 * Ly) & (y_1d <= 0.9 * Ly)).reshape(1, npy, 1) * \
                ((z_1d >= 0.1 * Lz) & (z_1d <= 0.9 * Lz)).reshape(1, 1, npz)
    bulk_mask = bulk_mask.astype(np.float32)
    bulk_list = [jax.device_put(bulk_mask[:, :, pid*npz_local:(pid+1)*npz_local], device=jax.devices()[pid]) for pid in range(num_dev)]
    bulk_mask_split = jax.device_put_sharded(bulk_list, jax.devices())
    del bulk_list, bulk_mask
    N_STEPS = 50

    steps_accumulated = 0 
    avg_field_dumped = False 
    
    nu_bot_sum = nu_top_sum = nu_vol_sum = nu_kin_sum = nu_th_sum = nu_wall_sum = re_rms_sum = 0.0
    nu_count = 0
    eps_moms_sum_global = np.zeros(4)
    ens_moms_sum_global = np.zeros(4)
    eps_moms_sum_main = np.zeros(4)
    ens_moms_sum_main = np.zeros(4)
    state['ub_split'], ub_tec_split, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = pmap_advance_fused(
        state['ub_split'], ub_tec_split, pid_array, 
        Jx, Jxx, Jy, Jyy, Jz_split, Jzz_split, bulk_mask_split, 2
    )

    maxiter = int(tf / dt)
    dx_uni = Lx / (npx - 1)
    half_N = npx // 2
    k_axis = np.fft.fftfreq(npx, d=dx_uni)[:half_N] * 2 * np.pi 
    Eu_sum_g = np.zeros(half_N)
    ET_sum_g = np.zeros(half_N)
    Eu_sum_m = np.zeros(half_N)
    ET_sum_m = np.zeros(half_N)
    spec_count = 0

    
    ipy_mid = npy // 2
    ipz_mid = npz // 2
    pid_mid = ipz_mid // npz_local      
    z_loc_mid = ipz_mid % npz_local    

    while state['ctime'] <= tf:
        old_ctime = state['ctime']
        ub_new_split, tec_new_split, r_norm, nu_bot, nu_top, nu_vol, nu_kin, nu_th, nu_wall, \
        g_vol, g_eps_moms, g_ens_moms, g_re_rms, \
        g_vol_main, g_eps_moms_main, g_ens_moms_main = pmap_advance_fused(
            state['ub_split'], ub_tec_split, pid_array, 
            Jx, Jxx, Jy, Jyy, Jz_split, Jzz_split, bulk_mask_split, N_STEPS
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

        def calc_moments(vol, eps_arr, ens_arr):
            avg_eps = eps_arr[0] / vol
            avg_ens = ens_arr[0] / vol
            eps_inst = np.array([
                avg_eps,
                (eps_arr[1] / vol) / (avg_eps**2 + 1e-16),
                (eps_arr[2] / vol) / (avg_eps**3 + 1e-16),
                (eps_arr[3] / vol) / (avg_eps**4 + 1e-16)
            ])
            ens_inst = np.array([
                avg_ens,
                (ens_arr[1] / vol) / (avg_ens**2 + 1e-16),
                (ens_arr[2] / vol) / (avg_ens**3 + 1e-16),
                (ens_arr[3] / vol) / (avg_ens**4 + 1e-16)
            ])
            return eps_inst, ens_inst


        eps_inst_g, ens_inst_g = calc_moments(g_vol[0].item(), np.asarray(g_eps_moms[0]), np.asarray(g_ens_moms[0]))
        eps_inst_m, ens_inst_m = calc_moments(g_vol_main[0].item(), np.asarray(g_eps_moms_main[0]), np.asarray(g_ens_moms_main[0]))

        if state['ctime'] >= t_start_avg:
            eps_moms_sum_global += eps_inst_g
            ens_moms_sum_global += ens_inst_g
            eps_moms_sum_main   += eps_inst_m
            ens_moms_sum_main   += ens_inst_m
            
            eps_avg_g = eps_moms_sum_global / nu_count
            ens_avg_g = ens_moms_sum_global / nu_count
            eps_avg_m = eps_moms_sum_main / nu_count
            ens_avg_m = ens_moms_sum_main / nu_count
        else:
            eps_avg_g, ens_avg_g, eps_avg_m, ens_avg_m = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)
        write_moments_log(script_dir, state['iter'], state['ctime'], eps_inst_g, eps_avg_g, ens_inst_g, ens_avg_g, prefix="Global")
        write_moments_log(script_dir, state['iter'], state['ctime'], eps_inst_m, eps_avg_m, ens_inst_m, ens_avg_m, prefix="Mainstream")

        if state['ctime'] >= t_end_avg and not avg_field_dumped:
            write_averaged_tecplot(script_dir, ub_tec_split, steps_accumulated)
            avg_field_dumped = True  
            ub_tec_split = jnp.zeros_like(state['ub_split'])

        if (state['iter'] % 1000000 == 0) or (state['iter'] >= maxiter):
            write_data(script_dir, state) 
            write_solu_tecplot(script_dir, state)
            if state['iter'] >= maxiter:
                break

        if state['iter'] % 100 == 0 and state['ctime'] >= t_spec_start:
            plane_list = [np.asarray(state['ub_split'][pid][:, ipy_mid, :, :]) for pid in range(num_dev)]
            plane_data = np.concatenate(plane_list, axis=1)
            Eu_g_inst, ET_g_inst, Eu_m_inst, ET_m_inst = compute_plane_averaged_spectrum_cpu(plane_data, x_1d, Lx, npx, z_1d, Lz)
            Eu_sum_g += Eu_g_inst
            ET_sum_g += ET_g_inst
            Eu_sum_m += Eu_m_inst
            ET_sum_m += ET_m_inst
            spec_count += 1

        if (state['iter'] % 200000 == 0) and (spec_count > 0):
            write_spectrum_log(script_dir, state['ctime'], k_axis, Eu_sum_g/spec_count, ET_sum_g/spec_count, prefix="Global")
            write_spectrum_log(script_dir, state['ctime'], k_axis, Eu_sum_m/spec_count, ET_sum_m/spec_count, prefix="Mainstream")

if __name__ == "__main__":
    main()