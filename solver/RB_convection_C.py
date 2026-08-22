
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import jit, random
from functools import partial
import os
import numpy as np 
import scipy.interpolate as spi 
import scipy.optimize as spo
jax.config.update('jax_default_matmul_precision', 'float32')
# ----------------- 全局与本地常量 -------------------
npx, npy, npz = 192 ,192, 256  # 网格数
num_dev = 4                 # 并行显卡数   
Lx, Ly, Lz = 1.0, 1.0, 1.0  # 物理域真实尺寸
gamma_z = 1.2               # z网格拉伸比例
dt = 2.5e-4                  # 时间步长
tf = 2000                   # 时间总数
Ma = 0.1
Pr = 1
Ra = 1e7
# -----------------------------------------------

# ----------------- 格式的阶数 -------------------
FLUX_ORDER = 7              # 通量迎风格式阶数 (可选: 3, 5, 7)
VISC_ORDER = 8              # 粘性项中心差分阶数 (可选: 4, 6, 8)
DIR_BND_ORDER = 4           # Dirichlet 边界提取内部点数 
NEU_BND_ORDER = 4           # Neumann 边界提取内部点数 
MAX_BND_ORDER = max(DIR_BND_ORDER, NEU_BND_ORDER)
# -----------------------------------------------

# ----------------- 重启文件 -----------------------
restart = 1              # 首次周期边界计算设为 0；仅可读取由本周期版本生成的重启文件
change = 1               # 是否需要网格插值
t_spec_start = 1600.0    # 能谱统计开始时间
t_start_avg = 1600.0     # 无量纲数统计开始时间
t_end_avg = 2000.0       # 无量纲数统计结束时间

script_dir = os.path.dirname(os.path.abspath(__file__))
namerestart = os.path.join(script_dir, 'RESTART0001499.dat.npz')

npx_old, npy_old, npz_old = 256, 256, 256   # 旧网格数
gamma_xy_old, gamma_z_old = 1.2, 1.2        # 旧网格拉伸比例
Lx_old, Ly_old, Lz_old = 1.0, 1.0, 1.0      # 旧网格尺寸
# --------------------------------------------------

Ma2_inv = 1.0 / (Ma**2)
npz_local = npz // num_dev    
# ----------------- 生成网格与度量(Metrics) -------------------
# x 和 y 是水平周期方向。周期网格不重复保存 x=Lx 和 y=Ly 端点。
dxi = 1.0 / npx
deta = 1.0 / npy
dzeta = 1.0 / (npz - 1)
xi_1d = np.arange(npx, dtype=np.float64) / npx
eta_1d = np.arange(npy, dtype=np.float64) / npy
zeta_1d = np.linspace(0, 1, npz)
# X 方向：均匀周期网格
x_1d = Lx * xi_1d
dx_dxi = np.full(npx, Lx, dtype=np.float64)
d2x_dxi2 = np.zeros(npx, dtype=np.float64)
Jx = jnp.array(dx_dxi).reshape(npx, 1, 1, 1)
Jxx = jnp.array(d2x_dxi2).reshape(npx, 1, 1, 1)
# Y 方向：均匀周期网格
y_1d = Ly * eta_1d
dy_deta = np.full(npy, Ly, dtype=np.float64)
d2y_deta2 = np.zeros(npy, dtype=np.float64)
Jy = jnp.array(dy_deta).reshape(1, npy, 1, 1)
Jyy = jnp.array(d2y_deta2).reshape(1, npy, 1, 1)
# Z 方向 
z_1d = Lz * 0.5 * (1.0 + np.tanh(gamma_z * (2.0 * zeta_1d - 1.0)) / np.tanh(gamma_z))
dz_dzeta = Lz * (gamma_z / np.tanh(gamma_z)) * (1.0 - np.tanh(gamma_z * (2.0 * zeta_1d - 1.0))**2)
d2z_dzeta2 = Lz * (gamma_z / np.tanh(gamma_z)) * (-4.0 * gamma_z) * (1.0 - np.tanh(gamma_z * (2.0 * zeta_1d - 1.0))**2) * np.tanh(gamma_z * (2.0 * zeta_1d - 1.0))
Jz = jnp.array(dz_dzeta).reshape(1, 1, npz, 1)
Jzz = jnp.array(d2z_dzeta2).reshape(1, 1, npz, 1)
# 切分 Z 向度量
Jz_split = np.moveaxis(Jz.reshape(1, 1, num_dev, npz_local, 1), 2, 0)
Jzz_split = np.moveaxis(Jzz.reshape(1, 1, num_dev, npz_local, 1), 2, 0)

# ----------------- 计算通量 -----------------
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

# -----------------边界外推 -----------------
def get_dirichlet_weights(pts, max_pts):
    """自动生成 Dirichlet 高阶外推系数矩阵 (不足 max_pts 的列自动补 0)"""
    W = np.zeros((4, max_pts))
    powers = list(range(pts))
    A = np.array([[float(i**p) for p in powers] for i in range(pts)])
    for k in range(1, 5):
        v = np.array([float((-k)**p) for p in powers])
        W[k-1, :pts] = np.linalg.solve(A.T, v)
    return W

def get_neumann_weights(pts, max_pts):
    """自动生成 Neumann 高阶外推系数矩阵 (不足 max_pts 的列自动补 0)"""
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
def extrapolate_z(f_in, W):
    return jnp.moveaxis(jnp.tensordot(W, f_in, axes=(1, 2)), 0, 2)

@jit
def apply_boundaries_and_halos(local_ub, pid):
    # x and y are periodic. Each ghost layer is copied from the opposite side.
    ubs_x = jnp.concatenate((local_ub[-4:, ...], local_ub, local_ub[:4, ...]), axis=0)
    ubs_xy = jnp.concatenate((ubs_x[:, -4:, ...], ubs_x, ubs_x[:, :4, ...]), axis=1)
    ubs = jnp.pad(ubs_xy, ((0,0), (0,0), (4,4), (0,0)), mode='constant')

    # 【1. X 方向边界】
    # 注意这里统一切片 MAX_BND_ORDER 个点，多余的点会被权重 0 自动抵消！

    # 【2. Y 方向边界】

    # 【3. Z 方向 PCIE 卡间通信】
    send_top, send_bot = ubs[:, :, -8:-4, :], ubs[:, :, 4:8, :]
    recv_bot = jax.lax.ppermute(send_top, axis_name='z', perm=[(i, (i+1)%num_dev) for i in range(num_dev)])
    recv_top = jax.lax.ppermute(send_bot, axis_name='z', perm=[(i, (i-1)%num_dev) for i in range(num_dev)])
    ubs = ubs.at[:, :, 0:4, :].set(recv_bot)
    ubs = ubs.at[:, :, -4:, :].set(recv_top)

    # 【4. 最底卡和最顶卡：Z向物理边界高精度处理】
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

# ----------------- 计算局部残差 ------------------------
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
    
    # === 粘性项的中心差分阶数 ===
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
    # x and y have no physical boundary nodes; all of their residuals advance.
    resid = jax.lax.cond(pid == 0, lambda r: r.at[:, :, 0, 1:5].set(0.0), lambda r: r, resid)
    resid = jax.lax.cond(pid == num_dev - 1, lambda r: r.at[:, :, -1, 1:5].set(0.0), lambda r: r, resid)

    return resid

# ----------------- 并行时间推进 (PMAP) ------------------
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
    # Periodic x/y grids do not contain duplicated end points.
    W_x = jnp.ones(npx)
    W_y = jnp.ones(npy)
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

    # =========== 全局 (Global) 积分 ===========
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

    # =========== 主流 (Mainstream) 积分 ===========
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

# ----------------- I/O 函数 ------------------
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

# ----------------- 中截面 (Mid-Plane) 阵列化能谱计算 ------------------------
def compute_plane_averaged_spectrum_cpu(plane_data, x_1d_stretched, Lx_val, npx_val, z_1d, Lz_val):
    # 1. 物理空间精准插值 (沿 X 轴)
    x_uniform = np.arange(npx_val, dtype=np.float64) * (Lx_val / npx_val)
    f_interp = spi.interp1d(x_1d_stretched, plane_data, axis=0, kind='cubic')
    plane_uni = f_interp(x_uniform)  # 结果形状: (npx, npz, 5)
    u_uni, v_uni, w_uni, t_uni = plane_uni[..., 1], plane_uni[..., 2], plane_uni[..., 3], plane_uni[..., 4]
    # 2. 去均值 (按每条线单独去均值)
    u_fluc = u_uni - np.mean(u_uni, axis=0)
    v_fluc = v_uni - np.mean(v_uni, axis=0)
    w_fluc = w_uni - np.mean(w_uni, axis=0)
    t_fluc = t_uni - np.mean(t_uni, axis=0)
    # 3. 施加汉宁窗 (Hanning Window) 抑制非周期边界泄露
    # No window is needed because x is periodic.
    u_win, v_win = u_fluc, v_fluc
    w_win, t_win = w_fluc, t_fluc
    # 4. 一维 FFT 沿 X 轴 (axis=0) 批量计算
    u_fft = np.fft.fft(u_win, axis=0)
    v_fft = np.fft.fft(v_win, axis=0)
    w_fft = np.fft.fft(w_win, axis=0)
    t_fft = np.fft.fft(t_win, axis=0)
    # 5. 计算各高度的 2D 功率谱密度矩阵
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



# ----------------- 主控流程 ----------------------
def main():
    sm_list = []
    am_list = []
    for pid in range(num_dev):
        sm_temp = np.ones((npx+8, npy+8, npz_local+8, 5), dtype=np.float32)
        am_temp = np.zeros((npx+8, npy+8, npz_local+8, 5), dtype=np.float32)
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
            # 1. 基础的完美线性导热剖面
            base_temp = (1.0 - z_local_coords / Lz).reshape(1, 1, npz_local)
            # 2. 生成随机扰动噪声 
            key = jax.random.PRNGKey(42 + pid) 
            # 生成振幅在 -0.05 到 0.05 之间的随机噪声
            noise = jax.random.uniform(key, shape=(npx, npy, npz_local), minval=-0.05, maxval=0.05)
            ub_temp[..., 4] = base_temp + noise
            if pid == 0:
                ub_temp[:, :, 0, 1:4] = 0.0    
                ub_temp[:, :, 0, 4] = 1.0     
            if pid == num_dev - 1:
                ub_temp[:, :, -1, 1:4] = 0.0   
                ub_temp[:, :, -1, 4] = 0.0     
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
    # There are no lateral boundary layers in the periodic directions. The
    # bulk region therefore excludes only the top and bottom thermal plates.
    z_bulk = ((z_1d >= 0.1 * Lz) & (z_1d <= 0.9 * Lz)).astype(np.float32)
    bulk_mask = np.broadcast_to(z_bulk.reshape(1, 1, npz),
                                (npx, npy, npz)).copy()
    bulk_list = [jax.device_put(bulk_mask[:, :, pid*npz_local:(pid+1)*npz_local], device=jax.devices()[pid]) for pid in range(num_dev)]
    bulk_mask_split = jax.device_put_sharded(bulk_list, jax.devices())
    del bulk_list, bulk_mask
    N_STEPS = 50

    # ====== 时间控制与累加器 ======
    steps_accumulated = 0 # 云图累加步数
    avg_field_dumped = False 
    
    nu_bot_sum = nu_top_sum = nu_vol_sum = nu_kin_sum = nu_th_sum = nu_wall_sum = re_rms_sum = 0.0
    nu_count = 0
    eps_moms_sum_global = np.zeros(4)
    ens_moms_sum_global = np.zeros(4)
    eps_moms_sum_main = np.zeros(4)
    ens_moms_sum_main = np.zeros(4)
    # ===============================

    print(f"JIT Compiling PMAP Fused Kernel (Batch: {N_STEPS} steps)... (等待约3-10分钟，不要中断！)")
    state['ub_split'], ub_tec_split, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = pmap_advance_fused(
        state['ub_split'], ub_tec_split, pid_array, 
        Jx, Jxx, Jy, Jyy, Jz_split, Jzz_split, bulk_mask_split, 2
    )
    print("Compilation complete. Firing on all cylinders!")

    maxiter = int(tf / dt)
    dx_uni = Lx / npx
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

        # 2. Nusselt 数时间平均逻辑
        res_v = r_norm[0].item()
        i_bot  = nu_bot[0].item()
        i_top  = nu_top[0].item()
        i_vol  = nu_vol[0].item()
        i_kin  = nu_kin[0].item()
        i_th   = nu_th[0].item()
        i_wall = nu_wall[0].item()
        i_re   = g_re_rms[0].item()         
        # 时间平均逻辑
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
# ============== 计算瞬时值并进行时间平均 ==============
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

        # 全局高阶矩计算
        eps_inst_g, ens_inst_g = calc_moments(g_vol[0].item(), np.asarray(g_eps_moms[0]), np.asarray(g_ens_moms[0]))
        # 主流高阶矩计算
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

# ============================================================================
# Differentiable closed-loop thermal control and bounded parameter optimization
# ============================================================================

# The uncontrolled result supplied with this case gives Bot_Nu_Avg=17.361081.
# Set RB_RECOMPUTE_NU0=1 to ignore that value and measure Nu0 again from the
# restart field. The default avoids repeating an already completed long run.
MEASURED_NU0 = float(os.environ.get("RB_NU0", "17.361081"))
RECOMPUTE_NU0 = os.environ.get("RB_RECOMPUTE_NU0", "0").strip() == "1"
PIPELINE_PROFILE = os.environ.get("RB_PIPELINE_PROFILE", "paper").strip().lower()
PIPELINE_RESUME = os.environ.get("RB_PIPELINE_RESUME", "1").strip() != "0"
PIPELINE_MODE = os.environ.get("RB_PIPELINE_MODE", "optimize").strip().lower()
PIPELINE_OUTPUT_TAG = "".join(
    character if character.isalnum() or character in "-_" else "_"
    for character in os.environ.get("RB_PIPELINE_OUTPUT_TAG", "").strip()
)
BETA_OPT_INITIAL = float(
    os.environ.get("RB_BETA_OPT_INITIAL", "30.0")
)
CONTROL_RESTART = os.environ.get(
    "RB_CONTROL_RESTART", "RESTART0002000.dat.npz"
)

# The feedback is refreshed every 0.005 free-fall time units and held fixed
# during the following SSPRK steps. This closely approximates continuous
# feedback while avoiding a global sensor reduction at every Runge--Kutta stage.
CONTROL_ACTION_TIME = float(os.environ.get("RB_CONTROL_ACTION_TIME", "0.005"))
CONTROL_ACTION_STEPS = int(round(CONTROL_ACTION_TIME / dt))
CONTROL_LOG_TIME = float(os.environ.get("RB_CONTROL_LOG_TIME", "0.05"))
CONTROL_LOG_BLOCKS = int(round(CONTROL_LOG_TIME / CONTROL_ACTION_TIME))

# A scales the paper's nondimensional wall-temperature actuation. A=1 reaches
# the paper-matched limit [-1, 1]. The beta search extends beyond the reported
# robust interval to determine whether the short-horizon objective has an
# interior stationary point or approaches the saturated-control limit.
AMPLITUDE_MIN, AMPLITUDE_MAX = 0.8, 1.0
BETA_MIN, BETA_MAX = 10.0, 50.0
INITIAL_CONTROL_PARAMETERS = (0.80, 20.0)
OPT_RAW_LEARNING_RATE = (0.20, 0.10)

AB_OPT_STARTS = tuple(
    tuple(float(component) for component in pair.split(":"))
    for pair in os.environ.get(
        "RB_AB_OPT_STARTS",
        (
            "0.82:14;0.82:22;0.82:30;0.82:38;0.82:46;"
            "0.90:14;0.90:22;0.90:30;0.90:38;0.90:46;"
            "0.98:14;0.98:22;0.98:30;0.98:38;0.98:46"
        ),
    ).split(";")
    if pair.strip()
)
AB_OPT_MAX_EVALUATIONS = int(
    os.environ.get("RB_AB_OPT_MAX_EVALUATIONS", "24")
)
AB_OPT_EVALUATIONS_PER_START = int(
    os.environ.get("RB_AB_OPT_EVALUATIONS_PER_START", "4")
)
AB_OPT_LOCAL_STARTS = int(
    os.environ.get("RB_AB_OPT_LOCAL_STARTS", "3")
)
AB_OPT_PROJECTED_GRADIENT_TOLERANCE = float(
    os.environ.get("RB_AB_OPT_PROJECTED_GRADIENT_TOLERANCE", "1.0e-4")
)
AB_OPT_FUNCTION_TOLERANCE = float(
    os.environ.get("RB_AB_OPT_FUNCTION_TOLERANCE", "1.0e-9")
)
AB_OPT_SCALE_A = float(
    os.environ.get(
        "RB_AB_OPT_SCALE_A", str(AMPLITUDE_MAX - AMPLITUDE_MIN)
    )
)
AB_OPT_SCALE_BETA = float(
    os.environ.get("RB_AB_OPT_SCALE_BETA", str(BETA_MAX - BETA_MIN))
)
FD_EPSILON_SWEEP = (
    (3.0e-3, 1.0e-3, 3.0e-4),
    (3.0e-2, 1.0e-2, 3.0e-3),
)

if PIPELINE_PROFILE == "quick":
    BASELINE_TRANSIENT_TIME = 0.25
    BASELINE_AVERAGING_TIME = 0.75
    VALIDATION_TRANSIENT_TIME = 0.50
    VALIDATION_AVERAGING_TIME = 1.00
    DIFF_WINDOW_TIME = 0.25
    DIFF_REWARD_START_TIME = 0.125
    OPT_ITERATIONS = 2
elif PIPELINE_PROFILE == "paper":
    BASELINE_TRANSIENT_TIME = 50.0
    BASELINE_AVERAGING_TIME = 100.0
    VALIDATION_TRANSIENT_TIME = 50.0
    VALIDATION_AVERAGING_TIME = 100.0
    DIFF_WINDOW_TIME = 5.0
    DIFF_REWARD_START_TIME = 2.5
    OPT_ITERATIONS = 8
else:
    raise ValueError("RB_PIPELINE_PROFILE must be 'quick' or 'paper'")

OPT_ITERATIONS = int(os.environ.get("RB_OPT_ITERATIONS", str(OPT_ITERATIONS)))
BETA_OPT_MAX_EVALUATIONS = int(
    os.environ.get("RB_BETA_OPT_MAX_EVALUATIONS", "16")
)
BETA_OPT_INITIAL_STEP = float(
    os.environ.get("RB_BETA_OPT_INITIAL_STEP", "1.0")
)
BETA_OPT_MAX_STEP = float(
    os.environ.get("RB_BETA_OPT_MAX_STEP", "2.5")
)
BETA_OPT_SIGN_CHECK_STEP = float(
    os.environ.get("RB_BETA_OPT_SIGN_CHECK_STEP", "0.5")
)
BETA_OPT_GRADIENT_TOLERANCE = float(
    os.environ.get("RB_BETA_OPT_GRADIENT_TOLERANCE", "1.0e-4")
)
BETA_OPT_INTERVAL_TOLERANCE = float(
    os.environ.get("RB_BETA_OPT_INTERVAL_TOLERANCE", "0.10")
)
VALIDATION_TRANSIENT_TIME = float(
    os.environ.get(
        "RB_VALIDATION_TRANSIENT_TIME", str(VALIDATION_TRANSIENT_TIME)
    )
)
VALIDATION_AVERAGING_TIME = float(
    os.environ.get(
        "RB_VALIDATION_AVERAGING_TIME", str(VALIDATION_AVERAGING_TIME)
    )
)

GRAD_CHECK_WINDOW_TIME = 0.05
GRAD_CHECK_REWARD_START_TIME = 0.025
DIFF_N_BLOCKS = int(round(DIFF_WINDOW_TIME / CONTROL_ACTION_TIME))
DIFF_REWARD_START_BLOCK = int(
    round(DIFF_REWARD_START_TIME / CONTROL_ACTION_TIME)
)
GRAD_CHECK_N_BLOCKS = int(round(GRAD_CHECK_WINDOW_TIME / CONTROL_ACTION_TIME))
GRAD_CHECK_REWARD_START_BLOCK = int(
    round(GRAD_CHECK_REWARD_START_TIME / CONTROL_ACTION_TIME)
)

# These globals are fixed before the first JAX trace. The sensor height follows
# the reference thermal-boundary-layer estimate z_delta=1/(2 Nu0).
CONTROL_NU0 = MEASURED_NU0
CONTROL_SENSOR_Z = 1.0 / (2.0 * MEASURED_NU0)
CONTROL_SENSOR_K0 = 0
CONTROL_SENSOR_K1 = 1
CONTROL_SENSOR_ALPHA = 0.0
CONTROL_SENSOR_PID = 0
CONTROL_SENSOR_LOCAL_K0 = 0
CONTROL_SENSOR_LOCAL_K1 = 1


def configure_control_context(nu0):
    global CONTROL_NU0
    global CONTROL_SENSOR_Z, CONTROL_SENSOR_K0, CONTROL_SENSOR_K1
    global CONTROL_SENSOR_ALPHA, CONTROL_SENSOR_PID
    global CONTROL_SENSOR_LOCAL_K0, CONTROL_SENSOR_LOCAL_K1

    if not np.isfinite(nu0) or nu0 <= 1.0:
        raise ValueError(f"Invalid uncontrolled Nusselt number: {nu0}")
    CONTROL_NU0 = float(nu0)
    CONTROL_SENSOR_Z = 1.0 / (2.0 * CONTROL_NU0)
    CONTROL_SENSOR_K1 = int(np.searchsorted(z_1d, CONTROL_SENSOR_Z))
    CONTROL_SENSOR_K1 = min(max(CONTROL_SENSOR_K1, 1), npz - 1)
    CONTROL_SENSOR_K0 = CONTROL_SENSOR_K1 - 1
    CONTROL_SENSOR_ALPHA = float(
        (CONTROL_SENSOR_Z - z_1d[CONTROL_SENSOR_K0])
        / (z_1d[CONTROL_SENSOR_K1] - z_1d[CONTROL_SENSOR_K0])
    )
    CONTROL_SENSOR_PID = CONTROL_SENSOR_K0 // npz_local
    CONTROL_SENSOR_LOCAL_K0 = CONTROL_SENSOR_K0 % npz_local
    CONTROL_SENSOR_LOCAL_K1 = CONTROL_SENSOR_K1 % npz_local
    if CONTROL_SENSOR_K1 // npz_local != CONTROL_SENSOR_PID:
        raise ValueError(
            "The two interpolation planes for the control sensor cross a GPU boundary"
        )


def _restart_path():
    if os.path.isabs(CONTROL_RESTART):
        return CONTROL_RESTART
    return os.path.join(script_dir, CONTROL_RESTART)


def load_control_restart():
    devices = jax.local_devices()
    if len(devices) < num_dev:
        raise RuntimeError(
            f"This case requires {num_dev} local devices, but JAX found {len(devices)}"
        )

    restart_path = _restart_path()
    if not os.path.exists(restart_path):
        raise FileNotFoundError(
            f"Restart file not found: {restart_path}. Put the Python file beside "
            "the restart or set RB_CONTROL_RESTART to its path."
        )

    with np.load(restart_path) as data:
        shape = tuple(data["ub"].shape)
    expected = (npx, npy, npz, 5)
    if shape == expected:
        loaded = read_data(restart_path)
    elif change == 1:
        print(f"[restart] interpolating {shape} -> {expected}")
        loaded = read_and_interpolate_data(restart_path, x_1d, y_1d, z_1d)
    else:
        raise ValueError(
            f"Restart field has shape {shape}, expected {expected}; set change=1 "
            "and configure the old-grid parameters before running."
        )

    host_shards = [
        np.asarray(loaded["ub_split"][pid], dtype=np.float32)
        for pid in range(num_dev)
    ]
    try:
        device_put_sharded = jax.device_put_sharded
    except AttributeError:
        # Newer JAX releases let pmap shard the leading device axis directly.
        sharded_state = jnp.asarray(np.stack(host_shards, axis=0))
    else:
        device_shards = [
            jax.device_put(host_shards[pid], device=devices[pid])
            for pid in range(num_dev)
        ]
        sharded_state = device_put_sharded(
            device_shards, devices[:num_dev]
        )
    return {
        "ctime": float(loaded["ctime"]),
        "iter": int(loaded["iter"]),
        "ub_split": sharded_state,
    }


def decode_control_parameters(raw_parameters):
    amplitude_fraction = jax.nn.sigmoid(raw_parameters[0])
    beta_fraction = jax.nn.sigmoid(raw_parameters[1])
    amplitude = AMPLITUDE_MIN + (
        AMPLITUDE_MAX - AMPLITUDE_MIN
    ) * amplitude_fraction
    beta = BETA_MIN + (BETA_MAX - BETA_MIN) * beta_fraction
    return jnp.stack((amplitude, beta))


def encode_control_parameters(physical_parameters):
    amplitude, beta = physical_parameters
    eps = 1.0e-6
    amplitude_fraction = np.clip(
        (amplitude - AMPLITUDE_MIN) / (AMPLITUDE_MAX - AMPLITUDE_MIN),
        eps,
        1.0 - eps,
    )
    beta_fraction = np.clip(
        (beta - BETA_MIN) / (BETA_MAX - BETA_MIN),
        eps,
        1.0 - eps,
    )
    return jnp.asarray(
        (
            np.log(amplitude_fraction / (1.0 - amplitude_fraction)),
            np.log(beta_fraction / (1.0 - beta_fraction)),
        ),
        dtype=jnp.float32,
    )


def enforce_controlled_walls(local_ub, pid, lower_wall_temperature):
    """Apply only the two physical z walls; x and y remain periodic."""

    def lower_wall(u):
        u = u.at[:, :, 0, 1:4].set(0.0)
        return u.at[:, :, 0, 4].set(lower_wall_temperature)

    def upper_wall(u):
        u = u.at[:, :, -1, 1:4].set(0.0)
        return u.at[:, :, -1, 4].set(0.0)

    local_ub = jax.lax.cond(pid == 0, lower_wall, lambda u: u, local_ub)
    return jax.lax.cond(
        pid == num_dev - 1, upper_wall, lambda u: u, local_ub
    )


@jit
def solve_zero_mean_threshold(signal, beta, weights):
    """Solve mean(tanh(beta*(signal-T0)))=0 with an implicit derivative."""

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


def build_feedback_control(local_ub, pid, Jx_metric, Jy_metric, parameters):
    amplitude, beta = parameters

    def read_sensor_plane(u):
        theta0 = u[:, :, CONTROL_SENSOR_LOCAL_K0, 4]
        theta1 = u[:, :, CONTROL_SENSOR_LOCAL_K1, 4]
        return (
            (1.0 - CONTROL_SENSOR_ALPHA) * theta0
            + CONTROL_SENSOR_ALPHA * theta1
        )

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
    wall_action = amplitude * jnp.tanh(
        beta * (sensor_fluctuation - threshold)
    )
    action_mean = jnp.sum(weights * wall_action) / weight_sum
    action_rms = jnp.sqrt(jnp.sum(weights * wall_action**2) / weight_sum)
    lower_wall_temperature = 1.0 + wall_action
    diagnostics = (
        threshold,
        action_mean,
        jnp.min(wall_action),
        jnp.max(wall_action),
        action_rms,
    )
    return lower_wall_temperature, diagnostics


def advance_control_action(
    local_ub,
    pid,
    lower_wall_temperature,
    Jx_metric,
    Jxx_metric,
    Jy_metric,
    Jyy_metric,
    Jz_local,
    Jzz_local,
):
    a30, a32 = 0.355909775, 0.644090224
    a40, a43 = 0.367933791, 0.632066208
    a52, a54 = 0.237593836, 0.762406163
    b10, b21 = 0.377268915, 0.377268915
    b32, b43, b54 = 0.242995220, 0.238458932, 0.287632146

    def apply_wall(u):
        return enforce_controlled_walls(
            u, pid, lower_wall_temperature
        )

    def residual(u):
        u = apply_wall(u)
        return calc_local_resid(
            u,
            pid,
            Jx_metric,
            Jxx_metric,
            Jy_metric,
            Jyy_metric,
            Jz_local,
            Jzz_local,
        )

    def one_step(_, u):
        u = apply_wall(u)
        u1 = u
        u = apply_wall(u + b10 * dt * residual(u))
        u = apply_wall(u + b21 * dt * residual(u))
        u2 = u
        u = apply_wall(a30 * u1 + a32 * u + b32 * dt * residual(u))
        u = apply_wall(a40 * u1 + a43 * u + b43 * dt * residual(u))
        u = a52 * u2 + a54 * u + b54 * dt * residual(u)
        return apply_wall(u)

    return jax.lax.fori_loop(
        0, CONTROL_ACTION_STEPS, one_step, local_ub
    )


def advance_uncontrolled_steps(
    local_ub,
    pid,
    Jx_metric,
    Jxx_metric,
    Jy_metric,
    Jyy_metric,
    Jz_local,
    Jzz_local,
    n_steps,
):
    a30, a32 = 0.355909775, 0.644090224
    a40, a43 = 0.367933791, 0.632066208
    a52, a54 = 0.237593836, 0.762406163
    b10, b21 = 0.377268915, 0.377268915
    b32, b43, b54 = 0.242995220, 0.238458932, 0.287632146

    def residual(u):
        return calc_local_resid(
            u,
            pid,
            Jx_metric,
            Jxx_metric,
            Jy_metric,
            Jyy_metric,
            Jz_local,
            Jzz_local,
        )

    def one_step(_, u):
        u1 = u
        u = u + b10 * dt * residual(u)
        u = u + b21 * dt * residual(u)
        u2 = u
        u = a30 * u1 + a32 * u + b32 * dt * residual(u)
        u = a40 * u1 + a43 * u + b43 * dt * residual(u)
        return a52 * u2 + a54 * u + b54 * dt * residual(u)

    return jax.lax.fori_loop(0, n_steps, one_step, local_ub)


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
    static_broadcasted_argnums=(8,),
    in_axes=(0, 0, None, None, None, None, 0, 0, None),
)
def pmap_uncontrolled_window(
    local_ub,
    pid,
    Jx_metric,
    Jxx_metric,
    Jy_metric,
    Jyy_metric,
    Jz_local,
    Jzz_local,
    n_steps,
):
    final_ub = advance_uncontrolled_steps(
        local_ub,
        pid,
        Jx_metric,
        Jxx_metric,
        Jy_metric,
        Jyy_metric,
        Jz_local,
        Jzz_local,
        n_steps,
    )
    nu_bottom, nu_top = controlled_wall_nusselt(
        final_ub, pid, Jx_metric, Jy_metric, Jz_local
    )
    return final_ub, nu_bottom, nu_top


@partial(
    jax.pmap,
    axis_name="z",
    static_broadcasted_argnums=(9, 10),
    in_axes=(0, 0, None, None, None, None, 0, 0, None, None, None),
)
def pmap_control_window(
    local_ub,
    pid,
    Jx_metric,
    Jxx_metric,
    Jy_metric,
    Jyy_metric,
    Jz_local,
    Jzz_local,
    physical_parameters,
    n_blocks,
    reward_start_block,
):
    zero = jnp.asarray(0.0, dtype=local_ub.dtype)
    initial_carry = (
        local_ub,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
    )

    def one_block(block, carry):
        u, nu_b_sum, nu_t_sum, count, _, _, _, _, _ = carry
        wall_temperature, diagnostics = build_feedback_control(
            u, pid, Jx_metric, Jy_metric, physical_parameters
        )
        u = advance_control_action(
            u,
            pid,
            wall_temperature,
            Jx_metric,
            Jxx_metric,
            Jy_metric,
            Jyy_metric,
            Jz_local,
            Jzz_local,
        )
        nu_bottom, nu_top = controlled_wall_nusselt(
            u, pid, Jx_metric, Jy_metric, Jz_local
        )
        include = block >= reward_start_block
        nu_b_sum = nu_b_sum + jnp.where(include, nu_bottom, 0.0)
        nu_t_sum = nu_t_sum + jnp.where(include, nu_top, 0.0)
        count = count + jnp.where(include, 1.0, 0.0)
        threshold, action_mean, action_min, action_max, action_rms = diagnostics
        return (
            u,
            nu_b_sum,
            nu_t_sum,
            count,
            threshold,
            action_mean,
            action_min,
            action_max,
            action_rms,
        )

    carry = jax.lax.fori_loop(0, n_blocks, one_block, initial_carry)
    (
        final_ub,
        nu_bottom_sum,
        nu_top_sum,
        sample_count,
        threshold,
        action_mean,
        action_min,
        action_max,
        action_rms,
    ) = carry
    sample_count = jnp.maximum(sample_count, 1.0)
    mean_nu_bottom = nu_bottom_sum / sample_count
    mean_nu_top = nu_top_sum / sample_count
    enhancement = (mean_nu_bottom - CONTROL_NU0) / CONTROL_NU0
    objective = -enhancement
    return (
        final_ub,
        objective,
        mean_nu_bottom,
        mean_nu_top,
        enhancement,
        threshold,
        action_mean,
        action_min,
        action_max,
        action_rms,
    )


def control_objective_physical(
    physical_parameters, initial_ub, pid_array, n_blocks, reward_start_block
):
    results = pmap_control_window(
        initial_ub,
        pid_array,
        Jx,
        Jxx,
        Jy,
        Jyy,
        Jz_split,
        Jzz_split,
        physical_parameters,
        n_blocks,
        reward_start_block,
    )
    return results[1][0]


def control_objective_raw(
    raw_parameters, initial_ub, pid_array, n_blocks, reward_start_block
):
    physical_parameters = decode_control_parameters(raw_parameters)
    return control_objective_physical(
        physical_parameters,
        initial_ub,
        pid_array,
        n_blocks,
        reward_start_block,
    )


def forward_mode_gradient(objective_fn, parameters):
    value = None
    gradients = []
    for index in range(parameters.shape[0]):
        direction = jnp.zeros_like(parameters).at[index].set(1.0)
        primal, tangent = jax.jvp(objective_fn, (parameters,), (direction,))
        primal.block_until_ready()
        if value is None:
            value = primal
        gradients.append(tangent)
    return value, jnp.stack(gradients)


def run_uncontrolled_baseline(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, "baseline_summary.npz")
    history_file = os.path.join(output_dir, "baseline_history.dat")
    if PIPELINE_RESUME and os.path.exists(summary_file):
        with np.load(summary_file) as data:
            return float(data["Nu_bottom"])

    if not RECOMPUTE_NU0:
        np.savez(
            summary_file,
            Nu_bottom=MEASURED_NU0,
            Nu_top=np.nan,
            samples=0,
            source="supplied long-time uncontrolled average",
        )
        with open(history_file, "w", encoding="utf-8") as stream:
            stream.write("Nu0 supplied through RB_NU0; no baseline rerun.\n")
        print(f"[baseline] using supplied Nu0={MEASURED_NU0:.6f}")
        return MEASURED_NU0

    state = load_control_restart()
    pid_array = jnp.arange(num_dev, dtype=jnp.int32)
    log_steps = int(round(CONTROL_LOG_TIME / dt))
    total_time = BASELINE_TRANSIENT_TIME + BASELINE_AVERAGING_TIME
    elapsed = 0.0
    bottom_sum = 0.0
    top_sum = 0.0
    count = 0
    with open(history_file, "w", encoding="utf-8") as stream:
        stream.write("elapsed_time\tNu_bottom\tNu_top\tNu_bottom_avg\tNu_top_avg\n")

    while elapsed < total_time - 0.5 * CONTROL_LOG_TIME:
        state["ub_split"], nu_bottom, nu_top = pmap_uncontrolled_window(
            state["ub_split"],
            pid_array,
            Jx,
            Jxx,
            Jy,
            Jyy,
            Jz_split,
            Jzz_split,
            log_steps,
        )
        elapsed += CONTROL_LOG_TIME
        bottom = float(nu_bottom[0])
        top = float(nu_top[0])
        if elapsed > BASELINE_TRANSIENT_TIME + 0.5 * CONTROL_LOG_TIME:
            bottom_sum += bottom
            top_sum += top
            count += 1
        bottom_avg = bottom_sum / max(count, 1)
        top_avg = top_sum / max(count, 1)
        with open(history_file, "a", encoding="utf-8") as stream:
            stream.write(
                f"{elapsed:.8f}\t{bottom:.10e}\t{top:.10e}\t"
                f"{bottom_avg:.10e}\t{top_avg:.10e}\n"
            )
        if int(round(elapsed / CONTROL_LOG_TIME)) % 20 == 0:
            print(
                f"[baseline] t={elapsed:.1f}/{total_time:.1f}, "
                f"Nu_b={bottom:.4f}, Nu_t={top:.4f}"
            )

    if count == 0:
        raise RuntimeError("The baseline averaging window produced no samples")
    bottom_avg = bottom_sum / count
    top_avg = top_sum / count
    np.savez(
        summary_file,
        Nu_bottom=bottom_avg,
        Nu_top=top_avg,
        samples=count,
        transient_time=BASELINE_TRANSIENT_TIME,
        averaging_time=BASELINE_AVERAGING_TIME,
        source="recomputed from restart",
    )
    print(
        f"[baseline] completed: Nu_bottom={bottom_avg:.6f}, "
        f"Nu_top={top_avg:.6f}"
    )
    return bottom_avg


def run_gradient_check(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, "gradient_check.dat")
    if PIPELINE_RESUME and os.path.exists(result_file):
        print(f"[gradient-check] reusing {result_file}")
        return

    state = load_control_restart()
    pid_array = jnp.arange(num_dev, dtype=jnp.int32)
    parameters = jnp.asarray(INITIAL_CONTROL_PARAMETERS, dtype=jnp.float32)
    objective_fn = lambda p: control_objective_physical(
        p,
        state["ub_split"],
        pid_array,
        GRAD_CHECK_N_BLOCKS,
        GRAD_CHECK_REWARD_START_BLOCK,
    )
    value, ad_gradient = forward_mode_gradient(objective_fn, parameters)
    names = ("amplitude", "beta")
    with open(result_file, "w", encoding="utf-8") as stream:
        stream.write(
            "parameter\tepsilon\tAD_gradient\tFD_gradient\trelative_error\n"
        )
        for index, name in enumerate(names):
            for epsilon in FD_EPSILON_SWEEP[index]:
                direction = jnp.zeros_like(parameters).at[index].set(epsilon)
                plus = objective_fn(parameters + direction)
                minus = objective_fn(parameters - direction)
                plus.block_until_ready()
                fd_value = float((plus - minus) / (2.0 * epsilon))
                ad_value = float(ad_gradient[index])
                scale = max(abs(ad_value), abs(fd_value), 1.0e-12)
                relative_error = abs(ad_value - fd_value) / scale
                stream.write(
                    f"{name}\t{epsilon:.12e}\t{ad_value:.12e}\t"
                    f"{fd_value:.12e}\t{relative_error:.12e}\n"
                )
                print(
                    f"[gradient-check] {name}, eps={epsilon:.1e}, "
                    f"AD={ad_value:.6e}, FD={fd_value:.6e}, "
                    f"error={relative_error:.3e}"
                )
    print(f"[gradient-check] objective={float(value):.8e}")


def run_parameter_optimization(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, "optimized_control_parameters.npz")
    history_file = os.path.join(output_dir, "optimization_history.dat")
    if PIPELINE_RESUME and os.path.exists(result_file):
        with np.load(result_file) as data:
            return (
                float(data["amplitude"]),
                float(data["beta"]),
                float(data["objective"]),
            )

    state = load_control_restart()
    pid_array = jnp.arange(num_dev, dtype=jnp.int32)
    raw_parameters = encode_control_parameters(INITIAL_CONTROL_PARAMETERS)
    learning_rate = jnp.asarray(OPT_RAW_LEARNING_RATE, dtype=jnp.float32)
    objective_fn = lambda p: control_objective_raw(
        p,
        state["ub_split"],
        pid_array,
        DIFF_N_BLOCKS,
        DIFF_REWARD_START_BLOCK,
    )

    first_moment = jnp.zeros_like(raw_parameters)
    second_moment = jnp.zeros_like(raw_parameters)
    adam_beta1, adam_beta2 = 0.9, 0.999
    best_value = np.inf
    best_raw = np.asarray(raw_parameters, dtype=np.float32).copy()
    with open(history_file, "w", encoding="utf-8") as stream:
        stream.write(
            "iteration\tobjective\tenhancement\tamplitude\tbeta\t"
            "gradient_raw_amplitude\tgradient_raw_beta\n"
        )

    for iteration in range(1, OPT_ITERATIONS + 1):
        value, gradient = forward_mode_gradient(objective_fn, raw_parameters)
        value_float = float(value)
        physical = np.asarray(decode_control_parameters(raw_parameters))
        if np.isfinite(value_float) and value_float < best_value:
            best_value = value_float
            best_raw = np.asarray(raw_parameters, dtype=np.float32).copy()
        with open(history_file, "a", encoding="utf-8") as stream:
            stream.write(
                f"{iteration}\t{value_float:.12e}\t{-value_float:.12e}\t"
                f"{physical[0]:.8f}\t{physical[1]:.8f}\t"
                f"{float(gradient[0]):.12e}\t{float(gradient[1]):.12e}\n"
            )
        print(
            f"[optimize] iteration={iteration}, J={value_float:.6e}, "
            f"eta={-100.0 * value_float:.3f}%, A={physical[0]:.6f}, "
            f"beta={physical[1]:.6f}, grad_raw={np.asarray(gradient)}"
        )
        if not np.all(np.isfinite(np.asarray(gradient))):
            raise FloatingPointError("Non-finite automatic-differentiation gradient")
        first_moment = (
            adam_beta1 * first_moment + (1.0 - adam_beta1) * gradient
        )
        second_moment = (
            adam_beta2 * second_moment + (1.0 - adam_beta2) * gradient**2
        )
        first_hat = first_moment / (1.0 - adam_beta1**iteration)
        second_hat = second_moment / (1.0 - adam_beta2**iteration)
        raw_parameters = raw_parameters - learning_rate * first_hat / (
            jnp.sqrt(second_hat) + 1.0e-8
        )

    final_value = objective_fn(raw_parameters)
    final_value.block_until_ready()
    final_value_float = float(final_value)
    final_physical = np.asarray(decode_control_parameters(raw_parameters))
    with open(history_file, "a", encoding="utf-8") as stream:
        stream.write(
            f"{OPT_ITERATIONS + 1}\t{final_value_float:.12e}\t"
            f"{-final_value_float:.12e}\t{final_physical[0]:.8f}\t"
            f"{final_physical[1]:.8f}\tnan\tnan\n"
        )
    if np.isfinite(final_value_float) and final_value_float < best_value:
        best_value = final_value_float
        best_raw = np.asarray(raw_parameters, dtype=np.float32).copy()

    best_physical = np.asarray(
        decode_control_parameters(jnp.asarray(best_raw)), dtype=np.float32
    )
    np.savez(
        result_file,
        amplitude=float(best_physical[0]),
        beta=float(best_physical[1]),
        raw_amplitude=float(best_raw[0]),
        raw_beta=float(best_raw[1]),
        objective=best_value,
        short_horizon_enhancement=-best_value,
        iterations=OPT_ITERATIONS,
        window_time=DIFF_WINDOW_TIME,
        reward_start_time=DIFF_REWARD_START_TIME,
        amplitude_bounds=(AMPLITUDE_MIN, AMPLITUDE_MAX),
        beta_bounds=(BETA_MIN, BETA_MAX),
    )
    print(
        f"[optimize] best short-window candidate: J={best_value:.8e}, "
        f"A={best_physical[0]:.6f}, beta={best_physical[1]:.6f}"
    )
    return float(best_physical[0]), float(best_physical[1]), best_value


def run_ab_lbfgsb_optimization(output_dir):
    """Single- or multi-start bound-constrained quasi-Newton optimization."""
    os.makedirs(output_dir, exist_ok=True)
    state = load_control_restart()
    pid_array = jnp.arange(num_dev, dtype=jnp.int32)
    lower = np.asarray((AMPLITUDE_MIN, BETA_MIN), dtype=np.float64)
    upper = np.asarray((AMPLITUDE_MAX, BETA_MAX), dtype=np.float64)
    parameter_scale = np.asarray(
        (AB_OPT_SCALE_A, AB_OPT_SCALE_BETA), dtype=np.float64
    )
    coordinate_upper = (upper - lower) / parameter_scale
    records = []
    accepted_records = []
    accepted_keys = set()
    cache = {}
    best_result = None

    class EvaluationBudgetReached(RuntimeError):
        pass

    def objective_physical(parameters):
        return control_objective_physical(
            parameters,
            state["ub_split"],
            pid_array,
            DIFF_N_BLOCKS,
            DIFF_REWARD_START_BLOCK,
        )

    def projected_gradient(normalized, gradient):
        projected = np.asarray(gradient, dtype=np.float64).copy()
        bound_tolerance = 1.0e-8
        for index in range(2):
            at_lower = normalized[index] <= bound_tolerance
            at_upper = (
                normalized[index]
                >= coordinate_upper[index] - bound_tolerance
            )
            if (at_lower and projected[index] > 0.0) or (
                at_upper and projected[index] < 0.0
            ):
                projected[index] = 0.0
        return projected

    def evaluate(normalized, start_index):
        nonlocal best_result
        normalized = np.clip(
            np.asarray(normalized, dtype=np.float64),
            0.0,
            coordinate_upper,
        )
        key = tuple(np.round(normalized, 10))
        if key in cache:
            return cache[key]
        if len(records) >= AB_OPT_MAX_EVALUATIONS:
            raise EvaluationBudgetReached

        physical = lower + parameter_scale * normalized
        value, gradient_physical = forward_mode_gradient(
            objective_physical,
            jnp.asarray(physical, dtype=jnp.float32),
        )
        value_float = float(value)
        gradient_physical = np.asarray(
            gradient_physical, dtype=np.float64
        )
        gradient_normalized = gradient_physical * parameter_scale
        projected = projected_gradient(normalized, gradient_normalized)
        projected_norm = float(np.linalg.norm(projected, ord=np.inf))
        if not np.isfinite(value_float) or not np.all(
            np.isfinite(gradient_physical)
        ):
            raise FloatingPointError(
                "Non-finite objective or automatic-differentiation gradient"
            )

        result = {
            "value": value_float,
            "normalized": normalized.copy(),
            "physical": physical.copy(),
            "gradient_physical": gradient_physical.copy(),
            "gradient_normalized": gradient_normalized.copy(),
            "projected_gradient_norm": projected_norm,
        }
        cache[key] = result
        records.append(
            (
                len(records) + 1,
                start_index,
                physical[0],
                physical[1],
                value_float,
                -value_float,
                gradient_physical[0],
                gradient_physical[1],
                gradient_normalized[0],
                gradient_normalized[1],
                projected_norm,
            )
        )
        if best_result is None or value_float < best_result["value"]:
            best_result = result
        print(
            f"[AB-optimize] evaluation={len(records):02d}, "
            f"start={start_index}, A={physical[0]:.8f}, "
            f"beta={physical[1]:.8f}, J={value_float:.6e}, "
            f"dJ/dA={gradient_physical[0]:.6e}, "
            f"dJ/dbeta={gradient_physical[1]:.6e}, "
            f"projected_grad={projected_norm:.6e}"
        )
        return result

    def scipy_value_and_gradient(normalized, start_index):
        result = evaluate(normalized, start_index)
        return result["value"], result["gradient_normalized"]

    def record_accepted_iteration(normalized, start_index):
        normalized = np.clip(
            np.asarray(normalized, dtype=np.float64),
            0.0,
            coordinate_upper,
        )
        cache_key = tuple(np.round(normalized, 10))
        accepted_key = (start_index, cache_key)
        if accepted_key in accepted_keys:
            return
        result = cache.get(cache_key)
        if result is None:
            result = evaluate(normalized, start_index)
        physical = result["physical"]
        gradient = result["gradient_physical"]
        accepted_keys.add(accepted_key)
        accepted_records.append(
            (
                len(accepted_records) + 1,
                start_index,
                physical[0],
                physical[1],
                result["value"],
                -result["value"],
                gradient[0],
                gradient[1],
                result["projected_gradient_norm"],
            )
        )
        print(
            f"[AB-accepted] iteration={len(accepted_records):02d}, "
            f"start={start_index}, A={physical[0]:.8f}, "
            f"beta={physical[1]:.8f}, J={result['value']:.6e}, "
            f"dJ/dA={gradient[0]:.6e}, "
            f"dJ/dbeta={gradient[1]:.6e}"
        )

    print(
        f"[AB-optimize] bounds: A=[{AMPLITUDE_MIN}, {AMPLITUDE_MAX}], "
        f"beta=[{BETA_MIN}, {BETA_MAX}], starts={AB_OPT_STARTS}, "
        f"parameter scales=({AB_OPT_SCALE_A}, {AB_OPT_SCALE_BETA}), "
        f"maximum unique evaluations={AB_OPT_MAX_EVALUATIONS}"
    )
    design_results = []
    for physical_start in AB_OPT_STARTS:
        physical_start = np.asarray(physical_start, dtype=np.float64)
        normalized_start = np.clip(
            (physical_start - lower) / parameter_scale,
            0.0,
            coordinate_upper,
        )
        design_results.append(evaluate(normalized_start, start_index=0))

    ranked_designs = sorted(
        design_results, key=lambda result: result["value"]
    )
    local_starts = ranked_designs[:AB_OPT_LOCAL_STARTS]
    print(
        "[AB-optimize] selected local starts: "
        + ", ".join(
            f"(A={result['physical'][0]:.4f}, "
            f"beta={result['physical'][1]:.4f})"
            for result in local_starts
        )
    )

    start_reports = []
    for start_index, start_result in enumerate(local_starts, start=1):
        if len(records) >= AB_OPT_MAX_EVALUATIONS:
            break
        normalized_start = start_result["normalized"]
        remaining = AB_OPT_MAX_EVALUATIONS - len(records)
        start_budget = min(AB_OPT_EVALUATIONS_PER_START, remaining)
        record_accepted_iteration(normalized_start, start_index)
        try:
            result = spo.minimize(
                lambda normalized: scipy_value_and_gradient(
                    normalized, start_index
                ),
                normalized_start,
                method="L-BFGS-B",
                jac=True,
                bounds=(
                    (0.0, coordinate_upper[0]),
                    (0.0, coordinate_upper[1]),
                ),
                callback=lambda normalized, index=start_index: (
                    record_accepted_iteration(normalized, index)
                ),
                options={
                    "maxiter": start_budget,
                    "maxfun": start_budget,
                    "maxls": 6,
                    "gtol": AB_OPT_PROJECTED_GRADIENT_TOLERANCE,
                    "ftol": AB_OPT_FUNCTION_TOLERANCE,
                },
            )
            report = (
                f"start={start_index}, success={int(result.success)}, "
                f"status={result.status}, message={result.message}"
            )
        except EvaluationBudgetReached:
            report = f"start={start_index}, global evaluation budget reached"
        start_reports.append(report)
        print(f"[AB-optimize] {report}")

    if best_result is None:
        raise RuntimeError("The A-beta optimizer did not evaluate any point")

    best_physical = best_result["physical"]
    best_gradient = best_result["gradient_physical"]
    best_projected_norm = best_result["projected_gradient_norm"]
    values = np.asarray(records, dtype=np.float64)
    history_file = os.path.join(output_dir, "ab_optimization_history.dat")
    np.savetxt(
        history_file,
        values,
        delimiter="\t",
        header=(
            "evaluation\tstart_index\tamplitude\tbeta\tobjective\t"
            "enhancement\tdJ_dA\tdJ_dbeta\tdJ_dA_scaled\t"
            "dJ_dbeta_scaled\tprojected_gradient_inf_norm"
        ),
        comments="",
        fmt="%.12e",
    )
    accepted_history_file = os.path.join(
        output_dir, "ab_accepted_trajectory.dat"
    )
    np.savetxt(
        accepted_history_file,
        np.asarray(accepted_records, dtype=np.float64),
        delimiter="\t",
        header=(
            "iteration\tstart_index\tamplitude\tbeta\tobjective\t"
            "enhancement\tdJ_dA\tdJ_dbeta\t"
            "projected_gradient_inf_norm"
        ),
        comments="",
        fmt="%.12e",
    )
    np.savez(
        os.path.join(output_dir, "optimized_ab_parameters.npz"),
        amplitude=best_physical[0],
        beta=best_physical[1],
        objective=best_result["value"],
        short_horizon_enhancement=-best_result["value"],
        gradient_amplitude=best_gradient[0],
        gradient_beta=best_gradient[1],
        projected_gradient_inf_norm=best_projected_norm,
        amplitude_bounds=(AMPLITUDE_MIN, AMPLITUDE_MAX),
        beta_bounds=(BETA_MIN, BETA_MAX),
        parameter_scale=(AB_OPT_SCALE_A, AB_OPT_SCALE_BETA),
        scaled_coordinate_upper=coordinate_upper,
        starts=np.asarray(AB_OPT_STARTS, dtype=np.float64),
        maximum_evaluations=AB_OPT_MAX_EVALUATIONS,
        evaluations_used=len(records),
        accepted_iterations=len(accepted_records),
        window_time=DIFF_WINDOW_TIME,
        reward_start_time=DIFF_REWARD_START_TIME,
        restart=_restart_path(),
    )
    with open(
        os.path.join(output_dir, "optimized_ab_summary.txt"),
        "w",
        encoding="utf-8",
    ) as stream:
        stream.write(f"restart\t{_restart_path()}\n")
        stream.write(f"evaluations_used\t{len(records)}\n")
        stream.write(f"accepted_iterations\t{len(accepted_records)}\n")
        stream.write(f"best_amplitude\t{best_physical[0]:.10e}\n")
        stream.write(f"best_beta\t{best_physical[1]:.10e}\n")
        stream.write(f"best_objective\t{best_result['value']:.10e}\n")
        stream.write(
            f"best_short_horizon_enhancement\t"
            f"{-best_result['value']:.10e}\n"
        )
        stream.write(f"dJ_dA\t{best_gradient[0]:.10e}\n")
        stream.write(f"dJ_dbeta\t{best_gradient[1]:.10e}\n")
        stream.write(
            f"projected_gradient_inf_norm\t"
            f"{best_projected_norm:.10e}\n"
        )
        for report in start_reports:
            stream.write(f"optimizer_report\t{report}\n")
    print(
        f"[AB-optimize] best candidate: A={best_physical[0]:.8f}, "
        f"beta={best_physical[1]:.8f}, "
        f"J={best_result['value']:.8e}, "
        f"projected_grad={best_projected_norm:.3e}"
    )
    return (
        float(best_physical[0]),
        float(best_physical[1]),
        float(best_result["value"]),
    )


def run_beta_optimization(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    state = load_control_restart()
    pid_array = jnp.arange(num_dev, dtype=jnp.int32)
    records = []
    evaluations = {}
    best_value = np.inf
    best_beta = np.nan

    def objective(beta_parameter):
        beta = beta_parameter[0]
        parameters = jnp.stack(
            (jnp.asarray(1.0, dtype=beta.dtype), beta)
        )
        return control_objective_physical(
            parameters,
            state["ub_split"],
            pid_array,
            DIFF_N_BLOCKS,
            DIFF_REWARD_START_BLOCK,
        )

    def evaluate(beta, stage, iteration, bracket_width=np.nan):
        nonlocal best_value, best_beta
        beta = float(np.clip(beta, BETA_MIN, BETA_MAX))
        cache_key = round(beta, 10)
        if cache_key in evaluations:
            return evaluations[cache_key]
        beta_parameter = jnp.asarray((beta,), dtype=jnp.float32)
        value, gradient = forward_mode_gradient(
            objective, beta_parameter
        )
        value_float = float(value)
        gradient_float = float(gradient[0])
        if not np.isfinite(value_float):
            raise FloatingPointError(
                f"Non-finite objective at beta={beta:.10g}"
            )
        if not np.isfinite(gradient_float):
            raise FloatingPointError(
                f"Non-finite automatic-differentiation gradient at "
                f"beta={beta:.10g}"
            )
        result = (beta, value_float, gradient_float)
        evaluations[cache_key] = result
        records.append(
            (
                len(records) + 1,
                stage,
                iteration,
                beta,
                value_float,
                -value_float,
                gradient_float,
                bracket_width,
            )
        )
        if value_float < best_value:
            best_value = value_float
            best_beta = beta
        print(
            f"[beta-optimize] evaluation={len(records):02d}, "
            f"stage={stage}, beta={beta:.8f}, J={value_float:.6e}, "
            f"dJ/dbeta={gradient_float:.6e}"
        )
        return result

    def find_minimum_bracket():
        results = list(evaluations.values())
        candidates = [
            (left, right)
            for left in results
            for right in results
            if left[2] < 0.0 < right[2] and left[0] < right[0]
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda pair: pair[1][0] - pair[0][0])

    def constrained_stationary(result):
        beta, _, gradient = result
        if (
            beta <= BETA_MIN + BETA_OPT_INTERVAL_TOLERANCE
            and gradient > 0.0
        ):
            return True, "lower-bound constrained optimum"
        if (
            beta >= BETA_MAX - BETA_OPT_INTERVAL_TOLERANCE
            and gradient < 0.0
        ):
            return True, "upper-bound constrained optimum"
        return False, ""

    print(
        f"[beta-optimize] A=1, initial beta={BETA_OPT_INITIAL}, "
        f"bounds=[{BETA_MIN}, {BETA_MAX}], "
        f"maximum gradient evaluations={BETA_OPT_MAX_EVALUATIONS}"
    )
    current = evaluate(
        BETA_OPT_INITIAL,
        stage=0,
        iteration=0,
    )
    stationary, reason = constrained_stationary(current)
    bracket = None
    previous = None
    step_size = BETA_OPT_INITIAL_STEP

    if not stationary and len(records) < BETA_OPT_MAX_EVALUATIONS:
        direction = 1.0 if current[2] <= 0.0 else -1.0
        first_step = (
            BETA_OPT_SIGN_CHECK_STEP
            if abs(current[2]) <= BETA_OPT_GRADIENT_TOLERANCE
            else step_size
        )
        trial_beta = float(
            np.clip(
                current[0] + direction * first_step,
                BETA_MIN,
                BETA_MAX,
            )
        )
        previous = current
        current = evaluate(
            trial_beta,
            stage=1,
            iteration=1,
        )
        bracket = find_minimum_bracket()
        stationary, reason = constrained_stationary(current)

    while (
        not stationary
        and bracket is None
        and len(records) < BETA_OPT_MAX_EVALUATIONS
    ):
        previous_beta, previous_value, previous_gradient = previous
        current_beta, current_value, current_gradient = current
        direction = 1.0 if current_gradient <= 0.0 else -1.0
        denominator = current_gradient - previous_gradient

        use_secant_prediction = False
        if (
            abs(denominator) > 1.0e-14
            and abs(current_beta - previous_beta) > 1.0e-14
        ):
            secant_beta = (
                current_beta
                - current_gradient
                * (current_beta - previous_beta)
                / denominator
            )
            secant_step = secant_beta - current_beta
            use_secant_prediction = (
                np.isfinite(secant_beta)
                and secant_step * direction > 0.0
                and abs(secant_step) <= BETA_OPT_MAX_STEP
            )

        if use_secant_prediction:
            trial_beta = secant_beta
            stage = 2
        else:
            if abs(current_gradient) <= BETA_OPT_GRADIENT_TOLERANCE:
                step_size = BETA_OPT_SIGN_CHECK_STEP
            elif current_value < previous_value:
                step_size = min(
                    BETA_OPT_MAX_STEP,
                    max(
                        BETA_OPT_INITIAL_STEP,
                        1.6 * abs(current_beta - previous_beta),
                    ),
                )
            else:
                step_size = max(
                    BETA_OPT_INTERVAL_TOLERANCE,
                    0.5 * abs(current_beta - previous_beta),
                )
            trial_beta = current_beta + direction * step_size
            stage = 1

        trial_beta = float(np.clip(trial_beta, BETA_MIN, BETA_MAX))
        if abs(trial_beta - current_beta) <= 1.0e-12:
            stationary, reason = constrained_stationary(current)
            if not stationary:
                reason = "no admissible update remains inside the bounds"
                stationary = True
            break

        previous = current
        current = evaluate(
            trial_beta,
            stage=stage,
            iteration=len(records),
        )
        bracket = find_minimum_bracket()
        stationary, reason = constrained_stationary(current)

    if bracket is not None:
        left, right = bracket
        while len(records) < BETA_OPT_MAX_EVALUATIONS:
            left_beta, _, left_gradient = left
            right_beta, _, right_gradient = right
            width = right_beta - left_beta
            if (
                abs(left_gradient) <= BETA_OPT_GRADIENT_TOLERANCE
                or abs(right_gradient) <= BETA_OPT_GRADIENT_TOLERANCE
                or width <= BETA_OPT_INTERVAL_TOLERANCE
            ):
                reason = "stationary point bracket converged"
                break

            denominator = right_gradient - left_gradient
            if abs(denominator) > 1.0e-14:
                trial_beta = (
                    right_beta
                    - right_gradient
                    * width
                    / denominator
                )
            else:
                trial_beta = 0.5 * (left_beta + right_beta)

            guard = 0.10 * width
            trial_beta = float(
                np.clip(
                    trial_beta,
                    left_beta + guard,
                    right_beta - guard,
                )
            )
            trial = evaluate(
                trial_beta,
                stage=3,
                iteration=len(records),
                bracket_width=width,
            )
            if trial[2] < 0.0:
                left = trial
            elif trial[2] > 0.0:
                right = trial
            if abs(trial[2]) <= BETA_OPT_GRADIENT_TOLERANCE:
                reason = "gradient tolerance reached with sign bracket"
                break

        bracket = (left, right)

    interior_minimum_verified = bracket is not None
    if interior_minimum_verified:
        negative_side, positive_side = bracket
        print(
            "[beta-optimize] sign verification: "
            f"beta_left={negative_side[0]:.8f}, "
            f"gradient_left={negative_side[2]:.6e}; "
            f"beta_right={positive_side[0]:.8f}, "
            f"gradient_right={positive_side[2]:.6e}"
        )
    else:
        negative_side = (np.nan, np.nan, np.nan)
        positive_side = (np.nan, np.nan, np.nan)

    if not reason:
        reason = "maximum gradient evaluations reached"
    print(f"[beta-optimize] stopping criterion: {reason}")

    values = np.asarray(records, dtype=np.float64)
    np.savetxt(
        os.path.join(output_dir, "beta_optimization_history.dat"),
        values,
        delimiter="\t",
        header=(
            "evaluation\tstage\titeration\tbeta\tobjective\t"
            "enhancement\tdJ_dbeta\tbracket_width"
        ),
        comments="",
        fmt="%.12e",
    )
    np.savez(
        os.path.join(output_dir, "optimized_beta.npz"),
        amplitude=1.0,
        beta=best_beta,
        objective=best_value,
        short_horizon_enhancement=-best_value,
        beta_bounds=(BETA_MIN, BETA_MAX),
        initial_beta=BETA_OPT_INITIAL,
        maximum_gradient_evaluations=BETA_OPT_MAX_EVALUATIONS,
        initial_step=BETA_OPT_INITIAL_STEP,
        maximum_step=BETA_OPT_MAX_STEP,
        sign_check_step=BETA_OPT_SIGN_CHECK_STEP,
        gradient_tolerance=BETA_OPT_GRADIENT_TOLERANCE,
        interval_tolerance=BETA_OPT_INTERVAL_TOLERANCE,
        number_of_evaluations=len(records),
        stopping_reason=reason,
        interior_minimum_verified=interior_minimum_verified,
        negative_side_beta=negative_side[0],
        negative_side_gradient=negative_side[2],
        positive_side_beta=positive_side[0],
        positive_side_gradient=positive_side[2],
        window_time=DIFF_WINDOW_TIME,
        reward_start_time=DIFF_REWARD_START_TIME,
        restart=_restart_path(),
    )
    with open(
        os.path.join(output_dir, "optimized_beta_summary.txt"),
        "w",
        encoding="utf-8",
    ) as stream:
        stream.write(f"restart\t{_restart_path()}\n")
        stream.write("amplitude\t1.0000000000e+00\n")
        stream.write(f"beta_min\t{BETA_MIN:.10e}\n")
        stream.write(f"beta_max\t{BETA_MAX:.10e}\n")
        stream.write(f"initial_beta\t{BETA_OPT_INITIAL:.10e}\n")
        stream.write(
            f"maximum_gradient_evaluations\t"
            f"{BETA_OPT_MAX_EVALUATIONS}\n"
        )
        stream.write(f"initial_step\t{BETA_OPT_INITIAL_STEP:.10e}\n")
        stream.write(f"maximum_step\t{BETA_OPT_MAX_STEP:.10e}\n")
        stream.write(
            f"sign_check_step\t{BETA_OPT_SIGN_CHECK_STEP:.10e}\n"
        )
        stream.write(
            f"gradient_tolerance\t"
            f"{BETA_OPT_GRADIENT_TOLERANCE:.10e}\n"
        )
        stream.write(
            f"interval_tolerance\t"
            f"{BETA_OPT_INTERVAL_TOLERANCE:.10e}\n"
        )
        stream.write(f"number_of_evaluations\t{len(records)}\n")
        stream.write(f"stopping_reason\t{reason}\n")
        stream.write(
            f"interior_minimum_verified\t"
            f"{int(interior_minimum_verified)}\n"
        )
        stream.write(
            f"negative_side_beta\t{negative_side[0]:.10e}\n"
        )
        stream.write(
            f"negative_side_gradient\t{negative_side[2]:.10e}\n"
        )
        stream.write(
            f"positive_side_beta\t{positive_side[0]:.10e}\n"
        )
        stream.write(
            f"positive_side_gradient\t{positive_side[2]:.10e}\n"
        )
        stream.write(f"best_beta\t{best_beta:.10e}\n")
        stream.write(f"best_objective\t{best_value:.10e}\n")
        stream.write(
            f"best_short_horizon_enhancement\t{-best_value:.10e}\n"
        )
    print(
        f"[beta-optimize] best candidate: A=1, beta={best_beta:.6f}, "
        f"J={best_value:.8e}"
    )
    return 1.0, best_beta, best_value


def run_optimized_validation(parameters, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, "optimized_validation_summary.npz")
    final_state_file = os.path.join(output_dir, "final_control_state.npz")
    history_file = os.path.join(output_dir, "optimized_validation_history.dat")
    if (
        PIPELINE_RESUME
        and os.path.exists(summary_file)
        and os.path.exists(final_state_file)
    ):
        with np.load(summary_file) as data:
            return {
                "Nu_bottom": float(data["Nu_bottom"]),
                "Nu_top": float(data["Nu_top"]),
                "enhancement": float(data["enhancement"]),
            }

    state = load_control_restart()
    pid_array = jnp.arange(num_dev, dtype=jnp.int32)
    physical_parameters = jnp.asarray(parameters, dtype=jnp.float32)
    total_time = VALIDATION_TRANSIENT_TIME + VALIDATION_AVERAGING_TIME
    elapsed = 0.0
    bottom_sum = 0.0
    top_sum = 0.0
    count = 0
    with open(history_file, "w", encoding="utf-8") as stream:
        stream.write(
            "elapsed_time\tNu_bottom_block\tNu_top_block\tNu_bottom_avg\t"
            "Nu_top_avg\tenhancement_avg\tthreshold\taction_mean\t"
            "action_min\taction_max\taction_rms\tamplitude\tbeta\n"
        )

    last_diagnostics = [np.nan] * 5
    while elapsed < total_time - 0.5 * CONTROL_LOG_TIME:
        results = pmap_control_window(
            state["ub_split"],
            pid_array,
            Jx,
            Jxx,
            Jy,
            Jyy,
            Jz_split,
            Jzz_split,
            physical_parameters,
            CONTROL_LOG_BLOCKS,
            0,
        )
        state["ub_split"] = results[0]
        elapsed += CONTROL_LOG_TIME
        bottom = float(results[2][0])
        top = float(results[3][0])
        if elapsed > VALIDATION_TRANSIENT_TIME + 0.5 * CONTROL_LOG_TIME:
            bottom_sum += bottom
            top_sum += top
            count += 1
        bottom_avg = bottom_sum / max(count, 1)
        top_avg = top_sum / max(count, 1)
        enhancement = (
            (bottom_avg - CONTROL_NU0) / CONTROL_NU0 if count else 0.0
        )
        last_diagnostics = [float(results[i][0]) for i in range(5, 10)]
        with open(history_file, "a", encoding="utf-8") as stream:
            stream.write(
                f"{elapsed:.8f}\t{bottom:.10e}\t{top:.10e}\t"
                f"{bottom_avg:.10e}\t{top_avg:.10e}\t{enhancement:.10e}\t"
                f"{last_diagnostics[0]:.10e}\t{last_diagnostics[1]:.10e}\t"
                f"{last_diagnostics[2]:.10e}\t{last_diagnostics[3]:.10e}\t"
                f"{last_diagnostics[4]:.10e}\t{parameters[0]:.8f}\t"
                f"{parameters[1]:.8f}\n"
            )
        if int(round(elapsed / CONTROL_LOG_TIME)) % 20 == 0:
            print(
                f"[validation] t={elapsed:.1f}/{total_time:.1f}, "
                f"Nu_b={bottom:.4f}, eta_avg={100.0 * enhancement:.3f}%"
            )

    if count == 0:
        raise RuntimeError("The controlled averaging window produced no samples")
    bottom_avg = bottom_sum / count
    top_avg = top_sum / count
    enhancement = (bottom_avg - CONTROL_NU0) / CONTROL_NU0
    wall_mismatch = (
        2.0 * abs(bottom_avg - top_avg) / (abs(bottom_avg) + abs(top_avg))
    )
    np.savez(
        summary_file,
        Nu0=CONTROL_NU0,
        Nu_bottom=bottom_avg,
        Nu_top=top_avg,
        enhancement=enhancement,
        wall_mismatch=wall_mismatch,
        amplitude=parameters[0],
        beta=parameters[1],
        samples=count,
        transient_time=VALIDATION_TRANSIENT_TIME,
        averaging_time=VALIDATION_AVERAGING_TIME,
        threshold=last_diagnostics[0],
        action_mean=last_diagnostics[1],
        action_min=last_diagnostics[2],
        action_max=last_diagnostics[3],
        action_rms=last_diagnostics[4],
    )

    device_array = np.asarray(state["ub_split"], dtype=np.float32)
    final_field = np.concatenate(
        [device_array[pid] for pid in range(num_dev)], axis=2
    )
    np.savez(
        final_state_file,
        ctime=float(state["ctime"] + total_time),
        iter=int(state["iter"] + round(total_time / dt)),
        ub=final_field,
        amplitude=parameters[0],
        beta=parameters[1],
        Nu0=CONTROL_NU0,
        Nu_bottom=bottom_avg,
        Nu_top=top_avg,
    )
    print(
        f"[validation] completed: Nu_bottom={bottom_avg:.6f}, "
        f"Nu_top={top_avg:.6f}, enhancement={100.0 * enhancement:.3f}%, "
        f"wall mismatch={100.0 * wall_mismatch:.3f}%"
    )
    return {
        "Nu_bottom": bottom_avg,
        "Nu_top": top_avg,
        "enhancement": enhancement,
    }


def write_control_summary(root_dir, parameters, short_objective, validation):
    filename = os.path.join(root_dir, "pipeline_summary.txt")
    with open(filename, "w", encoding="utf-8") as stream:
        stream.write(f"profile\t{PIPELINE_PROFILE}\n")
        stream.write(f"restart\t{_restart_path()}\n")
        stream.write(f"grid\t{npx}x{npy}x{npz}\n")
        stream.write(f"Ra\t{Ra:.8e}\n")
        stream.write(f"Pr\t{Pr:.8e}\n")
        stream.write(f"Nu0_bottom\t{CONTROL_NU0:.10e}\n")
        stream.write(f"sensor_z\t{CONTROL_SENSOR_Z:.10e}\n")
        stream.write(f"optimal_amplitude\t{parameters[0]:.10e}\n")
        stream.write(f"optimal_beta\t{parameters[1]:.10e}\n")
        stream.write(f"short_window_objective\t{short_objective:.10e}\n")
        stream.write(
            f"validated_Nu_bottom\t{validation['Nu_bottom']:.10e}\n"
        )
        stream.write(f"validated_Nu_top\t{validation['Nu_top']:.10e}\n")
        stream.write(
            f"validated_enhancement\t{validation['enhancement']:.10e}\n"
        )


def write_paper_reference_summary(root_dir, parameters, validation):
    filename = os.path.join(root_dir, "paper_reference_summary.txt")
    with open(filename, "w", encoding="utf-8") as stream:
        stream.write(f"profile\t{PIPELINE_PROFILE}\n")
        stream.write("mode\tpaper_reference\n")
        stream.write(f"restart\t{_restart_path()}\n")
        stream.write(f"grid\t{npx}x{npy}x{npz}\n")
        stream.write(f"Ra\t{Ra:.8e}\n")
        stream.write(f"Pr\t{Pr:.8e}\n")
        stream.write(f"Nu0_bottom\t{CONTROL_NU0:.10e}\n")
        stream.write(f"sensor_z\t{CONTROL_SENSOR_Z:.10e}\n")
        stream.write(f"paper_amplitude\t{parameters[0]:.10e}\n")
        stream.write(f"paper_beta\t{parameters[1]:.10e}\n")
        stream.write(
            f"validated_Nu_bottom\t{validation['Nu_bottom']:.10e}\n"
        )
        stream.write(f"validated_Nu_top\t{validation['Nu_top']:.10e}\n")
        stream.write(
            f"validated_enhancement\t{validation['enhancement']:.10e}\n"
        )


def write_continuation_summary(root_dir, parameters, validation):
    with np.load(_restart_path()) as restart:
        start_time = float(restart["ctime"])
    end_time = start_time + VALIDATION_AVERAGING_TIME
    np.savez(
        os.path.join(root_dir, "statistics_400t_summary.npz"),
        Nu0=CONTROL_NU0,
        amplitude=parameters[0],
        beta=parameters[1],
        start_time=start_time,
        end_time=end_time,
        averaging_time=VALIDATION_AVERAGING_TIME,
        Nu_bottom=validation["Nu_bottom"],
        Nu_top=validation["Nu_top"],
        enhancement=validation["enhancement"],
    )
    with open(
        os.path.join(root_dir, "statistics_400t_summary.txt"),
        "w",
        encoding="utf-8",
    ) as stream:
        stream.write(f"restart\t{_restart_path()}\n")
        stream.write(f"start_time\t{start_time:.10e}\n")
        stream.write(f"end_time\t{end_time:.10e}\n")
        stream.write(
            f"averaging_time\t{VALIDATION_AVERAGING_TIME:.10e}\n"
        )
        stream.write(f"paper_amplitude\t{parameters[0]:.10e}\n")
        stream.write(f"paper_beta\t{parameters[1]:.10e}\n")
        stream.write(f"Nu0\t{CONTROL_NU0:.10e}\n")
        stream.write(f"Nu_bottom\t{validation['Nu_bottom']:.10e}\n")
        stream.write(f"Nu_top\t{validation['Nu_top']:.10e}\n")
        stream.write(f"enhancement\t{validation['enhancement']:.10e}\n")


def validate_pipeline_configuration():
    if OPT_ITERATIONS < 1:
        raise ValueError("RB_OPT_ITERATIONS must be at least 1")
    if BETA_OPT_MAX_EVALUATIONS < 2:
        raise ValueError(
            "RB_BETA_OPT_MAX_EVALUATIONS must be at least 2"
        )
    if not BETA_MIN <= BETA_OPT_INITIAL <= BETA_MAX:
        raise ValueError(
            "RB_BETA_OPT_INITIAL must lie within the beta bounds"
        )
    if BETA_OPT_INITIAL_STEP <= 0.0:
        raise ValueError("RB_BETA_OPT_INITIAL_STEP must be positive")
    if BETA_OPT_MAX_STEP <= 0.0:
        raise ValueError("RB_BETA_OPT_MAX_STEP must be positive")
    if BETA_OPT_SIGN_CHECK_STEP <= 0.0:
        raise ValueError("RB_BETA_OPT_SIGN_CHECK_STEP must be positive")
    if BETA_OPT_GRADIENT_TOLERANCE <= 0.0:
        raise ValueError(
            "RB_BETA_OPT_GRADIENT_TOLERANCE must be positive"
        )
    if BETA_OPT_INTERVAL_TOLERANCE <= 0.0:
        raise ValueError(
            "RB_BETA_OPT_INTERVAL_TOLERANCE must be positive"
        )
    if AB_OPT_MAX_EVALUATIONS < 4:
        raise ValueError("RB_AB_OPT_MAX_EVALUATIONS must be at least 4")
    if AB_OPT_EVALUATIONS_PER_START < 2:
        raise ValueError(
            "RB_AB_OPT_EVALUATIONS_PER_START must be at least 2"
        )
    if AB_OPT_SCALE_A <= 0.0 or AB_OPT_SCALE_BETA <= 0.0:
        raise ValueError("The A-beta optimization scales must be positive")
    if AB_OPT_LOCAL_STARTS < 1:
        raise ValueError("RB_AB_OPT_LOCAL_STARTS must be at least 1")
    if AB_OPT_LOCAL_STARTS > len(AB_OPT_STARTS):
        raise ValueError(
            "RB_AB_OPT_LOCAL_STARTS cannot exceed the number of design points"
        )
    if not AB_OPT_STARTS:
        raise ValueError("RB_AB_OPT_STARTS must contain at least one point")
    if AB_OPT_MAX_EVALUATIONS < len(AB_OPT_STARTS):
        raise ValueError(
            "RB_AB_OPT_MAX_EVALUATIONS must cover every design point"
        )
    for start in AB_OPT_STARTS:
        if len(start) != 2:
            raise ValueError("Each RB_AB_OPT_STARTS entry must contain A:beta")
        if not AMPLITUDE_MIN <= start[0] <= AMPLITUDE_MAX:
            raise ValueError(f"Initial amplitude {start[0]} is out of bounds")
        if not BETA_MIN <= start[1] <= BETA_MAX:
            raise ValueError(f"Initial beta {start[1]} is out of bounds")
    if VALIDATION_TRANSIENT_TIME < 0.0 or VALIDATION_AVERAGING_TIME <= 0.0:
        raise ValueError(
            "Validation transient time must be nonnegative and averaging "
            "time must be positive"
        )
    tests = (
        (CONTROL_ACTION_STEPS * dt, CONTROL_ACTION_TIME, "control action"),
        (CONTROL_LOG_BLOCKS * CONTROL_ACTION_TIME, CONTROL_LOG_TIME, "log block"),
        (DIFF_N_BLOCKS * CONTROL_ACTION_TIME, DIFF_WINDOW_TIME, "AD window"),
    )
    for represented, requested, name in tests:
        if not np.isclose(represented, requested):
            raise ValueError(
                f"The {name} time ({requested}) must be an integer multiple "
                f"of its lower-level step; represented value is {represented}."
            )
    if CONTROL_ACTION_STEPS < 1 or CONTROL_LOG_BLOCKS < 1:
        raise ValueError("Control and logging blocks must contain at least one step")
    if not (0 <= DIFF_REWARD_START_BLOCK < DIFF_N_BLOCKS):
        raise ValueError("Invalid optimization reward window")
    if not (0 <= GRAD_CHECK_REWARD_START_BLOCK < GRAD_CHECK_N_BLOCKS):
        raise ValueError("Invalid gradient-check reward window")


def run_all_control_pipeline():
    validate_pipeline_configuration()
    if PIPELINE_MODE not in (
        "optimize",
        "paper_reference",
        "paper_continue",
        "beta_optimize",
        "ab_lbfgsb",
    ):
        raise ValueError(
            "RB_PIPELINE_MODE must be 'optimize', 'paper_reference', "
            "'paper_continue', 'beta_optimize', or 'ab_lbfgsb'"
        )
    case_tag = (
        f"{npx}x{npy}x{npz}_Ra{Ra:.0e}_Pr{Pr:g}"
        .replace("+", "")
        .replace(".", "p")
    )
    if PIPELINE_MODE == "paper_reference":
        root_name = f"control_paper_reference_{PIPELINE_PROFILE}_{case_tag}"
    elif PIPELINE_MODE == "paper_continue":
        root_name = f"control_paper_continuation_{PIPELINE_PROFILE}_{case_tag}"
    elif PIPELINE_MODE == "beta_optimize":
        root_name = f"control_beta_optimization_{PIPELINE_PROFILE}_{case_tag}"
    elif PIPELINE_MODE == "ab_lbfgsb":
        root_name = f"control_AB_lbfgsb_{PIPELINE_PROFILE}_{case_tag}"
    else:
        root_name = f"control_optimization_{PIPELINE_PROFILE}_{case_tag}"
    if PIPELINE_OUTPUT_TAG:
        root_name = f"{root_name}_{PIPELINE_OUTPUT_TAG}"
    root_dir = os.path.join(script_dir, root_name)
    os.makedirs(root_dir, exist_ok=True)
    print("=" * 76)
    print("Differentiable RB thermal-control calculation")
    print(
        f"mode={PIPELINE_MODE}, profile={PIPELINE_PROFILE}, "
        f"grid={npx}x{npy}x{npz}, "
        f"Ra={Ra:.3e}, Pr={Pr:g}, devices={jax.local_devices()}"
    )
    print("=" * 76)

    baseline_nu = run_uncontrolled_baseline(
        os.path.join(root_dir, "00_uncontrolled_baseline")
    )
    configure_control_context(baseline_nu)
    print(
        f"[sensor] Nu0={CONTROL_NU0:.6f}, z_delta={CONTROL_SENSOR_Z:.8f}, "
        f"indices=({CONTROL_SENSOR_K0}, {CONTROL_SENSOR_K1}), "
        f"alpha={CONTROL_SENSOR_ALPHA:.8f}"
    )

    if PIPELINE_MODE == "ab_lbfgsb":
        amplitude, beta, short_objective = run_ab_lbfgsb_optimization(
            os.path.join(root_dir, "01_AB_lbfgsb_optimization")
        )
        validation = run_optimized_validation(
            (amplitude, beta),
            os.path.join(root_dir, "02_AB_optimized_long_validation"),
        )
        write_control_summary(
            root_dir, (amplitude, beta), short_objective, validation
        )
        print("=" * 76)
        print("Bound-constrained A-beta L-BFGS-B optimization complete")
        print(
            f"Optimal candidate: A={amplitude:.6f}, beta={beta:.6f}"
        )
        print(
            f"Long-time validation: Nu_b={validation['Nu_bottom']:.6f}, "
            f"Nu_t={validation['Nu_top']:.6f}, "
            f"enhancement={100.0 * validation['enhancement']:.3f}%"
        )
        print(f"Results: {root_dir}")
        print("=" * 76)
        return

    if PIPELINE_MODE == "beta_optimize":
        amplitude, beta, short_objective = run_beta_optimization(
            os.path.join(root_dir, "01_beta_gradient_optimization")
        )
        validation = run_optimized_validation(
            (amplitude, beta),
            os.path.join(root_dir, "02_beta_optimized_long_validation"),
        )
        write_control_summary(
            root_dir, (amplitude, beta), short_objective, validation
        )
        print("=" * 76)
        print("Fixed-amplitude beta optimization complete")
        print(
            f"Optimal candidate: A=1.000000, beta={beta:.6f}"
        )
        print(
            f"Long-time validation: Nu_b={validation['Nu_bottom']:.6f}, "
            f"Nu_t={validation['Nu_top']:.6f}, "
            f"enhancement={100.0 * validation['enhancement']:.3f}%"
        )
        print(f"Results: {root_dir}")
        print("=" * 76)
        return

    if PIPELINE_MODE in ("paper_reference", "paper_continue"):
        paper_parameters = (1.0, 20.0)
        if PIPELINE_MODE == "paper_continue":
            validation_dir = "01_additional_400t_statistics"
        else:
            validation_dir = "01_paper_parameter_long_validation"
        validation = run_optimized_validation(
            paper_parameters,
            os.path.join(root_dir, validation_dir),
        )
        if PIPELINE_MODE == "paper_continue":
            write_continuation_summary(
                root_dir, paper_parameters, validation
            )
            print("=" * 76)
            print("Paper-parameter continuation complete")
            print(
                f"Statistics over {VALIDATION_AVERAGING_TIME:.1f}t: "
                f"Nu_b={validation['Nu_bottom']:.6f}, "
                f"Nu_t={validation['Nu_top']:.6f}, "
                f"enhancement={100.0 * validation['enhancement']:.3f}%"
            )
            print(f"Results: {root_dir}")
            print("=" * 76)
            return
        write_paper_reference_summary(
            root_dir, paper_parameters, validation
        )
        print("=" * 76)
        print("Paper-parameter validation complete")
        print("Paper parameters: A=1.000000, beta=20.000000")
        print(
            f"Long-time validation: Nu_b={validation['Nu_bottom']:.6f}, "
            f"Nu_t={validation['Nu_top']:.6f}, "
            f"enhancement={100.0 * validation['enhancement']:.3f}%"
        )
        print(f"Results: {root_dir}")
        print("=" * 76)
        return

    run_gradient_check(os.path.join(root_dir, "01_gradient_check"))
    amplitude, beta, short_objective = run_parameter_optimization(
        os.path.join(root_dir, "02_parameter_optimization")
    )
    validation = run_optimized_validation(
        (amplitude, beta),
        os.path.join(root_dir, "03_optimized_long_validation"),
    )
    write_control_summary(
        root_dir, (amplitude, beta), short_objective, validation
    )
    print("=" * 76)
    print("Pipeline complete")
    print(f"Optimal candidate: A={amplitude:.6f}, beta={beta:.6f}")
    print(
        f"Long-time validation: Nu_b={validation['Nu_bottom']:.6f}, "
        f"Nu_t={validation['Nu_top']:.6f}, "
        f"enhancement={100.0 * validation['enhancement']:.3f}%"
    )
    print(f"Results: {root_dir}")
    print("=" * 76)

# ---------------------------------------------------------------------------
# Final coupled damped-BFGS driver.  Keeping this configuration in a function
# preserves the separate global namespaces used by the original two files.
# ---------------------------------------------------------------------------
def _run_coupled_damped_bfgs_standalone() -> None:
    import os
    import sys

    import jax.numpy as jnp
    import numpy as np

    solver = sys.modules[__name__]

    INITIAL_A = float(os.environ.get("RB_AB_BFGS_INITIAL_A", "0.8"))
    INITIAL_BETA = float(os.environ.get("RB_AB_BFGS_INITIAL_BETA", "10.0"))
    MAX_EVALUATIONS = int(os.environ.get("RB_AB_BFGS_MAX_EVALUATIONS", "30"))
    PARAMETER_SCALE = np.asarray(
        (
            float(os.environ.get("RB_AB_BFGS_SCALE_A", "0.13")),
            float(os.environ.get("RB_AB_BFGS_SCALE_BETA", "18.0")),
        ),
        dtype=np.float64,
    )
    INITIAL_TRUST_RADIUS = float(
        os.environ.get("RB_AB_BFGS_INITIAL_TRUST_RADIUS", "0.25")
    )
    MIN_TRUST_RADIUS = float(
        os.environ.get("RB_AB_BFGS_MIN_TRUST_RADIUS", "0.02")
    )
    MAX_TRUST_RADIUS = float(
        os.environ.get("RB_AB_BFGS_MAX_TRUST_RADIUS", "1.0")
    )
    TRUST_SHRINK_FACTOR = float(
        os.environ.get("RB_AB_BFGS_TRUST_SHRINK_FACTOR", "0.5")
    )
    TRUST_GROW_FACTOR = float(
        os.environ.get("RB_AB_BFGS_TRUST_GROW_FACTOR", "1.25")
    )
    GRADIENT_TOLERANCE = float(
        os.environ.get("RB_AB_BFGS_GRADIENT_TOLERANCE", "1.0e-5")
    )
    OBJECTIVE_TOLERANCE = float(
        os.environ.get("RB_AB_BFGS_OBJECTIVE_TOLERANCE", "1.0e-8")
    )
    MAX_STAGNANT_STEPS = int(
        os.environ.get("RB_AB_BFGS_MAX_STAGNANT_STEPS", "3")
    )
    ARMIJO_CONSTANT = float(
        os.environ.get("RB_AB_BFGS_ARMIJO_CONSTANT", "1.0e-4")
    )
    BACKTRACK_FACTOR = float(
        os.environ.get("RB_AB_BFGS_BACKTRACK_FACTOR", "0.5")
    )
    MAX_LINE_SEARCH_EVALUATIONS = int(
        os.environ.get("RB_AB_BFGS_MAX_LINE_SEARCH", "6")
    )

    LOWER = np.asarray((solver.AMPLITUDE_MIN, solver.BETA_MIN), dtype=np.float64)
    UPPER = np.asarray((solver.AMPLITUDE_MAX, solver.BETA_MAX), dtype=np.float64)
    COORDINATE_LOWER = np.zeros(2, dtype=np.float64)
    COORDINATE_UPPER = (UPPER - LOWER) / PARAMETER_SCALE


    def projected_gradient(coordinates: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        projected = np.asarray(gradient, dtype=np.float64).copy()
        tolerance = 1.0e-10
        for index in range(2):
            at_lower = coordinates[index] <= COORDINATE_LOWER[index] + tolerance
            at_upper = coordinates[index] >= COORDINATE_UPPER[index] - tolerance
            if (at_lower and projected[index] > 0.0) or (
                at_upper and projected[index] < 0.0
            ):
                projected[index] = 0.0
        return projected


    def admissible_direction(
        coordinates: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        direction = np.asarray(direction, dtype=np.float64).copy()
        tolerance = 1.0e-10
        for index in range(2):
            if (
                coordinates[index] <= COORDINATE_LOWER[index] + tolerance
                and direction[index] < 0.0
            ):
                direction[index] = 0.0
            if (
                coordinates[index] >= COORDINATE_UPPER[index] - tolerance
                and direction[index] > 0.0
            ):
                direction[index] = 0.0
        return direction


    def damp_coupled_direction(
        direction: np.ndarray, trust_radius: float
    ) -> tuple[np.ndarray, float]:
        """Scale the complete coupled direction by one trust-region factor."""
        direction = np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm <= trust_radius or norm <= 1.0e-15:
            return direction, 1.0
        factor = trust_radius / norm
        return direction * factor, factor


    def validate_configuration() -> None:
        solver.validate_pipeline_configuration()
        initial = np.asarray((INITIAL_A, INITIAL_BETA), dtype=np.float64)
        if np.any(initial < LOWER) or np.any(initial > UPPER):
            raise ValueError("The initial A-beta pair is outside its bounds")
        if MAX_EVALUATIONS < 2:
            raise ValueError("RB_AB_BFGS_MAX_EVALUATIONS must be at least two")
        if np.any(PARAMETER_SCALE <= 0.0):
            raise ValueError("BFGS parameter scales must be positive")
        if not 0.0 < MIN_TRUST_RADIUS <= INITIAL_TRUST_RADIUS <= MAX_TRUST_RADIUS:
            raise ValueError("Trust radii must satisfy 0 < min <= initial <= max")
        if not 0.0 < TRUST_SHRINK_FACTOR < 1.0:
            raise ValueError("The trust-radius shrink factor must lie in (0, 1)")
        if TRUST_GROW_FACTOR <= 1.0:
            raise ValueError("The trust-radius growth factor must exceed one")
        if GRADIENT_TOLERANCE <= 0.0 or OBJECTIVE_TOLERANCE <= 0.0:
            raise ValueError("BFGS convergence tolerances must be positive")
        if not 0.0 < ARMIJO_CONSTANT < 1.0:
            raise ValueError("The Armijo constant must lie between zero and one")
        if not 0.0 < BACKTRACK_FACTOR < 1.0:
            raise ValueError("The backtracking factor must lie between zero and one")
        if MAX_LINE_SEARCH_EVALUATIONS < 1:
            raise ValueError("At least one line-search evaluation is required")


    def run_coupled_damped_bfgs(output_dir: str) -> tuple[float, float, float]:
        os.makedirs(output_dir, exist_ok=True)
        state = solver.load_control_restart()
        pid_array = jnp.arange(solver.num_dev, dtype=jnp.int32)
        evaluations = []
        accepted = []
        cache = {}
        best = None

        def objective_physical(parameters):
            return solver.control_objective_physical(
                parameters,
                state["ub_split"],
                pid_array,
                solver.DIFF_N_BLOCKS,
                solver.DIFF_REWARD_START_BLOCK,
            )

        def evaluate(coordinates: np.ndarray, stage: str) -> dict:
            nonlocal best
            coordinates = np.clip(
                np.asarray(coordinates, dtype=np.float64),
                COORDINATE_LOWER,
                COORDINATE_UPPER,
            )
            key = tuple(np.round(coordinates, 11))
            if key in cache:
                return cache[key]
            if len(evaluations) >= MAX_EVALUATIONS:
                raise RuntimeError("BFGS evaluation budget reached")

            physical = LOWER + PARAMETER_SCALE * coordinates
            value, gradient_physical = solver.forward_mode_gradient(
                objective_physical,
                jnp.asarray(physical, dtype=jnp.float32),
            )
            value = float(value)
            gradient_physical = np.asarray(gradient_physical, dtype=np.float64)
            if not np.isfinite(value) or not np.all(np.isfinite(gradient_physical)):
                raise FloatingPointError(
                    "Non-finite objective or automatic-differentiation gradient"
                )
            gradient_coordinates = gradient_physical * PARAMETER_SCALE
            projected = projected_gradient(coordinates, gradient_coordinates)
            result = {
                "coordinates": coordinates,
                "physical": physical,
                "value": value,
                "gradient_physical": gradient_physical,
                "gradient_coordinates": gradient_coordinates,
                "projected_gradient": projected,
                "projected_norm": float(np.linalg.norm(projected, ord=np.inf)),
            }
            cache[key] = result
            evaluations.append(
                (
                    len(evaluations) + 1,
                    physical[0],
                    physical[1],
                    value,
                    -value,
                    gradient_physical[0],
                    gradient_physical[1],
                    result["projected_norm"],
                    1.0 if stage == "initial" else 0.0,
                )
            )
            if best is None or value < best["value"]:
                best = result
            print(
                f"[AB-BFGS-eval] evaluation={len(evaluations):02d}, "
                f"stage={stage}, A={physical[0]:.8f}, beta={physical[1]:.8f}, "
                f"J={value:.6e}, dJ/dA={gradient_physical[0]:.6e}, "
                f"dJ/dbeta={gradient_physical[1]:.6e}"
            )
            return result

        def append_accepted(
            result: dict,
            delta: np.ndarray,
            raw_delta: np.ndarray,
            damping: float,
            trust_radius: float,
            model_ratio: float,
            alpha: float,
        ) -> None:
            physical = result["physical"]
            gradient = result["gradient_physical"]
            accepted.append(
                (
                    len(accepted) + 1,
                    physical[0],
                    physical[1],
                    delta[0],
                    delta[1],
                    result["value"],
                    -result["value"],
                    gradient[0],
                    gradient[1],
                    result["projected_norm"],
                    raw_delta[0],
                    raw_delta[1],
                    damping,
                    trust_radius,
                    model_ratio,
                    alpha,
                )
            )
            print(
                f"[AB-BFGS-accepted] iteration={len(accepted):02d}, "
                f"A={physical[0]:.8f}, beta={physical[1]:.8f}, "
                f"delta=({delta[0]:+.6f}, {delta[1]:+.6f}), "
                f"raw_delta=({raw_delta[0]:+.6f}, {raw_delta[1]:+.6f}), "
                f"J={result['value']:.6e}, dJ/dA={gradient[0]:.6e}, "
                f"dJ/dbeta={gradient[1]:.6e}, damping={damping:.4f}, "
                f"radius={trust_radius:.4f}, ratio={model_ratio:.4f}, "
                f"alpha={alpha:.4f}"
            )

        initial = np.asarray((INITIAL_A, INITIAL_BETA), dtype=np.float64)
        current = evaluate((initial - LOWER) / PARAMETER_SCALE, "initial")
        append_accepted(
            current,
            np.zeros(2, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            1.0,
            INITIAL_TRUST_RADIUS,
            np.nan,
            0.0,
        )
        inverse_hessian = np.eye(2, dtype=np.float64)
        trust_radius = INITIAL_TRUST_RADIUS
        stagnant_steps = 0
        stopping_reason = "maximum gradient evaluations reached"

        while len(evaluations) < MAX_EVALUATIONS:
            coordinates = current["coordinates"]
            gradient = current["gradient_coordinates"]
            projected = current["projected_gradient"]
            if current["projected_norm"] <= GRADIENT_TOLERANCE:
                stopping_reason = "projected-gradient tolerance reached"
                break

            raw_direction = -inverse_hessian @ projected
            raw_direction = admissible_direction(coordinates, raw_direction)
            if np.dot(gradient, raw_direction) >= -1.0e-14:
                inverse_hessian = np.eye(2, dtype=np.float64)
                raw_direction = admissible_direction(coordinates, -projected)
            raw_delta_physical = PARAMETER_SCALE * raw_direction
            direction, damping = damp_coupled_direction(
                raw_direction, trust_radius
            )
            if np.linalg.norm(direction, ord=np.inf) <= 1.0e-14:
                stopping_reason = "no admissible BFGS direction remains"
                break

            accepted_candidate = None
            accepted_alpha = 0.0
            alpha = 1.0
            for line_search_index in range(MAX_LINE_SEARCH_EVALUATIONS):
                if len(evaluations) >= MAX_EVALUATIONS:
                    break
                candidate_coordinates = np.clip(
                    coordinates + alpha * direction,
                    COORDINATE_LOWER,
                    COORDINATE_UPPER,
                )
                step_coordinates = candidate_coordinates - coordinates
                if np.linalg.norm(step_coordinates, ord=np.inf) <= 1.0e-14:
                    break
                candidate = evaluate(
                    candidate_coordinates,
                    f"line-search-{line_search_index + 1}",
                )
                armijo_rhs = current["value"] + ARMIJO_CONSTANT * np.dot(
                    gradient, step_coordinates
                )
                if candidate["value"] <= armijo_rhs:
                    accepted_candidate = candidate
                    accepted_alpha = alpha
                    break
                alpha *= BACKTRACK_FACTOR

            if accepted_candidate is None:
                stopping_reason = "Armijo line search did not find an improving step"
                break

            previous = current
            current = accepted_candidate
            step_coordinates = (
                current["coordinates"] - previous["coordinates"]
            )
            delta_physical = PARAMETER_SCALE * step_coordinates

            symmetric_inverse = 0.5 * (
                inverse_hessian + inverse_hessian.T
            )
            hessian_model = np.linalg.pinv(symmetric_inverse, rcond=1.0e-10)
            predicted_reduction = -float(
                np.dot(gradient, step_coordinates)
                + 0.5
                * np.dot(step_coordinates, hessian_model @ step_coordinates)
            )
            if predicted_reduction <= 1.0e-14:
                predicted_reduction = -float(
                    np.dot(gradient, step_coordinates)
                )
            actual_reduction = previous["value"] - current["value"]
            model_ratio = actual_reduction / max(predicted_reduction, 1.0e-14)
            used_radius = (
                np.linalg.norm(step_coordinates) >= 0.8 * trust_radius
            )
            accepted_radius = trust_radius
            if model_ratio < 0.25:
                trust_radius = max(
                    MIN_TRUST_RADIUS, trust_radius * TRUST_SHRINK_FACTOR
                )
            elif model_ratio > 0.75 and used_radius:
                trust_radius = min(
                    MAX_TRUST_RADIUS, trust_radius * TRUST_GROW_FACTOR
                )

            append_accepted(
                current,
                delta_physical,
                raw_delta_physical,
                damping * accepted_alpha,
                accepted_radius,
                model_ratio,
                accepted_alpha,
            )
            print(
                f"[AB-BFGS-trust] actual={actual_reduction:.6e}, "
                f"predicted={predicted_reduction:.6e}, "
                f"ratio={model_ratio:.4f}, next_radius={trust_radius:.4f}"
            )

            gradient_change = (
                current["gradient_coordinates"]
                - previous["gradient_coordinates"]
            )
            curvature = float(np.dot(gradient_change, step_coordinates))
            curvature_scale = (
                np.linalg.norm(gradient_change) * np.linalg.norm(step_coordinates)
            )
            if curvature > max(1.0e-12, 1.0e-10 * curvature_scale):
                rho = 1.0 / curvature
                identity = np.eye(2, dtype=np.float64)
                left = identity - rho * np.outer(
                    step_coordinates, gradient_change
                )
                right = identity - rho * np.outer(
                    gradient_change, step_coordinates
                )
                inverse_hessian = (
                    left @ inverse_hessian @ right
                    + rho * np.outer(step_coordinates, step_coordinates)
                )
            else:
                inverse_hessian = np.eye(2, dtype=np.float64)

            improvement = previous["value"] - current["value"]
            if improvement <= OBJECTIVE_TOLERANCE:
                stagnant_steps += 1
            else:
                stagnant_steps = 0
            if stagnant_steps >= MAX_STAGNANT_STEPS:
                stopping_reason = "accepted-step objective tolerance reached"
                break

        evaluation_values = np.asarray(evaluations, dtype=np.float64)
        accepted_values = np.asarray(accepted, dtype=np.float64)
        np.savetxt(
            os.path.join(output_dir, "bfgs_all_evaluations.dat"),
            evaluation_values,
            delimiter="\t",
            header=(
                "evaluation\tamplitude\tbeta\tobjective\tenhancement\t"
                "dJ_dA\tdJ_dbeta\tprojected_gradient_inf_norm\tis_initial"
            ),
            comments="",
            fmt="%.12e",
        )
        np.savetxt(
            os.path.join(output_dir, "bfgs_accepted_trajectory.dat"),
            accepted_values,
            delimiter="\t",
            header=(
                "iteration\tamplitude\tbeta\tdelta_A\tdelta_beta\tobjective\t"
                "enhancement\tdJ_dA\tdJ_dbeta\t"
                "projected_gradient_inf_norm\traw_delta_A\traw_delta_beta\t"
                "coupled_damping\ttrust_radius\tmodel_ratio\tline_search_alpha"
            ),
            comments="",
            fmt="%.12e",
        )

        best_physical = np.asarray(best["physical"], dtype=np.float64)
        best_gradient = np.asarray(best["gradient_physical"], dtype=np.float64)
        np.savez(
            os.path.join(output_dir, "optimized_coupled_damped_bfgs_parameters.npz"),
            amplitude=best_physical[0],
            beta=best_physical[1],
            objective=best["value"],
            short_horizon_enhancement=-best["value"],
            gradient_amplitude=best_gradient[0],
            gradient_beta=best_gradient[1],
            projected_gradient_inf_norm=best["projected_norm"],
            amplitude_bounds=(solver.AMPLITUDE_MIN, solver.AMPLITUDE_MAX),
            beta_bounds=(solver.BETA_MIN, solver.BETA_MAX),
            parameter_scale_A=PARAMETER_SCALE[0],
            parameter_scale_beta=PARAMETER_SCALE[1],
            initial_trust_radius=INITIAL_TRUST_RADIUS,
            final_trust_radius=trust_radius,
            evaluations_used=len(evaluations),
            accepted_iterations=len(accepted),
            stopping_reason=stopping_reason,
            restart=solver._restart_path(),
        )
        with open(
            os.path.join(output_dir, "coupled_damped_bfgs_summary.txt"),
            "w",
            encoding="utf-8",
        ) as stream:
            stream.write(f"restart\t{solver._restart_path()}\n")
            stream.write(f"evaluations_used\t{len(evaluations)}\n")
            stream.write(f"accepted_iterations\t{len(accepted)}\n")
            stream.write(f"stopping_reason\t{stopping_reason}\n")
            stream.write(f"parameter_scale_A\t{PARAMETER_SCALE[0]:.10e}\n")
            stream.write(
                f"parameter_scale_beta\t{PARAMETER_SCALE[1]:.10e}\n"
            )
            stream.write(
                f"initial_trust_radius\t{INITIAL_TRUST_RADIUS:.10e}\n"
            )
            stream.write(f"final_trust_radius\t{trust_radius:.10e}\n")
            stream.write(f"best_amplitude\t{best_physical[0]:.10e}\n")
            stream.write(f"best_beta\t{best_physical[1]:.10e}\n")
            stream.write(f"best_objective\t{best['value']:.10e}\n")
            stream.write(f"dJ_dA\t{best_gradient[0]:.10e}\n")
            stream.write(f"dJ_dbeta\t{best_gradient[1]:.10e}\n")

        print(f"[AB-BFGS] stopping criterion: {stopping_reason}")
        print(
            f"[AB-BFGS] best candidate: A={best_physical[0]:.8f}, "
            f"beta={best_physical[1]:.8f}, J={best['value']:.8e}"
        )
        return best_physical[0], best_physical[1], float(best["value"])


    def run_pipeline() -> None:
        validate_configuration()
        case_tag = (
            f"{solver.npx}x{solver.npy}x{solver.npz}_"
            f"Ra{solver.Ra:.0e}_Pr{solver.Pr:g}"
            .replace("+", "")
            .replace(".", "p")
        )
        root_name = f"control_AB_coupled_damped_BFGS_{solver.PIPELINE_PROFILE}_{case_tag}"
        if solver.PIPELINE_OUTPUT_TAG:
            root_name = f"{root_name}_{solver.PIPELINE_OUTPUT_TAG}"
        root_dir = os.path.join(solver.script_dir, root_name)
        os.makedirs(root_dir, exist_ok=True)

        print("=" * 76)
        print("Differentiable RB coupled damped-BFGS control optimization")
        print(
            f"initial=(A={INITIAL_A:.6f}, beta={INITIAL_BETA:.6f}), "
            f"bounds=A[{solver.AMPLITUDE_MIN}, {solver.AMPLITUDE_MAX}], "
            f"beta[{solver.BETA_MIN}, {solver.BETA_MAX}], "
            f"scales=({PARAMETER_SCALE[0]}, {PARAMETER_SCALE[1]}), "
            f"initial trust radius={INITIAL_TRUST_RADIUS}, "
            f"maximum evaluations={MAX_EVALUATIONS}"
        )
        print("=" * 76)

        baseline_nu = solver.run_uncontrolled_baseline(
            os.path.join(root_dir, "00_uncontrolled_baseline")
        )
        solver.configure_control_context(baseline_nu)
        print(
            f"[sensor] Nu0={solver.CONTROL_NU0:.6f}, "
            f"z_delta={solver.CONTROL_SENSOR_Z:.8f}, "
            f"indices=({solver.CONTROL_SENSOR_K0}, "
            f"{solver.CONTROL_SENSOR_K1}), "
            f"alpha={solver.CONTROL_SENSOR_ALPHA:.8f}"
        )

        amplitude, beta, short_objective = run_coupled_damped_bfgs(
            os.path.join(root_dir, "01_coupled_damped_BFGS_optimization")
        )
        validation = solver.run_optimized_validation(
            (amplitude, beta),
            os.path.join(root_dir, "02_optimized_long_validation"),
        )
        solver.write_control_summary(
            root_dir, (amplitude, beta), short_objective, validation
        )
        print("=" * 76)
        print("Coupled damped-BFGS optimization complete")
        print(f"Optimal candidate: A={amplitude:.6f}, beta={beta:.6f}")
        print(
            f"Long-time validation: Nu_b={validation['Nu_bottom']:.6f}, "
            f"Nu_t={validation['Nu_top']:.6f}, "
            f"enhancement={100.0 * validation['enhancement']:.3f}%"
        )
        print(f"Results: {root_dir}")
        print("=" * 76)

    run_pipeline()


if __name__ == "__main__":
    _run_coupled_damped_bfgs_standalone()
