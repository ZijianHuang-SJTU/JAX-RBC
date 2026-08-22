import jax
import jax.numpy as jnp
from jax import jit, random
from functools import partial
import os
import numpy as np 

npx, npy = 4001, 4001
hx = 1 / (npx - 1)
hy = 1 / (npy - 1)
dt = 2.5e-5
tf = 2000
Ma = 0.1
Pr = 0.7
Ra = 1e10
ipy_mid = (npy - 1) // 2

@partial(jit, static_argnames=['axis'])
def reconstruct_linear_7th(u_stencil, axis=0): 
    if axis == 1:
        u_stencil = jnp.moveaxis(u_stencil, 1, 0)
    uf = (-1/140) * u_stencil[0,...] + (5/84) * u_stencil[1,...] - (101/420) * u_stencil[2,...] + \
         (319/420) * u_stencil[3,...] + (107/210) * u_stencil[4,...] - (19/210) * u_stencil[5,...] + \
         (1/105) * u_stencil[6,...]
    return uf

@jit
def compute_resid_u(ub):
    ubs = jnp.pad(ub, ((4, 4), (4, 4), (0, 0))) 
    ubs = ubs.at[4:npx+4,0:4,0].set(ubs[4:npx+4,8:4:-1,0])
    ubs = ubs.at[4:npx+4,0:4,1].set(-ubs[4:npx+4,8:4:-1,1])
    ubs = ubs.at[4:npx+4,0:4,2].set(-ubs[4:npx+4,8:4:-1,2])
    ubs = ubs.at[4:npx+4,0:4,3].set(2-ubs[4:npx+4,8:4:-1,3])
    ubs = ubs.at[4:npx+4,npy+4:npy+8,0].set(ubs[4:npx+4,npy+2:npy-2:-1,0])
    ubs = ubs.at[4:npx+4,npy+4:npy+8,1].set(-ubs[4:npx+4,npy+2:npy-2:-1,1])
    ubs = ubs.at[4:npx+4,npy+4:npy+8,2].set(-ubs[4:npx+4,npy+2:npy-2:-1,2])
    ubs = ubs.at[4:npx+4,npy+4:npy+8,3].set(-ubs[4:npx+4,npy+2:npy-2:-1,3])
    ubs = ubs.at[0:4,4:npy+4,0].set(ubs[8:4:-1,4:npy+4,0])
    ubs = ubs.at[0:4,4:npy+4,1].set(-ubs[8:4:-1,4:npy+4,1])
    ubs = ubs.at[0:4,4:npy+4,2].set(-ubs[8:4:-1,4:npy+4,2])
    ubs = ubs.at[0:4,4:npy+4,3].set(ubs[8:4:-1,4:npy+4,3])    

    ubs = ubs.at[npx+4:npx+8,4:npy+4,0].set(ubs[npx+2:npx-2:-1,4:npy+4,0])
    ubs = ubs.at[npx+4:npx+8,4:npy+4,1].set(-ubs[npx+2:npx-2:-1,4:npy+4,1])
    ubs = ubs.at[npx+4:npx+8,4:npy+4,2].set(-ubs[npx+2:npx-2:-1,4:npy+4,2])
    ubs = ubs.at[npx+4:npx+8,4:npy+4,3].set(ubs[npx+2:npx-2:-1,4:npy+4,3])

    flux_x = jnp.zeros_like(ubs)
    flux_x = flux_x.at[:,:,0].set(ubs[:,:,1] * (ubs[:,:,0] + 1/Ma**2))
    flux_x = flux_x.at[:,:,1].set(ubs[:,:,0] + ubs[:,:,1]**2)
    flux_x = flux_x.at[:,:,2].set(ubs[:,:,1] * ubs[:,:,2])
    flux_x = flux_x.at[:,:,3].set(ubs[:,:,1] * ubs[:,:,3])
    u_c0, u_c1, u_c2 = ubs[:,:,0], ubs[:,:,1], ubs[:,:,2]
    lambda_x = jnp.abs(1.5 * u_c1) + jnp.sqrt(0.25 * u_c1**2 + u_c0 + 1/Ma**2)
    lambda_x = jnp.expand_dims(lambda_x, axis=-1)
    fp_x = 0.5 * (flux_x + lambda_x * ubs)
    fm_x = 0.5 * (flux_x - lambda_x * ubs)
    stencils_p_x = jnp.stack([fp_x[i:i + npx + 1, :, :] for i in range(7)], axis=0)
    stencils_m_x = jnp.stack([fm_x[i+1:i + npx + 2, :, :] for i in range(7)], axis=0)
    fluxp = reconstruct_linear_7th(stencils_p_x, axis=0) 
    fluxm = reconstruct_linear_7th(stencils_m_x[::-1, :,:], axis=0) 
    num_flux_x = fluxp + fluxm
    resid_u = -(num_flux_x[1:, 4:npy+4, :] - num_flux_x[:-1, 4:npy+4, :]) / hx
    
    flux_y = jnp.zeros_like(ubs)
    flux_y = flux_y.at[:,:,0].set(ubs[:,:,2] * (ubs[:,:,0] + 1/Ma**2))
    flux_y = flux_y.at[:,:,1].set(ubs[:,:,1] * ubs[:,:,2])
    flux_y = flux_y.at[:,:,2].set(ubs[:,:,0] + ubs[:,:,2]**2)
    flux_y = flux_y.at[:,:,3].set(ubs[:,:,2] * ubs[:,:,3])
    lambda_y = jnp.abs(1.5 * u_c2) + jnp.sqrt(0.25 * u_c2**2 + u_c0 + 1/Ma**2)
    lambda_y = jnp.expand_dims(lambda_y, axis=-1)
    fp_y = 0.5 * (flux_y + lambda_y * ubs)
    fm_y = 0.5 * (flux_y - lambda_y * ubs)
    stencils_p_y = jnp.stack([fp_y[:, i:i + npy + 1, :] for i in range(7)], axis=1)
    stencils_m_y = jnp.stack([fm_y[:, i+1:i + npy + 2, :] for i in range(7)], axis=1)
    fluxp_y = reconstruct_linear_7th(stencils_p_y, axis=1) 
    fluxm_y = reconstruct_linear_7th(stencils_m_y[:, ::-1, :], axis=1) 
    num_flux_y = fluxp_y + fluxm_y
    resid_u += -(num_flux_y[4:npx+4, 1:, :] - num_flux_y[4:npx+4, :-1, :]) / hy

    lap_x = ( -9 * ubs[0:npx+0, 4:npy+4, :] + 128 * ubs[1:npx+1, 4:npy+4, :] - 1008 * ubs[2:npx+2, 4:npy+4, :] + 8064 * ubs[3:npx+3, 4:npy+4, :] - 14350 * ubs[4:npx+4, 4:npy+4, :] + 8064 * ubs[5:npx+5, 4:npy+4, :] - 1008 * ubs[6:npx+6, 4:npy+4, :] + 128 * ubs[7:npx+7, 4:npy+4, :] - 9 * ubs[8:npx+8, 4:npy+4, :]) / (5040.0 * hx * hx)
    lap_y = ( -9 * ubs[4:npx+4, 0:npy+0, :] + 128 * ubs[4:npx+4, 1:npy+1, :] - 1008 * ubs[4:npx+4, 2:npy+2, :] + 8064 * ubs[4:npx+4, 3:npy+3, :] - 14350 * ubs[4:npx+4, 4:npy+4, :] + 8064 * ubs[4:npx+4, 5:npy+5, :] - 1008 * ubs[4:npx+4, 6:npy+6, :] + 128 * ubs[4:npx+4, 7:npy+7, :] - 9 * ubs[4:npx+4, 8:npy+8, :]) / (5040.0 * hy * hy)
    laplacian = lap_x + lap_y
    viscous_term_momentum = (Pr/Ra)**(0.5)* laplacian[:,:,0:3]
    resid_u = resid_u.at[:,:,0:3].add(viscous_term_momentum)
    diffusion_term_energy = (Ra * Pr)**(-0.5) * laplacian[:,:,3]
    resid_u = resid_u.at[:,:,3].add(diffusion_term_energy)
    buoyancy_force = ub[:,:,3]
    resid_u = resid_u.at[:,:,2].add(buoyancy_force)
    return resid_u

@jit
def ssp_rk3_step(ub):
    a30, a32 = 0.355909775063327, 0.644090224936674
    a40, a43 = 0.367933791638137, 0.632066208361863
    a52, a54 = 0.237593836598569, 0.762406163401431
    b10, b21, b32 = 0.377268915331368, 0.377268915331368, 0.242995220537396
    b43, b54 = 0.238458932846290, 0.287632146308408
    ub1 = ub
    resid = compute_resid_u(ub)
    ub = ub + b10 * dt * resid
    resid = compute_resid_u(ub)
    ub = ub + b21 * dt * resid
    ub2 = ub
    resid = compute_resid_u(ub)
    ub = a30 * ub1 + a32 * ub + b32 * dt * resid
    resid = compute_resid_u(ub)
    ub = a40 * ub1 + a43 * ub + b43 * dt * resid
    resid = compute_resid_u(ub)
    ub = a52 * ub2 + a54 * ub + b54 * dt * resid
    resid_norm = jnp.linalg.norm(resid) / (npx * npy)
    return ub, resid_norm

@jit
def calculate_spatial_mean_nusselt(ub, hy):
    theta = ub[:, :, 3]
    nu_local = -(-25*theta[:, 0] +48*theta[:, 1] -36*theta[:, 2]+16*theta[:, 3]-3*theta[:, 4]) / (12*hy)
    return jnp.mean(nu_local)


def write_nusselt_log(script_dir, iter, current_nu):
    log_filename = os.path.join(script_dir, "Nusselt_instant_log.dat")
    if not os.path.exists(log_filename):
        with open(log_filename, "w") as f:
            f.write("# Log of instantaneous Nusselt number\n")
            f.write("# Iter\t\tNu_instant\n")
    with open(log_filename, "a") as f:
        f.write(f"{iter:<12d}\t{current_nu:.8f}\n")


def write_solu_tecplot(script_dir, ctime, ub):
    ub_np = np.asarray(ub)
    decimal = f"{ctime - int(ctime):.4f}"[1:]
    filename = os.path.join(script_dir, f"TEC{int(ctime):07d}{decimal}.dat")

    with open(filename, "w") as f:
        f.write('TITLE = "Rayleigh-Benard Convection (JAX)"\n')
        f.write('VARIABLES = "X" "Y" "P" "U" "V" "T" \n')
        f.write(f"ZONE NODES={npx*npy},ELEMENTS={(npx-1)*(npy-1)},DATAPACKING=POINT,ZONETYPE=FEQUADRILATERAL\n")


        p, u, v, t = ub_np[:,:,0], ub_np[:,:,1], ub_np[:,:,2], ub_np[:,:,3]

        for ipy in range(npy):
            for ipx in range(npx):
                f.write(f"{ipx*hx:.8f} {ipy*hy:.8f} {p[ipx,ipy]:.8e} {u[ipx,ipy]:.8e} {v[ipx,ipy]:.8e} {t[ipx,ipy]:.8e} \n")

        for icy in range(npy - 1):
            for icx in range(npx - 1):
                ipc = icy * npx + icx
                f.write(f"{ipc+1} {ipc+2} {ipc+2+npx} {ipc+1+npx}\n")


def write_solu_tecplot_mean(script_dir, ctime, ub):
    ub_np = np.asarray(ub/((ctime-200)/2.5e-5))
    decimal = f"{ctime - int(ctime):.4f}"[1:]
    filename = os.path.join(script_dir, f"Mean_TEC{int(ctime):07d}{decimal}.dat")

    with open(filename, "w") as f:
        f.write('TITLE = "Rayleigh-Benard Convection (JAX)"\n')
        f.write('VARIABLES = "X" "Y" "P" "U" "V" "T" \n')
        f.write(f"ZONE NODES={npx*npy},ELEMENTS={(npx-1)*(npy-1)},DATAPACKING=POINT,ZONETYPE=FEQUADRILATERAL\n")


        p, u, v, t = ub_np[:,:,0], ub_np[:,:,1], ub_np[:,:,2], ub_np[:,:,3]

        for ipy in range(npy):
            for ipx in range(npx):
                f.write(f"{ipx*hx:.8f} {ipy*hy:.8f} {p[ipx,ipy]:.8e} {u[ipx,ipy]:.8e} {v[ipx,ipy]:.8e} {t[ipx,ipy]:.8e} \n")

        for icy in range(npy - 1):
            for icx in range(npx - 1):
                ipc = icy * npx + icx
                f.write(f"{ipc+1} {ipc+2} {ipc+2+npx} {ipc+1+npx}\n")


def write_data(script_dir, state):
    ctime = state['ctime']
    
    filename = os.path.join(script_dir, f"RESTART{int(ctime):07d}.dat.npz")
    np.savez(filename, ctime=state['ctime'], iter=state['iter'], ub=np.asarray(state['ub']))

def read_data(namerestart):
    if not os.path.exists(namerestart):
        raise FileNotFoundError(f"{namerestart} not found!")
    data = np.load(namerestart)
    state = {
        'ctime': data['ctime'].item(),
        'iter': data['iter'].item(),
        'ub': jnp.asarray(data['ub'])
    }
    return state

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    restart = 0
    namerestart = os.path.join(script_dir, 'RESTART0000299.dat.npz')
    ub_tec = jnp.zeros((npx, npy, 4))
    key = random.PRNGKey(0)

    if restart == 0:
        ub_initial = jnp.zeros((npx, npy, 4))
        y_coords = jnp.linspace(1, 0, npy)
        initial_temp = y_coords
        ub_initial = ub_initial.at[:,:,3].set(initial_temp)
        state = {'ctime': 0.0, 'iter': 0, 'ub': ub_initial}
    else:
        state = read_data(namerestart)

    print("JIT Compiling simulation and analysis functions... (This may take a minute)")
    step_fn = jit(ssp_rk3_step)
    nu_calc_fn = jit(calculate_spatial_mean_nusselt)


    state['ub'], _ = step_fn(state['ub']) 
    _ = nu_calc_fn(state['ub'], hy)

    state['ub'].block_until_ready()
    print("Compilation complete. Starting simulation.")

    maxiter = int(tf / dt)

    while state['ctime'] <= tf:
        ub_new, resid_norm = step_fn(state['ub'])
        ub_tec = ub_tec+ub_new
        state = {'ctime': state['ctime'] + dt, 'iter': state['iter'] + 1, 'ub': ub_new}
        
        ctime = state['ctime']
        current_nu = nu_calc_fn(ub_new, hy)
        nu_val = current_nu.item()

        if state['iter'] % 800 == 0:
            resid_norm_val = resid_norm.item()
            print(f"iter={state['iter']}, time={state['ctime']:.3f}, resid={resid_norm_val:.4e}, Nu_inst={nu_val:.5f}")
            write_nusselt_log(script_dir, state['iter'], nu_val)

        if (state['iter'] % 400000 == 0) or (state['iter'] >= maxiter):
            print(f"Writing output at iteration {state['iter']}...")
            write_solu_tecplot(script_dir, state['ctime'], state['ub'])
            write_data(script_dir, state) 
            write_solu_tecplot_mean(script_dir, ctime, ub_tec)
            if state['iter'] >= maxiter:
                break
    
    print("\n" + "="*50)
    print("Simulation finished.")
    print("="*50)

if __name__ == "__main__":
    main()