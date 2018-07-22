from __future__ import print_function
import numpy as np
from scipy.optimize import differential_evolution, fmin_l_bfgs_b
from time import time
import pyfde
# from joblib import Parallel, delayed
# import multiprocessing

# MATLAB DATA
# test_data(:, 1)=[29699.7651457787, 26099.7936129570, 21899.8268246651, 18299.8552918434, 15299.8790144920, 12299.9027371407, 9899.98318254948, 9899.92171525955, 8699.98522102833, 7499.94069337845, 7299.98759925365, 6099.98963773251, 5699.95492696762, 5099.99133646488, 4099.99303519726, 3899.96916055679, 3299.99439418316, 2699.97864961624, 2499.99575316906, 2099.98339414597, 1899.99677240849, 1499.98813867569, 1299.99779164791, 989.984488114715, 899.998471140862, 899.992883205414, 869.986368343234, 729.988561943173, 699.998810887337, 609.990442171693, 509.992009028792, 499.999150633812, 409.993575885892, 329.994829371572, 299.999490380287, 299.997627735138, 249.996082857251, 189.997022971511, 129.997963085771, 99.9998301267624, 89.9985898286104, 69.9989032000303, 49.9992165714502, 29.9995299428701, 9.99984331429005], 
# test_data(:, 2)=[34014.3136067845, 34015.6268615327, 34017.6038981704, 34019.8858110408, 34022.4546857903, 34026.0386860161, 34030.1705139903, 34030.1706411046, 34032.9354507078, 34036.4260725138, 34037.1003371573, 34041.9102207341, 34043.8852177442, 34047.3273640898, 34054.8877844363, 34056.7841807336, 34063.6062639026, 34072.8953233278, 34076.8074806208, 34086.4666353387, 34092.5505151447, 34108.6773850517, 34119.7761333256, 34144.1378965819, 34153.7825133224, 34153.7831622805, 34157.3686007005, 34177.2998751914, 34182.4282245778, 34200.3192250567, 34226.2458447706, 34229.3098360182, 34262.4312225684, 34304.1599701629, 34324.5175989063, 34324.5189687118, 34367.3448455800, 34442.6960507671, 34573.0069903759, 34684.9013264455, 34735.7731670130, 34872.8814136641, 35097.2730078582, 35552.9703151458, 37277.8916564037], 
# test_data(:, 3)=[-25.9717022518943, -28.3517826744616, -31.9372124114999, -36.0784306677522, -40.7431846845560, -47.2545930171735, -54.7643061339804, -54.7645372028052, -59.7907949573430, -66.1373938805904, -67.3634171158110, -76.1098701830045, -79.7015051163511, -85.9614655632051, -99.7117583520784, -103.160889711097, -115.569034930800, -132.464630099159, -139.580445852252, -157.149651971201, -168.215829278786, -197.549791997082, -217.737994532798, -262.051350588057, -279.594692460113, -279.595872901170, -286.117711080304, -322.372332440066, -331.700715866043, -364.244177054778, -411.404360064938, -416.977723212403, -477.225155925234, -553.129337259239, -590.159678390695, -590.162170054631, -668.062070968857, -805.125308871584, -1042.15993298586, -1245.69492732720, -1338.23044602444, -1587.62938932548, -1995.79615245484, -2824.70644580656, -5962.32668628457], 

# given frequency spectrum for measurments????
f = np.array([29699.7651457787, 26099.7936129570, 21899.8268246651, 
    18299.8552918434, 15299.8790144920, 12299.9027371407, 9899.98318254948, 
    9899.92171525955, 8699.98522102833, 7499.94069337845, 7299.98759925365, 
    6099.98963773251, 5699.95492696762, 5099.99133646488, 4099.99303519726, 
    3899.96916055679, 3299.99439418316, 2699.97864961624, 2499.99575316906, 
    2099.98339414597, 1899.99677240849, 1499.98813867569, 1299.99779164791, 
    989.984488114715, 899.998471140862, 899.992883205414, 869.986368343234, 
    729.988561943173, 699.998810887337, 609.990442171693, 509.992009028792, 
    499.999150633812, 409.993575885892, 329.994829371572, 299.999490380287, 
    299.997627735138, 249.996082857251, 189.997022971511, 129.997963085771, 
    99.9998301267624, 89.9985898286104, 69.9989032000303, 49.9992165714502, 
    29.9995299428701, 9.99984331429005])

n = len(f)

def otto_model_create_test_data(x):
    # Run the otto model to generate test data
    # Input:
    # x: 1D Numpy array or list with 5 elements as defined below 
    # x[0] = alpha CPE phase factor
    # x[1] = K CPE magnitude factor
    # x[2] = ren encapsulation resistance
    # x[3] = rex extracellular resistance
    # x[4] = am membrane area in cm**2
    #
    # example:
    # zr, zj = otto_model_create_test_data(x)
    # zr: 1D Numpy array of length f real component of z
    # zj: 1D Numpy array of length f imaginary component of z
    

    # glial encapsulation
    am = x[4]    # membrane area (cm**2)
    cm = 1e-6*am # cell membrane capaacitance (uf/cm**2)
    rm = 3.33/am # Cell membrane resistivity (ohm*cm**2)
    
    # 1j in Python is sqrt(-1.0)
    ecpe = 1.0 / (((1j*2*np.pi*f)**x[0])*(x[1]/1e6))
    ren = (x[2]*1e3) * np.ones(n)
    rex = (x[3]*1e3) * np.ones(n)
    
    # 2 parallel RC circuits in series
    cell_membrane = (1.0/((1j*2*np.pi*f*cm)+(1.0/rm)))+(1.0/((1j*2*np.pi*f*cm)+(1.0/rm)))
    
    # combine total impedances    
    ztot = ecpe + ren + (1.0 / ((1.0/(cell_membrane))+(1.0/rex)))
    return np.real(ztot), np.imag(ztot)

zr, zj = otto_model_create_test_data([0.68, 8.8, 34.0, 45.0, 89.0])

# generate noise
noise_mag = 1e-7
n1 = np.random.normal(loc=0.0, scale=np.mean(zr)*noise_mag, size=n)
n2 = np.random.normal(loc=0.0, scale=np.abs(np.mean(np.real(zj)))*noise_mag, size=n)
zr = zr + n1
zj = zj + n2

# This is the same as the MATLAB data... 

# measured impedence magnitude
zmag = np.sqrt((zr**2) + (zj**2))

# Now generate objective function

def otto_model_ec(x, zr=zr, zj=zj):
    # return the distance of the otto model for x from some data set
    # Input:
    # x: 1D Numpy array or list with 5 elements as defined below 
    # x[0] = alpha CPE phase factor
    # x[1] = K CPE magnitude factor
    # x[2] = ren encapsulation resistance
    # x[3] = rex extracellular resistance
    # x[4] = am membrane area in cm**2 
    
    # glial encapsulation
    am = x[4]    # membrane area (cm**2)
    cm = 1e-6*am # cell membrane capaacitance (uf/cm**2)
    rm = 3.33/am # Cell membrane resistivity (ohm*cm**2)
    
    # 1j in Python is sqrt(-1.0)
    ecpe = 1.0 / (((1j*2*np.pi*f)**x[0])*(x[1]/1e6))
    ren = (x[2]*1e3) * np.ones(n)
    rex = (x[3]*1e3) * np.ones(n)
    
    # 2 parallel RC circuits in series
    cell_membrane = (1.0/((1j*2*np.pi*f*cm)+(1.0/rm)))+(1.0/((1j*2*np.pi*f*cm)+(1.0/rm)))
    
    # combine total impedances    
    ztot = ecpe + ren + (1.0 / ((1.0/(cell_membrane))+(1.0/rex)))
    
    # e = (((zr-np.real(ztot))**2)/(np.abs(zmag)))+(((zj-np.imag(ztot))**2)/(np.abs(zmag)))
    # L2 = np.dot(e.T, e)
    
    L1 = np.sum(np.abs(zr-np.real(ztot))) + np.sum(np.abs(zj-np.imag(ztot)))
    return L1
def otto_model_ec_min(x, zr=zr, zj=zj):
    # return the distance of the otto model for x from some data set
    # Input:
    # x: 1D Numpy array or list with 5 elements as defined below 
    # x[0] = alpha CPE phase factor
    # x[1] = K CPE magnitude factor
    # x[2] = ren encapsulation resistance
    # x[3] = rex extracellular resistance
    # x[4] = am membrane area in cm**2 
    
    # glial encapsulation
    am = x[4]    # membrane area (cm**2)
    cm = 1e-6*am # cell membrane capaacitance (uf/cm**2)
    rm = 3.33/am # Cell membrane resistivity (ohm*cm**2)
    
    # 1j in Python is sqrt(-1.0)
    ecpe = 1.0 / (((1j*2*np.pi*f)**x[0])*(x[1]/1e6))
    ren = (x[2]*1e3) * np.ones(n)
    rex = (x[3]*1e3) * np.ones(n)
    
    # 2 parallel RC circuits in series
    cell_membrane = (1.0/((1j*2*np.pi*f*cm)+(1.0/rm)))+(1.0/((1j*2*np.pi*f*cm)+(1.0/rm)))
    
    # combine total impedances    
    ztot = ecpe + ren + (1.0 / ((1.0/(cell_membrane))+(1.0/rex)))
    
    # e = (((zr-np.real(ztot))**2)/(np.abs(zmag)))+(((zj-np.imag(ztot))**2)/(np.abs(zmag)))
    # L2 = np.dot(e.T, e)
    
    L1 = np.sum(np.abs(zr-np.real(ztot))) + np.sum(np.abs(zj-np.imag(ztot)))
    return L1


def opt_routine():
    results_x = np.zeros((10, 5))
    opts = np.zeros(10)
    for i in range(10):
        # run differential evolution
        solver = pyfde.ClassicDE(otto_model_ec_min, n_dim=5, n_pop=100, limits=bounds, minimize=True)
        solver.cr, solver.f = 1.0, 0.9
        best, fit = solver.run(n_it=1000)
        fit = fit*-1
        # polish with L BFGS
        res_bfgs = fmin_l_bfgs_b(otto_model_ec_min, best, fprime=None, args=(), approx_grad=True, bounds=bounds, m=10, factr=10000000.0, pgtol=1e-05, epsilon=1e-04, iprint=-1, maxfun=15000, maxiter=15000, disp=None, callback=None, maxls=20)
        # if polish better save polish results
        # print(best, fit)
        # print(res_bfgs)
        if res_bfgs[1] < fit:
            opts[i] = res_bfgs[1]
            results_x[i] = res_bfgs[0]
            print('Polish was better')
        else:
            opts[i] = fit
            results_x[i] = best            
            print('Polish did not help')
    # find the best result
    best_index = np.argmin(opts)
    best_opt = opts[best_index]
    best_x = results_x[best_index]
    return results_x, opts, best_x, best_opt
bounds = np.ones((5, 2))*1e-9
bounds[:, 1] = 100.0
bounds[4, 0] = 1.0
bounds[0, 1] = 1.0

t1 = time()
# res0 = differential_evolution(otto_model_ec, bounds, args=(), 
#             strategy='best1bin', maxiter=1000, popsize=100, tol=1e-60, 
#             mutation=(0.5, 1), recombination=0.7, seed=None, callback=None, 
#             disp=False, polish=False, init='latinhypercube', atol=-1)
t2 = time()
results_x, opts, best_x, best_opt = opt_routine()
t3 = time()

# res1 = fmin_slsqp(otto_model_ec, best, bounds=bounds, fprime=None, args=(), 
#     iter=100, acc=1e-3, iprint=1, disp=False, full_output=1, 
#     epsilon=1e-5)

# res1 = fmin_l_bfgs_b(otto_model_ec, best, fprime=None, args=(), approx_grad=1,
#     bounds=bounds, m=10, factr=10000000.0, pgtol=1e-05, epsilon=1e-05, 
#     iprint=-1, maxfun=15000, maxiter=15000, disp=None, callback=None, maxls=50)
t4 = time()
print('DE scipy runtime:', t2-t1)
# print(res0)
print("FDE runtime:", t4-t3)
# print(best, fit)
# # parralize
# res = []
# def gen_noise_run_model():
#     #  generate noise
#     noise_mag = 1e-7
#     n1 = np.random.normal(loc=0.0, scale=np.mean(zr)*noise_mag, size=n)
#     n2 = np.random.normal(loc=0.0, scale=np.abs(np.mean(np.real(zj)))*noise_mag, size=n)
#     zrn = zr + n1
#     zjn = zj + n2
#     res0 = differential_evolution(otto_model_ec, bounds, args=(zrn, zjn), 
#             strategy='best1bin', maxiter=10000, popsize=100, tol=1e-60, 
#             mutation=(0.5, 1), recombination=0.7, seed=None, callback=None, 
#             disp=False, polish=True, init='latinhypercube', atol=0)
#     # print(res0)
#     res.append(res0)
# num_cores = multiprocessing.cpu_count() 
# Parallel(n_jobs=num_cores)(delayed(gen_noise_run_model)() for i in range(10))  
# 
# print(res)
        
# res1 = fmin_slsqp(otto_model_ec, res0.x, bounds=bounds, fprime=None, args=(), 
#     iter=100, acc=1e-70, iprint=1, disp=True, full_output=1, 
#     epsilon=1.4901161193847656e-08)

# # plot the fitted data
# zr_x, zj_x = otto_model_create_test_data(res0.x)

# plt.figure()
# plt.plot(zr, -zj, '-', label='Test Data')
# plt.plot(zr_x, -zj_x, 'x', label='Optimization result')
# plt.xlabel(r'$Z_r (\Omega)$')
# plt.ylabel(r'$-Z_j (\Omega)$')
# plt.legend()
# plt.grid()
# plt.show()
# 
# # real residuals
# er = zr - zr_x
# # imaginary residuals
# ej = zj - zj_x
# 
# plt.figure()
# plt.semilogx(f, er, 'o')
# plt.xlabel('$f$')
# plt.ylabel('Real residuals')
# plt.grid()
# plt.show()
# 
# plt.figure()
# plt.semilogx(f, ej, 'o')
# plt.xlabel('$f$')
# plt.ylabel('Imaginary residuals')
# plt.grid()
# plt.show()
