import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# my data set
data_prefix = 'data/'
data_list = ['2018_02_25.csv', '2018_02_26.csv', '2018_02_27.csv',
             '2018_03_03.csv', '2018_03_10.csv']
for ind, data in enumerate(data_list):
    dataset = pd.read_csv(data_prefix+data)
    for i in range(2):
        j = i*3
        f = dataset.values[:, j]
        n = len(f)
        zr = dataset.values[:, j+1]
        zj = dataset.values[:, j+2]
        
        # sort the zr zj and f values
        f_ind = np.argsort(f)
        f = f[f_ind]
        zr = zr[f_ind]
        zj = zj[f_ind]

        # remove nans in zr and zj experimental data
        inds = np.where(np.isnan(np.log10(zj)))
        zj = np.delete(zj, inds)
        zr = np.delete(zr, inds)
        f = np.delete(f, inds)
        inds = np.where(np.isnan(np.log10(zr)))
        zj = np.delete(zj, inds)
        zr = np.delete(zr, inds)
        f = np.delete(f, inds)
        n = len(f)

        # calculate magnitude
        mag = np.sqrt(zr**2 + zj**2)

# plt.figure()
# plt.loglog(f, 1./mag, '.-')
# plt.show()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 7))
        ax1.loglog(f, 1./zj, '.-')
        ax1.set_xlabel('Frequency')
        ax1.set_ylabel(r'$\frac{1}{|Z|_j}$')

        ax2.loglog(f, 1./zr, '.-')
        ax2.set_xlabel('Frequency')
        ax2.set_ylabel(r'$\frac{1}{|Z|_r}$')

        ax3.loglog(f, 1./mag, '.-')
        ax3.set_xlabel('Frequency')
        ax3.set_ylabel(r'$\frac{1}{|Z|}$')


        ax4.loglog(zj, zr, 'xk')
        ax4.set_xlabel('$Z_r$')
        ax4.set_ylabel(r'$-Z_j$')


        fig.show()