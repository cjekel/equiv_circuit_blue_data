import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('2018_02_25.csv')

j = 0
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

plt.figure()
plt.loglog(f, zj, '.-')
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'$-Z_j (\Omega)$')
plt.grid(True)
plt.savefig('fzj.pdf', dpi=300, bbox_inches='tight')

plt.figure()
plt.loglog(f, zr, '.-')
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'$Z_r (\Omega)$')
plt.grid(True)
plt.savefig('fzr.pdf', dpi=300, bbox_inches='tight')

plt.figure()
plt.loglog(zr, zj, '.-')
plt.xlabel(r'$Z_r (\Omega)$')
plt.ylabel(r'$-Z_j (\Omega)$')
plt.grid(True)
plt.savefig('zrzj.pdf', dpi=300, bbox_inches='tight')

plt.show()
