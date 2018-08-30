from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

x = np.load('data/lk2_res_50_opts.npy')
labels = [r'$\alpha$ CPE', r'$K$ CPE', r'$r_{en}$ (k$\Omega)$', r'$r_{ex}$ (k$\Omega)$', r'$a_{m}$ (cm$^2$)']

fig, ax = plt.subplots(1, 5, sharey=False)
fig.set_size_inches(10, 5)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
for i in range(5):
    ax[i].boxplot(x[:, i])
    ax[i].get_xaxis().set_ticks([])
    ax[i].set_title(labels[i])
plt.savefig('figs/boxplots.pdf', bbox_inches='tight', dpi=300)
plt.show()