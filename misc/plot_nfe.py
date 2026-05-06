import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import get_cmap
# cm = get_cmap('inferno')

cm = mpl.colormaps['inferno']


invalid = np.array([
    0.9429,
    0.9895,
    0.9964,
    0.9992,
    0.9991,
    0.9999,
    0.9998,

])
fcd = np.array([
6.53730749774526,
4.318887279233685,
3.2564295509708074,
2.634944082585136,
2.168117544422799,
1.9206010199381183,
1.7140949069660678
])

nspdk = np.array([0.007974,
0.003997,
0.002791,
0.002078,
0.001587,
0.001342,
0.000872
])

invalid = 100*(1-invalid)
nspdk = nspdk * 1000

nfe = [2**i for i in range(4, 11)]
nfe = np.array(nfe
)


plt.xscale('log', base=2)
plt.xlabel('NFE', fontsize=18)
plt.xticks(nfe, labels=[str(x) for x in nfe], fontsize=16)
plt.yticks(fontsize=16)
plt.plot( nfe,  invalid, color=cm(float(0.15)), label='Invalid (%)', linewidth=5.0)
plt.plot( nfe,  fcd, color=cm(float(0.55)), label='FCD', linewidth=5.0)
plt.plot( nfe,  nspdk, color=cm(float(0.90)), label='NSPDK ($10^{3}$)', linewidth=5.0)
plt.ylim(0, 9)
plt.legend(fontsize=16)

plt.tight_layout()
plt.show()

marginal = np.array([0.22, 1.78, 0.995])
uniform = np.array([0.28, 1.82,  1.12])
# error = np.array([[0.16,  0.015, 0.0632], [0.27, 0.03, 0.1]])


# plt.xticks(nfe, labels=[str(x) for x in nfe], fontsize=16)
plt.yticks(fontsize=16)
plt.bar(['Invalid (%)', 'FCD', 'NSPDK ($10^3$)'],uniform,  capsize=5, color=cm(float(0.90)), ecolor=cm(float(0.75)), label='Uniform')
plt.bar(['Invalid (%)', 'FCD', 'NSPDK ($10^3$)'],marginal,  capsize=5,  color=cm(float(0.15)), ecolor=cm(float(0.35)), label='Marginal')
plt.xticks(fontsize=16)
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()