import numpy as np
from math import erfc, log
from matplotlib import pyplot as plt

EbN0 = np.array([2, 5, 8, 10])
SER = np.zeros(4)
constell = np.array ([-1, 1])/np.sqrt(2)

j = 0

for valores in EbN0:
    N0 = (10**(-valores/20))
    N = 10000000

    mensagem = np.random.choice(constell, N)
    mensCart = np.zeros([N, 2])
    mensCart[:,0] = mensagem

    #passagem pelo canal

    noise = (N0/2)*np.random.randn(N)

    recebido = mensagem + noise

    taxaErro = np.sum(np.sign(recebido) != np.sign(mensagem))/N
    SER[j] = taxaErro
    j += 1

SER[-1] += 1e-9

len = 100
EbN02 = np.linspace(-2, 15, len)
EbN0norm = (10**(EbN02/10))
BER = np.zeros(len)

for i in range(len):
    BER[i] = erfc(np.sqrt(EbN0norm[i]))/2

#plt.plot(EbN02, BER)
plt.plot(EbN0, SER)
plt.yscale('log')
plt.show()
