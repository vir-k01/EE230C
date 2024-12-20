import numpy as np
import matplotlib.pyplot as plt
from .data.py import DATA_TB11

gamma = np.array([0, 0])
M = np.array([np.pi/a, -np.pi/3**0.5/a])
K = np.array([2*np.pi/3/a, -2*np.pi/3**0.5/a])

path = [
    (gamma, M),
    (M, K),
    (K, gamma)
]

def compute_energy_eigs(kx, ky, data, a):
    '''
    Compute the energy eigenvalues for the TB Hamiltonian at a specified K-point.
    '''
    zero_point_energies = data['zero_point_energies'] #on-site energies
    sk_params = data['sk_params'] #slater-koster parameters
    
    xi = kx * a / 2
    eta = np.sqrt(3) * ky * a / 2
    cosphi = np.sqrt(4/7)
    sinphi = np.sqrt(3/7)
    del0 = zero_point_energies['D0']
    del1 = zero_point_energies['D1']
    del2 = zero_point_energies['D2']
    delp = zero_point_energies['Dp']
    delz = zero_point_energies['Dz']
    
    Vpds = sk_params['p(S)d(Mo)σ']
    Vpdp = sk_params['p(S)d(Mo)π']
    Vdds = sk_params['d(Mo)d(Mo)σ']
    Vddp = sk_params['d(Mo)d(Mo)π']
    Vddd = sk_params['d(Mo)d(Mo)δ']
    Vpps = sk_params['p(S)p(S)σ']
    Vppp = sk_params['p(S)p(S)π']

    l = np.zeros(5)
    l[0] = np.cos(2 * xi) + 2 * np.cos(xi) * np.cos(eta)
    l[1] = np.cos(2 * xi) - np.cos(xi) * np.cos(eta)
    l[2] = 2 * np.cos(2 * xi) + np.cos(xi) * np.cos(eta)
    l[3] = np.cos(xi) * np.cos(eta)
    l[4] = np.sin(xi) * np.sin(eta)
    d = np.sin(eta / 3) + 1j * np.cos(eta / 3)
    
    C = np.zeros(3, dtype=complex)
    C[0] = (2 * np.cos(xi) * np.cos(eta / 3) + np.cos(2 * eta / 3)) - 1j * (2 * np.cos(xi) * np.sin(eta / 3) - np.sin(2 * eta / 3))
    C[1] = (np.cos(xi) * np.cos(eta / 3) - np.cos(2 * eta / 3)) - 1j * (np.cos(xi) * np.sin(eta / 3) + np.sin(2 * eta / 3))
    C[2] = (np.cos(xi) * np.cos(eta / 3) + 2 * np.cos(2 * eta / 3)) - 1j * (np.cos(xi) * np.sin(eta / 3) - 2 * np.sin(2 * eta / 3))

    E = np.zeros(16)
    E[0]  = 0.5 * (-Vpds * (sinphi**2 - 0.5 * cosphi**2) + np.sqrt(3) * Vpdp * sinphi**2) * cosphi
    E[1]  = (-Vpds * (sinphi**2 - 0.5 * cosphi**2) - np.sqrt(3) * Vpdp * cosphi**2) * sinphi
    E[2]  = 0.25 * ((np.sqrt(3) / 2) * Vpds * cosphi**3 + Vpdp * cosphi * sinphi**2)
    E[3]  = 0.5 * ((np.sqrt(3) / 2) * Vpds * sinphi * cosphi**2 - Vpdp * sinphi * cosphi**2)
    E[4]  = (-3 / 4) * Vpdp * cosphi
    E[5]  = (-3 / 4) * Vpdp * sinphi
    E[6]  = 0.25 * (-np.sqrt(3) * Vpds * cosphi**2 - Vpdp * (1 - 2 * cosphi**2)) * sinphi
    E[7]  = 0.5 * (-np.sqrt(3) * Vpds * sinphi**2 - Vpdp * (1 - 2 * sinphi**2)) * cosphi
    E[8]  = 0.25 * (Vdds + 3 * Vddd)
    E[9]  = (-np.sqrt(3) / 4) * (Vdds - Vddd)
    E[10] = 0.25 * (3 * Vdds + Vddd)
    E[11] = Vddp
    E[12] = Vddp
    E[13] = Vddd
    E[14] = Vpps
    E[15] = Vppp
    
    H = np.zeros((11, 11), dtype=complex)
    H[0, 0] = del0 + 2 * E[8] * l[0]
    H[1, 1] = del1 + E[10] * l[2] + 3 * E[11] * l[3]
    H[2, 2] = del1 + E[11] * l[2] + 3 * E[10] * l[3]
    H[0, 1] = 2 * E[9] * l[1]
    H[0, 2] = -2 * np.sqrt(3) * E[9] * l[4]
    H[1, 2] = np.sqrt(3) * (E[10] - E[11]) * l[4]
    H[3, 3] = delp + E[14] * l[2] + 3 * E[15] * l[3] + E[15]
    H[4, 4] = delp + E[15] * l[2] + 3 * E[14] * l[3] + E[15]
    H[5, 5] = delz + 2 * E[15] * l[0] - E[14]
    H[3, 4] = -np.sqrt(3) * (E[14] - E[15]) * l[4]
    H[3, 5] = 0
    H[4, 5] = 0
    H[0, 3] = -2 * np.sqrt(6) * E[0] * np.sin(xi) * d
    H[0, 4] = 2 * np.sqrt(2) * E[0] * C[1]
    H[0, 5] = -np.sqrt(2) * E[1] * C[0]
    H[1, 3] = 2 * np.sqrt(6) * (E[2] - (1 / 3) * E[4]) * np.sin(xi) * d
    H[1, 4] = -2 * np.sqrt(2) * (E[2] * C[2] - 1j * E[4] * np.cos(xi) * d)
    H[1, 5] = 2 * np.sqrt(2) * E[3] * C[1]
    H[2, 3] = -2 * np.sqrt(2) * ((1 / 3) * E[4] * C[2] - 1j * 3 * E[2] * np.cos(xi) * d)
    H[2, 4] = H[1, 3]
    H[2, 5] = -2 * np.sqrt(6) * E[3] * np.sin(xi) * d
    
    H[6, 6] = del2 + E[11] * l[2] + 3 * E[13] * l[3]
    H[7, 7] = del2 + E[13] * l[2] + 3 * E[11] * l[3]
    H[6, 7] = np.sqrt(3) * (E[13] - E[11]) * l[4]
    H[8, 8] = delp + E[14] * l[2] + 3 * E[15] * l[3] - E[15]
    H[9, 9] = delp + E[15] * l[2] + 3 * E[14] * l[3] - E[15]
    H[10, 10] = delz + 2 * E[15] * l[0] + E[14]
    H[8, 9] = -np.sqrt(3) * (E[14] - E[15]) * l[4]
    H[8, 10] = 0
    H[9, 10] = 0
    H[6, 8] = 2 * np.sqrt(2) * ((1 / 3) * E[5] * C[2] - 1j * 3 * E[6] * np.cos(xi) * d)
    H[6, 9] = -2 * np.sqrt(6) * E[3] * np.sin(xi) * d
    H[6, 10] = -2 * np.sqrt(6) * E[7] * np.sin(xi) * d
    H[7, 8] = H[6, 9]
    H[7, 9] = -2 * np.sqrt(2) * (E[6] * C[2] - 1j * E[5] * np.cos(xi) * d)
    H[7, 10] = 2 * np.sqrt(2) * E[7] * C[1]
    
    for i in range(11):
        for j in range(i + 1, 11):
            H[j, i] = np.conjugate(H[i, j])

    eigvals = np.linalg.eigvalsh(H)
    return np.real(eigvals)
            
def compute_band_structure(path, data, Ntx=100, a=3.16):
    '''
    Sample Ntx # of points about a given path between 2 K-points and compute the energy eigenvalues (bands).
    '''
    Ehk = []
    Ehk_inter = []
    k_points = []
    kp = []
    for start, end in path:
        for i in range(Ntx):
            kx = (start[0] + (end[0] - start[0]) * i / (Ntx - 1))
            ky = (start[1] + (end[1] - start[1]) * i / (Ntx - 1))
            k = np.array([kx, ky])
            eigvals = compute_energy_eigs(kx, ky, data, a)
            Ehk_inter.append(eigvals)
            kp.append(np.linalg.norm(k))
        Ehk.append(Ehk_inter)
        k_points.append(kp)
    Ehk = np.array(Ehk)
    k_points = np.array(k_points)
    return Ehk, k_points

Ehk, k_points = compute_band_structure(path, DATA_TB11, Ntx, a)
fig = plt.figure()
for i, band in enumerate(Ehk.T):
    plt.plot([*band], c='k')
plt.grid(True)
plt.xlabel('Wave Vector', fontsize=12)
plt.ylabel('Energy [eV]', fontsize=12)
plt.ylim([-2.5, 9.5])
plt.xticks([0, Ntx, 2*Ntx, 3*Ntx], [r'$\Gamma$', 'M', 'K', r'$\Gamma$'])
plt.show()
fig.savefig('band_structure.png')
