import numpy as np
import matplotlib.pyplot as plt
from .overlaps import E_orb1_orb2
from .data import DATA_TB27

a = 3.12

mo_mo_nns = [[1, 0, 0], [-1, 0, 0], [0.5, np.sqrt(3)/2, 0], [-0.5, -np.sqrt(3)/2, 0], [0.5, -np.sqrt(3)/2, 0], [-0.5, np.sqrt(3)/2, 0]]
mo_s_nns = [1/2*np.array([1, -np.sqrt(3)/3, 1]), 1/2*np.array([1, -np.sqrt(3)/3, -1]), 1/2*np.array([-1, -np.sqrt(3)/3, -1]), 1/2*np.array([-1, -np.sqrt(3)/3, 1]), 1/2*np.array([0, 2*np.sqrt(3)/3, -1]),  1/2*np.array([0, 2*np.sqrt(3)/3, 1])]
s_s_nns_ip = [[0, 0, 1]]

def get_hamiltonian_element(kx: float, ky: float, orb1: str, orb2: str):
    '''
    Get the TB Hamiltonian element for the overlap between two orbitals (orb1 and orb2) at a specified K-point.
    Note: orb1 and orb2 are of the format: "(px_S_2)"
    '''
    sk_params = DATA_TB27.get('sk_params') #Intralyer Slater-Koster parameters
    zero_point_energies = DATA_TB27.get('zero_point_energies') #On-site interaction energies
    sk_params_interlayer = DATA_TB27.get('sk_params_interlayer', None) #Inter-layer interaction energies, if specified for S-S. 
  
    atom1 = orb1.split('_')[1]
    atom2 = orb2.split('_')[1]
    factor = 1
    E = 0
    
    in_plane = True
    params = sk_params
    if len(orb1.split('_')) == 3 and len(orb2.split('_')) == 3:
            if orb1.split('_')[-1] == orb2.split('_')[-1]:
                in_plane = False
                params = sk_params_interlayer if sk_params_interlayer else sk_params
    try:
        if '_'.join(orb1.split('_')[:2]) == '_'.join(orb2.split('_')[:2]):
            E = zero_point_energies[f'{orb1[0]}({atom1})'] if orb1 == orb2 else 0
    except:
        if orb1 == orb2:
            if orb1.startswith('dxy') or orb1.startswith('dx2y2'):
                E += zero_point_energies['D2']
            if orb1.startswith('dz2'):
                E += zero_point_energies['D0']
            if orb1.startswith('dxz') or orb1.startswith('dyz'):
                E += zero_point_energies['D1']
            if orb1.startswith('p') and in_plane:
                E += zero_point_energies['Dp']
            if orb1.startswith('dp') and not in_plane:
                E += zero_point_energies['Dz']
    
    c = 0 if E == 0 else 1
    
    if atom1 == 'Mo' and atom2 == 'Mo':
        nns = mo_mo_nns
    
    elif atom1 == 'S' and atom2 == 'S':
        nns = [*s_s_nns_ip, *mo_mo_nns]
        
    elif (atom1 == 'Mo' and atom2 == 'S') or (atom1 == 'S' and atom2 == 'Mo'):
        nns = mo_s_nns
        factor = 1
    
    for nn in nns:
        phase = np.exp(1j * factor* (nn[0] * kx * a + nn[1] * ky * a))
        nn = np.array(nn)/np.linalg.norm(nn)
        E += E_orb1_orb2(orb1, orb2, nn[0], nn[1], nn[2], params)*phase
    
    num_terms = len(nns) + c
    
    return E/num_terms

kmax = np.pi
Ntx = 50

gamma = np.array([0, 0])
M = np.array([np.pi/a, -np.pi/3**0.5/a])
K = np.array([2*np.pi/3/a, -2*np.pi/3**0.5/a])

path = [
    (gamma, M),
    (M, K),
    (K, gamma)
]

def compute_band_structure(path : tuple, Ntx: int):
    '''
    Find the energy eigenvalues for the TB Hamiltonian of a monolayer MoS2 at Ntx points along a specified path between K-points.
    '''
    Ehk = []
    Ehk_inter = []
    k_points = []
    kp = []
    c = 0
    
    orbitals = ['s', 'px', 'py', 'pz', 'dxy', 'dyz', 'dz2', 'dzx', 'dx2y2']
    atoms = ['Mo', 'S_1', 'S_2']
    states = [f'{orb}_{atom}' for orb in orbitals for atom in atoms]
  
    for start, end in path:
        for i in range(Ntx):
            kx = (start[0] + (end[0] - start[0]) * i / (Ntx - 1))
            ky = (start[1] + (end[1] - start[1]) * i / (Ntx - 1))
            k = np.array([kx, ky])
            H = np.zeros((len(states), len(states)), dtype=complex)
             
            for i, orb1 in enumerate(states):
                for j, orb2 in enumerate(states):
                    H[i, j] = np.round(get_hamiltonian_element(kx, ky, orb1, orb2), 6)
                    
            eigvals = np.linalg.eigvalsh(H)
            Ehk_inter.append((np.real(eigvals)))
            kp.append(np.linalg.norm(k))
        Ehk.append(Ehk_inter)
        k_points.append(kp)
        c += 1

    Ehk = np.array(Ehk)
    k_points = np.array(k_points)
    return Ehk, k_points

Ehk, k_points = compute_band_structure(path, Ntx)
plt.figure()
colors = plt.cm.tab10(np.linspace(0, 1, Ehk.shape[2]))
for band in Ehk.T[:, :, 1]:
    plt.plot([*band])

for band in Ehk.T[:, :, -1]:
    plt.plot([*band])

fig = plt.figure()
plt.grid(True)
plt.xlabel('Wave Vector', fontsize=12)
plt.ylabel('Energy [eV]', fontsize=12)
plt.ylim([-5, 5])
plt.xticks([0, Ntx, 2*Ntx, 3*Ntx], [r'$\Gamma$', 'M', 'K', r'$\Gamma$'])
plt.legend([f'Band {i+1}' for i in range(len(states))])
fig.savefig("band_structure.png")
