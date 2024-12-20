import numpy as np

def sp_overlap(l, m, n, orb1, orb2, sp_sigma):
    if 'x' in orb2:
        return l * sp_sigma
    elif 'y' in orb2:
        return m * sp_sigma
    elif 'z' in orb2:
        return n * sp_sigma
    
def pp_overlap(l,m,n,orb1,orb2,pp_sigma,pp_pi):
    map = {'x' : 0, 'y' : 1, 'z' : 2}
    o1 = orb1.split('_')[0][1:]
    o2 = orb2.split('_')[0][1:]
    #print(o1, o2)
    if map[o2] < map[o1]:
        orb2, orb1 = orb1, orb2
        l, m, n = -l, -m, -n
        
    if 'x' in orb1 and 'x' in orb2:
        val = l**2 * pp_sigma + (1 - l**2) * pp_pi
    elif 'x' in orb1 and 'y' in orb2:
        val = l * m * (pp_sigma - pp_pi)
    elif 'x' in orb1 and 'z' in orb2:
        val = l * n * (pp_sigma - pp_pi)
    elif 'y' in orb1 and 'y' in orb2:
        val = m**2 * pp_sigma + (1 - m**2) * pp_pi
    elif 'y' in orb1 and 'z' in orb2:
        val = m * n * (pp_sigma - pp_pi)
    elif 'z' in orb1 and 'z' in orb2:
        val = n**2 * pp_sigma + (1 - n**2) * pp_pi
    return val 


def sd_overlap(l, m, n, orb1, orb2, sd_sigma):
    if 'xy' in orb2:
        return np.sqrt(3) * l * m * sd_sigma
    elif 'yz' in orb2:
        return np.sqrt(3) * m * n * sd_sigma
    elif 'zx' in orb2:
        return np.sqrt(3) * l * n * sd_sigma
    elif 'x2y2' in orb2:
        return 0.5 * np.sqrt(3) * (l**2 - m**2) * sd_sigma
    elif 'z2' in orb2:
        return (n**2 - 0.5 * (l**2 + m**2)) * sd_sigma
    
def pd_overlap(l, m, n, orb1, orb2, pd_sigma, pd_pi):
    coord_ind_map = {'x' : 0, 'y' : 1, 'z' : 2}
    inds = [l, m, n]
    o1 = orb1.split('_')[0][1:]
    i = inds[coord_ind_map[o1]]
    
    if 'xy' in orb2:
        return np.sqrt(3) * l * m * i * pd_sigma + m*(1-2*l*i)*pd_pi
    elif 'yz' in orb2:
        return np.sqrt(3) * m * n * i * pd_sigma - 2 * m * n * i * pd_pi
    elif 'zx' in orb2:
        return np.sqrt(3) * l * n * i * pd_sigma + n*(1-2*l*i)*pd_pi
    elif 'x2y2' in orb2:
        term1 = 0.5 * np.sqrt(3) * (l**2 - m**2) * i * pd_sigma
        if o1 == 'x':
            term2 = l*(1-l**2 + m**2)*pd_pi
        if o1 == 'y':
            term2 = -m*(1+l**2 - m**2)*pd_pi
        if o1 == 'z':
            term2 = -n*(l**2 - m**2)*pd_pi
        return term1 + term2
    elif 'z2' in orb2:
        term1 = (n**2 - 0.5 * (l**2 + m**2)) * i * pd_sigma
        if o1 == 'x':
            term2 = -np.sqrt(3)*l*n**2*pd_pi
        if o1 == 'y':
            term2 = -np.sqrt(3)*m*n**2*pd_pi
        if o1 == 'z':
            term2 = -np.sqrt(3)*n*(l**2 + m**2)*pd_pi
        return term1 + term2

def dd_overlap(l, m, n, orb1, orb2, dd_sigma, dd_pi, dd_delta):
    coord_ind_map = {'x' : 0, 'y' : 1, 'z' : 2}
    map = {'xy' : 0, 'yz' : 1, 'zx' : 2, 'x2y2' : 3, 'z2' : 4}
    o1 = orb1.split('_')[0][1:]
    o2  = orb2.split('_')[0][1:]
    if map[o2] < map[o1]:
        orb2, orb1 = orb1, orb2
        o2, o1 = o1, o2
        l, m, n = -l, -m, -n
    
    inds = [l, m, n]

    if o1 not in ['z2', 'x2y2'] and o2 not in ['z2', 'x2y2']:
        i1 = inds[coord_ind_map[o1[0]]]
        j1 = inds[coord_ind_map[o1[1]]]
        k1 = set('xyz').difference(set(o1)).pop()
        k1 = inds[coord_ind_map[k1]]
        i2 = inds[coord_ind_map[o2[0]]]
        j2 = inds[coord_ind_map[o2[1]]]
        if o1 == o2:
            return 3*i1**2*j1**2*dd_sigma + (i1**2 + j1**2 - 4*i1**2*j1**2)*dd_pi + (k1**2 + i1**2*j1**2)*dd_delta
        else:
            c = set(o1).intersection(set(o2)).pop()
            uncommon = list(set([i1, j1, k1]) - set([c]))
            c = inds[coord_ind_map[c]]
            uncommon = set('xy').union(set('xz')) - set('xy').intersection(set('xz'))
            uncommon = [inds[coord_ind_map[c]] for c in list(uncommon)]
            return 3*i1*j1*i2*j2*dd_sigma + (uncommon[0]*uncommon[1])*(1-4*c**2)*dd_pi + uncommon[0]*uncommon[1]*(c**2 - 1)*dd_delta
    
    if o1 == 'xy' and o2 == 'x2y2':
        return 3/2*l*m*(l**2 - m**2)*dd_sigma + 2*l*m*(m**2 - l**2)*dd_pi + 1/2*l*m*(l**2 - m**2)*dd_delta

    if o1 == 'yz' and o2 == 'x2y2':
        return 3/2*m*n*(l**2 - m**2)*dd_sigma - m*n*(1 + 2*(l**2 - m**2))*dd_pi + m*n*(1 + 1/2*(l**2 - m**2))*dd_delta
    
    if o1 == 'zx' and o2 == 'x2y2':
        return 3/2*l*n*(l**2 - m**2)*dd_sigma + l*n*(1-2*(l**2 - m**2))*dd_pi - l*n*(1-1/2*(l**2 - m**2))*dd_delta

    if o1 == 'xy' and o2 == 'z2':
        return np.sqrt(3)*l*m*(n*82 - 1/2*(l**2 + m**2))*dd_sigma - 2*np.sqrt(3)*l*m*n**2*dd_pi +1/2*np.sqrt(3)*l*m*(1+n**2)*dd_delta
    
    if o1 == 'yz' and o2 == 'z2':
        return np.sqrt(3)*m*n*(n**2 - 1/2*(l**2 + m**2))*dd_sigma + np.sqrt(3)*m*n*(l**2 + m**2 - n**2)*dd_pi - 1/2*np.sqrt(3)*m*n*(l**2 + m**2)*dd_delta
    
    if o1 == 'zx' and o2 == 'z2':
        return np.sqrt(3)*l*n*(n**2 - 1/2*(l**2 + m**2))*dd_sigma + np.sqrt(3)*l*n*(l**2 + m**2 - n**2)*dd_pi - 1/2*np.sqrt(3)*l*n*(l**2 + m**2)*dd_delta       
    
    if o1 == 'x2y2' and o2 == 'x2y2':
        return 3/4*(l**2 - m**2)**2*dd_sigma + (l**2 + m**2 - (l**2 - m**2)**2)*dd_pi + (n**2 + 1/4*(l**2 - m**2)**2)*dd_delta
    
    if o1 == 'x2y2' and o2 == 'z2':
        return 1/2*np.sqrt(3)*(l**2 -m**2)*(n**2 - 1/2*(l**2 + m**2))*dd_sigma + np.sqrt(3)*(m**2 - l**2)*n**2*dd_pi + 1/4*np.sqrt(3)*(1+n**2)*(l**2 - m**2)*dd_delta
    
    if o1 == 'z2' and o2 == 'z2':
        return (n**2 - 1/2*(l**2 + m**2))**2*dd_sigma + 3*n**2*(l**2+m**2)*dd_pi + 3/4*(l**2+m**2)**2*dd_delta

def E_orb1_orb2(orb1, orb2, l, m, n, sk_params):
    map = {'s': 0, 'p': 1, 'd': 2}
    
    factor =1
    if map[orb1[0]] > map[orb2[0]]:
        orb1, orb2 = orb2, orb1
        l, m, n = -l, -m, -n
        factor = -1
    
    atom1 = orb1.split('_')[1]
    atom2 = orb2.split('_')[1]
    
    if orb1.startswith('s') and orb2.startswith('s'):
        try: 
            return sk_params['s(' + atom1 + ')s(' + atom2 + ')σ']
        except:
            atom1, atom2 = atom2, atom1
            return sk_params['s(' + atom1 + ')s(' + atom2 + ')σ']

    if orb1.startswith('s') and orb2.startswith('p'):
        try:
            sp_sigma = factor*sk_params['s(' + atom1 + ')p(' + atom2 + ')σ']
        except:
            atom1, atom2 = atom2, atom1
            sp_sigma = -factor*sk_params['p(' + atom1 + ')s(' + atom2 + ')σ']
        return sp_overlap(l, m, n, orb1, orb2, sp_sigma)
     
    if orb1.startswith('p') and orb2.startswith('p'):
        try:
            pp_sigma = sk_params['p(' + atom1 + ')p(' + atom2 + ')σ']
            pp_pi = sk_params['p(' + atom1 + ')p(' + atom2 + ')π']
        except:
            atom2, atom1 = atom1, atom2
            pp_sigma = -sk_params['p(' + atom1 + ')p(' + atom2 + ')σ']
            pp_pi = -sk_params['p(' + atom1 + ')p(' + atom2 + ')π']
        return pp_overlap(l, m, n, orb1, orb2, pp_sigma, pp_pi)
    
    if orb1.startswith('s') and orb2.startswith('d'):
        try:
            sd_sigma = sk_params['s(' + atom1 + ')d(' + atom2 + ')σ']
        except:
            atom1, atom2 = atom2, atom1
            sd_sigma = -sk_params['d(' + atom1 + ')s(' + atom2 + ')σ']
        return sd_overlap(l, m, n, orb1, orb2, sd_sigma)
    
    if orb1.startswith('p') and orb2.startswith('d'):
        try:
            pd_sigma = factor*sk_params['p(' + atom1 + ')d(' + atom2 + ')σ']
            pd_pi = factor*sk_params['p(' + atom1 + ')d(' + atom2 + ')π']
        except:
            atom2, atom1 = atom1, atom2
            pd_sigma = -factor*sk_params['d(' + atom1 + ')p(' + atom2 + ')σ']
            pd_pi = -factor*sk_params['d(' + atom1 + ')p(' + atom2 + ')π']
        return pd_overlap(l, m, n, orb1, orb2, pd_sigma, pd_pi)
    
    if orb1.startswith('d') and orb2.startswith('d'):
        try:
            dd_sigma = sk_params['d(' + atom1 + ')d(' + atom2 + ')σ']
            dd_pi = sk_params['d(' + atom1 + ')d(' + atom2 + ')π']
            dd_delta = sk_params['d(' + atom1 + ')d(' + atom2 + ')δ']
        except:
            atom2, atom1 = atom1, atom2
            dd_sigma = -sk_params['d(' + atom1 + ')d(' + atom2 + ')σ']
            dd_pi = -sk_params['d(' + atom1 + ')d(' + atom2 + ')π']
            dd_delta = -sk_params['d(' + atom1 + ')d(' + atom2 + ')δ']
        return dd_overlap(l, m, n, orb1, orb2, dd_sigma, dd_pi, dd_delta)
