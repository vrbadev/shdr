# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 12:54:50 2024

@author: vrbav
"""

ind_xyzw = ['x', 'y', 'z', 'w']
ind_rgba = ['r', 'g', 'b', 'a']
ind_stpq = ['s', 't', 'p', 'q']

def recursive(combs_len, vec_size, comb=[]):
    if len(comb) < combs_len:
        for i in range(vec_size):
            recursive(combs_len, vec_size, comb=comb+[i])
    else:
        print("vecX_view<%d, SCALAR, %s> %s, %s, %s%s;" % (vec_size, ', '.join('%s' % i for i in comb), ''.join(['%s' % ind_xyzw[i] for i in comb]), ''.join(['%s' % ind_rgba[i] for i in comb]), ''.join(['%s' % ind_stpq[i] for i in comb]), ', self' if combs_len == vec_size and all([comb[i+1] - comb[i] == 1 for i in range(len(comb)-1)]) else ''))
            

if __name__ == "__main__":
    vec_size = 4
    combs_max_len = 4
    
    for combs_len in range(1, combs_max_len + 1):
        recursive(combs_len, vec_size)
        
