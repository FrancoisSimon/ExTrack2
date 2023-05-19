
@njit
def LP_persistent(Cs, LocErr2, ds, hs, Dzs, nb_locs, nb_dims=2):
    Li = 0 # Log proba
    
    debug = 0
    
    state = 0
    d2s = ds**2
    Dz2s = Dzs**2
    h2s = hs**2
    nb_locs = len(Cs)
    persistent_disps = np.zeros((nb_locs-2, nb_dims))
    persistent_qs = np.zeros((nb_locs-2))
    
    #step 0:
    s2i = LocErr2 + d2s[state]
    x2i = h2s[state] + Dz2s[state]
    g2i = (Dz2s[state] *  s2i + s2i * h2s[state] + h2s[state] * Dz2s[state]) / (Dz2s[state] + h2s[state])
    
    alphai = h2s[state] / (h2s[state] + Dz2s[state])
    gammai = Cs[0]
    h2i = h2s[state]

    #step 1:
    
    #K1
    i=1
    ki = (Cs[i] - gammai) / alphai
    persistent_disps[i-1] = ki
    persistent_qs[i-1] = h2s[state]
    
    h2i = (g2i + LocErr2) / alphai**2
    Li += - np.log(alphai)

    #fs1
    ai = LocErr2 * alphai /(LocErr2 + g2i) + 1
    a2i = ai**2
    bi = (g2i * Cs[i] + LocErr2 * gammai) / (LocErr2 + g2i)
    s2i = (LocErr2 * g2i + g2i * d2s[state] + d2s[state] * LocErr2) / ((LocErr2 + g2i) * a2i)

    #ft1
    betai = x2i / (x2i + Dz2s[state])
    t2i = betai * Dz2s[state]
    #chi1
    x2i = x2i + Dz2s[state]
    #G1
    alphai = ai * betai * h2i / (h2i + t2i)
    gammai = bi + ai * t2i * ki / (h2i + t2i)
    g2i = a2i * (t2i *  s2i + s2i * h2i + h2i * t2i) / (t2i + h2i)
    
    #Q1
    kim1 = ki / betai
    q2i = (h2i + t2i) / betai**2
    Li += - np.log(betai)
    if debug:
        print('i', i)
        print('ki', ki, Cs[i] - Cs[i-1])
        print('qi', q2i**0.5)
        print('alphai', alphai)
        print('betai', betai)
        print('gammai', gammai)
    
    #step i:
    # int of G_{i-1}(r_i - (alpha_i * w_i + gamma_{i-1}) f_sigma(r_i - c_i) * f_d(r_{i+1} - r_i - w_i)) dr_i
    # = K_i(w_i - k_i) fsi(r_{i+1} - (a_i * w_i + b_i))        
    #Ki
    for i in range(2,  nb_locs - 2):
        ki = (Cs[i] - gammai) / alphai
        h2i = (g2i + LocErr2) / alphai**2
        Li += - np.log(alphai)
        
        #fsi
        ai = LocErr2 * alphai /(LocErr2 + g2i) + 1
        a2i = ai**2
        bi = (g2i * Cs[i] + LocErr2 * gammai) / (LocErr2 + g2i) 
        s2i = (LocErr2 * g2i + g2i * d2s[state] + d2s[state] * LocErr2) / ((LocErr2 + g2i) * a2i)

        # int of Q_{i-1}(w_i - k_{i-1}/beta_{i-1}) chi_{i-1}(w_i) K_i(w_i - k_i) fsi(r_{i+1} - (a_i * w_i + b_i)) fdz(w_{i+1} - w_i)    we need to reduce the 5 terms to 3: fusion of Q and K to L and K' and chi_{i-1} and fdz to chi_i and fti 
        # fusion of Qi-1 and Ki (Ki * Qi-1 -> Li * Ki)
        Li += np.sum(-0.5 * np.log(2*np.pi*(h2i + q2i)) - (ki - kim1)**2 / (2 * (h2i + q2i)))

        ki = (h2i * kim1 + q2i * ki) / (h2i + q2i)
        persistent_disps[i-1] = ki
        h2i = h2i * q2i / (q2i + h2i)
        
        #fti
        betai = x2i / (x2i + Dz2s[state])
        t2i = betai * Dz2s[state]
        #chii
        x2i = x2i + Dz2s[state]
        
        #Gi
        alphai = ai * betai * h2i / (h2i + t2i)
        gammai =  bi + ai * t2i * ki / (h2i + t2i)
        g2i = a2i * (t2i * s2i + s2i * h2i + h2i * t2i) / (t2i + h2i)

        #Qi
        kim1 = ki / betai
        q2i = (h2i + t2i) / betai**2
        Li += - np.log(betai)
        if debug:
            print('i', i)
            print('ki', ki, Cs[i] - Cs[i-1])
            print('qi', q2i**0.5)
            print('alphai', alphai)
            print('betai', betai)
            print('gammai', gammai)
        # variables needed for next step: alphai gammai g2i, kim1, q2i
        # gammai: most likely position after persistent motion
        # gi: std of the position after persistent motion (right before seeing the next position)

    #step n-1: (nothing changes for the dr int but the dw int does not have any fdz term)
    i = nb_locs - 2
    
    #Ki
    ki = (Cs[i] - gammai) / alphai
    h2i = (g2i + LocErr2) / alphai**2
    Li += - np.log(alphai)

    #fsi
    ai = LocErr2 * alphai /(LocErr2 + g2i) + 1
    a2i = ai**2
    bi = (g2i * Cs[i] + LocErr2 * gammai) / (LocErr2 + g2i) 
    s2i = (LocErr2 * g2i + g2i * d2s[state] + d2s[state] * LocErr2) / ((LocErr2 + g2i) * a2i)
    
    # fusion of Qi-1 and Ki (Ki * Qi-1 -> Li * Ki)
    Li += np.sum(-0.5 * np.log(2*np.pi*(h2i + q2i)) - (ki - kim1)**2 / (2 * (h2i + q2i)))
    ki = (h2i * kim1 + q2i * ki) / (h2i + q2i)
    persistent_disps[i-1] = ki
    h2i = h2i * q2i / (q2i + h2i)
    
    # int of chi_{i-1}(w_i) K_i(w_i - k_i) fsi(r_{i+1} - (a_i * w_i + b_i))
    #Qn-1
    Li += np.sum(-0.5 * np.log(2*np.pi*(h2i + x2i)) - (ki)**2 / (2 * (h2i + x2i)))

    #Gn-1
    gammai = bi + ai * x2i * ki / (h2i + x2i)
    g2i = a2i * (x2i * s2i + s2i * h2i + h2i * x2i) / (x2i + h2i)
    if debug:
        print('i', i)
        print('ki', ki, Cs[i] - Cs[i-1])
        print('alphai', alphai)
        print('betai', betai)
        print('gammai', gammai)
    
    #rn
    i = nb_locs - 1
    Li += np.sum(-0.5 * np.log(2*np.pi*(g2i + LocErr2)) - (Cs[i] - gammai)**2 / (2 * (g2i + LocErr2)))
    #persistent_disps[i-1] = Cs[i] - gammai
    
    # add a penalization term to force a displacement norm around h:
    disps = np.sum(persistent_disps**2, 1)**0.5
    #LP +=  -0.5*np.log(2*np.pi*Dzs[state]) - (np.abs(disps - hs[state])/(4*Dzs[state]))
    #Li += -np.sum(np.abs(disps - hs[state])/(2*10* 0.02))
    
    return Li, persistent_disps
