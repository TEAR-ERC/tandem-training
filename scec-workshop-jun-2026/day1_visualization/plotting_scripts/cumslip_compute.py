#!/usr/bin/env python3
'''
Functions related to plotting cumulative slip vs. depth plot
By Jeena Yun
Last modification: 2025.11.18.
'''
import numpy as np
yr2sec = 365*24*60*60

def compute_cumslip(var, event_info, **kwargs):
    """
    Compute cumulative slip profiles during creep and coseismic periods.
    
    Parameters:
    -----------
    var : array-like
        Tandem simulation output object
    
    event_info : dict
        Earthquake information dictionary
    
    **kwargs : dict
        Additional keyword options
    
    Returns:
    --------
    cumslip_outputs : dict
        Dictionary with cscreep, depcreep, cscoseis, and depcoseis arrays for cumulative slip and corresponding depths.
    """
    from scipy import interpolate
    options = {
        'dt_creep' : 5*yr2sec,
        'dt_coseismic' : 1,
        'print_on' : True
    }
    options.update(kwargs)

    outputs, dep = var.outputs, var.dep

    if options['print_on']: print('Compute cumulative slip vs. depth... ', end='')

    cscreep, depcreep = [], []
    cscoseis, depcoseis = [], []

    # Retreive event time info
    tstart_coseis, tend_coseis = event_info['tstart'], event_info['tend']

    # Now interpolate the cumulative slip using given event time ranges
    for i in range(len(dep)):
        z = abs(dep[i])
        time = np.array(outputs[i])[:, 0]
        cumslip = np.array(outputs[i])[:, 2]

        f = interpolate.interp1d(time, cumslip)

        # -------------------- Creep
        tcreep = np.arange(time[0], time[-1], options['dt_creep'])
        cscreep.append(f(tcreep))
        depcreep.append(z * np.ones(len(tcreep)))

        # -------------------- Coseismic
        if len(tstart_coseis) > 0:
            cs, depth = [], []
            for j in range(len(tstart_coseis)):
                if tend_coseis[j] - tstart_coseis[j] >= options['dt_coseismic']:
                    tcoseis = np.arange(tstart_coseis[j], tend_coseis[j] + options['dt_coseismic'], options['dt_coseismic'])
                else:
                    tcoseis = np.arange(tstart_coseis[j], tstart_coseis[j] + 2 * options['dt_coseismic'], options['dt_coseismic'])
                cs.append(f(tcoseis))
                depth.append(z * np.ones(len(tcoseis)))

            cscoseis.append([item for sublist in cs for item in sublist])
            depcoseis.append([item for sublist in depth for item in sublist])

    cumslip_outputs = {'cscreep':cscreep, 'depcreep':depcreep, 'cscoseis':cscoseis, 'depcoseis':depcoseis}
    if options['print_on'] : print('Done!')
    return cumslip_outputs
    
def compute_spinup(outputs, dep, cumslip_outputs, event_info, spin_up):
    """
    Shift cumulative slip outputs by a selected spin-up threshold.
    
    Parameters:
    -----------
    outputs : numpy.ndarray
        Tandem simulation output array
    
    dep : numpy.ndarray
        Depth information for Tandem simulation outputs
    
    cumslip_outputs : dict
        Cumulative slip output dictionary, created from cumslip_compute.py
    
    event_info : dict
        Earthquake information dictionary
    
    spin_up : array-like
        Spin-up mode and threshold
    
    Returns:
    --------
    spup_cumslip_outputs : dict
        Dictionary with new_init_Sl, new_init_dp, spup_evslip, spup_cscreep, spup_cscoseis, and spin_up_idx.
    """
    yr2sec = 365*24*60*60
    system_wide = event_info['system_wide']
    tstart = event_info['tstart']
    its_all = event_info['its_all']
    evslip = event_info['evslip']
    evdep = event_info['evdep']
    options = event_info['options']
    cscreep = cumslip_outputs['cscreep']
    cscoseis = cumslip_outputs['cscoseis']

    var_mode = spin_up[0]
    spin_up = float(spin_up[1])
    if spin_up == 0:
        spin_up_idx = 0
    else:
        if var_mode == 'm':
            if options['print_on']: print('Spin-up applied after slip > %2.2f m'%(spin_up))
            spin_up_idx = system_wide[np.where(evslip[system_wide] >= spin_up)[0][0]]
        elif var_mode == 'yrs':
            if options['print_on']: print('Spin-up applied after %2.2f yrs'%(spin_up))
            spin_up_idx = system_wide[np.where(tstart[system_wide]/yr2sec >= spin_up)[0][0]]

    spup_cscreep = np.copy(cscreep)
    spup_cscoseis = np.copy(cscoseis)
    spup_evslip = np.copy(evslip)

    if spin_up == 0:
        spin_up_idx = 0
        new_init_Sl = np.zeros(len(dep))
        new_init_dp = np.sort(abs(dep))
    else:
        new_init_Sl = []
        new_init_dp = []
        c = 0
        for i in range(len(dep)):
            z = abs(dep[i])        
            cumslip = np.array(outputs[i])[:, 2]

            new_init_Sl.append(cumslip[its_all[spin_up_idx]])
            new_init_dp.append(z)
            
            spup_cscreep[c] = cscreep[c] - new_init_Sl[-1]
            spup_cscoseis[c] = cscoseis[c] - new_init_Sl[-1]

            if np.isin(z, evdep):
                indx = np.where(z==evdep)[0]
                spup_evslip[indx] = evslip[indx] - new_init_Sl[-1]
            c += 1

    spup_cumslip_outputs = {'new_init_Sl':new_init_Sl, 'new_init_dp':new_init_dp, \
                            'spup_evslip':spup_evslip, 'spup_cscreep':spup_cscreep, 'spup_cscoseis':spup_cscoseis, 'spin_up_idx':spin_up_idx}

    return spup_cumslip_outputs

def SSE_event_times(outputs, dep, depth_range, event_info):
    """
    Identify slow-slip event time windows within a selected depth range.
    
    Parameters:
    -----------
    outputs : numpy.ndarray
        Tandem simulation output array
    
    dep : numpy.ndarray
        Depth information for Tandem simulation outputs
    
    depth_range : str
        Depth range used to identify shallow or deep SSEs
    
    event_info : dict
        Earthquake information dictionary
    
    Returns:
    --------
    sse_tstart, sse_tend, sse_evdep, its, ite, time, psr, cumslip, z, idep : tuple
        SSE timing, event depths, index ranges, time array, peak slip-rate history, cumulative slip array, sorted depths, and depth-index bounds.
    """
    tstart = event_info['tstart']
    tend = event_info['tend']
    options = event_info['options']
    ii = np.argsort(abs(dep))
    time = outputs[0, :, 0]
    cumslip = outputs[ii, :, 2]
    sliprate = abs(outputs[ii, :, 4])
    z = abs(dep[ii])

    if depth_range == 'shallow':
        Vths = -9.6
        target_depth = [0., 5.]
        if options['print_on']: print('Shallow SSEs (0 - 5 km)')
    elif depth_range == 'deep':
        Vths = -8.5
        target_depth = [10., 20.]
        if options['print_on']: print('Deep SSEs (10 - 20 km)')

    if len(target_depth) > 1:
        idep = [np.argmin(abs(z - abs(target_depth[0]))), np.argmin(abs(z - abs(target_depth[1])))]

    psr = np.log10(np.max(sliprate[idep[0]:idep[1], :], axis=0))
    ipsr = np.argmax(sliprate[idep[0]:idep[1], :], axis=0)

    # ----- Define events by peak sliprate
    events = np.where(np.logical_and(psr < -6, psr > Vths))[0]

    sse_tstart, sse_tend, sse_evdep = [], [], []
    if len(events) > 0:
        jumps = np.where(np.diff(events)>1)[0]+1

        tmp_its = events[np.hstack(([0], jumps))]
        tmp_ite = events[np.hstack((jumps-1, len(events)-1))]

        # ----- Remove events with too short duration
        kk = np.where(tmp_ite-tmp_its>=1)[0]
        tmp_its = tmp_its[kk]
        tmp_ite = tmp_ite[kk]

        # ----- Remove picks including coseismic events
        mpsr = np.array([max(psr[tmp_its[k]:tmp_ite[k]]) for k in range(len(tmp_its))])
        ii = np.where(mpsr < -6)[0]
        tmp_its = tmp_its[ii]
        tmp_ite = tmp_ite[ii]

        # ----- Remove picks including coseismic events
        mpsr = np.array([max(psr[tmp_its[k]:tmp_ite[k]]) for k in range(len(tmp_its))])
        ii = np.where(mpsr < -6)[0]
        tmp_its = tmp_its[ii]
        tmp_ite = tmp_ite[ii]

        # ----- Remove acceleration before coseismic events
        kk = np.array([len(np.where(tstart>=ste)[0]) for ste in time[tmp_ite]]) > 0
        nearest_end = np.log10(np.array([tstart[np.where(tstart>=ste)[0][0]]-ste for ste in time[tmp_ite[kk]]]))
        nearest_end = np.append(nearest_end, 10*np.ones(len(kk)-sum(kk)))
        ii = np.where(nearest_end > 6)[0]
        tmp_its = tmp_its[ii]
        tmp_ite = tmp_ite[ii]

        nearest_start = np.log10(np.array([sts - tend[np.where(tend<=sts)[0][-1]] for sts in time[tmp_its]]))
        ii = np.where(nearest_start > 7)[0]
        tmp_its = tmp_its[ii]
        tmp_ite = tmp_ite[ii]

        # ----- Merge peaks with unphysically close time
        interval = np.hstack(([1e8], time[tmp_its][1:]-time[tmp_ite][:-1]))
        its_filter = np.ones(len(tmp_its), dtype=bool)
        ite_filter = np.ones(len(tmp_ite), dtype=bool)
        for u, SRvar in enumerate(interval):
            if SRvar <= 3e7:
                its_filter[u] = False
                ite_filter[u-1] = False
        its = tmp_its[its_filter]
        ite = tmp_ite[ite_filter]
        
        sse_tstart = time[its]
        sse_tend = time[ite]
        sse_evdep = z[ipsr[its]]
    return sse_tstart, sse_tend, sse_evdep, its, ite, time, psr, cumslip, z, idep