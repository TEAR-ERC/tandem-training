#!/usr/bin/env python3
'''
Functions related to plotting cumulative slip vs. depth plot
By Jeena Yun
Last modification: 2025.01.13.
'''
import numpy as np
import matplotlib.pylab as plt
from scipy import interpolate

mypink = (230/255,128/255,128/255)
mydarkpink = (200/255,110/255,110/255)
myblue = (118/255,177/255,230/255)
myburgundy = (214/255,0,0)
mynavy = (17/255,34/255,133/255)
mylightblue = (218/255,230/255,240/255)
myygreen = (120/255,180/255,30/255)
mylavender = (170/255,100/255,215/255)
mydarkviolet = (145/255,80/255,180/255)
pptyellow = (255/255,217/255,102/255)

yr2sec = 365*24*60*60

def get_cumslip(save_dir,outputs,dep,Vths,dt_creep,dt_coseismic,print_on=True,save_on=True):
    fname = '%s/cumslip_outputs_Vths_%1.0e_dtcreep_%d_dtcoseis_%02d.npy'%(save_dir,Vths,dt_creep/yr2sec,dt_coseismic*10)
    import os
    if os.path.exists(fname):
        cumslip_outputs = dict(np.load(fname,allow_pickle=True).item())
    else:
        cumslip_outputs = compute_cumslip(outputs,dep,Vths,dt_creep,dt_coseismic,print_on,save_on)
        if save_on: np.save(fname,cumslip_outputs)
    return cumslip_outputs

def plot_cumslip_basic(save_dir,cumslip_outputs,save_on=True):
    plt.rcParams['font.size'] = '15'
    fig,ax = plt.subplots(figsize = (8,6))
    ax.plot(cumslip_outputs['cscoseis'], cumslip_outputs['depcoseis'], color=mydarkpink, lw=1)
    ax.plot(cumslip_outputs['cscreep'], cumslip_outputs['depcreep'], color='0.62', lw=1)
    ax.set_xlabel('Cumulative Slip [m]', fontsize=17)
    ax.set_ylabel('Depth [km]', fontsize=17)
    xl = ax.get_xlim()
    ax.set_xlim(0,xl[1])
    ax.set_ylim(max([np.max(np.absolute(cumslip_outputs['depcoseis'])),np.max(np.absolute(cumslip_outputs['depcreep']))]),0)
    # ax.set_ylim(yl[1],0)
    if save_on:
        plt.savefig('%s/cumslip.png'%(save_dir))

def event_times(dep,outputs,Vths,dt_coseismic,print_on=True):
    time = np.array(outputs[0][:,0])
    sliprate = abs(np.array([outputs[i][:,4] for i in np.argsort(abs(dep))]))
    z = np.sort(abs(dep))

    psr = np.max(sliprate,axis=0)
    ipsr = np.argmax(sliprate,axis=0)

    # Define events by peak sliprate
    events = np.where(psr > Vths)[0]

    if len(events) > 0:
        jumps = np.where(np.diff(events)>1)[0]+1

        tmp_tstart = time[events][np.hstack(([0],jumps))]
        tmp_tend = time[events][np.hstack((jumps-1,len(events)-1))]
        tmp_evdep = ipsr[events][np.hstack(([0],jumps))]

        # ----- Remove events with too short duration
        ii = np.where(tmp_tend-tmp_tstart>=dt_coseismic)[0]
        tstart = tmp_tstart[ii]
        tend = tmp_tend[ii]
        evdep = z[tmp_evdep[ii]]

        its_all = np.array([np.argmin(abs(time-t)) for t in tstart])
        ite_all = np.array([np.argmin(abs(time-t)) for t in tend])
        evdep = z[ipsr[its_all]]
        tstart = time[its_all]

        # ----- Remove events if it is only activated at specific depth: likely to be unphysical
        num_active_dep = np.array([np.sum(np.sum(sliprate[:,its_all[k]:ite_all[k]]>Vths,axis=1)>0) for k in range(len(tstart))])
        if len(num_active_dep>1) > 0:
            if print_on: print('Remove single-depth activated event:',np.where(num_active_dep==1)[0])
            tstart = tstart[num_active_dep>1]
            tend = tend[num_active_dep>1]
            evdep = evdep[num_active_dep>1]
        else:
            if print_on: print('All events activate more than one depth')

    else:
        tstart, tend, evdep = [],[],[]

    return tstart, tend, evdep

def compute_cumslip(outputs,dep,Vths,dt_creep,dt_coseismic,print_on):
    cscreep = []
    depcreep = []
    cscoseis = []
    depcoseis = []
    fault_slip = []

    # Obtain globally min. event start times and max. event tend times
    tstart_coseis, tend_coseis, evdep = event_times(dep,outputs,Vths,dt_coseismic,print_on)
    evslip = np.zeros(tstart_coseis.shape)

    # Now interpolate the cumulative slip using given event time ranges
    for i in np.argsort(abs(dep)):
        z = abs(dep[i])
        time = np.array(outputs[i])[:,0]
        cumslip = np.array(outputs[i])[:,2]

        f = interpolate.interp1d(time,cumslip)

        # -------------------- Creep
        tcreep = np.arange(time[0],time[-1],dt_creep)
        cscreep.append(f(tcreep))
        depcreep.append(z*np.ones(len(tcreep)))

        # -------------------- Coseismic
        cs = []
        depth = []
        Dbar = []
        for j in range(len(tstart_coseis)):
            tcoseis = np.arange(tstart_coseis[j],tend_coseis[j],dt_coseismic)
            cs.append(f(tcoseis))
            depth.append(z*np.ones(len(tcoseis)))
            Dbar.append(f(tcoseis)[-1]-f(tcoseis)[0])

        cscoseis.append([item for sublist in cs for item in sublist])
        depcoseis.append([item for sublist in depth for item in sublist])
        fault_slip.append(Dbar)

        # -------------------- Event detph
        if np.isin(z,evdep):
            indx = np.where(z==evdep)[0]
            evslip[indx] = f(tstart_coseis)[indx]

    cumslip_outputs = {'tstart_coseis':tstart_coseis, 'tend_coseis':tend_coseis,\
                        'evslip':evslip, 'evdep':evdep,'fault_slip':fault_slip,\
                            'cscreep':cscreep, 'depcreep':depcreep,\
                                'cscoseis':cscoseis, 'depcoseis':depcoseis}
    
    return cumslip_outputs