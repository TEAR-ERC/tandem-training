#!/usr/bin/env python3
'''
Functions related to plotting spatio-temporal evolution of variables as an image
By Jeena Yun
Last modification: 2025.01.13.
'''
import numpy as np
import matplotlib as mpl
import cmcrameri.cm as cram

def auto_cmap_for_sliprate():
    Vths = 1e-2                 # Slip rate threshold defining coseismic rupture [m/s]
    vmin,vmax = 1e-12,1e1       # Min. and Max. range for colormap
    Vp = 1e-9                   # Plate loading rate (Vp)
    V0 = 1e-6                   # Reference slip rate (V0)
    params = {'vmin':vmin, 'vmax':vmax, 'Vths':Vths, 'Vp':Vp, 'V0':V0}
    cmap_n,_ = generate_cmap('sliprate',params)
    norm = mpl.colors.LogNorm(vmin, vmax)
    return cmap_n,norm

def generate_cmap(image,params):
    cb_dict = {'state': r'State Variable $\psi$', 'slip': 'Cumulative Slip [m]', 'shearT': 'Shear Traction [MPa]', 'delshearT': 'Shear Traction Change [MPa]', 'sliprate': 'Slip Rate [m/s]', 'normalT': 'Normal Stress [MPa]', 'delnormalT': 'Normal Stress Change [MPa]'} 
    if image == 'sliprate':
        cm = mpl.colormaps['RdYlBu_r']
        col_list = [cm(i)[0:3] for i in [0.15,0.5,0.8,0.9]]
        col_list = [cm(0.15)[0:3],mpl.colormaps['jet'](0.67),cm(0.8)[0:3],cm(0.9)[0:3]]
        col_list.insert(0,(0,0,0))
        col_list.insert(3,mpl.colors.to_rgb('w'))
        col_list.append(mpl.colormaps['turbo'](0))
        float_list = [0,mpl.colors.LogNorm(params['vmin'],params['vmax'])(params['Vp']),mpl.colors.LogNorm(params['vmin'],params['vmax'])(params['V0'])]
        [float_list.append(k) for k in np.linspace(mpl.colors.LogNorm(params['vmin'],params['vmax'])(params['Vths']),1,4)]
        cmap_n = get_continuous_cmap(col_list,input_hex=False,float_list=float_list)
    elif image == 'shearT': # ---- plotting total stress
        cmap_n = cram.davos
    elif image == 'delshearT': # ---- plotting stress change
        cm = cram.vik
        col_list = [cm(i) for i in np.linspace(0,1,6)]
        col_list.insert(3,mpl.colors.to_rgb('w'))
        float_list = [mpl.colors.Normalize(params['vmin'],params['vmax'])(i) for i in np.linspace(params['vmin'],0,4)]
        [float_list.append(mpl.colors.Normalize(params['vmin'],params['vmax'])(k)) for k in np.linspace(0,params['vmax'],4)[1:]]
        cmap_n = get_continuous_cmap(col_list,input_hex=False,float_list=float_list)
    elif image == 'normalT': # ---- plotting total stress
        cmap_n = cram.davos
    elif image == 'delnormalT': # ---- plotting stress change
        cm = cram.vik
        col_list = [cm(i) for i in np.linspace(0,1,6)]
        col_list.insert(3,mpl.colors.to_rgb('w'))
        float_list = [mpl.colors.Normalize(params['vmin'],params['vmax'])(i) for i in np.linspace(params['vmin'],0,4)]
        [float_list.append(mpl.colors.Normalize(params['vmin'],params['vmax'])(k)) for k in np.linspace(0,params['vmax'],4)[1:]]
        cmap_n = get_continuous_cmap(col_list,input_hex=False,float_list=float_list)
    elif image == 'statevar':
        cmap_n = 'magma'
    cb_label = cb_dict[image]
    return cmap_n,cb_label

def get_continuous_cmap(col_list,input_hex=False,float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in col_list.
        If float_list is provided, each color in col_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        col_list: list of color code strings
        float_list: list of floats between 0 and 1, same length as col_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    if input_hex:
        rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in col_list]
    else:
        rgb_list = col_list.copy()

    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mpl.colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]
