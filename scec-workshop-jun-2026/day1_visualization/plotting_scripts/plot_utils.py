#!/usr/bin/env python3
"""
Collection of customized plot-related functions and constants
Last modification: 2025.03.09.
by Jeena Yun
"""
import numpy as np
import matplotlib.pylab as plt

class Figpref:
    def __init__(self):
        # My colors
        """
        Initialize the object.
        
        Parameters:
        -----------
        self : object
            Current object instance
        
        Returns:
        --------
        None
        """
        self.myburgundy = (214/255, 0, 0)
        self.mydarkpink = (200/255, 110/255, 110/255)
        self.mypink = (230/255, 128/255, 128/255)
        self.mypalepink = (235/255, 180/255, 180/255)
        self.myorange = (1, 149/255, 33/255)
        self.mypeach = (255/255, 210/255, 190/255)
        self.myyellow = (1, 221/255, 51/255)
        self.myolive = (120/255, 180/255, 30/255)
        self.mygreen = (102/255, 153/255, 26/255)
        self.mylightmint = (225/255, 245/255, 240/255)
        self.mymint = (70/255, 225/255, 165/255)
        self.mylightblue = (218/255, 230/255, 240/255)
        self.myblue = (118/255, 177/255, 230/255)
        self.myashblue = (225/255, 230/255, 230/255)
        self.myteal = (85/255, 170/255, 170/255)
        self.mynavy = (17/255, 34/255, 133/255)
        self.myviolet = (130/255, 6/255, 214/255)
        self.mylavender = (170/255, 100/255, 215/255)
        self.mylilac = (200/255, 165/255, 230/255)
        self.mydarkviolet = (145/255, 80/255, 180/255)
        self.mybrown = (104/255, 54/255, 46/255)
        self.mygrey = (158/255, 158/255, 158/255)
        self.mydarkgrey = (89/255, 89/255, 89/255)

        self.pptlightyellow = (255/255, 242/255, 204/255)
        self.pptyellow = (255/255, 217/255, 102/255)
        self.pptlightorange = (250/255, 200/255, 175/255)
        self.pptorange = (235/255, 125/255, 50/255)
        self.pptlightgreen = (226/255, 240/255, 217/255)
        self.pptgreen = (115/255, 175/255, 70/255)
        self.pptgreen = (169/255, 209/255, 142/255)

        self.yr2sec = 365*24*60*60

        self.default_options = {
            'target_depth': None,
            'dep': None,
            'fig_title': None,
            'xlab': '',
            'ylab': '',
            'toff': 0,
            'sz': 25,
            'lw': 1.5,
            'ls': '-',
            'col': 'k',
            'fc': 'k',
            'ec': 'k',
            'lab': None,
            'prefix': None,
            'legend_txt_fs': 11,
            'axis_fs': 13,
            'tit_fs': 15,
            'zorder': 3,
            'Dths': 30,
            'scatter': False,
            'shallow': False,
            'deep': False,
            'log10': True,
            'grid_on': True,
            'plot_in_sec': True,
            'ylim': False,
            'no_xlab': False,
            'no_ylab': False,
            # 'abs_on': False,
            'abs_y': False,
            'ax': None,
            'title_on': True,
            'print_on': True,
            'save_on': True
        }

    def set_time(self, t, plot_in_sec):
        # --- Set x-axis scale and x-label
        """
        Convert simulation time to plotting units and create an axis label.
        
        Parameters:
        -----------
        self : object
            Current object instance
        
        t : array-like
            Time array
        
        plot_in_sec : bool
            Flag for plotting time in seconds
        
        Returns:
        --------
        time, xlab : tuple
            Converted time array and x-axis label.
        """
        if plot_in_sec:
            time, xlab = np.array(t), 'Time [s]'
        else:
            time, xlab = np.array(t)/self.yr2sec, 'Time [yr]'
        return time, xlab
    
    def plot_matfric(self, ax, dat, z, xlab, ylab, col='k', lw=2, axis_fs=13, lab='', ls='-', grid_on=True):
        """
        Plot material or frictional properties versus depth.
        
        Parameters:
        -----------
        self : object
            Current object instance
        
        ax : matplotlib.axes.Axes
            Axes object to plot on
        
        dat : array-like
            Data array to plot
        
        z : array-like
            Depth coordinate array
        
        xlab : str
            x-axis label
        
        ylab : str
            y-axis label
        
        col : str
            Line or annotation color
        
        lw : float
            Line width
        
        axis_fs : int
            Axis label font size
        
        lab : str
            Plot legend label
        
        ls : str
            Line style
        
        grid_on : bool
            Flag for drawing the plot grid
        
        Returns:
        --------
        ax, xl, yl : tuple
            Axes object and x/y limits after plotting.
        """
        ax.plot(dat, z, lw=lw, color=col, label=lab, zorder=3, linestyle=ls)
        ax.set_xlabel(xlab, fontsize=axis_fs)
        ax.set_ylabel(ylab, fontsize=axis_fs)
        if grid_on: ax.grid(True, alpha=0.5)
        ax.invert_yaxis()
        plt.tight_layout()
        xl = ax.get_xlim()
        yl = ax.get_ylim()
        return ax, xl, yl

    def decor(self, ax, xl, kink_z, col='0.5', lw=1.5):
        """
        Add depth marker lines to a plot.
        
        Parameters:
        -----------
        self : object
            Current object instance
        
        ax : matplotlib.axes.Axes
            Axes object to plot on
        
        xl : tuple
            x-axis limits
        
        kink_z : array-like
            Depths of mesh-size transition markers
        
        col : str
            Line or annotation color
        
        lw : float
            Line width
        
        Returns:
        --------
        ax : matplotlib.axes.Axes
            Decorated axes object.
        """
        for z in kink_z:
            ax.hlines(-z, xl[0], xl[1], color=col, linestyles='--', lw=lw)
        ax.set_xlim(xl)
        return ax

    def plot_profile(self, ax, y, var, **kwargs):
        """
        Plot a profile variable versus depth.
        
        Parameters:
        -----------
        self : object
            Current object instance
        
        ax : matplotlib.axes.Axes
            Axes object to plot on
        
        y : array-like
            Dependent variable array
        
        var : array-like
            Tandem simulation output object
        
        **kwargs : dict
            Additional keyword options
        
        Returns:
        --------
        ax : matplotlib.axes.Axes
            Axes object containing the profile plot.
        """
        options = self.default_options.copy()
        options.update(kwargs)
        if options['abs_y']:
            plot_y = abs(y)
        else:
            plot_y = y
        if options['ls'] == '-o':
            ax.plot(var, plot_y, options['ls'], color=options['col'], lw=options['lw'], label=options['lab'], zorder=options['zorder'])
        elif options['scatter']:
            ax.scatter(var, plot_y, s=options['sz'], fc=options['fc'], ec=options['ec'], lw=options['lw'], label=options['lab'], zorder=options['zorder'])
        else:
            ax.plot(var, plot_y, color=options['col'], lw=options['lw'], label=options['lab'], linestyle=options['ls'], zorder=options['zorder'])
        if not options['no_xlab']: ax.set_xlabel(options['xlab'], fontsize=options['axis_fs'])
        if not options['no_ylab']: ax.set_ylabel('Depth [km]', fontsize=options['axis_fs'])
        if options['ylim']: ax.set_ylim(max(plot_y), 0)
        if options['grid_on']: ax.grid(True, alpha=0.5)
        return ax

    def plot_timeserise(self, ax, t, var, **kwargs):
        """
        Plot a variable through time.
        
        Parameters:
        -----------
        self : object
            Current object instance
        
        ax : matplotlib.axes.Axes
            Axes object to plot on
        
        t : array-like
            Time array
        
        var : array-like
            Tandem simulation output object
        
        **kwargs : dict
            Additional keyword options
        
        Returns:
        --------
        ax : matplotlib.axes.Axes
            Axes object containing the time-series plot.
        """
        options = self.default_options.copy()
        options.update(kwargs)
        if abs(options['toff']) > 0: t += options['toff']
        if options['ls'] == '-o':
            ax.plot(t, var, options['ls'], color=options['col'], lw=options['lw'], label=options['lab'], linestyle=options['ls'], zorder=options['zorder'])
        elif options['scatter']:
            ax.scatter(t, var, s=options['sz'], fc=options['fc'], ec=options['ec'], lw=options['lw'], label=options['lab'], zorder=options['zorder'])
        else:
            ax.plot(t, var, color=options['col'], lw=options['lw'], label=options['lab'], linestyle=options['ls'], zorder=options['zorder'])
        if not options['no_xlab']: ax.set_xlabel(options['xlab'], fontsize=options['axis_fs'])
        if not options['no_ylab']: ax.set_ylabel(options['ylab'], fontsize=options['axis_fs'])
        if options['grid_on']: ax.grid(True, alpha=0.5)
        if options['fig_title'] is not None and options['title_on']: ax.set_title(options['fig_title'], fontsize=options['tit_fs'], fontweight = 'bold')
        return ax

    def decor_mesh_size(self, ax, **kwargs):
        """
        Decorate a mesh-size plot.
        
        Parameters:
        -----------
        self : object
            Current object instance
        
        ax : matplotlib.axes.Axes
            Axes object to plot on
        
        **kwargs : dict
            Additional keyword options
        
        Returns:
        --------
        None
        """
        options = self.default_options.copy()
        options.update(kwargs)

        ax.set_xlabel('Element Size [km]', fontsize=options['axis_fs'])
        ax.set_ylabel('Depth [km]', fontsize=options['axis_fs'])
        ax.hlines(75, 0, 500, color=options['col'], linestyles='--')
        if 'no_xlim' not in options:
            ax.text(4.8, 76.5, 'End of the fault', ha='right', va='top', fontsize=options['legend_txt_fs'], color=options['col'], fontweight='bold')
            ax.set_xlim(0, 5)
        ax.grid(True, alpha=0.5)

    def get_continuous_cmap(self, col_list, input_hex=False, float_list=None):
        """
        Create a continuous matplotlib colormap from a list of colors.
        
        Parameters:
        -----------
        self : object
            Current object instance
        
        col_list : list
            Color list used to build the colormap
        
        input_hex : bool
            Flag indicating whether colors are hex strings
        
        float_list : list
            Color anchor positions between 0 and 1
        
        Returns:
        --------
        cmp : matplotlib.colors.LinearSegmentedColormap
            Continuous colormap created from the input colors.
        """
        import matplotlib.colors as mcolors
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
            rgb_list = [self.rgb_to_dec(self.hex_to_rgb(i)) for i in col_list]
        else:
            rgb_list = col_list.copy()

        if float_list:
            pass
        else:
            float_list = list(np.linspace(0, 1, len(rgb_list)))
            
        cdict = dict()
        for num, col in enumerate(['red', 'green', 'blue']):
            col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
            cdict[col] = col_list
        cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
        return cmp

    def hex_to_rgb(self, value):
        """
        Convert a hex color string to RGB values.
        
        Parameters:
        -----------
        self : object
            Current object instance
        
        value : str
            Color value to convert
        
        Returns:
        --------
        rgb : tuple
            RGB color values.
        """
        value = value.strip("#") # removes hash symbol if present
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    def rgb_to_dec(self, value):
        """
        Convert RGB values to decimal color values.
        
        Parameters:
        -----------
        self : object
            Current object instance
        
        value : str
            Color value to convert
        
        Returns:
        --------
        rgb_decimal : list
            RGB values scaled to decimal color values.
        """
        return [v/256 for v in value]