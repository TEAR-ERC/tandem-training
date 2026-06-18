#!/usr/bin/env python3
'''
Functions related to plotting spatio-temporal evolution of variables as an image
By Jeena Yun & Benjamin Winjum
Last modification: 2026.06.16.
'''
from ipywidgets import Dropdown, Label, Output, VBox
from IPython.display import display, clear_output
import copy
from pathlib import Path
from glob import glob

def select_output_dir(base_dir_str=None):
    """
    Create a widget for selecting a Tandem output directory.
    
    Parameters:
    -----------
    base_dir_str : string
        Output directory containing Tandem jobs. 
        Default = None
    
    Returns:
    --------
    get_selected_job : callable
        Callback that returns the selected job name and full output path.
    """

    if base_dir_str is not None:
        # Base directory whose subfolders you want to list
        base_dir = Path(base_dir_str) 
    else:
        # If not defined, use default base directory
        base_dir = Path("../")

    default_job_name = 'depthVarying'
    # Gather job names
    job_names = sorted(
        [d.name for d in base_dir.iterdir() if d.is_dir()]
    )
    # Make default job name to be the first option
    if default_job_name in job_names:
        job_names.remove(default_job_name)

    job_names.insert(0, default_job_name)

    # Do not show pycache for cleaness
    unwanted_keys = ['__pycache__', '.ipynb_checkpoints', '.vscode']
    for bad_key in unwanted_keys:
        if bad_key in job_names: job_names.remove(bad_key)

    description_str = 'Folder:'
    # Create a dropdown menu
    dropdown = Dropdown(
        options=job_names,
        description=description_str,
        layout={'width': '300px'},
        style={'description_width': 'initial'}
    )

    state = {
        "full_path": base_dir / default_job_name,
        "job_name" : default_job_name,
    }

    info_label = Label(f"Selected job name: {state['job_name']} (default)")

    def on_change(change):
        if change["name"] == "value" and change["new"] is not None:
            job_name = change["new"]

            state["job_name"] = job_name
            state["full_path"] = base_dir / job_name

            info_label.value = f"Selected job name: {state['job_name']}"

    dropdown.observe(on_change, names="value")

    display(VBox([dropdown, info_label]))

    def get_selected_job():
        return state["job_name"], state["full_path"]

    return get_selected_job

def select_evolution_type(plot_fn, plot_options=None):
    """
    Create a widget for selecting a spatio-temporal evolution plot.
    
    Parameters:
    -----------
    plot_fn : callable
        Plotting function called by the widget
    
    plot_options : dict
        Options passed to the plotting function
    
    Returns:
    --------
    dropdown, info_label, out : tuple
        Dropdown widget, status label, and output widget for evolution plots.
    """

    evolution_types = ['', 'State Variable', 'Cumulative Slip', 'Slip Rate', 'Shear Stress', 'Shear Stress Change', 'Normal Stress', 'Normal Stress Change']
    args ={
        'State Variable': 'state',
        'Cumulative Slip' : 'slip',
        'Slip Rate' : 'sliprate',
        'Shear Stress' : 'shearT',
        'Shear Stress Change' : 'delshearT',
        'Normal Stress' : 'normalT',
        'Normal Stress Change' : 'delnormalT'
    }
    # Create a dropdown menu
    dropdown = Dropdown(
        options=evolution_types,
        description = 'Evolution type:',
        layout={'width': '300px'},
        style={'description_width': 'initial'}
    )
    info_label = Label("No type selected yet")
    out = Output()

    def render(selected_readable_name):
        with out:
            clear_output(wait=True)
            if not selected_readable_name:
                print("Nothing evolutoin type selected — choose one from the dropdown.")
                return

            target_variable = args[selected_readable_name]
            
            info_label.value = f"Selected evolutoin type: {selected_readable_name}  (target: {target_variable})"

            call_options = copy.deepcopy(plot_options)
            call_options['target_variable'] = target_variable

            plot_fn(**call_options)

    def on_change(change):
        if change["name"] == "value" and change["new"] is not None:
            render(change["new"])
            
    dropdown.observe(on_change, names="value")
    
    # display the UI (calls display immediately)
    display(VBox([dropdown, info_label, out]))

    # return the widget objects if caller wants to manipulate them
    return dropdown, info_label, out

def select_timeseries_type(plot_fn, plot_options=None):
    """
    Create a widget for selecting a time-series plot.
    
    Parameters:
    -----------
    plot_fn : callable
        Plotting function called by the widget
    
    plot_options : dict
        Options passed to the plotting function
    
    Returns:
    --------
    dropdown, info_label, out : tuple
        Dropdown widget, status label, and output widget for time series plots.
    """

    timeseries_types = ['', 'State Variable', 'Cumulative Slip', 'Slip Rate', 'Shear Stress', 'Normal Stress']
    args ={
        'State Variable': 'state',
        'Cumulative Slip' : 'slip',
        'Slip Rate' : 'sliprate',
        'Shear Stress' : 'shearT',
        'Normal Stress' : 'normalT',
    }

    # Create a dropdown menu
    dropdown = Dropdown(
        options=timeseries_types,
        description='Time series type:',
        layout={'width': '300px'},
        style={'description_width': 'initial'}
    )
    info_label = Label("No type selected yet")
    out = Output()

    def render(selected_readable_name):
        with out:
            clear_output(wait=True)
            if not selected_readable_name:
                print("No time series selected — choose one from the dropdown.")
                return

            target_variable = args[selected_readable_name]
            
            info_label.value = f"Selected time series type: {selected_readable_name}  (target: {target_variable})"

            call_options = copy.deepcopy(plot_options)
            call_options['target_variable'] = target_variable

            plot_fn(target_variable, **call_options)

    def on_change(change):
        if change["name"] == "value" and change["new"] is not None:
            render(change["new"])
            
    dropdown.observe(on_change, names="value")
    
    # display the UI (calls display immediately)
    display(VBox([dropdown, info_label, out]))

    # return the widget objects if caller wants to manipulate them
    return dropdown, info_label, out

def select_mesh(save_dir):
    """
    Detect a mesh file from a given directory.
    
    Parameters:
    -----------
    save_dir : str
        Directory containing Tandem simulation input files
    
    Returns:
    --------
    mesh_path : str
        Selected mesh file path.
    """
    mesh_path = glob(str(save_dir / '*.msh'))

    if len(mesh_path) > 1:
        raise NameError(f'More than one mesh file (*.msh) found in {str(save_dir)}')
    elif len(mesh_path) == 0:
        raise ValueError(f'No mesh file (*.msh) found in {str(save_dir)}')
    
    print(f"Mesh file: {mesh_path[0].split('/')[-1]}")
    return mesh_path[0]

def select_lua_scenario(save_dir):
    """
    Detect a Lua library and scenario from a given directory.
    
    Parameters:
    -----------
    save_dir : str
        Directory containing Tandem simulation input files
    
    Returns:
    --------
    lua_lib_name, lua_scenario_name : tuple
        Selected Lua library file name and scenario file name.
    """
    toml_path = glob(str(save_dir / '*.toml'))

    if len(toml_path) > 1:
        raise NameError(f'More than one parameter file (*.toml) found in {str(save_dir)}')
    elif len(toml_path) == 0:
        raise ValueError(f'No parameter file (*.toml) found in {str(save_dir)}')

    print(f"Parameter file: {toml_path[0].split('/')[-1]}")
    with open(toml_path[0], 'r') as fid:
        for line in fid:
            if '#' in line:
                info = line.split('#')[0].strip()
            else:
                info = line
            
            if 'lib' in info:
                lua_lib_name = info.split('\"')[1]
            if 'scenario' in info:
                lua_scenario_name = info.split('\"')[1]

    return lua_lib_name, lua_scenario_name