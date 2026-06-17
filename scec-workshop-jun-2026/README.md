# SCEC Tandem Training Workshop — June 2026

Hands-on tutorial materials for the [2026 SCEC Tandem Training Workshop](https://www.scec.org/events/2026-tandem-training-workshop/), held at the Scripps Institution of Oceanography, La Jolla, California, June 22–24, 2026.

---

## Workshop Sessions

| Session | Topic | Instructor | Time |
|---|---|---|---|
| A | Preparing Tandem input files and launching simulations | Yohai Magen | Tue 09:00–10:15 |
| B | Postprocessing and visualizing Tandem simulations | Jeena Yun | Tue 11:00–12:15 |
| C | Generating meshes for Tandem with Gmsh | Bar Oryan | Wed 08:30–09:45 |
| D | Running and compiling Tandem on HPC: Quakeworx and external clusters | Jeena Yun & Bar Oryan | Wed 12:45–13:45 |

---

## Repository Structure

```
.
├── day1_input_files/          # Session A: input files and tutorial notebook
│   ├── runing_bp3.ipynb       # Jupyter notebook — full Session A workflow
│   ├── bp3_QD.toml            # Step 1 – plain quasi-dynamic run (no pre-computed GFs)
│   ├── bp3_QDGreen_1.toml     # Step 2 – QDGreen: compute and checkpoint GFs
│   └── bp3_QDGreen_2.toml     # Step 3 – QDGreen: load pre-computed GFs
│
├── day1_visualization/        # Session B: post-processing and visualization notebook
│   ├── plot_tandem_results.ipynb   # Jupyter notebook — Session B probe outputs workflow
│   └── plotting_scripts/      # Python scripts for post-processing and visualization
│       ├── cumslip_compute.py # Functions to compute cumulative slip for a given time interval
│       ├── cumslip_plot.py    # Plot the cumulative slip output computed from cumslip_compute.py
│       ├── event_analyze.py   # Functions to automatically classify earthquake events from continuous Tandem outputs
│       ├── faultoutputs_image.py  # Plot spatio-temporal evolution of on-fault variables
│       ├── plot_utils.py      # Commonly used utilities for plotting scripts
│       ├── read_mesh.py       # Read, process, and plot Gmsh mesh file
│       ├── read_outputs.py    # Read and process raw Tandem outputs
│       └── select_widget.py   # Tools to support widget features in the plot_tandem_results.ipynb
│
├── uniform/                   # BP3 reference setup — uniform elastic properties
│   ├── bp3.geo                # Gmsh geometry script (adjustable dip, fault/surface resolution)
│   ├── bp3.lua                # Lua library: uniform Vs = 4.0 km/s, ρ = 2.9 g/cm³
│   ├── bp3.toml               # Tandem config (QDGreen, dip=30° reverse, 550 yr run)
│   ├── petsc_config.cfg       # PETSc solver settings (RK5DP adaptive time-stepping, MUMPS LU)
│   ├── msh_files/             # Pre-generated meshes at resolutions 0.05–0.50 km
│   │   └── bp3_res_<R>.msh
│   └── gf/                    # Pre-computed discrete Green's function checkpoints
│       └── gf_res_<R>/        # One directory per mesh resolution (0.25–0.50 km)
│           ├── gf_mat.bin
│           ├── gf_vec.bin
│           └── gf_facet_labels.bin
│
├── depthVarying/              # BP3 extension — depth-varying elastic properties
│   ├── bp3.geo                # Gmsh geometry script (same as uniform/)
│   ├── bp3.lua                # Lua library: exponential Vs profile (1.2→4.0 km/s) and ρ
│   ├── bp3.toml               # Tandem config (QDGreen, dip=30° reverse, 550 yr run)
│   └── petsc_config.cfg       # PETSc solver settings
│
└── meshing_tutorial/          # Session C: Gmsh meshing tutorial
    └── bp3_mesh/
        ├── bp3.geo            # Gmsh geometry script for BP3 mesh
        ├── bp3_python.ipynb   # Jupyter notebook — Gmsh Python API workflow
        ├── cascadia_profile.npz  # Cascadia subduction zone profile data
        ├── domainOutput.py    # Helper script for domain output
        └── gmsh_sc.png        # Screenshot of Gmsh mesh
```

---

## Workshop Information

- **Dates:** June 22–24, 2026
- **Location:** Scripps Institution of Oceanography, La Jolla, California
- **Website:** https://www.scec.org/events/2026-tandem-training-workshop/
- **Organizers:** Bar Oryan, Yohai Magen, Alice Gabriel, Jeena Yun, Dave May (UC San Diego)
- **Funded by:** Statewide California Earthquake Center (SCEC)
