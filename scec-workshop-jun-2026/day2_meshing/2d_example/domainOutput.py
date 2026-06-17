#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 13:02:52 2024

@author: bar
"""

#%%

import numpy as np
import meshio
import matplotlib.pyplot as plt
from collections import defaultdict
#%%

DEFAULT_GROUPS = {1: "surface", 3: "fault", 5: "dirichlet"}

class GmshMesh2D:
    def __init__(self, file_path):
        """
        Initialize the GmshMesh object.

        Parameters:
            file_path (str): Path to the GMSH mesh file (typically a .msh file).
        """
        self.file_path = file_path
        self.mesh = None
        self.physical_curves = {}           # phys_id -> list of line-segment arrays
        self.physical_surfaces = {}         # phys_id -> list of triangle arrays
        self.phys_id_to_name = {}           # phys_id -> group name (curves)
        self.phys_surface_id_to_name = {}   # phys_id -> group name (surfaces)
        self.dim=2
        self.read_mesh()

    def read_mesh(self) -> None:
        """Read the file and fill .mesh / .physical_curves / .phys_id_to_name"""
        self.mesh = meshio.read(self.file_path)

        # 1) map phys-ID -> name for curves (dim=1) and surfaces (dim=2)
        if self.mesh.field_data:
            for name, (phys_id, dim) in self.mesh.field_data.items():
                if dim == 1:   # curve
                    self.phys_id_to_name[phys_id] = name
                elif dim == 2: # surface
                    self.phys_surface_id_to_name[phys_id] = name

        # 2) extract line and triangle elements that carry a physical tag
        if "gmsh:physical" in self.mesh.cell_data:
            for cb, phys in zip(self.mesh.cells,
                                self.mesh.cell_data["gmsh:physical"]):
                if cb.type == "line":
                    for seg, tag in zip(cb.data, phys):
                        self.physical_curves.setdefault(tag, []).append(seg)
                elif cb.type == "triangle":
                    for tri, tag in zip(cb.data, phys):
                        self.physical_surfaces.setdefault(tag, []).append(tri)

        # 3) ensure default IDs exist (possibly empty lists)
        if not self.physical_curves:           # nothing found → fallback groups
            for pid, name in DEFAULT_GROUPS.items():
                self.phys_id_to_name.setdefault(pid, name)
                self.physical_curves.setdefault(pid, [])
    
    def plot_mesh(self,ax=None):
        if self.mesh is None:
            raise ValueError("Mesh not loaded. Call read_mesh() first.")
        if ax is None:
            fig, ax = plt.subplots()

        # plot triangles
        
        for cb in self.mesh.cells:
            if cb.type == "triangle":
                tris = self.mesh.points[cb.data]
                for tri in tris:
                    pts = tri[:, :2]
                    pts = np.vstack([pts, pts[0]])
                    ax.plot(pts[:,0], pts[:,1], color="lightgrey", linewidth=0.25,alpha=0.7,zorder=0)

    def plot_physical_surfaces(self, ax=None, alpha=0.3, showLabel=True):
        """
        Fills each physical surface (2D domain) with a distinct color.
        Call this before plot_physical_curves so curves are drawn on top.
        """
        if self.mesh is None:
            raise ValueError("Mesh not loaded. Call read_mesh() first.")
        if ax is None:
            fig, ax = plt.subplots()

        cmap = plt.get_cmap("Set2")
        for i, (phys_id, tris) in enumerate(self.physical_surfaces.items()):
            label = self.phys_surface_id_to_name.get(phys_id, f"Surface {phys_id}")
            color = cmap(i % 20)
            first = True
            for tri in tris:
                pts = self.mesh.points[tri, :2]
                poly = plt.Polygon(pts, closed=True, facecolor=color,
                                   edgecolor='none', alpha=alpha,
                                   label=label if first else None, zorder=1)
                ax.add_patch(poly)
                first = False

        if showLabel and self.physical_surfaces:
            ax.legend()

    def plot_physical_curves(self, ax=None, show_triangles=True,color=None,showLabel=True):
        """
        Plots the physical curves, or the default 1/3/5 groups if none were in the file.
        Optionally overlays triangles in light grey.
        """
        if self.mesh is None:
            raise ValueError("Mesh not loaded. Call read_mesh() first.")
        if ax is None:
            fig, ax = plt.subplots()
        
        setColorAuto=False
        if color is None:
            setColorAuto=True

        # plot triangles
        if show_triangles:
            self.plot_mesh(ax=ax)

        cmap = plt.get_cmap("Set1")
        for i, (phys, segs) in enumerate(self.physical_curves.items()):
            label = self.phys_id_to_name.get(phys, f"Physical {phys}")
            
            if setColorAuto:
                color = cmap(i % 10)
                
            for j, seg in enumerate(segs):
                pts = self.mesh.points[seg, :2]
                if j == 0:
                    ax.plot(pts[:,0], pts[:,1], color=color, linewidth=2, label=label)
                else:
                    ax.plot(pts[:,0], pts[:,1], color=color, linewidth=2)

        #ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        if showLabel:
            ax.legend()
        #plt.title("Physical Curves with Mesh Triangles")
        
    def get_curve_points(self, phys_id, dim=None):
        """
        Extracts the endpoint coordinates of every line‐segment in a physical curve.
    
        Parameters
        ----------
        mesh : meshio.Mesh
            A meshio mesh object with mesh.points of shape (N,3).
        physical_curves : dict[int, list[array]]
            Mapping each physical‐curve ID to a list of 1D int‐arrays (node indices).
        phys_id : int
            The physical‐curve ID whose segments you want.
        dim : {2,3}, default 2
            Number of coordinates to return per node (2 → XY only, 3 → XYZ).
    
        Returns
        -------
        List[np.ndarray]
            A list of length-2 arrays of shape (2, dim), one per segment.
        """
        if dim is None:
            dim=self.dim
        #pts_list = []
        # grab the list of segments (each seg is e.g. array([n0, n1], dtype=int))
        pts_list: list[np.ndarray] = []
    
        for element in self.physical_curves.get(phys_id, []):
            # mesh.points[element] has shape (Nnodes, 3); slice → (Nnodes, dim)
            pts_list.append(self.mesh.points[element, :dim])
    
        return pts_list
