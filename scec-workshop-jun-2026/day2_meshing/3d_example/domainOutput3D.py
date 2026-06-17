import meshio
import numpy as np
import pyvista as pv

# Default colour palette for up to 10 tags
_COLORS = [
    'crimson', 'steelblue', 'forestgreen', 'darkorange',
    'mediumpurple', 'gold', 'teal', 'salmon', 'slategray', 'peru'
]


class GmshMesh3D:
    """
    Read a Gmsh v2.2 .msh file and visualize physical surface and volume groups
    using PyVista.

    Parameters
    ----------
    file_path : str
        Path to the .msh file.

    Usage
    -----
    mesh = GmshMesh3D('bp_subduction_3d.msh')
    pl = mesh.plot(
        surface_tags = [1, 3, 5],   # physical surface IDs to show
        volume_tags  = [6, 7],      # physical volume IDs to show
        show_mesh    = True,        # show triangle/tet edges
    )
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.points = None

        # tag -> list of element node arrays
        self.surf_by_tag = {}    # triangles
        self.vol_by_tag  = {}    # tetrahedra

        # tag -> name string (from field_data)
        self.tag_to_name = {}

        self._read()

    # ------------------------------------------------------------------
    def _read(self):
        m = meshio.read(self.file_path)
        self.points = m.points

        # Build tag -> name from field_data
        for name, (phys_id, dim) in m.field_data.items():
            self.tag_to_name[int(phys_id)] = name

        # Sort elements into surf_by_tag and vol_by_tag
        for cb, phys in zip(m.cells, m.cell_data['gmsh:physical']):
            for elem, tag in zip(cb.data, phys):
                tag = int(tag)
                if cb.type == 'triangle':
                    self.surf_by_tag.setdefault(tag, []).append(elem)
                elif cb.type == 'tetra':
                    self.vol_by_tag.setdefault(tag, []).append(elem)

    # ------------------------------------------------------------------
    def _make_surface(self, tag):
        """Build a pyvista PolyData for a surface tag."""
        tris  = np.array(self.surf_by_tag[tag])
        faces = np.hstack([np.full((len(tris), 1), 3), tris]).ravel()
        return pv.PolyData(self.points, faces)

    def _make_volume(self, tag):
        """Build a pyvista UnstructuredGrid for a volume tag."""
        tets      = np.array(self.vol_by_tag[tag])
        cells     = np.hstack([np.full((len(tets), 1), 4), tets]).ravel()
        celltypes = np.full(len(tets), pv.CellType.TETRA)
        return pv.UnstructuredGrid(cells, celltypes, self.points)

    # ------------------------------------------------------------------
    def plot(self,
             surface_tags=None,
             volume_tags=None,
             show_mesh=True,
             opacity_surface=0.9,
             opacity_volume=0.3,
             window_size=(1000, 700)):
        """
        Plot physical surface and volume groups.

        Parameters
        ----------
        surface_tags : list[int] or None
            Physical surface IDs to plot. None = all surfaces.
        volume_tags : list[int] or None
            Physical volume IDs to plot. None = all volumes.
        show_mesh : bool
            Show triangle/tet edges. Default True.
        opacity_surface : float
            Opacity for surface meshes (0=transparent, 1=opaque).
        opacity_volume : float
            Opacity for volume meshes.
        window_size : tuple
            Plotter window size in pixels.
        """
        if surface_tags is None:
            surface_tags = list(self.surf_by_tag.keys())
        if volume_tags is None:
            volume_tags = list(self.vol_by_tag.keys())

        pl = pv.Plotter(window_size=window_size)
        color_idx = 0

        # --- Surface groups ---
        for tag in surface_tags:
            if tag not in self.surf_by_tag:
                print(f"Warning: surface tag {tag} not found, skipping")
                continue
            mesh  = self._make_surface(tag)
            name  = self.tag_to_name.get(tag, f"surface_{tag}")
            color = _COLORS[color_idx % len(_COLORS)]
            color_idx += 1
            pl.add_mesh(mesh,
                        color=color,
                        opacity=opacity_surface,
                        show_edges=show_mesh,
                        label=f"[surf {tag}] {name}")

        # --- Volume groups ---
        for tag in volume_tags:
            if tag not in self.vol_by_tag:
                print(f"Warning: volume tag {tag} not found, skipping")
                continue
            mesh  = self._make_volume(tag)
            name  = self.tag_to_name.get(tag, f"volume_{tag}")
            color = _COLORS[color_idx % len(_COLORS)]
            color_idx += 1
            pl.add_mesh(mesh,
                        color=color,
                        opacity=opacity_volume,
                        show_edges=show_mesh,
                        label=f"[vol  {tag}] {name}")

        # Axes and legend
        pl.add_axes(line_width=3)
        pl.add_legend(bcolor='white', border=True, size=(0.25, 0.25))
        pl.show_grid()

        return pl

    # ------------------------------------------------------------------
    def summary(self):
        """Print a summary of all physical groups in the file."""
        print(f"File: {self.file_path}")
        print(f"Points: {len(self.points)}")
        print()
        print("Surface groups (dim=2):")
        for tag, elems in sorted(self.surf_by_tag.items()):
            name = self.tag_to_name.get(tag, '?')
            print(f"  tag={tag}  n_triangles={len(elems):6d}  name='{name}'")
        print()
        print("Volume groups (dim=3):")
        for tag, elems in sorted(self.vol_by_tag.items()):
            name = self.tag_to_name.get(tag, '?')
            print(f"  tag={tag}  n_tets={len(elems):6d}  name='{name}'")