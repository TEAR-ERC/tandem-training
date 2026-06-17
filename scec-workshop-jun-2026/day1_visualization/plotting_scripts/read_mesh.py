import meshio
import matplotlib.pyplot as plt
import numpy as np
'''
Read, process, and plot Gmsh mesh file used for Tandem simulation
Modified by Bar Oryan and Jeena Yun
Last modification: 2026.01.14.
'''

DEFAULT_GROUPS = {1: "surface", 3: "fault", 5: "dirichlet"}

class GmshMesh2D:
    def __init__(self, file_path):
        """
        Initialize the object.
        
        Parameters:
        -----------
        self : object
            Current object instance
        
        file_path : str
            Gmsh mesh file path
        
        Returns:
        --------
        None
        """
        self.file_path = file_path
        self.mesh = None
        self.physical_curves = {}      # phys_id -> list of line‐segment arrays
        self.phys_id_to_name = {}      # phys_id -> group name
        self.dim=2
        self.read_mesh()

    def read_mesh(self) -> None:
        """
        Read curves, points, and elements from a Gmsh mesh file.
        
        Parameters:
        -----------
        self : object
            Current object instance
        
        Returns:
        --------
        None
        """
        self.mesh = meshio.read(self.file_path)

        # 1) map phys-ID → name  (only for 1-D groups)
        if self.mesh.field_data:
            for name, (phys_id, dim) in self.mesh.field_data.items():
                if dim == 1:                 # 1-D = curve
                    self.phys_id_to_name[phys_id] = name

        # 2) extract line elements that carry a physical tag
        if "gmsh:physical" in self.mesh.cell_data:
            for cb, phys in zip(self.mesh.cells,
                                self.mesh.cell_data["gmsh:physical"]):
                if cb.type == "line":
                    for seg, tag in zip(cb.data, phys):
                        self.physical_curves.setdefault(tag, []).append(seg)

        # 3) ensure default IDs exist (possibly empty lists)
        if not self.physical_curves:           # nothing found → fallback groups
            for pid, name in DEFAULT_GROUPS.items():
                self.phys_id_to_name.setdefault(pid, name)
                self.physical_curves.setdefault(pid, [])

    def plot_physical_curves(self, ax=None, show_triangles=True, save_dir=None):
        """
        Plot physical curves from a Gmsh mesh file.
        
        Parameters:
        -----------
        self : object
            Current object instance
        
        ax : matplotlib.axes.Axes
            Axes object to plot on
        
        show_triangles : bool
            Flag for plotting mesh triangles
        
        save_dir : str
            Directory containing Tandem simulation outputs
        
        Returns:
        --------
        None
        """
        if self.mesh is None:
            raise ValueError("Mesh not loaded. Call read_mesh() first.")
        if ax is None:
            import plot_utils
            p_utils = plot_utils.Figpref()
            plt.rcParams['font.size'] = '11'
            fig, ax = plt.subplots(figsize=(10, 5))

        # plot triangles
        xmin, xmax = 1e10, -1e10
        ymin, ymax = 1e10, -1e10
        if show_triangles:
            for cb in self.mesh.cells:
                if cb.type == "triangle":
                    tris = self.mesh.points[cb.data]
                    for tri in tris:
                        pts = tri[:, :2]
                        pts = np.vstack([pts, pts[0]])
                        ax.plot(pts[:, 0], pts[:, 1], color="lightgrey", linewidth=0.25, alpha=0.7)
                        xmin, xmax = self.update_min_max(xmin, xmax, pts[:, 0])
                        ymin, ymax = self.update_min_max(ymin, ymax, pts[:, 1])

        # cmap = plt.get_cmap("tab10")
        for i, (phys, segs) in enumerate(self.physical_curves.items()):
            # label = self.phys_id_to_name.get(phys, f"Physical {phys}")
            # color = cmap(i % 10)
            if phys == 1:
                color = p_utils.myblue
                label = 'Free surface'
            elif phys == 3:
                color = 'k'
                label = 'Fault (rate-and-state friction)'
            elif phys == 5:
                color = p_utils.mypink
                label = 'Dirichlet'
            for j, seg in enumerate(segs):
                pts = self.mesh.points[seg, :2]
                if j == 0:
                    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2, label=label)
                else:
                    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2)
                xmin, xmax = self.update_min_max(xmin, xmax, pts[:, 0])
                ymin, ymax = self.update_min_max(ymin, ymax, pts[:, 1])

        #ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("X [km]", fontsize=13)
        ax.set_ylabel("Y [km]", fontsize=13)
        xl = xmax - xmin
        yl = ymax - ymin
        ax.set_xlim(xmin-xl*0.01, xmax+xl*0.01)
        ax.set_ylim(ymin-yl*0.01, ymax+yl*0.01)
        ax.legend(fontsize=11, loc='lower right', bbox_to_anchor=(0.98, 0.02), bbox_transform=ax.transAxes)
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        #plt.title("Physical Curves with Mesh Triangles")
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig('%s/mesh_BC.png'%(save_dir), dpi=300)
        plt.show()
        
    def get_curve_points(self, phys_id, dim=None):
        """
        Get coordinates for points belonging to a physical curve.
        
        Parameters:
        -----------
        self : object
            Current object instance
        
        phys_id : int
            Physical curve identifier
        
        dim : int
            Mesh entity dimension
        
        Returns:
        --------
        pts_list : list
            List of point-coordinate arrays for the selected physical curve.
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
    
    def update_min_max(self, current_min, current_max, val):
        """
        Update minimum and maximum values with a new candidate value.
        
        Parameters:
        -----------
        self : object
            Current object instance
        
        current_min : float
            Current minimum value
        
        current_max : float
            Current maximum value
        
        val : float
            Candidate value used to update the limits
        
        Returns:
        --------
        new_min, new_max : tuple
            Updated minimum and maximum values.
        """
        new_min = current_min
        new_max = current_max
        if current_min > np.min(val):
            new_min = np.min(val)
        if current_max < np.max(val):
            new_max = np.max(val)
        return new_min, new_max