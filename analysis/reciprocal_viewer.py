from typing import Sequence, Tuple
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mplcursors
from ipywidgets import FloatSlider, IntSlider, Checkbox, HBox, VBox, interactive_output, Dropdown, ToggleButtons, Output
from ipywidgets import SelectMultiple
from IPython.display import display, clear_output, HTML

class ReciprocalViewer:
    def __init__(self, img_array, qxy, qz, calculator,
                 target_q=None, tol=0.1,
                 init_rot_x=0, init_rot_y=0, init_rot_z=0,
                 xlim: Tuple[float, float] = None,
                 ylim: Tuple[float, float] = None):
        self.img_array = img_array
        self.qxy       = qxy
        self.qz        = qz
        self.calc      = calculator
        self.target_q  = target_q
        self.tol       = tol
        self._a_rot = self._b_rot = self._c_rot = None

        self.xlim = xlim
        self.ylim = ylim

        # initialize sliders for lattice and orientation
        a0, b0, c0 = (calculator.a_len, calculator.b_len, calculator.c_len)
        α0, β0, γ0 = (calculator.alpha_deg, calculator.beta_deg, calculator.gamma_deg)
        self._sliders = {
            'a_len':     FloatSlider(min=0.1, max=2*a0,   value=a0,     description='a'),
            'b_len':     FloatSlider(min=0.1, max=2*b0,   value=b0,     description='b'),
            'c_len':     FloatSlider(min=0.1, max=2*c0,   value=c0,     description='c'),
            'alpha_deg': FloatSlider(min=1,   max=179,    value=α0,     description='α'),
            'beta_deg':  FloatSlider(min=1,   max=179,    value=β0,     description='β'),
            'gamma_deg': FloatSlider(min=1,   max=179,    value=γ0,     description='γ'),
            'rot_x':     IntSlider(min=-180, max=180,   value=init_rot_x, description='rot_x'),
            'rot_y':     IntSlider(min=-180, max=180,   value=init_rot_y, description='rot_y'),
            'rot_z':     IntSlider(min=-180, max=180,   value=init_rot_z, description='rot_z'),
            'draw_cell': Checkbox(value=True, description='Draw Cell')
        }

    @staticmethod
    def rot_matrix(u: np.ndarray, theta: float) -> np.ndarray:
        ux, uy, uz = u
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [c + ux*ux*(1-c),    ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
            [uy*ux*(1-c) + uz*s, c + uy*uy*(1-c),    uy*uz*(1-c) - ux*s],
            [uz*ux*(1-c) - uy*s, uz*uy*(1-c) + ux*s, c + uz*uz*(1-c)]
        ])

    @staticmethod
    def rotate_lattice(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                       rotation_axis: Sequence[float], rotation_degrees: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        axis = np.asarray(rotation_axis) / np.linalg.norm(rotation_axis)
        theta = np.deg2rad(rotation_degrees)
        R = ReciprocalViewer.rot_matrix(axis, theta)
        return R @ a, R @ b, R @ c

    def _update(self, **kwargs):
        clear_output(wait=True)

        # Create figure for GIWAXS image and unit cell
        fig = plt.figure(figsize=(15, 6))

        # Update lattice, compute peaks
        self.calc.set_lattice(kwargs['a_len'], kwargs['b_len'], kwargs['c_len'],
                              kwargs['alpha_deg'], kwargs['beta_deg'], kwargs['gamma_deg'])
        a, b, c = self.calc.a_vec, self.calc.b_vec, self.calc.c_vec
        a_rot, b_rot, c_rot = self.rotate_lattice(a, b, c, [1,0,0], kwargs['rot_x'])
        a_rot, b_rot, c_rot = self.rotate_lattice(a_rot, b_rot, c_rot, [0,1,0], kwargs['rot_y'])
        a_rot, b_rot, c_rot = self.rotate_lattice(a_rot, b_rot, c_rot, [0,0,1], kwargs['rot_z'])
        self._a_rot, self._b_rot, self._c_rot = a_rot, b_rot, c_rot
        self.calc.update_reciprocal(a_rot, b_rot, c_rot)
        peaks = self.calc.find_peaks(target_q=self.target_q, tol=self.tol)

        # Plot GIWAXS image
        ax = fig.add_subplot(121)
        vmin, vmax = np.percentile(self.img_array, 1), np.percentile(self.img_array, 99.3)
        ax.imshow(self.img_array,
                  norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax),
                  cmap='turbo', extent=(self.qxy.min(), self.qxy.max(), self.qz.min(), self.qz.max()),
                  origin='lower', aspect='auto')

        # Apply any preset limits
        if self.xlim is not None:
            ax.set_xlim(*self.xlim)
        if self.ylim is not None:
            ax.set_ylim(*self.ylim)

        ax.set_xlabel('$q_{xy}$ (Å⁻¹)', size=12)
        ax.set_ylabel('$q_{z}$ (Å⁻¹)', size=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(prune='both'))

        # Overlay peaks
        pts = []
        for hkl, q_mag, chi in peaks:
            cr = np.radians(chi)
            x = q_mag * np.sin(cr)
            y = q_mag * np.cos(cr)
            p, = ax.plot(x, y, 'o', color='white', markersize=5, markeredgecolor='black')
            p.set_gid(hkl)
            pts.append(p)
        
        
        mplc = mplcursors.cursor(pts, hover=True)
        mplc.connect('add', lambda sel: sel.annotation.set_text(sel.artist.get_gid()))
        mplc.connect('add', lambda sel: sel.annotation.set_position((0.1,0.1)))

        # Optional 3D cell
        if kwargs['draw_cell'] and self._a_rot is not None:
            ax3d = fig.add_subplot(1, 2, 2, projection='3d')
            ax3d.view_init(elev=15, azim=45)
            ax3d.set_proj_type('ortho')
            verts = np.array([[0,0,0], self._a_rot, self._b_rot, self._c_rot,
                              self._a_rot+self._b_rot, self._a_rot+self._c_rot,
                              self._b_rot+self._c_rot, self._a_rot+self._b_rot+self._c_rot])
            faces = [[verts[i] for i in face] for face in [(0,1,4,2),(0,1,5,3),(0,2,6,3),
                                                          (7,4,1,5),(7,6,2,4),(7,5,3,6)]]
            pc = Poly3DCollection(faces, facecolors='cyan', edgecolors='black', alpha=0.8)
            ax3d.add_collection3d(pc)
            cen = verts.mean(axis=0)
            lim = max(np.linalg.norm(v) for v in verts[1:4]) * 1.5
            ax3d.set_xlim(cen[0]-lim, cen[0]+lim)
            ax3d.set_ylim(cen[1]-lim, cen[1]+lim)
            ax3d.set_zlim(cen[2]-lim, cen[2]+lim)
            ax3d.set_box_aspect((1,1,1))

        plt.tight_layout()
        plt.show()

        # Get actual plot limits for filtering
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        # Build table rows only for displayed peaks
        rows = []
        for hkl, q_mag, chi in peaks:
            cr = np.radians(chi)
            qxy_val = q_mag * np.sin(cr)
            qz_val  = q_mag * np.cos(cr)
            if not (x_min <= qxy_val <= x_max and y_min <= qz_val <= y_max):
                continue
            h, k, l = hkl
            rows.append({
                'q_xy': round(qxy_val, 2),
                'q_z':  round(qz_val, 2),
                'h':     int(h),
                'k':     int(k),
                'l':     int(l),
                'hkl':   f"{hkl}"
            })
        df = pd.DataFrame(rows, columns=['q_xy', 'q_z', 'h', 'k', 'l', 'hkl'])

        # Interactive table controls
        sort_dd = Dropdown(options=df.columns.tolist(), description='Sort by:')
        order_dd = ToggleButtons(options=[('Asc', True), ('Desc', False)], description='Order:')
        out = Output()

        def refresh(change=None):
            out.clear_output(wait=True)
            sorted_df = df.sort_values(by=sort_dd.value, ascending=order_dd.value)
            html = sorted_df.to_html(index=False)
            with out:
                display(HTML(f'<div style="max-height:200px; overflow:auto;">{html}</div>'))

        sort_dd.observe(refresh, names='value')
        order_dd.observe(refresh, names='value')
        refresh()
        display(VBox([HBox([sort_dd, order_dd]), out]))

    def show(self):
        ui = VBox([
            HBox([self._sliders[k] for k in ('a_len','b_len','c_len')]),
            HBox([self._sliders[k] for k in ('alpha_deg','beta_deg','gamma_deg')]),
            HBox([self._sliders[k] for k in ('rot_x','rot_y','rot_z','draw_cell')])
        ])
        out = interactive_output(self._update, self._sliders)
        display(ui, out)
        init = {k: v.value for k, v in self._sliders.items()}
        with out:
            self._update(**init)
