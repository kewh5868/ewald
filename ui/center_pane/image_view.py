"""
ImageCanvas: composite widget displaying a main 2D image frame with four subplots below.
The main frame has default q_xy vs q_z axes; subplots:
 - q_xy vs Intensity
 - q_z vs Intensity
 - q_r vs Intensity
 - q_xy vs q_z (small)
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar, FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import xarray as xr

from ...dataclass.single_image import SingleImage

class ImageCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # --- Figure, Canvas, and Toolbar ---
        self.fig = Figure(facecolor='darkgray')
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # --- Axis selector ---
        self.ax_selector = QComboBox()
        self.ax_selector.addItems(['q_xy vs q_z', 'pixel coords'])
        self.ax_selector.currentTextChanged.connect(self.on_axis_change)

        # --- Layout toolbar + selector ---
        topbar = QHBoxLayout()
        topbar.addWidget(self.toolbar)
        topbar.addWidget(self.ax_selector)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(topbar)
        main_layout.addWidget(self.canvas)

        # --- Create axes grid ---
        gs = GridSpec(2, 4, figure=self.fig, height_ratios=[4, 1], hspace=0.4, wspace=0.5)
        self.ax_main    = self.fig.add_subplot(gs[0, :])
        self.ax_qxy     = self.fig.add_subplot(gs[1, 0])
        self.ax_qz      = self.fig.add_subplot(gs[1, 1])
        self.ax_qr      = self.fig.add_subplot(gs[1, 2])
        self.ax_small2d = self.fig.add_subplot(gs[1, 3])

        # --- Initial setup ---
        self._setup_main()
        self._setup_subplots()

    def _setup_main(self):
        ax = self.ax_main
        ax.set_facecolor('darkgray')
        ax.set_xlabel(r'$q_{xy}\,\mathrm{(\AA^{-1})}$')
        ax.set_ylabel(r'$q_{z}\,\mathrm{(\AA^{-1})}$')
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)

    def _setup_subplots(self):
        # q_xy vs Intensity
        self.ax_qxy.set_facecolor('white')
        self.ax_qxy.set_xlabel(r'$q_{xy}\,\mathrm{(\AA^{-1})}$')
        self.ax_qxy.set_ylabel('Intensity')
        # q_z vs Intensity
        self.ax_qz.set_facecolor('white')
        self.ax_qz.set_xlabel(r'$q_{z}\,\mathrm{(\AA^{-1})}$')
        self.ax_qz.set_ylabel('Intensity')
        # q_r vs Intensity
        self.ax_qr.set_facecolor('white')
        self.ax_qr.set_xlabel(r'$q_{r}\,\mathrm{(\AA^{-1})}$')
        self.ax_qr.set_ylabel('Intensity')
        # small 2D
        self.ax_small2d.set_facecolor('darkgray')
        self.ax_small2d.set_xlabel(r'$q_{xy}\,\mathrm{(\AA^{-1})}$')
        self.ax_small2d.set_ylabel(r'$q_{z}\,\mathrm{(\AA^{-1})}$')
        self.ax_small2d.set_xlim(0, 3)
        self.ax_small2d.set_ylim(0, 3)

    def clear(self):
        """Clear all axes to initial state."""
        self.ax_main.cla()
        self._setup_main()
        for ax in (self.ax_qxy, self.ax_qz, self.ax_qr, self.ax_small2d):
            ax.cla()
        self._setup_subplots()
        self.canvas.draw()

    def displayImage(self, data, extent=None, cmap='viridis'):
        """Display data on the main axis and reset limits."""
        self.ax_main.cla()
        self._setup_main()
        if extent is None:
            extent = [0, data.shape[1] * (3.0/data.shape[1]), 0, data.shape[0] * (3.0/data.shape[0])]
        self.ax_main.imshow(data, origin='lower', extent=extent, aspect='auto', cmap=cmap)
        self.ax_main.set_xlim(extent[0], extent[1])
        self.ax_main.set_ylim(extent[2], extent[3])
        self.canvas.draw()

    def overlayPeaks(self, qxy, qz, **kwargs):
        """Overlay Bragg peaks on the main axis."""
        self.ax_main.scatter(qxy, qz, **kwargs)
        self.canvas.draw()

    def update1D(self, axis, x, y, **kwargs):
        """Update one of the 1D subplots: 'qxy','qz','qr'."""
        mapping = {'qxy': self.ax_qxy, 'qz': self.ax_qz, 'qr': self.ax_qr}
        ax = mapping.get(axis)
        if ax is None:
            return
        ax.cla()
        ax.set_ylabel('Intensity')
        ax.plot(x, y, **kwargs)
        self.canvas.draw()

    def update2Dsmall(self, data, extent=None, cmap='viridis'):
        """Display data on the small 2D subplot."""
        self.ax_small2d.cla()
        self.ax_small2d.set_facecolor('darkgray')
        if extent is None:
            extent = [0, data.shape[1] * (3.0/data.shape[1]), 0, data.shape[0] * (3.0/data.shape[0])]
        self.ax_small2d.imshow(data, origin='lower', extent=extent, aspect='auto', cmap=cmap)
        self.ax_small2d.set_xlim(extent[0], extent[1])
        self.ax_small2d.set_ylim(extent[2], extent[3])
        self.canvas.draw()

    def on_axis_change(self, text: str):
        """Switch main‐plot labels (and re‐draw if needed)."""
        if text == 'q_xy vs q_z':
            self.ax_main.set_xlabel('q_xy')
            self.ax_main.set_ylabel('q_z')
        else:
            self.ax_main.set_xlabel('pixel x')
            self.ax_main.set_ylabel('pixel y')
        self.canvas.draw()

    def displayReciprocal(self, recip_ds: xr.Dataset, cmap='viridis'):
        """
        Display a reciprocal‐space xarray Dataset on the main 2D axes,
        plus its q_xy and q_z 1D projections on the subplots.
        """
        # choose DataArray
        da = recip_ds if not isinstance(recip_ds, xr.Dataset) else recip_ds[list(recip_ds.data_vars)[0]]
        # alias lookup
        aliases = {
            'qxy': {'qxy','q_xy','qip','QXY','Qip'},
            'qz':  {'qz','q_z','qoop','QZ','Qoop'}
        }
        coords_map = {name.lower(): name for name in da.coords}
        # find qxy and qz keys robustly
        for key, alias_set in aliases.items():
            match = next((a for a in alias_set if a.lower() in coords_map), None)
            if not match:
                raise KeyError(f"No {key} axis found among coords {list(coords_map)}")
            # resolve to actual coord name
            aliases[key] = coords_map[match.lower()]
        qxy_key = aliases['qxy']
        qz_key  = aliases['qz']
        data = da.values
        q_xy = da.coords[qxy_key].values
        q_z  = da.coords[qz_key].values
        extent = [float(q_xy.min()), float(q_xy.max()), float(q_z.min()), float(q_z.max())]
        # display
        self.displayImage(data, extent=extent, cmap=cmap)
        I_qxy = da.sum(dim=qz_key).values
        I_qz  = da.sum(dim=qxy_key).values
        self.update1D('qxy', q_xy, I_qxy)
        self.update1D('qz',  q_z,  I_qz)

    def displaySingleImage(self, img: SingleImage, **kwargs):
        self.displayReciprocal(img.recip_DS, **kwargs)
