from PyQt5.QtWidgets import QTextEdit, QMainWindow, QApplication, QAction, QWidget, QFrame, QVBoxLayout, QHBoxLayout, QAction, QFileDialog, QToolBar, QPushButton, QSizePolicy, QSlider, QMessageBox, QTableWidget, QTableWidgetItem, QLabel, QLineEdit, QGridLayout, QAbstractItemView
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QWindow

import sys, os
from io import StringIO
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.text import Text
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.interpolate import griddata
# from lmfit.models import Model
import lmfit
from lmfit.lineshapes import gaussian2d, lorentzian
from scipy.interpolate import griddata
from lmfit import Model, Parameters
from scipy.stats import multivariate_normal
from scipy.ndimage import rotate
from scipy.ndimage import affine_transform
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import gmean
import hdbscan
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN

script_dir = os.path.dirname(os.path.abspath(__file__)) # Get the directory of the current script
parent_dir = os.path.dirname(script_dir) # Get the parent directory
# main_dir = os.path.join(parent_dir, 'main') # Construct the path to the /main/ directory
main_dir = os.path.join(parent_dir, 'pywaxs') # Construct the path to the /main/ directory
sys.path.append(main_dir) # Add the /main/ directory to sys.path

class XrayApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ds = None  # Initialize dataset to None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('X-ray Data Analysis')
        self.setGeometry(100, 100, 1600, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Left frame and layout (50%)
        self.left_frame = QFrame()
        self.left_layout = QVBoxLayout()
        self.left_frame.setLayout(self.left_layout)
        self.layout.addWidget(self.left_frame, stretch=1)

        # Main frame for X-ray plot (canvas1)
        self.main_frame = QFrame()
        self.main_layout = QVBoxLayout()
        self.main_frame.setLayout(self.main_layout)
        self.left_layout.addWidget(self.main_frame)

        self.fig1, self.ax1 = plt.subplots()
        self.ax1.axis('off')
        self.canvas1 = MyCanvas(self.fig1, main_app=self, add_subplot=True)
        self.main_layout.addWidget(self.canvas1)

        self.nav1 = MyNavigationToolbar(self.canvas1, self)
        self.main_layout.insertWidget(0, self.nav1)

        # Right frame and layout (50%)
        self.output_frame = QFrame()
        self.output_layout = QVBoxLayout()
        self.layout.addWidget(self.output_frame, stretch=1)
        self.output_frame.setLayout(self.output_layout)

        self.fig2 = plt.figure(figsize=(8, 4))
        # Modify the gridspec to adjust spacing
        gs = GridSpec(2, 2, width_ratios=[3, 1], hspace=0.5, wspace=0.3)
        self.ax2 = self.fig2.add_subplot(gs[0, 0])
        self.ax3 = self.fig2.add_subplot(gs[1, 0])
        self.ax4 = self.fig2.add_subplot(gs[0, 1])
        self.ax5 = self.fig2.add_subplot(gs[1, 1])

        axes_dict = {'ax2': self.ax2, 'ax3': self.ax3, 'ax4': self.ax4, 'ax5': self.ax5}
        self.canvas2 = MyCanvas(self.fig2, main_app=self, add_subplot=False, axes=axes_dict)
        self.nav2 = GaussianFitToolbar(self.canvas2, self)
        self.nav2.setFixedWidth(600)
        self.canvas2.ax2.set_title('Active ROI')
        self.canvas2.ax3.set_title('1D Integration')  # Set title for ax3
        self.canvas2.ax4.set_title('Fit')
        self.canvas2.ax5.set_title('Residual')

        fontsize = 12
        self.canvas2.ax2.set_xlabel('$q_{xy}$ (Å$^{-1}$)', fontsize=fontsize)
        self.canvas2.ax2.set_ylabel('$q_{z}$ (Å$^{-1}$)', fontsize=fontsize)
        self.canvas2.ax3.set_xlabel('$q_{z}$ (Å$^{-1}$)', fontsize=fontsize)  # Set xlabel for ax3
        self.canvas2.ax3.set_ylabel('Intensity', fontsize=fontsize)  # Set ylabel for ax3
        self.canvas2.ax4.set_xlabel('$q_{xy}$ (Å$^{-1}$)', fontsize=fontsize)
        self.canvas2.ax4.set_ylabel('$q_{z}$ (Å$^{-1}$)', fontsize=fontsize)
        self.canvas2.ax5.set_xlabel('$q_{xy}$ (Å$^{-1}$)', fontsize=fontsize)
        self.canvas2.ax5.set_ylabel('$q_{z}$ (Å$^{-1}$)', fontsize=fontsize)
        
        self.output_layout.addWidget(self.nav2)
        self.output_layout.addWidget(self.canvas2, stretch=2)  # 2/3 of the vertical space

        self.fig2.tight_layout(pad=3.0)
        self.canvas2.draw()  # To refresh the canvas and apply the layout update

        # Stat boxes (1/3 of the vertical space)
        # self.widget3 = QWidget()
        # self.grid3 = QGridLayout()
        # self.widget3.setLayout(self.grid3)
        # self.add_stat_box('Amplitude', 1, 0)
        # self.add_stat_box('Center X', 2, 0)
        # self.add_stat_box('Center Y', 3, 0)
        # self.add_stat_box('Chi-square', 1, 1)
        # self.add_stat_box('Reduced Chi-square', 2, 1)
        # self.add_stat_box('R-squared', 3, 1)
        # self.add_stat_box('Sigma X', 1, 2)
        # self.add_stat_box('Sigma Y', 2, 2)
        # self.add_stat_box('FWHM X', 3, 2)
        # self.add_stat_box('FWHM Y', 4, 2)

        # Replace the stat boxes with QTextEdit for scrollable output
        self.output_text_edit = QTextEdit()
        self.output_text_edit.setReadOnly(True)  # Make it read-only if you don't want the user to edit it
        self.output_text_edit.ensureCursorVisible()  # Ensure the cursor is visible so it scrolls automatically

        # For the QTextEdit sizing, make sure it does not interfere with the matplotlib canvas
        # You might need to adjust the stretch factors or widget sizes
        self.output_layout.addWidget(self.output_text_edit, stretch=0)  # Adjust the stretch factor if needed

        # Redraw the canvas after layout adjustments
        self.canvas2.draw_idle()

        # self.output_layout.addWidget(self.widget3, stretch=1)

        # Initialize the table with additional columns for fit statistics
        self.roi_table = QTableWidget()
        self.header_labels = ['ROI #', 'q_{xy} (min)', 'q_{xy} (max)', 'q_{z} (min)', 'q_{z} (max)', 
                            'Amplitude', 'Center X', 'Center Y', 'Sigma X', 'Sigma Y',
                            'Chi-square', 'Reduced Chi-square', 'R-squared', 'FWHM X', 'FWHM Y']
        self.roi_table.setColumnCount(len(self.header_labels))
        self.roi_table.setHorizontalHeaderLabels(self.header_labels)
        self.roi_table.setWordWrap(True)
        self.roi_table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.main_layout.addWidget(self.roi_table)

        self.roi_selector = None
        self.coords = None

        # Add table to the layout
        self.roi_table.cellClicked.connect(self.on_table_click)
   
    def load_project(self):
        # Check if a project is already loaded
        if hasattr(self, 'ds') and self.ds is not None:
            reply = QMessageBox.question(self, 'Save Current Project', 
                                        'Would you like to save the current project before loading a new one?', 
                                        QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, 
                                        QMessageBox.Cancel)

            if reply == QMessageBox.Yes:
                options = QFileDialog.Options()
                filepath, _ = QFileDialog.getSaveFileName(self, "Save Current Project Data", "", "NetCDF Files (*.nc);;All Files (*)", options=options)
                if filepath:
                    if not filepath.endswith('.nc'):
                        filepath += '.nc'
                    try:
                        # Assume you have a method called export_data that takes care of saving
                        self.canvas1.export_data(filepath)
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"An error occurred while saving: {e}", QMessageBox.Ok)
                        return  # If save fails, don't proceed to load a new project

            elif reply == QMessageBox.Cancel:
                return  # Don't proceed to load a new project

            # Clear the current dataset only if we are sure that a new project will be loaded
            # self.ds = None

        # Now you can call your loadData function here
        self.loadData()

    def loadData(self):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self, "Load Data", "", "NetCDF Files (*.nc);;All Files (*)", options=options)
        if file:
            try:

                self.canvas1.init_plot() # initialize canvas1
                # self.canvas2.init_plot()
                self.ds = None
                self.ds = xr.open_dataset(file, engine='h5netcdf')

                try:
                    if self.ds['intensity'] is None:  #or self.ds['peak_positions'] is None:
                            raise ValueError("Intensity or peak positions are None.")
                    
                    # Extract coordinate names
                    coord_names_intensity = list(self.ds['intensity'].dims)
                    coord_names_peak = list(self.ds['peak_positions'].dims)
                    
                    self.coords = {
                        'xlabel': coord_names_intensity[1] if len(coord_names_intensity) > 1 else None,  
                        'ylabel': coord_names_intensity[0] if len(coord_names_intensity) > 0 else None,
                    }

                    # Assuming `plot_data` is a method in your canvas object
                    self.canvas1.plot_data(self.ds['intensity'], self.ds['peak_positions'], self.coords)
                    self.roi_selector = self.canvas1.roi_selector
                except Exception as e:
                    QMessageBox.critical(self.window, "Error", f"An error occurred while plotting: {e}", QMessageBox.Ok)
        
                if self.ds:
                    max_intensity = self.ds['intensity'].max()
                    max_intensity = int(max_intensity.values)
                    # self.slider_vmin.setMaximum(max_intensity)
                    # self.slider_vmax.setMaximum(max_intensity)
            
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while loading the project: {e}", QMessageBox.Ok)

        print("Loaded dataset:", self.ds)
        print("Intensity dimensions:", self.ds['intensity'].dims)
        print("Peak_positions dimensions:", self.ds['peak_positions'].dims)

    def update_roi_table(self):
        print("Updating ROI table")  # Debugging print

        # print(f"ID of ROISelector in XrayApp: {id(self.roi_selector)}")
        # print(f"ID of rectangles in XrayApp: {id(self.roi_selector.rectangles)}")
        # print(f"ID of ROISelector in canvas1: {id(self.canvas1.roi_selector)}")
        # print(f"ID of rectangles in canvas1: {id(self.canvas1.roi_selector.rectangles)}")

        self.roi_table.setRowCount(0)  # Clear the table
        self.roi_selector = self.canvas1.roi_selector

        for i, rect in enumerate(self.roi_selector.rectangles):
            self.roi_table.insertRow(i)
            self.roi_table.setItem(i, 0, QTableWidgetItem(str(i)))
            
            # Retrieve the stored coordinates of the rectangle
            coords = rect.get('coords', {})
            x1, y1, x2, y2 = coords.get('x1', 0), coords.get('y1', 0), coords.get('x2', 0), coords.get('y2', 0)
            print(f"update_roi_table: (x1, y1):{x1:3.2f}, {y1:3.2f}) --> (*x2, y2):{x2:3.2f}, {y2:3.2f})")

            self.roi_table.setItem(i, 1, QTableWidgetItem(f"{x1:.2f}"))
            self.roi_table.setItem(i, 2, QTableWidgetItem(f"{x2:.2f}"))
            self.roi_table.setItem(i, 3, QTableWidgetItem(f"{y1:.2f}"))
            self.roi_table.setItem(i, 4, QTableWidgetItem(f"{y2:.2f}"))

    def on_table_click(self, row, column):
        # Reset the edge color of previously active rectangle, if any
        if self.canvas1.roi_selector.active_rect:
            self.canvas1.roi_selector.active_rect['main'].set_edgecolor('black')

        # Get the new active rectangle based on the clicked row in the table
        new_active_rect = self.canvas1.roi_selector.rectangles[row]

        # Set its edge color to blue to highlight it
        new_active_rect['main'].set_edgecolor('blue')

        # Update the active rectangle in the roi_selector
        self.canvas1.roi_selector.active_rect = new_active_rect

        # Redraw the canvas to reflect the changes
        self.canvas1.draw()

        # *** Modified part starts here ***
        # Check if the active rectangle exists before updating the sliced data array
        if self.canvas1.roi_selector.active_rect is not None:
            if hasattr(self.canvas2, 'colorbar') and self.canvas2.colorbar:
                ax = self.canvas2.colorbar.ax
                if ax:
                    self.canvas2.colorbar.remove()
                    self.canvas2.colorbar = None  # Reset to None either way

            self.update_sliced_dataarray()
        else:
            self.canvas2.ax2.clear()
            if self.canvas2.colorbar is not None:
                ax = self.canvas2.colorbar.ax  # Get the axis associated with the colorbar
                if ax is not None:
                    self.canvas2.colorbar.remove()
                self.canvas2.colorbar = None  # Reset to None either way
            self.canvas2.draw()
        # *** Modified part ends here ***

    def update_sliced_dataarray(self):
        """Updates the sliced dataarray based on the coordinates of the selected ROI."""
        if self.canvas1.roi_selector.active_rect is None:
            self.canvas2.ax2.clear()
            if self.canvas2.colorbar is not None:
                ax = self.canvas2.colorbar.ax  # Get the axis associated with the colorbar
                if ax is not None:
                    self.canvas2.colorbar.remove()
                self.canvas2.colorbar = None  # Reset to None either way
            self.canvas2.draw()
            return
        # Retrieve the stored coordinates of the rectangle
        coords = self.canvas1.roi_selector.active_rect.get('coords', {})
        x1, y1, x2, y2 = coords.get('x1', 0), coords.get('y1', 0), coords.get('x2', 0), coords.get('y2', 0)
        
        # Map pixel positions to actual dataset coordinates
        dim1 = 'q_xy'
        dim2 = 'q_z'
        
        # - Uncomment if using pixel dimensions
        # q_xy_min = self.ds[dim1][int(np.floor(x1))].values
        # q_xy_max = self.ds[dim1][int(np.ceil(x2))].values
        # q_z_min = self.ds[dim2][int(np.floor(y1))].values
        # q_z_max = self.ds[dim2][int(np.ceil(y2))].values

        # - Use for q-space coordinates, not pixel dimensions.
        q_xy_min = min(x1, x2)
        q_xy_max = max(x1, x2)
        q_z_min = min(y1, y2)
        q_z_max = max(y1, y2)

        # Find the nearest actual dataset coordinates for each boundary
        q_xy_min_nearest = self.ds[dim1].sel({dim1: q_xy_min}, method='nearest').values
        q_xy_max_nearest = self.ds[dim1].sel({dim1: q_xy_max}, method='nearest').values
        q_z_min_nearest = self.ds[dim2].sel({dim2: q_z_min}, method='nearest').values
        q_z_max_nearest = self.ds[dim2].sel({dim2: q_z_max}, method='nearest').values

        # Use these nearest values to slice the data array
        self.sliced_ds = self.ds.sel(
            q_z=slice(min(q_z_min_nearest, q_z_max_nearest), max(q_z_min_nearest, q_z_max_nearest)), 
            q_xy=slice(min(q_xy_min_nearest, q_xy_max_nearest), max(q_xy_min_nearest, q_xy_max_nearest))
        )

        if hasattr(self.canvas2, 'colorbar') and self.canvas2.colorbar:
            ax = self.canvas2.colorbar.ax
            if ax:
                self.canvas2.colorbar.remove()
                self.canvas2.colorbar = None  # Reset to None either way

        # Update ax2
        self.canvas2.ax2.clear()

        # Get extents based on actual dataset coordinates
        extent = [q_xy_min, q_xy_max, q_z_min, q_z_max]

        # Calculate vmin and vmax
        img_values = self.ds['intensity'].values
        vmin = np.nanpercentile(img_values, 10)
        vmax = np.nanpercentile(img_values, 99)

        im = self.canvas2.ax2.imshow(
            self.sliced_ds['intensity'].values, 
            cmap='turbo',
            origin='lower',
            extent=extent,
            aspect='auto',
            vmin=vmin, vmax=vmax
        )

        if self.canvas2.colorbar is not None:
            ax = self.canvas2.colorbar.ax  # Get the axis associated with the colorbar
            if ax is not None:
                self.canvas2.colorbar.remove()
            else:
                self.canvas2.colorbar = None  # Reset to None if ax is None

        # if self.coords:
        #     xlabel = self.coords.get('xlabel', None)
        #     ylabel = self.coords.get('ylabel', None)
        #     if xlabel:
        #         self.canvas2.ax2.set_xlabel(xlabel)
        #     if ylabel:
        #         self.canvas2.ax2.set_ylabel(ylabel)

        self.canvas2.colorbar = self.canvas2.fig.colorbar(im, ax=self.canvas2.ax2)
        self.canvas2.draw()

        # *** Modified part starts here ***
        if self.canvas2.colorbar is not None:
            ax = self.canvas2.colorbar.ax  # Get the axis associated with the colorbar
            if ax is not None:
                self.canvas2.colorbar.remove()
            self.canvas2.colorbar = None  # Reset to None either way
        
        self.canvas2.ax2.set_title('Active ROI')
    
        # Update the colorbar
        self.canvas2.colorbar = self.canvas2.fig.colorbar(im, ax=self.canvas2.ax2)
        
        # Update the 1D integration plot on ax3
        self.update_1d_integration_plot()

        # Redraw the canvas to reflect updates
        self.canvas2.draw()

    def delete_active_roi(self):
        if self.canvas1.roi_selector.active_rect:
            self.canvas1.roi_selector.active_rect['main'].remove()
            self.canvas1.roi_selector.active_rect['close_box'].remove()
            self.canvas1.roi_selector.active_rect = None
            self.canvas2.ax2.clear()
            self.canvas2.draw()

    def add_stat_box(self, label, row, col):
        lbl = QLabel(label)
        le = QLineEdit(self)
        le.setReadOnly(True)
        setattr(self, f"stat_{label.replace(' ', '_').lower()}", le)  # Save as attribute
        self.grid3.addWidget(lbl, 2*row-1, col)
        self.grid3.addWidget(le, 2*row, col)

    def update_stat_boxes(self, fit_stats_dict):
        for key, value in fit_stats_dict.items():
            stat_box_name = f"stat_{key.lower()}"
            stat_box = getattr(self, stat_box_name, None)
            if stat_box is not None:
                stat_box.setText(str(value))

    def perform_gaussian_fit(self):
        fit_instance = Gaussian2DFit(self.sliced_ds)
        fit_instance.perform_combined_fit(fit_method='gaussianpolarcartesian')
        
        # Create a StringIO object to capture the fit report
        fit_report_io = StringIO()
        fit_report_str = lmfit.fit_report(fit_instance.result)
        fit_report_io.write(fit_report_str)
        
        # Retrieve the string from StringIO object
        fit_report_str = fit_report_io.getvalue()
        
        # Print detailed fit report to the QTextEdit
        self.output_text_edit.setText(fit_report_str)
    
        # Print detailed fit report
        lmfit.report_fit(fit_instance.result)
        
        fit_instance.plot_fit(self.canvas2.axes['ax4'], self.canvas2.axes['ax5'])
        
        # Get the fit statistics
        fit_stats_dict = fit_instance.fit_statistics()
        
        # Update stat boxes
        self.update_stat_boxes(fit_stats_dict)

        # Identify the active ROI row
        # active_row = self.roi_selector  # Assuming roi_selector contains the index of the active ROI
        active_row = self.roi_table.currentRow()
        # Check if there are existing values and prompt user for overwrite
        if self.roi_table.item(active_row, 5) is not None:  # Assuming the first fit parameter is in column 5
            reply = QMessageBox.question(self, 'Overwrite Existing Values', 
                                        "Do you want to overwrite existing fit values for this ROI?",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return

        # Update the table with new fit parameters
        for col, header in enumerate(self.header_labels[5:]):  # Starting from 'Amplitude'
            if header.replace(' ', '_').replace('-', '_').lower() in fit_stats_dict:
                new_item = QTableWidgetItem(str(fit_stats_dict[header.replace(' ', '_').replace('-', '_').lower()]))
                self.roi_table.setItem(active_row, col+5, new_item)  # col+5 because we start from 'Amplitude' which is at index 5
                        
    def display_fit_statistics(self, stats):
        # Assuming you have QLineEdit widgets stored in a dictionary self.stat_boxes
        if stats is not None:
            self.stat_boxes['Amplitude'].setText(str(stats['amplitude']))
            self.stat_boxes['Center X'].setText(str(stats['centerx']))
            self.stat_boxes['Center Y'].setText(str(stats['centery']))
            self.stat_boxes['Chi-square'].setText(str(stats['chisqr']))
            self.stat_boxes['Reduced Chi-square'].setText(str(stats['redchi']))
            self.stat_boxes['R-squared'].setText(str(stats['rsquared']))
            self.stat_boxes['Sigma X'].setText(str(stats['sigmax']))
            self.stat_boxes['Sigma Y'].setText(str(stats['sigmay']))
            self.stat_boxes['FWHM X'].setText(str(stats['fwhmx']))
            self.stat_boxes['FWHM Y'].setText(str(stats['fwhmy']))

    def update_1d_integration_plot(self):
        # Ensure the dataset is not None
        if self.ds is None:
            return
        
        # Sum along 'q_xy' axis for each 'q_z' value
        integrated_intensity = self.sliced_ds.sum(dim='q_xy')

        # Clear the previous plot
        self.canvas2.ax3.clear()
    
        # Convert the Dataset to a DataArray
        integrated_dataarray = integrated_intensity.to_array()

        # Now you can plot using the DataArray
        # You need to ensure that 'integrated_dataarray' has two dimensions: one for 'q_z' and one for the values.
        self.canvas2.ax3.plot(integrated_dataarray['q_z'].values, integrated_dataarray[0].values, label='Integrated Intensity')  # Index 0 assumes that you want the first variable in the dataset.
    
        # Plot the integrated intensity
        # self.canvas2.ax3.plot(integrated_intensity['q_z'], integrated_intensity, label='Integrated Intensity')
        # self.canvas2.ax3.plot(integrated_intensity['q_z'].values, integrated_intensity['intensity'].values, label='Integrated Intensity')
        
        # Set labels and title
        fontsize = 12
        self.canvas2.ax3.set_xlabel('$q_{z}$ (Å$^{-1}$)', fontsize=fontsize)
        self.canvas2.ax3.set_ylabel('Integrated Intensity', fontsize=fontsize)
        self.canvas2.ax3.set_title('1D Integration')

class MyCanvas(FigureCanvas):
    def __init__(self, fig, main_app=None, add_subplot=True, axes=None, subplot_dims=None):
        super(MyCanvas, self).__init__(fig)
        self.main_app = main_app
        self.fig = fig

        if add_subplot:
            self.ax = self.fig.add_subplot(111)
        elif axes is not None:
            if isinstance(axes, list):
                self.axes = axes
                # Dynamically set named attributes for axes
                for i, ax in enumerate(axes, start=2):
                    setattr(self, f'ax{i}', ax)
            elif isinstance(axes, dict):
                self.axes = axes
                for key, ax in axes.items():
                    setattr(self, key, ax)
            else:
                raise ValueError("Invalid type for axes. Should be either list or dictionary.")
        else:
            self.ax = None  # No axes provided

        if subplot_dims:
            self.subplot_dims = subplot_dims

        self.intensity = None
        self.colorbar = None

        self.roi_selector = None  # Initialize roi_selector to None, will be set later
        self.rectangles = []  # Initialize rectangles

        self.mpl_connect('key_press_event', self.on_key)
        
    def init_plot(self):
        '''Add an init_plot method to initialize a blank plot with no data. This can be called to reset the plot before adding new data.'''
        self.ax.clear()
        
        # Clear or reset other attributes
        self.peak_positions = None
        self.intensity = None
        self.scatter = None

        # self.roi_selector.rectangles = []  # Clear the existing rectangles in ROISelector
        
        # Remove the colorbar if it exists
        if self.colorbar:
            self.colorbar.remove()
            self.colorbar = None
        
        self.draw()
        # self.roi_selector = ROISelector(self.ax, self.main_app)
        # self.roi_selector.activate()
        
        # Initialize ROISelector only if not already initialized
        # if self.roi_selector is None:
        #     self.roi_selector = ROISelector(self.ax, self.main_app)

        # self.roi_selector.activate()

    def plot_data(self, intensity, peak_positions, coords=None):
        ''' 
        plot_data:
            Purpose:
            Plot the 2D intensity data.

            Implementation:
            Clears the existing axis.
            Plots the 2D heatmap using imshow.
            Adds or updates the colorbar.
        '''
        # self.init_plot()  # Reset the plot
            # Validation
        if intensity is None:
            print("Error: Intensity is None.")
            return
        if peak_positions is None:
            print("Error: Peak Positions are None.")
            return
        if not isinstance(intensity, xr.DataArray):
            print("Error: Intensity is not an xarray DataArray.")
            return
        if not isinstance(peak_positions, xr.DataArray):
            print("Error: Peak Positions is not an xarray DataArray.")
            return

        # Get extents based on xarray coordinates
        extent = [
            intensity.coords[intensity.dims[1]].min(),
            intensity.coords[intensity.dims[1]].max(),
            intensity.coords[intensity.dims[0]].min(),
            intensity.coords[intensity.dims[0]].max(),
        ]
        
        # Calculate vmin and vmax
        img_values = intensity.values
        vmin = np.nanpercentile(img_values, 10)
        vmax = np.nanpercentile(img_values, 99)
        
        # Change color map and set contrast
        im = self.ax.imshow(intensity.values, cmap='turbo', 
                            origin='lower', 
                            extent=extent, 
                            aspect='auto', 
                            vmin=vmin, vmax=vmax)
        
        # Add or update the colorbar
        if self.colorbar is not None:
            self.colorbar.remove()
        self.colorbar = self.fig.colorbar(im, ax=self.ax)
        
        # If coordinates for labels are provided, set them
        if coords:
            xlabel = coords.get('xlabel', None)
            ylabel = coords.get('ylabel', None)
            if xlabel:
                self.ax.set_xlabel(xlabel)
            if ylabel:
                self.ax.set_ylabel(ylabel)

        # Find the coordinates where peak_positions is 1
        y, x = np.where(peak_positions.values == 1)
        
        # Convert to actual coordinate values
        y_vals = peak_positions.coords[peak_positions.dims[0]].values[y]
        x_vals = peak_positions.coords[peak_positions.dims[1]].values[x]
        
        # Initialize facecolors
        num_points = len(x_vals)
        # initial_colors = np.full((num_points, 4), [1, 0, 0, 1])  # Initial color is red for all points
        
        # self.scatter = self.ax.scatter(x_vals, y_vals, facecolors=initial_colors)
        # self.facecolors = initial_colors

        # Update MyCanvas Intensity & Peak Positions variables.
        self.intensity = intensity
        self.peak_positions = peak_positions
        
        # If coordinates for labels are provided, set them
        if coords:
            xlabel = coords.get('xlabel', None)
            ylabel = coords.get('ylabel', None)
            if xlabel:
                self.ax.set_xlabel(xlabel)
            if ylabel:
                self.ax.set_ylabel(ylabel)

        self.draw()
        
        # if self.roi_selector is None:
        self.roi_selector = ROISelector(self.ax, self.main_app)
        
        # Use the existing ROISelector
        if self.roi_selector is not None:
            self.roi_selector.activate()        

    def on_key(self, event):
        if event.key in ['R', 'r']:
            self.roi_selector.activate()  # Assuming roi_selector is accessible as an attribute
        elif event.key in ['Q', 'q']:
            self.roi_selector.deactivate()

    def setup_rectangle_selector(self):
        ''' 
        setup_rectangle_selector:
            Purpose:
            Sets up the RectangleSelector for the plot.

            Implementation:
            Initializes a RectangleSelector with specific properties and binds it to the line_select_callback.

            Considerations:
            Call this method to initialize the RectangleSelector before using it.

            Attributes:
            RS (RectangleSelector): Initialized and set up for use.
        '''

        self.RS = RectangleSelector(self.ax, self.line_select_callback,
                                    useblit=True, button=[1, 3],
                                    minspanx=5, minspany=5,
                                    spancoords='pixels',
                                    interactive=True)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_rect_pick)

class MyNavigationToolbar(NavigationToolbar2QT):
    def __init__(self, canvas, parent, coordinates=True):
        super(MyNavigationToolbar, self).__init__(canvas, parent, coordinates)
        self.window = parent  # Assuming the parent window is passed as `parent`
        
        script_dir = os.path.dirname(os.path.realpath(__file__))
        
        # Create custom buttons
        custom_buttons = [
            {'name': 'Load Data', 'icon': os.path.join(script_dir, 'icons/loaddata_icon.png'), 'function': self.window.load_project},
            {'name': 'Toggle ROI Selector', 'icon': os.path.join(script_dir, 'icons/roi_icon.png'), 'function': self.toggle_roi_selector, 'toggle': True}  # Add your own icon path for ROI
        ]
        
        for button in custom_buttons:
            action = QAction(QIcon(button['icon']), '', self)
            action.setToolTip(button['name'])
            if button.get('toggle', False):
                action.setCheckable(True)
            action.triggered.connect(button['function'])
            self.addAction(action)

        # Move custom buttons to the front
        for i in range(len(custom_buttons)):
            action = self.actions()[-1]
            self.removeAction(action)
            self.insertAction(self.actions()[0], action)

    # Toggle function for ROI Selector
    def toggle_roi_selector(self, checked):
        print("toggle_roi_selector called")  # Debugging print
        if self.window.canvas1.roi_selector is None:
            print("roi_selector is None")
        else:
            print("roi_selector is an instance of ROISelector")
        if checked:
            self.window.canvas1.roi_selector.activate()
            self.window.canvas1.draw()
        else:
            self.window.canvas1.roi_selector.deactivate()

class ROISelector:
    def __init__(self, ax, window):
        self.ax = ax
        self.window = window
        print(f"Initializing ROI Selector with ax: {self.ax}, {type(self.ax)}")

        self.rectangles = []  # List to store all rectangles
        self.rectangle_selector = RectangleSelector(self.ax, self.on_select,
                                                    useblit=True, button=[1, 3], 
                                                    minspanx=5, minspany=5,
                                                    spancoords='pixels', interactive=True)
        self.rectangle_selector.set_active(False)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_click)
        self.closing_action = False
        self.active_rect = None

        self.close_box_visible = False  # Add this line
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)  # Add this line

    def activate(self):
        print("ROISelector activate called")  # Debugging print
        self.rectangle_selector.set_active(True)
        print(f"RectangleSelector active state: {self.rectangle_selector.active}")

    def deactivate(self):
        print("ROISelector deactivate called")  # Debugging print
        self.rectangle_selector.set_active(False)
        print(f"RectangleSelector active state: {self.rectangle_selector.active}")

    def on_select(self, eclick, erelease):
        print(f"Window reference: {self.window}")  # Debugging print
        print(f"ROISelector in on_select: {id(self)}, Rectangles: {id(self.rectangles)}")

        # Check if a close action is happening
        if self.closing_action:
            self.closing_action = False
            return
        
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print(f"on_select: (x1, y1):{x1:3.2f}, {y1:3.2f}) --> (*x2, y2):{x2:3.2f}, {y2:3.2f})")

        # Create the main rectangle
        new_rect = Rectangle((min(x1, x2), min(y1, y2)), np.abs(x1 - x2), np.abs(y1 - y2),
                            edgecolor='black', facecolor='none', linewidth=1.5)

        # Create the close box
        close_box_size = 0.15
        close_box_center_x = min(x1, x2) - 0.6 * close_box_size
        close_box_center_y = max(y1, y2) + 0.6 * close_box_size
        close_box = Rectangle((close_box_center_x - close_box_size / 2, close_box_center_y - close_box_size / 2), 
                        close_box_size, close_box_size, 
                        edgecolor='black', facecolor='gray')  # Solid gray fill

        # Store the rectangle and its actual data coordinates in a dictionary
        self.rectangles.append({
            'main': new_rect,
            'close_box': close_box,
            'coords': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        })

        # Add a Line2D to represent the 'X' inside the close box
        x = close_box_center_x
        y = close_box_center_y
        half_size = close_box_size / 2
        x_line = [x - half_size, x + half_size]
        y_line = [y - half_size, y + half_size]
        close_x1 = self.ax.add_line(Line2D(x_line, y_line, color='black', linewidth=1))

        x_line = [x + half_size, x - half_size]
        close_x2 = self.ax.add_line(Line2D(x_line, y_line, color='black', linewidth=1))

        self.ax.add_patch(new_rect)
        self.ax.add_patch(close_box)
        self.ax.figure.canvas.draw()
        self.window.update_roi_table()

        # Update the stored rectangle dictionary to include the 'X'
        self.rectangles[-1].update({
            'close_x1': close_x1,
            'close_x2': close_x2,
        })

    def on_click(self, event):
        was_active = self.rectangle_selector.active  # Store the previous state
        self.rectangle_selector.set_active(False)  # Temporarily deactivate

        clicked_any = False  # Flag to know if any rectangle was clicked

        for rect in self.rectangles:
            main_rect = rect['main']
            close_box = rect['close_box']
            close_x1 = rect.get('close_x1', None)
            close_x2 = rect.get('close_x2', None)

            # Remove rectangle if the close box is clicked
            if close_box.contains(event)[0]:
                self.closing_action = True  # Set the flag
                main_rect.remove()
                close_box.remove()
                self.rectangles.remove(rect)
                self.ax.figure.canvas.draw()
                clicked_any = True

                # Remove the 'X' as well
                if close_x1:
                    close_x1.remove()
                if close_x2:
                    close_x2.remove()

                break  # Only remove the first rectangle that matches

            # Toggle active rectangle if the main rectangle is clicked
            elif main_rect.contains(event)[0]:
                if self.active_rect:  # Reset the previous active rectangle
                    self.active_rect['main'].set_edgecolor('black')
                
                self.active_rect = rect  # Set the new active rectangle
                main_rect.set_edgecolor('blue')  # Highlight the active rectangle
                
                # Coordinates of the selected ROI
                x1, y1, x2, y2 = main_rect.get_extents().get_points().flatten()
                print(f"on_click: (x1, y1):{x1:3.2f}, {y1:3.2f}) --> (*x2, y2):{x2:3.2f}, {y2:3.2f})")
                self.window.update_sliced_dataarray(x1, y1, x2, y2)
                
                self.ax.figure.canvas.draw()
                clicked_any = True
                break  # Only activate the first rectangle that matches

        self.rectangle_selector.set_active(was_active)  # Restore the previous state

        if not clicked_any:
            # Reset the previous active rectangle if none was clicked
            if self.active_rect:
                self.active_rect['main'].set_edgecolor('black')
            self.active_rect = None
        
        self.window.update_roi_table()

    def update_close_box_visibility(self, visible):
        """Updates the visibility of the close box."""
        for rect in self.rectangles:
            close_box = rect['close_box']
            close_box.set_visible(visible)
        self.ax.figure.canvas.draw()

    def on_mouse_move(self, event):
        for rect in self.rectangles:
            close_box = rect['close_box']
            close_x1 = rect.get('close_x1', None)
            close_x2 = rect.get('close_x2', None)

            if close_box.contains(event)[0]:
                # Show the close box and 'X'
                close_box.set_visible(True)
                if close_x1:
                    close_x1.set_visible(True)
                if close_x2:
                    close_x2.set_visible(True)
            else:
                # Hide the close box and 'X'
                close_box.set_visible(False)
                if close_x1:
                    close_x1.set_visible(False)
                if close_x2:
                    close_x2.set_visible(False)
        
        self.ax.figure.canvas.draw()

class Gaussian2DFit:
    def __init__(self, data_array: xr.DataArray):
        # Check for NaN values and replace them with zeros
        nan_mask = np.isnan(data_array['intensity'].values)
        self.nan_mask = xr.DataArray(nan_mask, coords=data_array['intensity'].coords, name='nan_mask')
        data_array['intensity'] = xr.where(xr.DataArray(nan_mask, coords=data_array['intensity'].coords), 0, data_array['intensity'])

        self.data_array = data_array
        self.result = None
        self.model = None
        self.residuals = None
        self.percent_residuals = None

    @staticmethod
    def combined_gaussian_2D(q_xy, q_z, fwhm_qr, fwhm_chi, center_qr, center_chi, fwhm_qxy, fwhm_qz, weight):
        # Convert to qr and chi
        qr = np.sqrt(q_xy**2 + q_z**2)
        chi = np.degrees(np.pi / 2 - np.arctan2(q_z, q_xy))

        # Convert FWHM to sigma
        sigma_qr = fwhm_qr / np.sqrt(8 * np.log(2))
        sigma_chi = fwhm_chi / np.sqrt(8 * np.log(2))
        sigma_qxy = fwhm_qxy / np.sqrt(8 * np.log(2))
        sigma_qz = fwhm_qz / np.sqrt(8 * np.log(2))

        # Gaussian in polar and Cartesian coordinates
        polar_gaussian = np.exp(-((qr - center_qr)**2 / (2 * sigma_qr ** 2) + (chi - center_chi)**2 / (2 * sigma_chi ** 2)))
        cartesian_gaussian = np.exp(-((q_xy - center_qr*np.cos(np.radians(center_chi)))**2 / (2 * sigma_qxy ** 2) + 
                                       (q_z - center_qr*np.sin(np.radians(center_chi)))**2 / (2 * sigma_qz ** 2)))

        # Combined Gaussian with weight
        return (1 - weight) * polar_gaussian + weight * cartesian_gaussian

    def construct_model(self):
        self.model = Model(self.combined_gaussian_2D, independent_vars=['q_xy', 'q_z'])

    def initial_peak_identification(self, img_xr, threshold_ratio):
        # Initialize variables
        max_intensity = np.max(img_xr)
        mean_intensity = np.mean(img_xr)
        median_intensity = np.median(img_xr)
        min_intensity = np.min(img_xr)
        noise_level = np.std(img_xr[img_xr < median_intensity])
        snr = (max_intensity - median_intensity) / noise_level
        # threshold_ratio = initial_threshold_ratio
        found_peaks = False

        # Adaptive thresholding and sigma adjustment
        while not found_peaks and threshold_ratio <= 1.0:
            sigma1 = snr * 0.5
            sigma2 = snr * 1.5
            img_smooth1 = gaussian_filter(np.nan_to_num(img_xr), sigma=sigma1)
            img_smooth2 = gaussian_filter(np.nan_to_num(img_xr), sigma=sigma2)

            dog = img_smooth1 - img_smooth2
            threshold = max(threshold_ratio * max_intensity, 0.1 * max_intensity)

            # Identify peak locations
            peaks = np.where(dog >= threshold)

            if peaks[0].size > 0:
                found_peaks = True
                break
            else:
                threshold_ratio += 0.1  # Increment threshold ratio

        # Proceed with clustering if peaks are found
        if found_peaks:
            coords = np.column_stack(peaks)
            clustering = DBSCAN(eps=3, min_samples=2).fit(coords)
            labels = clustering.labels_
            # cluster_means = [geometric_mean(coords[labels == i], axis=0) for i in set(labels) if i != -1]
            
            # Calculate geometric mean of each cluster
            cluster_means = [gmean(coords[labels == i], axis=0) for i in set(labels) if i != -1]

            if cluster_means:
                center_estimate = max(cluster_means, key=lambda x: img_xr[tuple(x.astype(int))])
            else:
                center_estimate = [np.nan, np.nan]
        else:
            center_estimate = [np.nan, np.nan]

        return center_estimate, dog, peaks

    def initial_fit(self):
        q_xy_vals = self.data_array.coords['q_xy'].values
        q_z_vals = self.data_array.coords['q_z'].values
        q_xy, q_z = np.meshgrid(q_xy_vals, q_z_vals)
        intensity = self.data_array['intensity'].values

        # Determine the bounds of q_{xy} and q_{z} based on the data
        q_xy_min, q_xy_max = np.min(q_xy_vals), np.max(q_xy_vals)
        q_z_min, q_z_max = np.min(q_z_vals), np.max(q_z_vals)

        # Ensure shapes match
        if q_xy.shape != intensity.shape or q_z.shape != intensity.shape:
            raise ValueError("Shape mismatch between q_{xy}, q_{z}, and intensity arrays.")

        # Ensure the model is expecting these variables
        if 'q_xy' not in self.model.independent_vars or 'q_z' not in self.model.independent_vars:
            raise ValueError("Model does not expect q_{xy} and q_{z} as independent variables.")

        # Convert to qr and chi
        qr_vals = np.sqrt(q_xy ** 2 + q_z ** 2)
        chi_vals = np.degrees(np.pi / 2 - np.arctan2(q_z, q_xy))

        # Calculate the bounds of qr and chi
        qr_min, qr_max = np.min(qr_vals), np.max(qr_vals)
        chi_min, chi_max = np.min(chi_vals), np.max(chi_vals)

        # Determine the window size as a percentage of the range
        qr_window = (qr_max - qr_min) * 0.70  # 10% of qr range
        chi_window = (chi_max - chi_min) * 0.70  # 10% of chi range

        # Find the maximum intensity index
        max_intensity_idx = np.unravel_index(np.argmax(intensity), intensity.shape)
        # Convert the maximum intensity index to qr and chi values
        center_qr_est, center_chi_est = qr_vals[max_intensity_idx], chi_vals[max_intensity_idx]

        # Create a mask based on the dynamic window size
        mask = (qr_vals > center_qr_est - qr_window) & (qr_vals < center_qr_est + qr_window) & \
            (chi_vals > center_chi_est - chi_window) & (chi_vals < center_chi_est + chi_window)

        # Estimate FWHM based on the masked region
        fwhm_qr_est = 2 * np.sqrt(np.mean((qr_vals[mask] - center_qr_est) ** 2))
        fwhm_chi_est = 2 * np.sqrt(np.mean((chi_vals[mask] - center_chi_est) ** 2))

        # Call the peak identification method
        # center_estimate, dog, peaks = self.initial_peak_identification(intensity, threshold_ratio=0.5)
        center_estimate = np.nan

        # if not np.isnan(center_estimate).any():
        #     # Convert the center estimate to qr and chi values
        #     center_qr_est, center_chi_est = self._convert_to_qr_chi(center_estimate)
        # else:
        #     # Fallback to maximum intensity if no peaks are found
        #     max_intensity_idx = np.unravel_index(np.argmax(intensity), intensity.shape)
        #     center_qr_est, center_chi_est = qr_vals[max_intensity_idx], chi_vals[max_intensity_idx]

        # fwhm_qr_est = 2.0 * np.sqrt(np.average((qr_vals - center_qr_est) ** 2, weights=intensity))
        # fwhm_chi_est = 2.0 * np.sqrt(np.average((chi_vals - center_chi_est) ** 2, weights=intensity))

        # Check for NaN or Inf in intensity and weights
        if np.any(np.isnan(intensity)) or np.any(np.isinf(intensity)):
            print("Warning: NaN or Inf detected in intensity.")

        if np.any(np.isnan(qr_vals)) or np.any(np.isinf(qr_vals)) or np.any(np.isnan(chi_vals)) or np.any(np.isinf(chi_vals)):
            print("Warning: NaN or Inf detected in qr_vals or chi_vals.")

        # Set up initial parameters with dynamic bounds
        params = self.model.make_params(I0=np.max(intensity), 
                                        center_qr=center_qr_est, 
                                        center_chi=center_chi_est, 
                                        fwhm_qr=fwhm_qr_est, 
                                        fwhm_chi=fwhm_chi_est)

        # Set the bounds dynamically based on the qr and chi ranges
        params['center_qr'].set(value=center_qr_est, min=qr_min, max=qr_max, vary=True)
        params['center_chi'].set(value=center_chi_est, min=chi_min, max=chi_max, vary=True)

        # Set the initial value and bounds for fwhm_qr
        params['fwhm_qr'].set(value=fwhm_qr_est, min=0, max=qr_max - qr_min)

        # Set the initial value and bounds for fwhm_chi
        params['fwhm_chi'].set(value=fwhm_chi_est, min=0, max=chi_max - chi_min)

        params.add('fwhm_qxy', value=0.2, min=0, vary=False)
        params.add('fwhm_qz', value=0.2, min=0, vary=False)
        params.add('weight', value=0, min=0, max=1, vary=False)

        error = np.sqrt(intensity + 1)

        # Perform initial fit
        self.result = self.model.fit(intensity.ravel(), q_xy=q_xy.ravel(), q_z=q_z.ravel(), params=params, weights=1/np.sqrt(error.ravel()))

    def intermediate_fit(self):
        # Unlock weight and lock fwhm_qr, fwhm_chi, center_qr, center_chi
        self.result.params['weight'].set(value=0.5, min=0, max=1, vary=True)
        for param in ['fwhm_qr', 'fwhm_chi', 'center_qr', 'center_chi']:
            self.result.params[param].set(vary=False)

        # Perform intermediate fit
        q_xy_vals = self.data_array.coords['q_xy'].values
        q_z_vals = self.data_array.coords['q_z'].values
        q_xy, q_z = np.meshgrid(q_xy_vals, q_z_vals)
        intensity = self.data_array['intensity'].values
        # intensity = self.data_array['intensity'].values.ravel()
        error = np.sqrt(intensity + 1)

        self.result = self.model.fit(intensity.ravel(), q_xy=q_xy.ravel(), q_z=q_z.ravel(), params=self.result.params, weights=1/np.sqrt(error.ravel()))

    def final_fit(self):
        # Unlock all parameters and adjust bounds
        for param in self.result.params:
            self.result.params[param].set(vary=True)
            if param != 'weight':
                self.result.params[param].set(min=self.result.params[param].value * 0.8, 
                                            max=self.result.params[param].value * 1.2)

        # Set FWHM of qxy and qz to qr and lock them
        fwhm_qr_current_value = self.result.params['fwhm_qr'].value
        self.result.params['fwhm_qxy'].set(value=fwhm_qr_current_value, vary=False)
        self.result.params['fwhm_qz'].set(value=fwhm_qr_current_value, vary=False)

        # Perform final fit
        q_xy_vals = self.data_array.coords['q_xy'].values
        q_z_vals = self.data_array.coords['q_z'].values
        q_xy, q_z = np.meshgrid(q_xy_vals, q_z_vals)
        intensity = self.data_array['intensity'].values

        error = np.sqrt(intensity + 1)

        self.result = self.model.fit(intensity.ravel(), q_xy=q_xy.ravel(), q_z=q_z.ravel(), params=self.result.params, weights=1/np.sqrt(error.ravel()))

    def perform_combined_fit(self, fit_method='gaussianpolarcartesian'):
        self.construct_model()

        if fit_method == 'gaussianpolar':
            self.initial_fit()
        elif fit_method == 'gaussianpolarcartesian':
            self.initial_fit()
            self.intermediate_fit()
            self.final_fit()
        elif fit_method == 'gaussiancartesian':
            # Placeholder for gaussiancartesian fitting method
            pass
        else:
            print("Invalid fitting method selected. Defaulting to gaussianpolarcartesian.")
            self.initial_fit()
            self.intermediate_fit()
            self.final_fit()

    def get_fit_result(self):
        return self.result

    def calculate_residuals(self):
        if self.result is None:
            return None

        x, y = np.meshgrid(self.data_array.coords['q_xy'], self.data_array.coords['q_z'])
        fit_data = self.model.func(x.flatten(), y.flatten(), **self.result.best_values)
        self.residuals = self.data_array['intensity'].values.flatten() - fit_data

        # Calculate residuals as percentage of maximum intensity
        max_intensity = np.max(self.data_array['intensity'].values)
        self.percent_residuals = (self.residuals / max_intensity) * 100

        return self.percent_residuals

    def fit_statistics(self):
        if self.result is None:
            return None

        # Create an empty dictionary to store fit statistics
        fit_stats_dict = {}

        # Chi-square
        fit_stats_dict['chi_square'] = self.result.chisqr

        # Akaike Information Criterion
        fit_stats_dict['aic'] = self.result.aic

        # Bayesian Information Criterion
        fit_stats_dict['bic'] = self.result.bic

        # Parameters and their uncertainties
        for param_name, param in self.result.params.items():
            fit_stats_dict[f"{param_name}_value"] = param.value
            fit_stats_dict[f"{param_name}_stderr"] = param.stderr

        return fit_stats_dict

    def plot_fit(self, ax_fit, ax_residual):
        if self.result is None:
            return

        # Generate grid for plotting
        x, y = np.meshgrid(self.data_array.coords['q_xy'], self.data_array.coords['q_z'])
        x_plot, y_plot = np.meshgrid(
            np.linspace(x.min(), x.max(), 100),
            np.linspace(y.min(), y.max(), 100)
        )

        # Font size settings
        fontsize = 12

        # Plot Fit
        ax_fit.clear()  # Clear the axes
        fit_data = self.model.func(x_plot, y_plot, **self.result.best_values)
        fit_plot = ax_fit.pcolor(x_plot, y_plot, fit_data, shading='auto')
        ax_fit.set_aspect('equal', 'box')  # Fixed aspect ratio

        # Only set labels and titles if they are not set
        if not ax_fit.title.get_text():
            ax_fit.set_title('Fit', fontsize=fontsize)
        if not ax_fit.xaxis.get_label_text():
            ax_fit.set_xlabel('$q_{xy}$ (Å$^{-1}$)', fontsize=fontsize)
        if not ax_fit.yaxis.get_label_text():
            ax_fit.set_ylabel('$q_{z}$ (Å$^{-1}$)', fontsize=fontsize)

        # Create an area for the colorbar
        divider = make_axes_locatable(ax_fit)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(fit_plot, cax=cax)

        # Plot Residuals
        ax_residual.clear()  # Clear the axes
        residuals = self.calculate_residuals()
        Z_residual = griddata((x.flatten(), y.flatten()), residuals.flatten(), (x_plot, y_plot), method='linear', fill_value=0)
        residual_plot = ax_residual.pcolor(x_plot, y_plot, Z_residual, shading='auto')
        ax_residual.set_aspect('equal', 'box')  # Fixed aspect ratio

        # Only set labels and titles if they are not set
        if not ax_residual.title.get_text():
            ax_residual.set_title('Residual', fontsize=fontsize)
        if not ax_residual.xaxis.get_label_text():
            ax_residual.set_xlabel('$q_{xy}$ (Å$^{-1}$)', fontsize=fontsize)
        if not ax_residual.yaxis.get_label_text():
            ax_residual.set_ylabel('$q_{z}$ (Å$^{-1}$)', fontsize=fontsize)

        # Create an area for the colorbar
        divider = make_axes_locatable(ax_residual)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(residual_plot, cax=cax)

        # Update canvas
        ax_fit.figure.canvas.draw_idle()
        ax_residual.figure.canvas.draw_idle()
        
    # def plot_fit(self, ax_fit, ax_residual):
    #     if self.result is None:
    #         return

    #     # Clear the previous plots
    #     ax_fit.clear()
    #     ax_residual.clear()

    #     x, y = np.meshgrid(self.data_array.coords['q_xy'], self.data_array.coords['q_z'])
    #     # z = self.data_array['intensity'].values

    #     # Generate grid for plotting
    #     x_plot, y_plot = np.meshgrid(
    #         np.linspace(x.min(), x.max(), 100),
    #         np.linspace(y.min(), y.max(), 100)
    #     )

    #     # Font size settings
    #     fontsize = 12

    #     # Fit plot
    #     fit_data = self.model.func(x_plot, y_plot, **self.result.best_values)
    #     ax_fit.pcolor(x_plot, y_plot, fit_data, shading='auto')
    #     ax_fit.set_title('Fit')
    #     ax_fit.set_xlabel('$q_{xy}$ (Å$^{-1}$)', fontsize=fontsize)
    #     ax_fit.set_ylabel('$q_{z}$ (Å$^{-1}$)', fontsize=fontsize)

    #     # Residual plot
    #     residuals = self.calculate_residuals()
    #     Z_residual = griddata((x.flatten(), y.flatten()), residuals.flatten(), (x_plot, y_plot), method='linear', fill_value=0)
    #     ax_residual.pcolor(x_plot, y_plot, Z_residual, shading='auto')
    #     ax_residual.set_title('Residuals')
    #     ax_residual.set_xlabel('$q_{xy}$ (Å$^{-1}$)', fontsize=fontsize)
    #     ax_residual.set_ylabel('$q_{z}$ (Å$^{-1}$)', fontsize=fontsize)

    #     # Update canvas
    #     ax_fit.figure.canvas.draw_idle()
    #     ax_residual.figure.canvas.draw_idle()

    '''
        # fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # # Set the contrast scaling based on percentiles of the main data array
        # vmin = np.nanpercentile(z, 10)
        # vmax = np.nanpercentile(z, 99)

        # # Original Data
        # im0 = axs[0].imshow(z, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        # axs[0].set_title('Original Data')
        # axs[0].set_xlabel('$q_{xy}$ (Å$^{-1}$)', fontsize=fontsize)
        # axs[0].set_ylabel('$q_{z}$ (Å$^{-1}$)', fontsize=fontsize)
        # fig.colorbar(im0, ax=axs[0], shrink=0.5)

        # # Fit Data
        # fit_data = self.model.func(x, y, **self.result.best_values).reshape(x.shape)
        # im1 = axs[1].imshow(fit_data, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis', vmin=vmin, vmax=vmax, interpolation='none')
        # axs[1].set_title('Fit')
        # axs[1].set_xlabel('$q_{xy}$ (Å$^{-1}$)', fontsize=fontsize)
        # axs[1].set_ylabel('$q_{z}$ (Å$^{-1}$)', fontsize=fontsize)
        # fig.colorbar(im1, ax=axs[1], shrink=0.5)

        # # Percent Residuals
        # percent_residuals = self.percent_residuals.reshape(x.shape)
        # im2 = axs[2].imshow(percent_residuals, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis', vmin=np.nanpercentile(percent_residuals, 10), vmax=np.nanpercentile(percent_residuals, 99))
        # axs[2].set_title('Percent Residuals')
        # axs[2].set_xlabel('$q_{xy}$ (Å$^{-1}$)', fontsize=fontsize)
        # axs[2].set_ylabel('$q_{z}$ (Å$^{-1}$)', fontsize=fontsize)
        # fig.colorbar(im2, ax=axs[2], shrink=0.5)
    '''

class GaussianFitToolbar(NavigationToolbar2QT):
    def __init__(self, canvas, parent, coordinates=True):
        super(GaussianFitToolbar, self).__init__(canvas, parent, coordinates)
        
        # Assuming the parent window is passed as `parent`
        self.window = parent  
        script_dir = os.path.dirname(os.path.realpath(__file__))

        # Create custom buttons
        custom_buttons = [
            {'name': '2D Gaussian Fit', 'icon': os.path.join(script_dir, 'icons/gaussian_icon.png'), 'function': self.window.perform_gaussian_fit},
            # {'name': 'Toggle ROI Selector', 'icon': os.path.join(script_dir, 'icons/roi_icon.png'), 'function': self.toggle_roi_selector, 'toggle': True}  # Add your own icon path for ROI
        ]
        
        for button in custom_buttons:
            action = QAction(QIcon(button['icon']), '', self)
            action.setToolTip(button['name'])
            if button.get('toggle', False):
                action.setCheckable(True)
            action.triggered.connect(button['function'])
            self.addAction(action)

        # Move custom buttons to the front
        for i in range(len(custom_buttons)):
            action = self.actions()[-1]
            self.removeAction(action)
            self.insertAction(self.actions()[0], action)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = XrayApp()
    ex.show()
    sys.exit(app.exec_())


''' Canvas Setup (Refactored)
    # self.setWindowTitle('X-ray Data Analysis')
    # self.setGeometry(100, 100, 1600, 800)

    # self.central_widget = QWidget()
    # self.setCentralWidget(self.central_widget)
    # self.layout = QHBoxLayout()
    # self.central_widget.setLayout(self.layout)

    # # Left part for canvas1 (50% of window)
    # self.left_frame = QFrame()
    # self.left_layout = QVBoxLayout()
    # self.left_frame.setLayout(self.left_layout)
    # self.layout.addWidget(self.left_frame, stretch=1)

    # # Main canvas1 setup
    # self.fig1, self.ax1 = plt.subplots()
    # self.ax1.axis('off')
    # self.canvas1 = MyCanvas(self.fig1, main_app=self, add_subplot=True)
    # self.nav1 = MyNavigationToolbar(self.canvas1, self)
    # self.main_layout = QVBoxLayout(self.left_frame)
    # self.main_layout.addWidget(self.nav1)
    # self.main_layout.addWidget(self.canvas1)

    # # ROI table
    # self.roi_table = QTableWidget()
    # self.roi_table.setColumnCount(5)
    # self.roi_table.setHorizontalHeaderLabels(['ROI #', 'q_{xy} (min)', 'q_{xy} (max)', 'q_{z} (min)', 'q_{z} (max)'])
    # self.main_layout.addWidget(self.roi_table)
    
    # # Right part for canvas2 and stats (50% of window)
    # self.right_frame = QFrame()
    # self.right_layout = QVBoxLayout()
    # self.right_frame.setLayout(self.right_layout)
    # self.layout.addWidget(self.right_frame, stretch=1)
    
    # # Canvas2 setup
    # self.fig2, self.axs = plt.subplots(2, 2, figsize=(8, 4))
    # self.ax2, self.ax3, self.ax4, self.ax5 = self.axs.flatten()
    # self.ax3.axis('off')
    # self.canvas2 = MyCanvas(self.fig2, main_app=self, add_subplot=False)
    # self.nav2 = MyNavigationToolbar(self.canvas2, self)
    # self.nav2.setFixedWidth(600)
    # self.right_layout.addWidget(self.nav2)
    # self.right_layout.addWidget(self.canvas2, stretch=2)

    # # Stats setup
    # self.stats_widget = QWidget()
    # self.stats_layout = QGridLayout()
    # self.stats_widget.setLayout(self.stats_layout)
    # self.add_stat_box('Amplitude', 1, 0)
    # self.add_stat_box('Center X', 2, 0)
    # self.add_stat_box('Center Y', 3, 0)
    # self.add_stat_box('Chi-square', 1, 1)
    # self.add_stat_box('Reduced Chi-square', 2, 1)
    # self.add_stat_box('R-squared', 3, 1)
    # self.add_stat_box('Sigma X', 1, 2)
    # self.add_stat_box('Sigma Y', 2, 2)
    # self.add_stat_box('FWHM X', 3, 2)
    # self.add_stat_box('FWHM Y', 4, 2)
    # self.right_layout.addWidget(self.stats_widget, stretch=1)


    def initUI(self):
        self.setWindowTitle('X-ray Data Analysis')
        self.setGeometry(100, 100, 1600, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Create a frame for the left part (plot + sliders)
        self.left_frame = QFrame()
        self.left_layout = QVBoxLayout()
        self.left_frame.setLayout(self.left_layout)
        self.layout.addWidget(self.left_frame)

        # Create main frame for X-ray plot
        self.main_frame = QFrame()
        self.main_layout = QVBoxLayout()
        self.main_frame.setLayout(self.main_layout)
        self.left_layout.addWidget(self.main_frame)

        # Create matplotlib figure and attach it to main frame
        self.fig1, self.ax1 = plt.subplots()
        self.ax1.axis('off')

        self.canvas1 = MyCanvas(self.fig1, main_app=self, add_subplot=True)
        # self.canvas1 = MyCanvas(self.fig1, main_app=self, add_subplot=True)
        self.main_layout.addWidget(self.canvas1)

        self.nav1 = MyNavigationToolbar(self.canvas1, self)
        self.main_layout.insertWidget(0, self.nav1)

        # Initialize the output frame and layout
        self.output_frame = QFrame()
        self.output_layout = QVBoxLayout()
        self.output_frame.setLayout(self.output_layout)
        self.layout.addWidget(self.output_frame)  # add output_frame to main layout
        
        # Create matplotlib figure and set up GridSpec
        self.fig2 = plt.figure(figsize=(8, 4))  # Adjust figsize as needed
        gs = GridSpec(2, 2, width_ratios=[3, 2])  # ax2 will be twice as wide
        
        self.ax2 = self.fig2.add_subplot(gs[0, 0])
        self.ax3 = self.fig2.add_subplot(gs[1, 0])  # You can leave this blank
        self.ax4 = self.fig2.add_subplot(gs[0, 1])
        self.ax5 = self.fig2.add_subplot(gs[1, 1])
        
        self.ax3.axis('off')  # Turn off ax3 if not used

        self.output_layout = QVBoxLayout()
        self.output_frame.setLayout(self.output_layout)
        
        # Use setStretch to control the vertical size
        self.output_layout.setStretch(0, 3)  # For canvas2
        self.output_layout.setStretch(1, 2)  # For widget3

        # Create a QWidget for ax3 and set it to a grid layout
        self.widget3 = QWidget()
        self.grid3 = QGridLayout()
        self.widget3.setLayout(self.grid3)

        # Add 'Fit 2D Gaussian' button
        self.btn_fit = QPushButton('Fit 2D Gaussian', self)
        self.btn_fit.clicked.connect(self.perform_gaussian_fit)
        self.grid3.addWidget(self.btn_fit, 0, 0)

        # Add boxes and labels for fit statistics
        self.add_stat_box('Amplitude', 1, 0)
        self.add_stat_box('Center X', 2, 0)
        self.add_stat_box('Center Y', 3, 0)
        self.add_stat_box('Chi-square', 1, 1)
        self.add_stat_box('Reduced Chi-square', 2, 1)
        self.add_stat_box('R-squared', 3, 1)
        self.add_stat_box('Sigma X', 1, 2)
        self.add_stat_box('Sigma Y', 2, 2)
        self.add_stat_box('FWHM X', 3, 2)
        self.add_stat_box('FWHM Y', 4, 2)

        # Add the QWidget for ax3 into your layout
        self.output_layout.addWidget(self.widget3)
        
        # Create a dictionary to map custom names to axes
        axes_dict = {'ax2': self.ax2, 'ax3': self.ax3, 'ax4': self.ax4, 'ax5': self.ax5}
        self.canvas2 = MyCanvas(self.fig2, main_app=self, add_subplot=False, axes=axes_dict)

        self.output_layout.addWidget(self.canvas2)  # Now this should work
        self.nav2 = NavigationToolbar(self.canvas2, self)
        self.nav2.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
        self.nav2.setFixedWidth(600)  # set to a suitable widthThere 
        self.output_layout.insertWidget(0, self.nav2)

        # Create QTableWidget object
        self.roi_table = QTableWidget()
        self.roi_table.setColumnCount(5)
        self.roi_table.setHorizontalHeaderLabels(['ROI #', 'q_{xy} (min)', 'q_{xy} (max)', 'q_{z} (min)', 'q_{z} (max)'])
        
        # Add table to the layout
        self.main_layout.addWidget(self.roi_table)
        self.roi_table.cellClicked.connect(self.on_table_click)
        # self.roi_table.cellClicked.connect(self.row_was_selected)
        # self.roi_table.itemSelectionChanged.connect(self.row_was_selected)

        self.roi_selector = None
        self.coords = None

        # # Fix the heights of the canvases
        # self.canvas1.setFixedHeight(400)
        # self.canvas2.setFixedHeight(400)

        # # Align the toolbars at the top
        # self.main_layout.setAlignment(self.nav1, Qt.AlignTop)
        # self.output_layout.setAlignment(self.nav2, Qt.AlignTop)

        # # Allow canvases to expand but keep them at the same height
        # self.main_layout.setStretchFactor(self.canvas1, 1)
        # self.output_layout.setStretchFactor(self.canvas2, 1)
        # self.roi_selector = ROISelector(self.ax1)
        # self.roi_selector = ROISelector(self.ax1, self)
        # self.canvas1.roi_selector = self.roi_selector
        # self.roi_selector.activate()
        # print(f"ROISelector in XrayApp: {self.roi_selector}, active: {self.roi_selector.rectangle_selector.active}")
        # print(f"ROISelector in canvas1: {self.canvas1.roi_selector}, active: {self.canvas1.roi_selector.rectangle_selector.active}")

        # def onclick(event):
        #     print(f"Mouse click event: {event}")
            
        # self.fig1.canvas.mpl_connect('button_press_event', onclick)
'''

''' GaussianFit2D Method (Refactored)
    class Gaussian2DFit:
        # Adapted From: https://lmfit.github.io/lmfit-py/examples/example_two_dimensional_peak.html
        def __init__(self, data_array: xr.DataArray):
            self.data_array = data_array
            self.result = None

        @staticmethod
        def gaussian2d_rotated(x, y, amplitude, centerx, centery, sigmax, sigmay, rotation):
            xp = (x - centerx) * np.cos(rotation) - (y - centery) * np.sin(rotation)
            yp = (x - centerx) * np.sin(rotation) + (y - centery) * np.cos(rotation)
            g = amplitude * np.exp(-((xp/sigmax)**2 + (yp/sigmay)**2) / 2.)
            return g
        
        def construct_model(self):
            # self.model = lmfit.models.Gaussian2dModel()
            self.model = Model(self.gaussian2d_rotated, independent_vars=['x', 'y'])

        def perform_fit(self):
            x_vals = self.data_array.coords['q_xy'].values
            y_vals = self.data_array.coords['q_z'].values
            x, y = np.meshgrid(x_vals, y_vals)
            z = self.data_array['intensity'].values

            params = self.model.make_params(amplitude=np.max(z), centerx=np.mean(x_vals), centery=np.mean(y_vals),
                                            sigmax=np.std(x_vals), sigmay=np.std(y_vals), rotation=0)
            params['rotation'].set(value = .1, min=0, max=np.pi)
            params['sigmax'].set(min=0)
            params['sigmay'].set(min=0)

            error = np.sqrt(z + 1)
            self.result = self.model.fit(z.ravel(), x=x.ravel(), y=y.ravel(), params=params, weights=1/np.sqrt(error.ravel()))

        def calculate_residuals(self):
            if self.result is None:
                return None

            x, y = np.meshgrid(self.data_array.coords['q_xy'], self.data_array.coords['q_z'])
            fit_data = self.model.func(x.flatten(), y.flatten(), **self.result.best_values)
            return self.data_array['intensity'].values.flatten() - fit_data

        def fit_statistics(self):
            if self.result is None:
                return None

            # Create an empty dictionary to store fit statistics
            fit_stats_dict = {}

            # Chi-square
            fit_stats_dict['chi_square'] = self.result.chisqr

            # Akaike Information Criterion
            fit_stats_dict['aic'] = self.result.aic

            # Bayesian Information Criterion
            fit_stats_dict['bic'] = self.result.bic

            # Parameters and their uncertainties
            for param_name, param in self.result.params.items():
                fit_stats_dict[f"{param_name}_value"] = param.value
                fit_stats_dict[f"{param_name}_stderr"] = param.stderr

            return fit_stats_dict

        def plot_fit(self, ax_fit, ax_residual):
            if self.result is None:
                return

            # Clear the previous plots
            ax_fit.clear()
            ax_residual.clear()

            x, y = np.meshgrid(self.data_array.coords['q_xy'], self.data_array.coords['q_z'])

            # Generate grid for plotting
            x_plot, y_plot = np.meshgrid(
                np.linspace(x.min(), x.max(), 100),
                np.linspace(y.min(), y.max(), 100)
            )

            # Fit plot
            fit_data = self.model.func(x_plot, y_plot, **self.result.best_values)
            ax_fit.pcolor(x_plot, y_plot, fit_data, shading='auto')
            ax_fit.set_title('Fit')

            # Residual plot
            residuals = self.calculate_residuals()
            Z_residual = griddata((x.flatten(), y.flatten()), residuals.flatten(), (x_plot, y_plot), method='linear', fill_value=0)
            ax_residual.pcolor(x_plot, y_plot, Z_residual, shading='auto')
            ax_residual.set_title('Residuals')

            # Update canvas
            ax_fit.figure.canvas.draw_idle()
            ax_residual.figure.canvas.draw_idle()
'''

''' Perform Gaussian Fit (Refactored)
    # def perform_gaussian_fit(self):
    #     fit_instance = Gaussian2DFit(self.sliced_ds)
    #     fit_instance.construct_model()
    #     fit_instance.perform_fit()
        
    #     # Print detailed fit report
    #     lmfit.report_fit(fit_instance.result)
        
    #     fit_instance.plot_fit(self.canvas2.axes['ax4'], self.canvas2.axes['ax5'])
        
    #     # Get the fit statistics
    #     stats = fit_instance.fit_statistics()
    #     # self.display_fit_statistics(stats)

    #     # Get fit statistics
    #     fit_stats_dict = fit_instance.fit_statistics()
        
    #     # Update stat boxes
    #     self.update_stat_boxes(fit_stats_dict)

        # # Find the active row in the table (assuming it's selected)
        # active_row = self.roi_table.currentRow()

        # if active_row != -1:  # A row is selected
        #     # Check for existing fit values in the active row
        #     existing_value = self.roi_table.item(active_row, 5)  # Check one of the fit-related columns
        #     if existing_value and existing_value.text():  
        #         # Prompt user for overwriting
        #         msg = QMessageBox()
        #         msg.setIcon(QMessageBox.Warning)
        #         msg.setText("Do you want to overwrite existing fit values for the active ROI?")
        #         msg.setWindowTitle("Overwrite Confirmation")
        #         msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        #         retval = msg.exec_()
        #         if retval != QMessageBox.Yes:
        #             return  # Do not overwrite, return early

        #     # Update the table with new fit statistics
        #     for col_idx, label in enumerate(self.header_labels[5:], start=5):
        #         item_value = fit_stats_dict.get(label.replace(' ', '_').lower(), "")
        #         self.roi_table.setItem(active_row, col_idx, QTableWidgetItem(str(item_value)))
'''
