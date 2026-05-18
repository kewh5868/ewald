from PyQt6.QtCore import QObject
from .roi_selector import ROISelector

class ROIManager(QObject):
    """
    Manages ROI drawing and table updates using ROISelector.
    """
    def __init__(self, image_canvas, peak_table_view):
        super().__init__(image_canvas)
        self.canvas = image_canvas.canvas
        self.ax = image_canvas.ax_main
        self.roi_model = peak_table_view.roi_model

        # allow the table model to call back into this manager
        self.roi_model.manager = self

        # Initialize ROISelector for persistent ROI drawing
        self.selector = ROISelector(self.ax, window=self)
        # Start with selector disabled
        self.selector.enable_selector(False)

    def enable_selector(self, enable=True):
        """Enable or disable ROI drawing mode."""
        self.selector.enable_selector(enable)

    def clear_all(self):
        """
        Remove every ROI patch from the axes and clear both the selector and the ROI table.
        """
        # Remove every rectangle & its close-box from the axes
        for rect_dict in list(self.selector.rectangles):
            rect_dict['main'].remove()
            rect_dict['close_box'].remove()
            rect_dict['close_x1'].remove()
            rect_dict['close_x2'].remove()
        # Drop all stored rectangles
        self.selector.rectangles.clear()
        # Clear the ROI table entries
        self.roi_model.clear()
        # Refresh the canvas
        self.canvas.draw_idle()

    def update_roi_table(self):
        """Update the ROI table model based on current rectangles."""
        # Clear existing entries
        self.roi_model.clear()
        # Add current ROIs from the selector
        for rect_dict in self.selector.rectangles:
            main_rect = rect_dict['main']
            x0, y0 = main_rect.get_xy()
            w, h = main_rect.get_width(), main_rect.get_height()
            cx, cy = x0 + w/2, y0 + h/2
            # Corner coordinates
            c1 = (x0, y0)
            c2 = (x0 + w, y0)
            c3 = (x0 + w, y0 + h)
            c4 = (x0, y0 + h)
            # Insert into the ROI table model; linked_hkl left empty
            self.roi_model.add_roi('Box', cx, cy, c1, c2, c3, c4, '')