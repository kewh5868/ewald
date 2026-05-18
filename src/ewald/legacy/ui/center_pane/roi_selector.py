from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector

class ROISelector:
    """
    Allows drawing, persisting, and deleting rectangular ROIs on a Matplotlib Axes,
    with a small "close" box to remove each ROI.
    """
    def __init__(self, ax, window):
        self.ax = ax
        self.window = window
        self.rectangles = []      # List of dicts storing each ROI's patches
        self.active_rect = None

        # Configure RectangleSelector but start inactive
        self.rectangle_selector = RectangleSelector(
            self.ax, self.on_select,
            useblit=True, button=[1, 3],  # left or right click & drag
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )
        self.rectangle_selector.set_active(False)

        # Connect events for deletion box hover & click
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_click)
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def enable_selector(self, enabled: bool):
        """
        Enable or disable the ROI drawing mode (RectangleSelector).
        """
        self.rectangle_selector.set_active(enabled)

    def on_select(self, eclick, erelease):
        """
        Callback when user finishes dragging out a rectangle.
        Creates a persistent ROI patch plus a small delete box.
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, y1, x2, y2):
            return

        # Compute rectangle bounds
        x_min, y_min = min(x1, x2), min(y1, y2)
        width, height = abs(x2 - x1), abs(y2 - y1)

        # Main ROI patch
        main_rect = Rectangle((x_min, y_min), width, height,
                              edgecolor='black', facecolor='none', linewidth=1.5)
        self.ax.add_patch(main_rect)

        # Delete (close) box in top-right corner of ROI
        cb_size = min(width, height) * 0.1
        cb_x = x_min + width - cb_size
        cb_y = y_min + height - cb_size
        close_box = Rectangle((cb_x, cb_y), cb_size, cb_size,
                              edgecolor='black', facecolor='none', linewidth=1)
        self.ax.add_patch(close_box)

        # "X" lines inside the close box, initially hidden
        close_x1 = self.ax.plot([cb_x, cb_x + cb_size], [cb_y, cb_y + cb_size],
                                linewidth=1, visible=False)[0]
        close_x2 = self.ax.plot([cb_x + cb_size, cb_x], [cb_y, cb_y + cb_size],
                                linewidth=1, visible=False)[0]

        # Store all parts together
        self.rectangles.append({
            'main': main_rect,
            'close_box': close_box,
            'close_x1': close_x1,
            'close_x2': close_x2
        })

        # Redraw and update GUI table
        self.ax.figure.canvas.draw_idle()
        if hasattr(self.window, 'update_roi_table'):
            self.window.update_roi_table()

    def on_click(self, event):
        """
        Detect clicks on any ROI's close-box to delete that ROI.
        """
        if event.inaxes != self.ax:
            return

        for i, rect in enumerate(list(self.rectangles)):
            contains, _ = rect['close_box'].contains(event)
            if contains:
                # Remove all artists
                for artist in (rect['main'], rect['close_box'],
                               rect['close_x1'], rect['close_x2']):
                    artist.remove()
                # Drop from list
                self.rectangles.remove(rect)
                # Refresh canvas and table
                self.ax.figure.canvas.draw_idle()
                if hasattr(self.window, 'update_roi_table'):
                    self.window.update_roi_table()
                break

    def on_mouse_move(self, event):
        """
        Toggle visibility of the "X" when hovering over the close box.
        """
        if event.inaxes != self.ax:
            return

        update_needed = False
        for rect in self.rectangles:
            contains, _ = rect['close_box'].contains(event)
            # Always show the close box edge
            rect['close_box'].set_visible(True)
            # Show or hide the X lines
            rect['close_x1'].set_visible(contains)
            rect['close_x2'].set_visible(contains)
            update_needed = True

        if update_needed:
            self.ax.figure.canvas.draw_idle()
