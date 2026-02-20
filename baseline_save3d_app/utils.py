"""
Utility Functions
"""

def _hex_to_rgb(hex_color):
    """Convert hex to RGB (0-1 range) for PyVista"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    
def _hex_to_rgba(hex_color):
    """Convert hex to RGBA (0-1 range) for napari labels"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    a = 1.0  # Fully opaque
    return [r, g, b, a]