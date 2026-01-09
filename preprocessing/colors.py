"""
Color Utilities

Functions for color conversion and automatic color assignment.
Uses Golden Ratio method for visually distinct colors.
"""

import colorsys
import numpy as np


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple (0-1 range)"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def rgb_to_hex(rgb):
    """Convert RGB tuple (0-1 range) to hex color"""
    return '#{:02x}{:02x}{:02x}'.format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    )


def color_distance(c1, c2):
    """Calculate color distance in RGB space"""
    r1, g1, b1 = hex_to_rgb(c1) if isinstance(c1, str) else c1
    r2, g2, b2 = hex_to_rgb(c2) if isinstance(c2, str) else c2
    return np.sqrt((r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2)


def generate_rainbow_colors(n, saturation=0.75, value=0.85):
    """
    Generate n visually distinct colors using Golden Ratio method
    
    Ensures adjacent indices get maximally different colors,
    critical when CC labeling assigns nearby IDs to adjacent structures.
    """
    colors = []
    
    # Golden ratio conjugate ≈ 0.618 - mathematically optimal spacing
    golden_ratio_conjugate = 0.618033988749895
    
    hue = 0.0
    for i in range(n):
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb_to_hex(rgb))
        hue = (hue + golden_ratio_conjugate) % 1.0
    
    return colors


def assign_colors(label_names, user_colors=None, min_distance=0.3):
    """
    Assign colors to labels, respecting user specifications
    
    Parameters:
    -----------
    label_names : list
        List of label names
    user_colors : dict or None
        User-specified colors {label_name: hex_color}
    min_distance : float
        Minimum color distance to avoid similar colors
    
    Returns:
    --------
    colors : dict
        Complete color mapping {label_name: hex_color}
    """
    if user_colors is None:
        user_colors = {}
    
    result = {}
    used_colors = list(user_colors.values())
    
    # First, assign user-specified colors
    for name in label_names:
        if name in user_colors:
            result[name] = user_colors[name]
    
    # Generate rainbow colors for unassigned labels
    unassigned = [name for name in label_names if name not in result]
    
    if unassigned:
        # Generate more colors than needed to allow selection
        n_candidates = max(len(unassigned) * 3, 12)
        rainbow = generate_rainbow_colors(n_candidates)
        
        for name in unassigned:
            # Find a color that's far enough from all used colors
            best_color = None
            best_min_dist = -1
            
            for candidate in rainbow:
                if candidate in used_colors:
                    continue
                
                # Calculate minimum distance to any used color
                if used_colors:
                    min_dist = min(color_distance(candidate, c) for c in used_colors)
                else:
                    min_dist = float('inf')
                
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_color = candidate
            
            if best_color is None:
                # Fallback: just pick first unused
                for candidate in rainbow:
                    if candidate not in used_colors:
                        best_color = candidate
                        break
            
            if best_color is None:
                # Final fallback: generate random color
                best_color = rgb_to_hex((np.random.random(), np.random.random(), np.random.random()))
            
            result[name] = best_color
            used_colors.append(best_color)
    
    return result
