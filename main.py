#!/usr/bin/env python3
"""
3D ASCII Cube Animation
Displays rotating 3D cubes in the terminal using ASCII characters.

Features:
    - Diagonal rotation on all three axes
    - Distance-based character rendering
    - Adjustable animation speed
    - Configurable runtime
    - Multiple cubes when terminal width allows
    - Colored cubes with different colors
    - Reduced flickering for smoother animation

Command line arguments:
    --sleep-time: Sleep time between frames in seconds (default: 0.1)
                  Lower values result in faster animation.
    --runtime: Total runtime in seconds (default: infinite)
               If not provided, the animation runs until interrupted.

Character representation:
    - '#': Closest points to the viewpoint
    - 'O': Close points
    - '*': Medium-close points
    - 'o': Medium-far points
    - '+': Far points
    - '.': Furthest points from the viewpoint
"""

import numpy as np
import time
import os
import math
import argparse
import random
import sys
from typing import List, Tuple, Dict, Any

# ANSI color codes for terminal colors
COLORS = {
    'red': '\033[31m',
    'green': '\033[32m',
    'yellow': '\033[33m',
    'blue': '\033[34m',
    'magenta': '\033[35m',
    'cyan': '\033[36m',
    'white': '\033[37m',
    'bright_red': '\033[91m',
    'bright_green': '\033[92m',
    'bright_yellow': '\033[93m',
    'bright_blue': '\033[94m',
    'bright_magenta': '\033[95m',
    'bright_cyan': '\033[96m',
    'reset': '\033[0m'
}

# List of color names for random selection
COLOR_NAMES = list(COLORS.keys())
COLOR_NAMES.remove('reset')  # Don't use reset as a cube color

# Define the vertices of a cube centered at the origin
def create_cube(size: float = 2.0) -> np.ndarray:
    """Create vertices of a cube centered at the origin."""
    half_size = size / 2
    vertices = np.array([
        [-half_size, -half_size, -half_size],  # 0: back bottom left
        [half_size, -half_size, -half_size],   # 1: back bottom right
        [half_size, half_size, -half_size],    # 2: back top right
        [-half_size, half_size, -half_size],   # 3: back top left
        [-half_size, -half_size, half_size],   # 4: front bottom left
        [half_size, -half_size, half_size],    # 5: front bottom right
        [half_size, half_size, half_size],     # 6: front top right
        [-half_size, half_size, half_size]     # 7: front top left
    ])
    return vertices

# Define the edges of the cube (pairs of vertex indices)
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # back face
    (4, 5), (5, 6), (6, 7), (7, 4),  # front face
    (0, 4), (1, 5), (2, 6), (3, 7)   # connecting edges
]

def rotation_matrix(angle_x: float, angle_y: float, angle_z: float) -> np.ndarray:
    """Create a 3D rotation matrix for the given angles (in radians).

    This is an optimized version that calculates the combined rotation matrix directly
    instead of creating separate matrices and multiplying them.
    """
    # Precompute sine and cosine values
    cx, sx = np.cos(angle_x), np.sin(angle_x)
    cy, sy = np.cos(angle_y), np.sin(angle_y)
    cz, sz = np.cos(angle_z), np.sin(angle_z)

    # Calculate combined rotation matrix directly
    # This is equivalent to rz @ ry @ rx but more efficient
    return np.array([
        [cy*cz, -cy*sz, sy],
        [sx*sy*cz + cx*sz, -sx*sy*sz + cx*cz, -sx*cy],
        [-cx*sy*cz + sx*sz, cx*sy*sz + sx*cz, cx*cy]
    ])

def project_3d_to_2d(points: np.ndarray, distance: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D points to 2D using perspective projection.

    Returns:
        Tuple containing:
        - 2D projected points (x, y)
        - Z-coordinates for distance calculations
    """
    # Extract x, y, z coordinates from points
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Calculate projection factors for all points at once
    factors = distance / (distance + z)

    # Apply projection to all points at once
    px = x * factors
    py = y * factors

    # Stack x and y coordinates to create 2D points
    projected_points = np.column_stack((px, py))

    return projected_points, z

def render_cube(vertices_2d: np.ndarray, z_coords: np.ndarray, canvas: List[List[tuple]], width: int, height: int, offset_x: int = 0, color: str = None) -> None:
    """Render the cube as ASCII characters in a 2D grid.

    Args:
        vertices_2d: 2D projected vertices of the cube
        z_coords: Z-coordinates for depth information
        canvas: The canvas to draw on
        width: Width of the canvas
        height: Height of the canvas
        offset_x: Horizontal offset from the center (default: 0)
        color: Color name to use for this cube (default: None)
    """
    # Scale and center the projected points to fit the terminal
    scale = min(width, height) * 0.3
    center_x, center_y = width // 2 + offset_x, height // 2

    # Draw edges
    for start_idx, end_idx in EDGES:
        start = vertices_2d[start_idx]
        end = vertices_2d[end_idx]

        # Get z-coordinates for depth information
        z1 = z_coords[start_idx]
        z2 = z_coords[end_idx]

        # Scale and center the points
        x1, y1 = int(start[0] * scale + center_x), int(start[1] * scale + center_y)
        x2, y2 = int(end[0] * scale + center_x), int(end[1] * scale + center_y)

        # Draw a line using Bresenham's algorithm with depth information
        draw_line(canvas, x1, y1, x2, y2, z1, z2, color)

def draw_thick_point(canvas: List[List[tuple]], x: int, y: int, char: str, z: float, width: int, height: int, color: str = None) -> None:
    """Draw a single point with thickness by adding characters in adjacent positions.

    Args:
        canvas: The canvas to draw on (contains tuples of (character, depth, color))
        x: X-coordinate of the point
        y: Y-coordinate of the point
        char: Character to draw
        z: Depth value (smaller values are closer to the viewer)
        width: Width of the canvas
        height: Height of the canvas
        color: Color name to use for this point (default: None)
    """
    # Check if point is within canvas bounds
    if not (0 <= x < width and 0 <= y < height):
        return

    # Draw the center point if it's closer than what's already there
    current_char, current_z, current_color = canvas[y][x]
    if z < current_z:  # Only draw if this point is closer
        canvas[y][x] = (char, z, color)

        # Draw adjacent points (horizontal and vertical) if they're closer
        # Only process adjacent points if the center point was drawn
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                current_char, current_z, current_color = canvas[ny][nx]
                if z < current_z:  # Only draw if this point is closer
                    canvas[ny][nx] = (char, z, color)

def draw_line(canvas: List[List[tuple]], x1: int, y1: int, x2: int, y2: int, z1: float, z2: float, color: str = None) -> None:
    """Draw a line on the canvas using Bresenham's algorithm with depth-based characters.

    Uses different ASCII characters based on the distance to the viewpoint:
    - '#' for closest points
    - 'O' for close points
    - '*' for medium-close points
    - 'o' for medium-far points
    - '+' for far points
    - '.' for furthest points

    The edges are drawn with thickness by adding characters in adjacent positions.
    Points are only drawn if they're closer than what's already there (z-buffer approach).

    Args:
        canvas: The canvas to draw on
        x1, y1: Start point coordinates
        x2, y2: End point coordinates
        z1, z2: Depth values for start and end points
        color: Color name to use for this line (default: None)
    """
    # Get canvas dimensions once
    width = len(canvas[0])
    height = len(canvas)

    # Check if both points are outside the canvas bounds
    if ((x1 < 0 and x2 < 0) or (x1 >= width and x2 >= width) or 
        (y1 < 0 and y2 < 0) or (y1 >= height and y2 >= height)):
        return

    # Precompute character lookup based on z-coordinate
    def get_char(z):
        if z > 0.8:  # Furthest points
            return '°'
        elif z > 0.3:  # Far points
            return '+'
        elif z > -0.1:  # Medium-far points
            return 'o'
        elif z > -0.5:  # Medium-close points
            return '*'
        elif z > -0.9:  # Close points
            return 'O'
        else:  # Closest points
            return '#'

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    # Total steps for interpolation
    steps = max(dx, dy)
    if steps == 0:
        # If it's a single point, just draw it and return
        draw_thick_point(canvas, x1, y1, get_char(z1), z1, width, height, color)
        return

    # Current step for interpolation
    step = 0

    # Precompute step increment for z interpolation
    z_step = (z2 - z1) / steps if steps > 0 else 0
    z = z1

    while True:
        # Choose character based on z-coordinate (distance from viewpoint)
        char = get_char(z)

        # Draw the point with thickness, passing the z-coordinate and color for depth checking
        draw_thick_point(canvas, x1, y1, char, z, width, height, color)

        if x1 == x2 and y1 == y2:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

        # Increment z using precomputed step
        step += 1
        z = z1 + z_step * step

def position_cursor_at_top() -> None:
    """Position the cursor at the top of the screen and clear to the end of screen.

    This reduces flickering by not clearing the entire screen between frames.
    Instead, we just move the cursor to the top and clear from there to the end.
    """
    # Use ANSI escape codes:
    # \033[H - Move cursor to home position (top-left corner)
    # \033[2J - Clear entire screen
    # \033[3J - Clear scrollback buffer
    # \033[?25l - Hide cursor
    sys.stdout.write("\033[H\033[2J\033[3J\033[?25l")
    sys.stdout.flush()

def get_terminal_size() -> Tuple[int, int]:
    """Get the terminal size."""
    try:
        columns, lines = os.get_terminal_size()
        return columns, lines
    except (AttributeError, OSError):
        # Fallback if terminal size cannot be determined
        return 80, 24

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='3D ASCII Cube Animation')
    parser.add_argument('--sleep-time', type=float, default=0.1,
                        help='Sleep time between frames in seconds (default: 0.1)')
    parser.add_argument('--runtime', type=float, default=None,
                        help='Total runtime in seconds (default: infinite)')
    return parser.parse_args()

def calculate_cubes_count(width: int, height: int) -> int:
    """Calculate how many cubes can fit horizontally in the terminal.

    Args:
        width: Terminal width
        height: Terminal height

    Returns:
        Number of cubes that can fit horizontally
    """
    # Each cube needs approximately 30 columns of space
    # This is an estimate based on the scale factor and cube size
    cube_width = int(min(width, height) * 0.3 * 2.5)

    # Ensure at least 5 columns of space between cubes
    spacing = 5

    # Calculate how many cubes can fit
    if cube_width > 0:
        return max(1, (width + spacing) // (cube_width + spacing))
    return 1

def main() -> None:
    """Main function to run the 3D cube animation."""
    # Parse command line arguments
    args = parse_args()

    # Create cube vertices
    cube = create_cube(1.8)

    # Animation parameters
    rotation_speed = 0.05  # radians per frame

    # Initialize rotation angles for multiple cubes
    # We'll start with angles for one cube and add more as needed
    rotation_angles = [(0, 0, 0)]

    # Initialize cube colors
    cube_colors = [random.choice(COLOR_NAMES)]

    # Track start time if runtime is specified
    start_time = time.time() if args.runtime is not None else None

    # Switch to alternate screen buffer
    # This creates a separate screen that doesn't affect the main terminal
    sys.stdout.write("\033[?1049h")
    sys.stdout.flush()

    # Clear screen
    sys.stdout.write("\033[2J")
    sys.stdout.flush()

    try:
        # Initial setup
        width, height = get_terminal_size()
        num_cubes = calculate_cubes_count(width, height)

        # Ensure we have enough rotation angles for all cubes
        while len(rotation_angles) < num_cubes:
            # Add a new cube with slightly different starting angles for variety
            offset = len(rotation_angles) * 0.5
            rotation_angles.append((offset, offset, offset))

            # Assign a random color to the new cube
            # Make sure it's different from the previous cube's color
            if len(cube_colors) > 0:
                available_colors = [c for c in COLOR_NAMES if c != cube_colors[-1]]
                if not available_colors:  # If all colors are used, reset
                    available_colors = COLOR_NAMES
                cube_colors.append(random.choice(available_colors))
            else:
                cube_colors.append(random.choice(COLOR_NAMES))

        # Calculate the spacing between cube centers
        if num_cubes > 1:
            cube_spacing = width // num_cubes
        else:
            cube_spacing = 0

        # Pre-calculate the first frame
        canvas = [[(' ', float('inf'), None) for _ in range(width)] for _ in range(height)]

        # Render each cube for the first frame
        for i in range(num_cubes):
            # Get rotation angles for this cube
            angle_x, angle_y, angle_z = rotation_angles[i]

            # Create rotation matrix
            rot_matrix = rotation_matrix(angle_x, angle_y, angle_z)

            # Apply rotation to cube vertices
            rotated_cube = np.dot(cube, rot_matrix.T)

            # Project 3D points to 2D
            projected_cube, z_coords = project_3d_to_2d(rotated_cube)

            # Calculate horizontal offset for this cube
            if num_cubes > 1:
                offset_x = (i * cube_spacing) - (width // 2) + (cube_spacing // 2)
            else:
                offset_x = 0

            # Render cube directly onto the main canvas with its color
            render_cube(projected_cube, z_coords, canvas, width, height, offset_x, cube_colors[i])

        # Position cursor at top of screen (don't clear it)
        position_cursor_at_top()

        while True:
            # Check if runtime has expired
            if start_time is not None and time.time() - start_time >= args.runtime:
                # Position cursor at the bottom of the screen and print message
                sys.stdout.write(f"\033[{height};1HRuntime completed.")
                sys.stdout.flush()
                # Small delay to ensure message is visible
                time.sleep(0.5)
                break

            # Print the current frame
            # Build all lines first and then join them with newlines
            frame_lines = []
            for row in canvas:
                # Use a list to build each line for better performance
                line_parts = []
                for char, _, color in row:
                    if color:
                        line_parts.append(COLORS[color] + char + COLORS['reset'])
                    else:
                        line_parts.append(char)
                frame_lines.append(''.join(line_parts))

            # Ensure we have exactly 'height' lines
            if len(frame_lines) < height:
                # Pad with empty lines
                frame_lines.extend([""] * (height - len(frame_lines)))
            elif len(frame_lines) > height:
                # Truncate
                frame_lines = frame_lines[:height]

            # Join all lines with cursor positioning commands
            output = []
            for i, line in enumerate(frame_lines):
                # Position cursor at the beginning of line i+1 (1-indexed)
                output.append(f"\033[{i+1};1H{line}")

            # Write all lines at once and flush
            sys.stdout.write(''.join(output))
            sys.stdout.flush()

            # Start calculating the next frame while the current frame is being displayed
            # Get terminal size for the next frame
            new_width, new_height = get_terminal_size()

            # Check if terminal size has changed
            if new_width != width or new_height != height:
                width, height = new_width, new_height
                num_cubes = calculate_cubes_count(width, height)

                # Ensure we have enough rotation angles for all cubes
                while len(rotation_angles) < num_cubes:
                    # Add a new cube with slightly different starting angles for variety
                    offset = len(rotation_angles) * 0.5
                    rotation_angles.append((offset, offset, offset))

                    # Assign a random color to the new cube
                    if len(cube_colors) > 0:
                        available_colors = [c for c in COLOR_NAMES if c != cube_colors[-1]]
                        if not available_colors:  # If all colors are used, reset
                            available_colors = COLOR_NAMES
                        cube_colors.append(random.choice(available_colors))
                    else:
                        cube_colors.append(random.choice(COLOR_NAMES))

                # Remove extra colors if we have too many
                cube_colors = cube_colors[:num_cubes]

                # Calculate the spacing between cube centers
                if num_cubes > 1:
                    cube_spacing = width // num_cubes
                else:
                    cube_spacing = 0

            # Create an empty canvas for all cubes with infinite depth for the next frame
            next_canvas = [[(' ', float('inf'), None) for _ in range(width)] for _ in range(height)]

            # Render each cube for the next frame
            for i in range(num_cubes):
                # Get rotation angles for this cube
                angle_x, angle_y, angle_z = rotation_angles[i]

                # Update rotation angles for diagonal rotation
                angle_x += rotation_speed
                angle_y += rotation_speed
                angle_z += rotation_speed
                rotation_angles[i] = (angle_x, angle_y, angle_z)

                # Create rotation matrix
                rot_matrix = rotation_matrix(angle_x, angle_y, angle_z)

                # Apply rotation to cube vertices
                rotated_cube = np.dot(cube, rot_matrix.T)

                # Project 3D points to 2D
                projected_cube, z_coords = project_3d_to_2d(rotated_cube)

                # Calculate horizontal offset for this cube
                if num_cubes > 1:
                    offset_x = (i * cube_spacing) - (width // 2) + (cube_spacing // 2)
                else:
                    offset_x = 0

                # Render cube directly onto the next canvas with its color
                render_cube(projected_cube, z_coords, next_canvas, width, height, offset_x, cube_colors[i])

            # Wait before next frame
            time.sleep(args.sleep_time)

            # Swap canvases
            canvas = next_canvas

    except KeyboardInterrupt:
        # Show cursor again before exiting
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
        print("\nAnimation stopped.")
    finally:
        # Ensure cursor is visible even if another exception occurs
        sys.stdout.write("\033[?25h")

        # Restore main screen buffer
        sys.stdout.write("\033[?1049l")

        sys.stdout.flush()

if __name__ == "__main__":
    # import profile
    # profile.run('main()', sort='tottime')
    main()
