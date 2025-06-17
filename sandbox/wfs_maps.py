import numpy as np
import math
import matplotlib.pyplot as plt

def display_mask(mask, title="Mask Visualization", cmap='viridis'):
    """
    Display a mask using matplotlib's imshow
    
    Args:
        mask (numpy.ndarray): Boolean or integer mask to display
        title (str, optional): Title for the plot. Defaults to "Mask Visualization".
        cmap (str, optional): Colormap to use. Defaults to 'viridis'.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap=cmap)
    plt.title(title)
    plt.colorbar(label='Value')
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def display_numbered_mask(grid, outside_value=None, title="Numbered Mask", cmap='viridis', show_grid=True):
    """
    Display a numbered grid using matplotlib's imshow with annotations and optional grid lines
    
    Args:
        grid (numpy.ndarray): Numbered grid to display
        outside_value (any, optional): Value used for positions outside the mask. Defaults to None.
        title (str, optional): Title for the plot. Defaults to "Numbered Mask".
        cmap (str, optional): Colormap to use. Defaults to 'viridis'.
        show_grid (bool, optional): Whether to display grid lines. Defaults to True.
    """
    plt.figure(figsize=(10, 10))
    
    # Create a copy of the grid for visualization and ensure it's float type
    vis_grid = np.array(grid, dtype=float)
    
    # Handle outside values - convert to NaN for better visualization
    if outside_value is not None:
        # Create a mask of outside values
        mask = grid == outside_value
        # Set those positions to NaN
        vis_grid[mask] = np.nan
    
    # Display the grid
    img = plt.imshow(vis_grid, cmap=cmap)
    plt.title(title)
    plt.colorbar(img, label='Tile Number')
    
    # Add text annotations for tile numbers
    height, width = grid.shape
    for y in range(height):
        for x in range(width):
            # Check if this position is not an outside value before adding text
            value = grid[y, x]
            if value is not None and value != outside_value:
                # Handle potential type issues safely
                try:
                    text_value = str(int(value))
                except (TypeError, ValueError):
                    text_value = str(value)
                
                plt.text(x, y, text_value, 
                         ha="center", va="center", 
                         color="red", fontweight="bold")
    
    # Add grid lines if requested
    if show_grid:
        # Add grid lines at cell boundaries
        plt.grid(True, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Set ticks at the cell boundaries (with 0.5 offset for proper alignment)
        ax = plt.gca()
        ax.set_xticks(np.arange(-0.5, width, 1))
        ax.set_yticks(np.arange(-0.5, height, 1))
        
        # Hide tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Turn off minor ticks
        ax.minorticks_off()
    else:
        plt.grid(False)
    
    plt.tight_layout()
    plt.show()

def create_circular_mask(width, height=None, center_x=None, center_y=None, radius=None):
    """
    Creates a circular binary mask.
    
    Args:
        width (int): Width of the 2D array
        height (int, optional): Height of the 2D array. Defaults to width if not specified.
        center_x (float, optional): X coordinate of circle center. Defaults to width/2.
        center_y (float, optional): Y coordinate of circle center. Defaults to height/2.
        radius (float, optional): Radius of the circle. Defaults to min(width, height)/2.
        
    Returns:
        numpy.ndarray: 2D boolean array with True for positions inside the circle
    """
    # Set default values
    if height is None:
        height = width
    if center_x is None:
        center_x = width / 2
    if center_y is None:
        center_y = height / 2
    if radius is None:
        radius = min(width, height) / 2
    
    # Create a grid of coordinates
    y, x = np.ogrid[:height, :width]
    
    # Calculate distances from center for all positions
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Create the mask
    mask = dist_from_center <= radius
    
    return mask

def create_circular_tile_numbering(width, height=None, center_x=None, center_y=None, radius=None, outside_value=None, mask=None):
    """
    Creates a 2D array with sequential numbering for tiles within a circular mask.
    
    Args:
        width (int): Width of the 2D array (required if mask is not provided)
        height (int, optional): Height of the 2D array. Defaults to width if not specified.
        center_x (float, optional): X coordinate of circle center. Defaults to width/2.
        center_y (float, optional): Y coordinate of circle center. Defaults to height/2.
        radius (float, optional): Radius of the circle. Defaults to min(width, height)/2.
        outside_value (any, optional): Value for cells outside the circle. Defaults to None.
        mask (numpy.ndarray, optional): Pre-computed boolean mask with True for positions to number.
            If provided, all other shape/position parameters are ignored.
        
    Returns:
        numpy.ndarray: 2D array with numbered tiles inside circular mask
    """
    # If mask is provided, use it directly
    if mask is not None:
        height, width = mask.shape
        # Initialize the 2D array with outside_value
        grid = np.full((height, width), outside_value)
        
        # Counter for tile numbering
        tile_number = 1
        
        # Iterate through each cell in the grid
        for y in range(height):
            for x in range(width):
                # If inside the mask, assign the next tile number
                if mask[y, x]:
                    grid[y, x] = tile_number
                    tile_number += 1
        
        return grid
    
    # If no mask provided, generate one
    # Set default values
    if height is None:
        height = width
    if center_x is None:
        center_x = width / 2
    if center_y is None:
        center_y = height / 2
    if radius is None:
        radius = min(width, height) / 2
    
    # Create the circular mask
    mask = create_circular_mask(width, height, center_x, center_y, radius)
    
    # Initialize the 2D array with outside_value
    grid = np.full((height, width), outside_value)
    
    # Counter for tile numbering
    tile_number = 1
    
    # Iterate through each cell in the grid
    for y in range(height):
        for x in range(width):
            # If inside the circle, assign the next tile number
            if mask[y, x]:
                grid[y, x] = tile_number
                tile_number += 1
    
    return grid

def print_circular_grid(grid, outside_char=' '):
    """
    Print the tile numbering grid in a readable format.
    
    Args:
        grid (numpy.ndarray): The 2D array with tile numbering
        outside_char (str, optional): Character to display for cells outside the circle. Defaults to ' '.
    """
    for y in range(grid.shape[0]):
        row = ''
        for x in range(grid.shape[1]):
            if grid[y, x] is None or grid[y, x] == outside_char:
                row += f"{outside_char:>4}"
            else:
                row += f"{int(grid[y, x]):>4}"
        print(row)

def create_rowwise_circular_tile_numbering(width, height=None, center_x=None, center_y=None, radius=None, outside_value=None, mask=None):
    """
    Creates a 2D array with sequential row-by-row numbering for tiles within a circular mask.
    
    Args:
        width (int): Width of the 2D array (required if mask is not provided)
        height (int, optional): Height of the 2D array. Defaults to width if not specified.
        center_x (float, optional): X coordinate of circle center. Defaults to width/2.
        center_y (float, optional): Y coordinate of circle center. Defaults to height/2.
        radius (float, optional): Radius of the circle. Defaults to min(width, height)/2.
        outside_value (any, optional): Value for cells outside the circle. Defaults to None.
        mask (numpy.ndarray, optional): Pre-computed boolean mask with True for positions to number.
            If provided, all other shape/position parameters are ignored.
        
    Returns:
        numpy.ndarray: 2D array with numbered tiles inside circular mask in row-wise order
    """
    # If mask is provided, use it directly
    if mask is not None:
        height, width = mask.shape
        
        # Initialize grid with outside_value
        grid = np.full((height, width), outside_value)
        
        # Counter for tile numbering
        tile_number = 1
        
        # Iterate through each cell in row-wise order
        for y in range(height):
            for x in range(width):
                if mask[y, x]:
                    grid[y, x] = tile_number
                    tile_number += 1
        
        return grid
    
    # If no mask provided, generate one using the standard parameters
    # Set default values
    if height is None:
        height = width
    if center_x is None:
        center_x = width / 2
    if center_y is None:
        center_y = height / 2
    if radius is None:
        radius = min(width, height) / 2
    
    # Create the circular mask
    mask = create_circular_mask(width, height, center_x, center_y, radius)
    
    # Initialize grid with outside_value
    grid = np.full((height, width), outside_value)
    
    # Counter for tile numbering
    tile_number = 1
    
    # Iterate through each cell in row-wise order
    for y in range(height):
        for x in range(width):
            if mask[y, x]:
                grid[y, x] = tile_number
                tile_number += 1
    
    return grid

def extract_external_ring(mask, method='morphological'):
    """
    Extract the external ring of tiles from a circular mask.
    
    Args:
        mask (numpy.ndarray): Boolean mask with True for positions inside the circle
        method (str): Method to use for ring extraction:
                     'morphological' - Uses erosion to get exactly 1-tile thick ring (default)
                     'neighbor' - Uses neighbor checking (may produce thicker rings)
        
    Returns:
        numpy.ndarray: Boolean mask with True only for external ring positions
    """
    if method == 'morphological':
        return _extract_ring_morphological(mask)
    elif method == 'neighbor':
        return _extract_ring_neighbor(mask)
    else:
        raise ValueError("Method must be 'morphological' or 'neighbor'")

def _extract_ring_morphological(mask):
    """
    Extract external ring using morphological operations to ensure 1-tile thickness.
    """
    from scipy import ndimage
    
    # Create a structuring element for 4-connectivity (cross shape)
    # This helps ensure we get a clean 1-pixel boundary
    struct_4 = np.array([[0, 1, 0],
                         [1, 1, 1],
                         [0, 1, 0]], dtype=bool)
    
    # Alternative: 8-connectivity (square shape) - uncomment if preferred
    # struct_8 = np.ones((3, 3), dtype=bool)
    
    try:
        # Erode the mask by 1 pixel
        eroded_mask = ndimage.binary_erosion(mask, structure=struct_4)
        
        # The external ring is the difference between original and eroded mask
        external_ring = mask & ~eroded_mask
        
        return external_ring
    
    except ImportError:
        # Fallback to manual implementation if scipy is not available
        print("Warning: scipy not available, using manual morphological operations")
        return _extract_ring_manual_morphological(mask)

def _extract_ring_manual_morphological(mask):
    """
    Manual implementation of morphological ring extraction without scipy.
    """
    height, width = mask.shape
    eroded_mask = np.zeros_like(mask, dtype=bool)
    
    # Manual erosion with 4-connectivity
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if mask[y, x]:
                # Check 4-connected neighbors (up, down, left, right)
                if (mask[y-1, x] and mask[y+1, x] and 
                    mask[y, x-1] and mask[y, x+1]):
                    eroded_mask[y, x] = True
    
    # External ring is original minus eroded
    external_ring = mask & ~eroded_mask
    return external_ring

def _extract_ring_neighbor(mask):
    """
    Extract external ring using neighbor checking (original method).
    This may produce rings that are 2 tiles thick in some areas.
    """
    height, width = mask.shape
    external_ring = np.zeros_like(mask, dtype=bool)
    
    # Define 8-directional neighbors (including diagonals)
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    for y in range(height):
        for x in range(width):
            # If current position is inside the mask
            if mask[y, x]:
                # Check if it has at least one neighbor outside the mask
                is_external = False
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    # Check bounds and if neighbor is outside the mask
                    if (ny < 0 or ny >= height or nx < 0 or nx >= width or not mask[ny, nx]):
                        is_external = True
                        break
                
                if is_external:
                    external_ring[y, x] = True
    
    return external_ring

def get_external_ring_tiles(grid, mask=None, outside_value=None, method='morphological'):
    """
    Get the tile numbers and positions of the external ring from a numbered grid.
    
    Args:
        grid (numpy.ndarray): Numbered grid with tile numbers
        mask (numpy.ndarray, optional): Original circular mask. If not provided, 
                                       will be inferred from grid and outside_value
        outside_value (any, optional): Value used for positions outside the circle
        method (str): Method for ring extraction:
                     'morphological' - Ensures exactly 1-tile thick ring (default)
                     'neighbor' - May produce thicker rings in some areas
        
    Returns:
        tuple: (ring_positions, ring_numbers) where:
               - ring_positions is a list of (y, x) coordinates
               - ring_numbers is a list of corresponding tile numbers
    """
    # If mask is not provided, infer it from the grid
    if mask is None:
        if outside_value is not None:
            mask = grid != outside_value
        else:
            mask = grid != None
    
    # Extract the external ring mask
    ring_mask = extract_external_ring(mask, method=method)
    
    # Get positions and numbers
    ring_positions = []
    ring_numbers = []
    
    height, width = grid.shape
    for y in range(height):
        for x in range(width):
            if ring_mask[y, x]:
                ring_positions.append((y, x))
                ring_numbers.append(grid[y, x])
    
    return ring_positions, ring_numbers

def print_external_ring_info(ring_positions, ring_numbers):
    """
    Print information about the external ring tiles.
    
    Args:
        ring_positions (list): List of (y, x) coordinates
        ring_numbers (list): List of corresponding tile numbers
    """
    print(f"External Ring contains {len(ring_numbers)} tiles:")
    print("Position (y, x) -> Tile Number")
    print("=" * 30)
    for pos, num in zip(ring_positions, ring_numbers):
        print(f"({pos[0]:2d}, {pos[1]:2d})     ->     {num}")
    
    print(f"\nTile numbers in external ring: {sorted(ring_numbers)}")

def visualize_external_ring(grid, ring_positions=None, outside_value=None, title="External Ring Highlighted"):
    """
    Visualize the external ring by highlighting it in the grid.
    
    Args:
        grid (numpy.ndarray): Original numbered grid
        ring_positions (list): List of (y, x) coordinates for ring positions
        outside_value (any, optional): Value used for positions outside the circle
        title (str, optional): Title for the plot
    """
    # Create a copy of the grid for visualization
    vis_grid = np.array(grid, dtype=float)
    
    # Handle outside values - convert to NaN for better visualization
    if outside_value is not None:
        mask = grid == outside_value
        vis_grid[mask] = np.nan
    
    plt.figure(figsize=(12, 10))
    
    # Display the grid
    img = plt.imshow(vis_grid, cmap='Spectral')
    plt.title(title)
    #plt.colorbar(img, label='Actuator Number')
    
    # Add text annotations for all tile numbers
    height, width = grid.shape
    for y in range(height):
        for x in range(width):
            value = grid[y, x]
            if value is not None and value != outside_value:
                try:
                    text_value = str(int(value))
                except (TypeError, ValueError):
                    text_value = str(value)
                
                if ring_positions is not None:
                    # Check if this position is in the ring positions
                    if (y, x) in ring_positions:
                        color = "black"
                        fontweight = "bold"
                        fontsize = 12
                        circle = plt.Circle((x, y), 0.4, color='black', fill=False, linewidth=3)
                        plt.gca().add_patch(circle)
                    else:
                        color = "black"
                        fontweight = "bold"
                        fontsize = 12
                else:
                    color = "black"
                    fontweight = "bold"
                    fontsize = 12
                
                plt.text(x, y, text_value, 
                         ha="center", va="center", 
                         color=color, fontweight=fontweight, fontsize=fontsize)
    
    # add legend for external ring
    if ring_positions is not None:
        # draw a circle for the external ring
        circle = plt.Circle((1.5, 2), 0.4, color='black', fill=False, linewidth=3)
        plt.gca().add_patch(circle)
        plt.text(1.5, 1, "Masked Actuators", color='black', fontsize=12, ha='center', va='center', fontweight='bold')
    # add ring as part of the legend
    # Add grid lines
    plt.grid(True, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, width, 1))
    ax.set_yticks(np.arange(-0.5, height, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.minorticks_off()
    
    plt.tight_layout()
    plt.show()
# Example usage
if __name__ == "__main__":
    from pyTomoAO.tomographicReconstructor import tomographicReconstructor
    rec = tomographicReconstructor("../examples/benchmark/reconstructor_config_revolt.yaml")
    dmMask = rec.dmParams.validActuators
    wfsMask = rec.lgsWfsParams.validLLMap
    
    dmMap = create_circular_tile_numbering(width=None, height=None, mask=dmMask)
    wfsMap = create_circular_tile_numbering(width=None, height=None, mask=wfsMask)
    print("DM Map:")
    print_circular_grid(dmMap, '·')
    print("WFS Map:")
    print_circular_grid(wfsMap, '·')
    # Extract external ring information
    ring_positions, ring_numbers = get_external_ring_tiles(dmMap, dmMask, outside_value=None)
    
    # Visualize the external ring
    print("\nVisualizing external ring (red circles and bold red numbers):")
    visualize_external_ring(dmMap, ring_positions, outside_value=None, title="DM Actuators Map")
        # Visualize the external ring
    print("\nVisualizing external ring (red circles and bold red numbers):")
    visualize_external_ring(wfsMap, outside_value=None, title="WFS Actuators Map")
    
    # work on the reconstructor
    rec.build_reconstructor(alpha=10)
    rec.assemble_reconstructor_and_fitting(nChannels=1, slopesOrder="keck",
                                        scalingFactor=2.3e5, stretch_factor=1.1, 
                                        rotation=1, flip=1)
    
    rec.mask_DM_actuators(np.array(ring_numbers)-1)
    
    FR = rec.FR
    
    plt.imshow(FR)