"""Visualize molecular dimers from CO2 crystal using matplotlib."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from space_groups import generate_molecular_dimers

# Path to CIF file
TEST_CIF_PATH = Path(
    "/home/awallace43/projects/x23_dmetcalf_2022_si/cifs/carbon_dioxide.cif"
)

# Element colors
COLORS = {'C': 'gray', 'O': 'red', 'N': 'blue', 'H': 'white', 'S': 'yellow'}
SIZES = {'C': 200, 'O': 180, 'N': 180, 'H': 80, 'S': 220}


def plot_dimer(dimer, ax, title=""):
    """Plot a molecular dimer in 3D."""
    ax.clear()
    
    # Get coordinates and symbols for both molecules
    for mol, alpha in [(dimer.molecule_a, 1.0), (dimer.molecule_b, 0.8)]:
        coords = mol.cart_coords
        symbols = mol.symbols
        
        for i, (coord, sym) in enumerate(zip(coords, symbols)):
            color = COLORS.get(sym, 'purple')
            size = SIZES.get(sym, 150)
            ax.scatter(
                coord[0], coord[1], coord[2],
                c=color, s=size, alpha=alpha, edgecolors='black'
            )
    
    # Draw bonds within each molecule (C-O bonds for CO2)
    for mol in [dimer.molecule_a, dimer.molecule_b]:
        coords = mol.cart_coords
        symbols = mol.symbols
        c_idx = symbols.index('C')
        o_indices = [i for i, s in enumerate(symbols) if s == 'O']
        
        for o_idx in o_indices:
            ax.plot(
                [coords[c_idx][0], coords[o_idx][0]],
                [coords[c_idx][1], coords[o_idx][1]],
                [coords[c_idx][2], coords[o_idx][2]],
                'k-', linewidth=2
            )
    
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(title)
    
    # Equal aspect ratio
    all_coords = np.vstack([dimer.molecule_a.cart_coords, 
                           dimer.molecule_b.cart_coords])
    max_range = np.ptp(all_coords, axis=0).max() / 2
    mid = all_coords.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


# Generate molecular dimers
dimers, crystal, molecules = generate_molecular_dimers(
    TEST_CIF_PATH, radius=10.0
)

print(f"Found {len(dimers)} unique dimers within 10 Å")
print()

# Create figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Save each dimer as an image
for i, dimer in enumerate(dimers):
    title = f"Dimer {i}: d={dimer.distance:.2f} Å, mult={dimer.multiplicity}"
    plot_dimer(dimer, ax, title)
    
    filename = f"dimer_{i:02d}_{dimer.distance:.2f}A.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved {filename}")

plt.close()

# Also create a summary figure with all dimers
n_dimers = len(dimers)
cols = min(3, n_dimers)
rows = (n_dimers + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows),
                         subplot_kw={'projection': '3d'})
if n_dimers == 1:
    axes = np.array([[axes]])
elif rows == 1:
    axes = axes.reshape(1, -1)

for i, dimer in enumerate(dimers):
    row, col = i // cols, i % cols
    ax = axes[row, col]
    title = f"d={dimer.distance:.2f}Å, m={dimer.multiplicity}"
    plot_dimer(dimer, ax, title)

# Hide empty subplots
for i in range(n_dimers, rows * cols):
    row, col = i // cols, i % cols
    axes[row, col].set_visible(False)

plt.tight_layout()
plt.savefig('all_dimers_summary.png', dpi=150, bbox_inches='tight')
print("\nSaved all_dimers_summary.png")
