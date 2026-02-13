"""Visualize all dimer partners (with multiplicity) around a reference molecule.

Shows all symmetry-equivalent copies, not just unique representatives.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ase import Atoms
from ase.visualize import view

from space_groups import generate_molecular_dimers, parse_cif_file

# Path to CIF file
TEST_CIF_PATH = Path(
    "/home/awallace43/projects/x23_dmetcalf_2022_si/cifs/carbon_dioxide.cif"
)

# Generate molecular dimers - we need to get ALL copies, not just unique
# For this, we'll regenerate with the full expansion
from space_groups import _identify_molecules_in_cell

crystal, asu_symbols, asu_frac_coords = parse_cif_file(TEST_CIF_PATH)

# Get all molecules in the central cell
central_molecules = _identify_molecules_in_cell(
    crystal, asu_symbols, asu_frac_coords
)

print(f"Found {len(central_molecules)} molecules in unit cell")

# Reference molecule
ref_mol = central_molecules[0]
ref_centroid = ref_mol.centroid_cart

# Generate ALL neighbors within radius (not just unique)
radius = 10.0
n_a = int(np.ceil(radius / crystal.a)) + 1
n_b = int(np.ceil(radius / crystal.b)) + 1
n_c = int(np.ceil(radius / crystal.c)) + 1

all_neighbors = []

for i in range(-n_a, n_a + 1):
    for j in range(-n_b, n_b + 1):
        for k in range(-n_c, n_c + 1):
            translation = np.array([float(i), float(j), float(k)])

            for mol in central_molecules:
                # Skip reference molecule in central cell
                if (i, j, k) == (0, 0, 0) and mol.molecule_index == 0:
                    continue

                trans_frac = mol.frac_coords + translation
                trans_cart = crystal.to_cartesian(trans_frac)
                trans_centroid = np.mean(trans_cart, axis=0)

                dist = np.linalg.norm(trans_centroid - ref_centroid)
                if dist <= radius:
                    all_neighbors.append({
                        'symbols': mol.symbols,
                        'cart_coords': trans_cart,
                        'centroid': trans_centroid,
                        'distance': dist,
                    })

# Sort by distance
all_neighbors.sort(key=lambda x: x['distance'])

print(f"Found {len(all_neighbors)} total neighbors within {radius} Å")

# Group by distance for coloring
distance_groups = {}
tol = 0.05
for neighbor in all_neighbors:
    d = neighbor['distance']
    # Find matching group
    found = False
    for key in distance_groups:
        if abs(key - d) < tol:
            distance_groups[key].append(neighbor)
            found = True
            break
    if not found:
        distance_groups[d] = [neighbor]

print(f"\nDistance shells:")
for d in sorted(distance_groups.keys()):
    count = len(distance_groups[d])
    print(f"  {d:.2f} Å: {count} molecules")

# Build visualization
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))

fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# Plot reference molecule in black
for sym, pos in zip(ref_mol.symbols, ref_mol.cart_coords):
    size = 200 if sym == 'C' else 150
    ax.scatter(pos[0], pos[1], pos[2], c='black', s=size,
               edgecolors='white', linewidths=1, zorder=10)

# Draw bonds for reference
c_idx = ref_mol.symbols.index('C')
o_indices = [i for i, s in enumerate(ref_mol.symbols) if s == 'O']
coords = ref_mol.cart_coords
for o_idx in o_indices:
    ax.plot([coords[c_idx][0], coords[o_idx][0]],
            [coords[c_idx][1], coords[o_idx][1]],
            [coords[c_idx][2], coords[o_idx][2]],
            'k-', linewidth=3, zorder=10)

ax.scatter([], [], [], c='black', s=100, label='Reference')

# Plot each shell with different colors
for shell_idx, d in enumerate(sorted(distance_groups.keys())):
    neighbors = distance_groups[d]
    color = COLORS[shell_idx % len(COLORS)]

    for neighbor in neighbors:
        for sym, pos in zip(neighbor['symbols'], neighbor['cart_coords']):
            size = 120 if sym == 'C' else 80
            ax.scatter(pos[0], pos[1], pos[2], c=[color], s=size,
                       alpha=0.7, edgecolors='gray', linewidths=0.3)

        # Draw bonds
        syms = neighbor['symbols']
        coords = neighbor['cart_coords']
        c_idx = syms.index('C')
        o_indices = [i for i, s in enumerate(syms) if s == 'O']
        for o_idx in o_indices:
            ax.plot([coords[c_idx][0], coords[o_idx][0]],
                    [coords[c_idx][1], coords[o_idx][1]],
                    [coords[c_idx][2], coords[o_idx][2]],
                    color=color, linewidth=1.5, alpha=0.7)

    # Legend entry
    ax.scatter([], [], [], c=[color], s=80,
               label=f'd={d:.2f}Å (n={len(neighbors)})')

ax.set_xlabel('X (Å)', fontsize=12)
ax.set_ylabel('Y (Å)', fontsize=12)
ax.set_zlabel('Z (Å)', fontsize=12)
ax.set_title('CO₂ Crystal: All Neighbors within 10 Å\n(colored by distance shell)',
             fontsize=14)
ax.legend(loc='upper left', fontsize=9)

# Equal aspect ratio
all_pos = [ref_mol.cart_coords] + [n['cart_coords'] for n in all_neighbors]
all_pos = np.vstack(all_pos)
max_range = np.ptp(all_pos, axis=0).max() / 2 * 1.1
mid = all_pos.mean(axis=0)
ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

plt.tight_layout()
plt.savefig('all_neighbors_full.png', dpi=150, bbox_inches='tight')
print("\nSaved all_neighbors_full.png")

# Create ASE Atoms for GUI
all_symbols = list(ref_mol.symbols)
all_positions = list(ref_mol.cart_coords)

for neighbor in all_neighbors:
    all_symbols.extend(neighbor['symbols'])
    all_positions.extend(neighbor['cart_coords'])

atoms = Atoms(symbols=all_symbols, positions=all_positions)
print(f"\nTotal atoms for visualization: {len(atoms)}")

print("\nOpening ASE GUI...")
view(atoms)
