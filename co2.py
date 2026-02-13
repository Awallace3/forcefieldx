"""Visualize symmetry-unique dimers only (one per shell)."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.visualize import view

from space_groups import generate_molecular_dimers

# Path to CIF file
TEST_CIF_PATH = Path(
    "/home/awallace43/projects/x23_dmetcalf_2022_si/cifs/carbon_dioxide.cif"
)

# Generate unique molecular dimers
dimers, crystal, molecules = generate_molecular_dimers(
    TEST_CIF_PATH, radius=30.0
)

print(f"Found {len(dimers)} symmetry-unique dimers within 30 Å")
print("\nUnique dimers:")
for i, d in enumerate(dimers):
    print(f"  {i}: d={d.distance:.2f} Å, multiplicity={d.multiplicity}")

# Reference molecule
ref_mol = dimers[0].molecule_a

# Build visualization with unique dimers only
COLORS = plt.cm.tab10(np.linspace(0, 1, 30))

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot reference molecule in purple
PURPLE = '#800080'
for sym, pos in zip(ref_mol.symbols, ref_mol.cart_coords):
    size = 250 if sym == 'C' else 180
    ax.scatter(pos[0], pos[1], pos[2], c=PURPLE, s=size,
               edgecolors='white', linewidths=1.5, zorder=10)

# Draw bonds for reference
c_idx = ref_mol.symbols.index('C')
o_indices = [i for i, s in enumerate(ref_mol.symbols) if s == 'O']
coords = ref_mol.cart_coords
for o_idx in o_indices:
    ax.plot([coords[c_idx][0], coords[o_idx][0]],
            [coords[c_idx][1], coords[o_idx][1]],
            [coords[c_idx][2], coords[o_idx][2]],
            color=PURPLE, linewidth=4, zorder=10)

ax.scatter([], [], [], c=PURPLE, s=120, label='Reference')

# Plot each unique dimer partner with different colors
for i, dimer in enumerate(dimers):
    color = COLORS[i % len(COLORS)]
    mol_b = dimer.molecule_b

    for sym, pos in zip(mol_b.symbols, mol_b.cart_coords):
        size = 180 if sym == 'C' else 120
        ax.scatter(pos[0], pos[1], pos[2], c=[color], s=size,
                   alpha=0.85, edgecolors='gray', linewidths=0.5)

    # Draw bonds
    syms = mol_b.symbols
    coords = mol_b.cart_coords
    c_idx = syms.index('C')
    o_indices = [j for j, s in enumerate(syms) if s == 'O']
    for o_idx in o_indices:
        ax.plot([coords[c_idx][0], coords[o_idx][0]],
                [coords[c_idx][1], coords[o_idx][1]],
                [coords[c_idx][2], coords[o_idx][2]],
                color=color, linewidth=2.5, alpha=0.85)

    # Legend entry
    ax.scatter([], [], [], c=[color], s=100,
               label=f'd={dimer.distance:.2f}Å (×{dimer.multiplicity})')

ax.set_xlabel('X (Å)', fontsize=12)
ax.set_ylabel('Y (Å)', fontsize=12)
ax.set_zlabel('Z (Å)', fontsize=12)
ax.set_title(
    'CO₂ Crystal: Symmetry-Unique Dimers (10 Å)\n'
    '(one representative per shell, multiplicity shown)',
    fontsize=13
)
# ax.legend(loc='upper left', fontsize=9)

# Equal aspect ratio
all_coords = [ref_mol.cart_coords]
for d in dimers:
    all_coords.append(d.molecule_b.cart_coords)
all_pos = np.vstack(all_coords)
max_range = np.ptp(all_pos, axis=0).max() / 2 * 1.2
mid = all_pos.mean(axis=0)
ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

plt.tight_layout()
plt.savefig('unique_dimers.png', dpi=150, bbox_inches='tight')
print("\nSaved unique_dimers.png")

# Create ASE Atoms for GUI
all_symbols = list(ref_mol.symbols)
all_positions = list(ref_mol.cart_coords)

for dimer in dimers:
    all_symbols.extend(dimer.molecule_b.symbols)
    all_positions.extend(dimer.molecule_b.cart_coords)

atoms = Atoms(symbols=all_symbols, positions=all_positions)
print(f"\nTotal atoms: {len(atoms)} ({len(dimers) + 1} molecules)")

print("\nOpening ASE GUI...")
view(atoms)
