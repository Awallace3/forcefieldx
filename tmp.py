import numpy as np
import pytest
from pathlib import Path

from space_groups import (
    SymOp,
    Crystal,
    Monomer,
    DimerPair,
    MolecularCluster,
    MolecularDimer,
    parse_symop_xyz,
    parse_cif_file,
    cif_to_molecule,
    get_crystal_info,
    remove_duplicate_atoms,
    generate_unique_dimers,
    generate_molecular_dimers,
    get_dimer_molecules,
    _parse_fraction,
    _remove_duplicate_monomers,
    _build_bond_graph,
    _find_connected_components,
    _identify_molecules_in_cell,
)

import qcelemental as qcel

TEST_CIF_PATH = Path("/home/awallace43/projects/x23_dmetcalf_2022_si/cifs/carbon_dioxide.cif")
dimers = get_dimer_molecules(TEST_CIF_PATH, radius=30.0)
print(dimers)
print(len(dimers))
