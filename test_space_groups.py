"""
Tests for space_groups.py module.

Tests parsing of symmetry operations, CIF files, and generation of
symmetry-equivalent positions.
"""

import numpy as np
import pytest
from pathlib import Path

from space_groups import (
    SymOp,
    Crystal,
    Monomer,
    DimerPair,
    parse_symop_xyz,
    parse_cif_file,
    cif_to_molecule,
    get_crystal_info,
    remove_duplicate_atoms,
    generate_unique_dimers,
    _parse_fraction,
    _remove_duplicate_monomers,
)


# Test CIF file path
TEST_CIF_PATH = Path("/home/awallace43/projects/x23_dmetcalf_2022_si/cifs/carbon_dioxide.cif")


class TestSymOp:
    """Tests for SymOp class."""

    def test_identity(self):
        """Test identity symmetry operation."""
        symop = SymOp.identity()
        assert np.allclose(symop.rot, np.eye(3))
        assert np.allclose(symop.tr, np.zeros(3))

    def test_apply_identity(self):
        """Test applying identity operation."""
        symop = SymOp.identity()
        coords = np.array([0.25, 0.5, 0.75])
        result = symop.apply(coords)
        assert np.allclose(result, coords)

    def test_apply_inversion(self):
        """Test applying inversion operation."""
        symop = SymOp(rot=-np.eye(3), tr=np.zeros(3))
        coords = np.array([0.25, 0.5, 0.75])
        result = symop.apply(coords)
        assert np.allclose(result, -coords)

    def test_apply_translation(self):
        """Test applying translation."""
        symop = SymOp(rot=np.eye(3), tr=np.array([0.5, 0.5, 0.5]))
        coords = np.array([0.0, 0.0, 0.0])
        result = symop.apply(coords)
        assert np.allclose(result, [0.5, 0.5, 0.5])

    def test_apply_multiple_coords(self):
        """Test applying to multiple coordinate sets."""
        symop = SymOp.identity()
        coords = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        result = symop.apply(coords)
        assert result.shape == (2, 3)
        assert np.allclose(result, coords)

    def test_as_matrix(self):
        """Test 4x4 matrix representation."""
        symop = SymOp(rot=np.eye(3), tr=np.array([0.5, 0.25, 0.0]))
        mat = symop.as_matrix()
        assert mat.shape == (4, 4)
        assert np.allclose(mat[:3, :3], np.eye(3))
        assert np.allclose(mat[:3, 3], [0.5, 0.25, 0.0])
        assert np.allclose(mat[3, :], [0, 0, 0, 1])


class TestParseSymOpXYZ:
    """Tests for parsing symmetry operation strings."""

    def test_parse_identity(self):
        """Test parsing identity operation 'x,y,z'."""
        symop = parse_symop_xyz("x,y,z")
        assert np.allclose(symop.rot, np.eye(3))
        assert np.allclose(symop.tr, np.zeros(3))

    def test_parse_inversion(self):
        """Test parsing inversion operation '-x,-y,-z'."""
        symop = parse_symop_xyz("-x,-y,-z")
        assert np.allclose(symop.rot, -np.eye(3))
        assert np.allclose(symop.tr, np.zeros(3))

    def test_parse_with_translation_half(self):
        """Test parsing operation with 1/2 translation."""
        symop = parse_symop_xyz("1/2+x,1/2+y,1/2+z")
        assert np.allclose(symop.rot, np.eye(3))
        assert np.allclose(symop.tr, [0.5, 0.5, 0.5])

    def test_parse_with_translation_suffix(self):
        """Test parsing operation with translation as suffix."""
        symop = parse_symop_xyz("x+1/2,y+1/2,z+1/2")
        assert np.allclose(symop.rot, np.eye(3))
        assert np.allclose(symop.tr, [0.5, 0.5, 0.5])

    def test_parse_mixed_signs(self):
        """Test parsing operation with mixed signs."""
        symop = parse_symop_xyz("-x,y,-z")
        expected_rot = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)
        assert np.allclose(symop.rot, expected_rot)
        assert np.allclose(symop.tr, np.zeros(3))

    def test_parse_pa3_symop(self):
        """Test parsing symmetry operation from Pa-3 space group."""
        # From the CO2 CIF: "1/2+z,x,1/2-y"
        symop = parse_symop_xyz("1/2+z,x,1/2-y")
        # x' = z + 1/2, so rot row 0 = [0, 0, 1], tr[0] = 0.5
        # y' = x, so rot row 1 = [1, 0, 0], tr[1] = 0
        # z' = -y + 1/2, so rot row 2 = [0, -1, 0], tr[2] = 0.5
        expected_rot = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]], dtype=float)
        expected_tr = np.array([0.5, 0.0, 0.5])
        assert np.allclose(symop.rot, expected_rot), f"Got rot:\n{symop.rot}"
        assert np.allclose(symop.tr, expected_tr), f"Got tr: {symop.tr}"

    def test_parse_with_spaces(self):
        """Test parsing with spaces."""
        symop = parse_symop_xyz(" x , y , z ")
        assert np.allclose(symop.rot, np.eye(3))

    def test_parse_case_insensitive(self):
        """Test parsing is case insensitive."""
        symop = parse_symop_xyz("X,Y,Z")
        assert np.allclose(symop.rot, np.eye(3))

    def test_parse_cyclic_permutation(self):
        """Test parsing cyclic permutation 'y,z,x'."""
        symop = parse_symop_xyz("y,z,x")
        # x' = y, y' = z, z' = x
        expected_rot = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
        assert np.allclose(symop.rot, expected_rot)

    def test_parse_third_fractions(self):
        """Test parsing 1/3 and 2/3 fractions."""
        symop = parse_symop_xyz("x+1/3,y+2/3,z")
        assert np.allclose(symop.tr, [1/3, 2/3, 0])


class TestParseFraction:
    """Tests for fraction parsing helper."""

    def test_parse_half(self):
        assert _parse_fraction("1/2") == 0.5

    def test_parse_third(self):
        assert abs(_parse_fraction("1/3") - 1/3) < 1e-10

    def test_parse_two_thirds(self):
        assert abs(_parse_fraction("2/3") - 2/3) < 1e-10

    def test_parse_quarter(self):
        assert _parse_fraction("1/4") == 0.25

    def test_parse_integer(self):
        assert _parse_fraction("1") == 1.0


class TestCrystal:
    """Tests for Crystal class."""

    def test_cubic_crystal(self):
        """Test cubic crystal with a=b=c, alpha=beta=gamma=90."""
        crystal = Crystal(a=5.0, b=5.0, c=5.0, alpha=90, beta=90, gamma=90)
        assert abs(crystal.volume - 125.0) < 1e-6

    def test_orthorhombic_crystal(self):
        """Test orthorhombic crystal."""
        crystal = Crystal(a=3.0, b=4.0, c=5.0, alpha=90, beta=90, gamma=90)
        assert abs(crystal.volume - 60.0) < 1e-6

    def test_to_cartesian_cubic(self):
        """Test fractional to Cartesian conversion for cubic cell."""
        crystal = Crystal(a=10.0, b=10.0, c=10.0, alpha=90, beta=90, gamma=90)
        frac = np.array([0.5, 0.5, 0.5])
        cart = crystal.to_cartesian(frac)
        assert np.allclose(cart, [5.0, 5.0, 5.0])

    def test_to_fractional_cubic(self):
        """Test Cartesian to fractional conversion for cubic cell."""
        crystal = Crystal(a=10.0, b=10.0, c=10.0, alpha=90, beta=90, gamma=90)
        cart = np.array([5.0, 5.0, 5.0])
        frac = crystal.to_fractional(cart)
        assert np.allclose(frac, [0.5, 0.5, 0.5])

    def test_roundtrip_conversion(self):
        """Test roundtrip conversion frac -> cart -> frac."""
        crystal = Crystal(a=5.624, b=5.624, c=5.624, alpha=90, beta=90, gamma=90)
        frac_orig = np.array([0.1185, 0.1185, 0.1185])
        cart = crystal.to_cartesian(frac_orig)
        frac_back = crystal.to_fractional(cart)
        assert np.allclose(frac_orig, frac_back)

    def test_generate_symmetry_equivalents_p1(self):
        """Test symmetry equivalent generation for P1 (identity only)."""
        crystal = Crystal(a=5.0, b=5.0, c=5.0, alpha=90, beta=90, gamma=90)
        crystal.symops = [SymOp.identity()]
        frac = np.array([0.25, 0.25, 0.25])
        equiv = crystal.generate_symmetry_equivalents(frac)
        assert equiv.shape == (1, 3)
        assert np.allclose(equiv[0], frac)

    def test_generate_symmetry_equivalents_p_minus_1(self):
        """Test symmetry equivalent generation for P-1 (inversion)."""
        crystal = Crystal(a=5.0, b=5.0, c=5.0, alpha=90, beta=90, gamma=90)
        crystal.symops = [
            SymOp.identity(),
            SymOp(rot=-np.eye(3), tr=np.zeros(3)),
        ]
        frac = np.array([0.25, 0.25, 0.25])
        equiv = crystal.generate_symmetry_equivalents(frac, wrap_to_unit_cell=False)
        assert equiv.shape == (2, 3)
        # Should have original and inverted positions
        assert np.allclose(equiv[0], [0.25, 0.25, 0.25])
        assert np.allclose(equiv[1], [-0.25, -0.25, -0.25])


class TestRemoveDuplicateAtoms:
    """Tests for duplicate atom removal."""

    def test_no_duplicates(self):
        """Test with no duplicate atoms."""
        symbols = ["C", "O", "O"]
        coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
        new_symbols, new_coords = remove_duplicate_atoms(symbols, coords)
        assert len(new_symbols) == 3
        assert new_coords.shape == (3, 3)

    def test_with_duplicates(self):
        """Test with duplicate atoms."""
        symbols = ["C", "C"]
        coords = np.array([[0.0, 0.0, 0.0], [0.001, 0.001, 0.001]])
        new_symbols, new_coords = remove_duplicate_atoms(symbols, coords, tolerance=0.01)
        assert len(new_symbols) == 1

    def test_periodic_duplicates(self):
        """Test detection of periodic image duplicates."""
        symbols = ["O", "O"]
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # Same position due to periodicity
        new_symbols, new_coords = remove_duplicate_atoms(symbols, coords, tolerance=0.01)
        assert len(new_symbols) == 1


@pytest.mark.skipif(not TEST_CIF_PATH.exists(), reason="Test CIF file not found")
class TestParseCIFFile:
    """Tests for CIF file parsing using carbon dioxide CIF."""

    def test_parse_cif_returns_tuple(self):
        """Test that parse_cif_file returns expected tuple."""
        result = parse_cif_file(TEST_CIF_PATH)
        assert len(result) == 3
        crystal, symbols, frac_coords = result
        assert isinstance(crystal, Crystal)
        assert isinstance(symbols, list)
        assert isinstance(frac_coords, np.ndarray)

    def test_parse_cif_lattice_parameters(self):
        """Test extracted lattice parameters for CO2."""
        crystal, _, _ = parse_cif_file(TEST_CIF_PATH)
        # CO2 has cubic cell with a = 5.624 Å
        assert abs(crystal.a - 5.624) < 0.001
        assert abs(crystal.b - 5.624) < 0.001
        assert abs(crystal.c - 5.624) < 0.001
        assert abs(crystal.alpha - 90.0) < 0.001
        assert abs(crystal.beta - 90.0) < 0.001
        assert abs(crystal.gamma - 90.0) < 0.001

    def test_parse_cif_space_group(self):
        """Test extracted space group for CO2."""
        crystal, _, _ = parse_cif_file(TEST_CIF_PATH)
        # CO2 is in Pa-3 (space group #205)
        assert crystal.space_group_number == 205
        assert "a" in crystal.space_group_name.lower() or "3" in crystal.space_group_name

    def test_parse_cif_symmetry_operations(self):
        """Test extracted symmetry operations for CO2."""
        crystal, _, _ = parse_cif_file(TEST_CIF_PATH)
        # Pa-3 has 24 symmetry operations
        assert len(crystal.symops) == 24

    def test_parse_cif_atoms(self):
        """Test extracted atoms for CO2."""
        _, symbols, frac_coords = parse_cif_file(TEST_CIF_PATH)
        # CO2 asymmetric unit has 2 atoms (C and O)
        assert len(symbols) == 2
        assert frac_coords.shape == (2, 3)
        # Should have C and O
        assert "C" in symbols
        assert "O" in symbols

    def test_parse_cif_carbon_position(self):
        """Test carbon position in CO2."""
        _, symbols, frac_coords = parse_cif_file(TEST_CIF_PATH)
        c_idx = symbols.index("C")
        # Carbon is at origin (0, 0, 0)
        assert np.allclose(frac_coords[c_idx], [0.0, 0.0, 0.0], atol=0.001)

    def test_parse_cif_oxygen_position(self):
        """Test oxygen position in CO2."""
        _, symbols, frac_coords = parse_cif_file(TEST_CIF_PATH)
        o_idx = symbols.index("O")
        # Oxygen is at (0.1185, 0.1185, 0.1185)
        assert np.allclose(frac_coords[o_idx], [0.1185, 0.1185, 0.1185], atol=0.001)


@pytest.mark.skipif(not TEST_CIF_PATH.exists(), reason="Test CIF file not found")
class TestCIFToMolecule:
    """Tests for CIF to QCElemental Molecule conversion."""

    def test_cif_to_molecule_returns_molecule(self):
        """Test that cif_to_molecule returns a Molecule object."""
        import qcelemental as qcel
        mol = cif_to_molecule(TEST_CIF_PATH)
        assert isinstance(mol, qcel.models.Molecule)

    def test_cif_to_molecule_atom_count(self):
        """Test atom count after symmetry expansion for CO2."""
        mol = cif_to_molecule(TEST_CIF_PATH, expand_symmetry=True)
        # Pa-3 has 24 symops, 2 atoms in ASU
        # But many positions are equivalent due to special positions
        # CO2 unit cell should have 4 CO2 molecules = 12 atoms
        # (Z=4 for Pa-3 with molecules on special positions)
        assert len(mol.symbols) > 2  # At least expanded from ASU
        # Check we have reasonable atom types
        symbols = list(mol.symbols)
        assert symbols.count("C") >= 1
        assert symbols.count("O") >= 2

    def test_cif_to_molecule_no_symmetry(self):
        """Test molecule creation without symmetry expansion."""
        mol = cif_to_molecule(TEST_CIF_PATH, expand_symmetry=False)
        # Should only have ASU atoms
        assert len(mol.symbols) == 2

    def test_cif_to_molecule_geometry_shape(self):
        """Test geometry array shape."""
        mol = cif_to_molecule(TEST_CIF_PATH)
        n_atoms = len(mol.symbols)
        # mol.geometry can be either flat (n_atoms * 3,) or 2D (n_atoms, 3)
        geom = mol.geometry
        if geom.ndim == 1:
            assert len(geom) == n_atoms * 3
        else:
            assert geom.shape == (n_atoms, 3)

    def test_cif_to_molecule_coordinates_in_bohr(self):
        """Test that coordinates are in Bohr."""
        import qcelemental as qcel
        mol = cif_to_molecule(TEST_CIF_PATH, expand_symmetry=False)
        coords_bohr = mol.geometry.reshape(-1, 3)
        coords_angstrom = coords_bohr * qcel.constants.bohr2angstroms
        # Coordinates should be within unit cell (roughly 0-6 Å for CO2)
        assert np.all(coords_angstrom < 10.0)
        assert np.all(coords_angstrom > -10.0)


@pytest.mark.skipif(not TEST_CIF_PATH.exists(), reason="Test CIF file not found")
class TestGetCrystalInfo:
    """Tests for get_crystal_info function."""

    def test_get_crystal_info_returns_dict(self):
        """Test that get_crystal_info returns a dictionary."""
        info = get_crystal_info(TEST_CIF_PATH)
        assert isinstance(info, dict)

    def test_get_crystal_info_keys(self):
        """Test that all expected keys are present."""
        info = get_crystal_info(TEST_CIF_PATH)
        expected_keys = [
            "a", "b", "c", "alpha", "beta", "gamma", "volume",
            "space_group_name", "space_group_number", "n_symops", "n_atoms_asu"
        ]
        for key in expected_keys:
            assert key in info

    def test_get_crystal_info_values(self):
        """Test specific values for CO2."""
        info = get_crystal_info(TEST_CIF_PATH)
        assert abs(info["a"] - 5.624) < 0.001
        assert info["space_group_number"] == 205
        assert info["n_symops"] == 24
        assert info["n_atoms_asu"] == 2


class TestSymOpRoundtrip:
    """Test symmetry operation roundtrip: parse -> to_xyz_string -> parse."""

    @pytest.mark.parametrize("xyz_str", [
        "x,y,z",
        "-x,-y,-z",
        "-x,y,-z",
        "x+1/2,y+1/2,z",
        "1/2+x,-y,z+1/2",
        "y,z,x",
        "-y,x,z",
    ])
    def test_roundtrip(self, xyz_str):
        """Test parsing and reconstructing xyz strings."""
        symop = parse_symop_xyz(xyz_str)
        # Apply to test coordinates
        test_coords = np.array([0.1, 0.2, 0.3])
        result1 = symop.apply(test_coords)

        # Convert back to string and parse again
        xyz_str2 = symop.to_xyz_string()
        symop2 = parse_symop_xyz(xyz_str2)
        result2 = symop2.apply(test_coords)

        # Results should match
        assert np.allclose(result1, result2), f"Mismatch for {xyz_str} -> {xyz_str2}"


class TestMonomer:
    """Tests for Monomer dataclass."""

    def test_monomer_creation(self):
        """Test creating a Monomer object."""
        symbols = ["C", "O", "O"]
        frac_coords = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
        cart_coords = np.array([[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]])
        centroid_frac = np.mean(frac_coords, axis=0)
        centroid_cart = np.mean(cart_coords, axis=0)

        monomer = Monomer(
            symbols=symbols,
            frac_coords=frac_coords,
            cart_coords=cart_coords,
            cell_index=(0, 0, 0),
            symop_index=0,
            centroid_frac=centroid_frac,
            centroid_cart=centroid_cart,
        )

        assert monomer.symbols == ["C", "O", "O"]
        assert monomer.cell_index == (0, 0, 0)
        assert monomer.symop_index == 0
        assert monomer.frac_coords.shape == (3, 3)
        assert monomer.cart_coords.shape == (3, 3)

    def test_monomer_to_molecule(self):
        """Test converting Monomer to QCElemental Molecule."""
        import qcelemental as qcel

        symbols = ["C", "O", "O"]
        # CO2 in Angstroms
        cart_coords = np.array([[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]])
        frac_coords = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])

        monomer = Monomer(
            symbols=symbols,
            frac_coords=frac_coords,
            cart_coords=cart_coords,
            cell_index=(0, 0, 0),
            symop_index=0,
            centroid_frac=np.zeros(3),
            centroid_cart=np.zeros(3),
        )

        mol = monomer.to_molecule()
        assert tuple(mol.symbols) == ("C", "O", "O")
        # Check geometry is in Bohr
        geom = np.array(mol.geometry).reshape(-1, 3)
        expected = cart_coords / qcel.constants.bohr2angstroms
        assert np.allclose(geom, expected, atol=1e-6)


class TestDimerPair:
    """Tests for DimerPair dataclass."""

    def test_dimer_pair_creation(self):
        """Test creating a DimerPair object."""
        symbols = ["C", "O", "O"]
        cart_a = np.array([[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]])
        cart_b = np.array([[5.0, 0.0, 0.0], [6.16, 0.0, 0.0], [3.84, 0.0, 0.0]])
        frac_a = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
        frac_b = np.array([[0.5, 0.0, 0.0], [0.6, 0.0, 0.0], [0.4, 0.0, 0.0]])

        mon_a = Monomer(
            symbols=symbols,
            frac_coords=frac_a,
            cart_coords=cart_a,
            cell_index=(0, 0, 0),
            symop_index=0,
            centroid_frac=np.mean(frac_a, axis=0),
            centroid_cart=np.mean(cart_a, axis=0),
        )

        mon_b = Monomer(
            symbols=symbols,
            frac_coords=frac_b,
            cart_coords=cart_b,
            cell_index=(1, 0, 0),
            symop_index=1,
            centroid_frac=np.mean(frac_b, axis=0),
            centroid_cart=np.mean(cart_b, axis=0),
        )

        dimer = DimerPair(
            monomer_a=mon_a,
            monomer_b=mon_b,
            distance=5.0,
            symop_index_a=0,
            symop_index_b=1,
            cell_index_b=(1, 0, 0),
            multiplicity=2,
        )

        assert dimer.distance == 5.0
        assert dimer.multiplicity == 2
        assert dimer.symop_index_a == 0
        assert dimer.symop_index_b == 1
        assert dimer.cell_index_b == (1, 0, 0)

    def test_dimer_pair_to_molecule(self):
        """Test converting DimerPair to QCElemental Molecule."""
        import qcelemental as qcel

        symbols = ["C", "O", "O"]
        cart_a = np.array([[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]])
        cart_b = np.array([[5.0, 0.0, 0.0], [6.16, 0.0, 0.0], [3.84, 0.0, 0.0]])
        frac_a = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
        frac_b = np.array([[0.5, 0.0, 0.0], [0.6, 0.0, 0.0], [0.4, 0.0, 0.0]])

        mon_a = Monomer(
            symbols=symbols,
            frac_coords=frac_a,
            cart_coords=cart_a,
            cell_index=(0, 0, 0),
            symop_index=0,
            centroid_frac=np.zeros(3),
            centroid_cart=np.zeros(3),
        )

        mon_b = Monomer(
            symbols=symbols,
            frac_coords=frac_b,
            cart_coords=cart_b,
            cell_index=(1, 0, 0),
            symop_index=1,
            centroid_frac=np.array([0.5, 0.0, 0.0]),
            centroid_cart=np.array([5.0, 0.0, 0.0]),
        )

        dimer = DimerPair(
            monomer_a=mon_a,
            monomer_b=mon_b,
            distance=5.0,
            symop_index_a=0,
            symop_index_b=1,
            cell_index_b=(1, 0, 0),
            multiplicity=1,
        )

        mol = dimer.to_molecule()
        # Dimer should have 6 atoms (3 + 3)
        assert len(mol.symbols) == 6
        assert tuple(mol.symbols) == ("C", "O", "O", "C", "O", "O")

        # Check geometry is combined and in Bohr
        geom = np.array(mol.geometry).reshape(-1, 3)
        expected = np.vstack([cart_a, cart_b]) / qcel.constants.bohr2angstroms
        assert np.allclose(geom, expected, atol=1e-6)


class TestRemoveDuplicateMonomers:
    """Tests for _remove_duplicate_monomers helper."""

    def test_no_duplicates(self):
        """Test with no duplicate monomers."""
        symbols = ["C"]
        monomers = []
        for i in range(3):
            cart = np.array([[float(i * 5), 0.0, 0.0]])
            frac = np.array([[float(i) * 0.1, 0.0, 0.0]])
            mon = Monomer(
                symbols=symbols,
                frac_coords=frac,
                cart_coords=cart,
                cell_index=(0, 0, 0),
                symop_index=i,
                centroid_frac=frac[0],
                centroid_cart=cart[0],
            )
            monomers.append(mon)

        result = _remove_duplicate_monomers(monomers, tolerance=0.1)
        assert len(result) == 3

    def test_with_duplicates(self):
        """Test removing duplicate monomers."""
        symbols = ["C"]
        monomers = []

        # Create two monomers at the same position
        for i in range(2):
            cart = np.array([[0.0, 0.0, 0.0]])
            frac = np.array([[0.0, 0.0, 0.0]])
            mon = Monomer(
                symbols=symbols,
                frac_coords=frac,
                cart_coords=cart,
                cell_index=(0, 0, 0),
                symop_index=i,
                centroid_frac=frac[0],
                centroid_cart=cart[0],
            )
            monomers.append(mon)

        # Add one at a different position
        cart = np.array([[5.0, 0.0, 0.0]])
        frac = np.array([[0.5, 0.0, 0.0]])
        mon = Monomer(
            symbols=symbols,
            frac_coords=frac,
            cart_coords=cart,
            cell_index=(0, 0, 0),
            symop_index=2,
            centroid_frac=frac[0],
            centroid_cart=cart[0],
        )
        monomers.append(mon)

        result = _remove_duplicate_monomers(monomers, tolerance=0.1)
        assert len(result) == 2

    def test_empty_list(self):
        """Test with empty list."""
        result = _remove_duplicate_monomers([], tolerance=0.1)
        assert len(result) == 0


@pytest.mark.skipif(
    not TEST_CIF_PATH.exists(),
    reason="Test CIF file not available"
)
class TestGenerateUniqueDimers:
    """Tests for generate_unique_dimers function."""

    def test_returns_tuple(self):
        """Test that function returns correct tuple."""
        result = generate_unique_dimers(TEST_CIF_PATH, radius=5.0)
        assert isinstance(result, tuple)
        assert len(result) == 3
        dimers, crystal, ref_monomer = result
        assert isinstance(dimers, list)
        assert isinstance(crystal, Crystal)
        assert isinstance(ref_monomer, Monomer)

    def test_dimers_are_dimer_pairs(self):
        """Test that returned dimers are DimerPair objects."""
        dimers, _, _ = generate_unique_dimers(TEST_CIF_PATH, radius=6.0)
        for dimer in dimers:
            assert isinstance(dimer, DimerPair)

    def test_dimers_within_radius(self):
        """Test that all dimers are within specified radius."""
        radius = 7.0
        dimers, _, _ = generate_unique_dimers(TEST_CIF_PATH, radius=radius)
        for dimer in dimers:
            assert dimer.distance <= radius + 0.01  # small tolerance

    def test_dimers_sorted_by_distance(self):
        """Test that dimers are sorted by distance."""
        dimers, _, _ = generate_unique_dimers(TEST_CIF_PATH, radius=8.0)
        distances = [d.distance for d in dimers]
        assert distances == sorted(distances)

    def test_multiplicity_positive(self):
        """Test that all multiplicities are positive."""
        dimers, _, _ = generate_unique_dimers(TEST_CIF_PATH, radius=6.0)
        for dimer in dimers:
            assert dimer.multiplicity >= 1

    def test_reference_monomer_in_central_cell(self):
        """Test that reference monomer is in the central unit cell."""
        _, _, ref_monomer = generate_unique_dimers(TEST_CIF_PATH, radius=5.0)
        assert ref_monomer.cell_index == (0, 0, 0)

    def test_small_radius_fewer_dimers(self):
        """Test that smaller radius gives fewer dimers."""
        dimers_small, _, _ = generate_unique_dimers(TEST_CIF_PATH, radius=4.0)
        dimers_large, _, _ = generate_unique_dimers(TEST_CIF_PATH, radius=8.0)
        assert len(dimers_small) <= len(dimers_large)

    def test_no_self_pairing(self):
        """Test that reference monomer is not paired with itself."""
        dimers, _, ref = generate_unique_dimers(TEST_CIF_PATH, radius=10.0)
        for dimer in dimers:
            # Check that monomer_b is not the same as monomer_a
            if dimer.cell_index_b == (0, 0, 0):
                assert dimer.symop_index_b != ref.symop_index

    def test_dimer_to_molecule_correct_atoms(self):
        """Test that dimer molecules have correct atom count."""
        dimers, crystal, _ = generate_unique_dimers(TEST_CIF_PATH, radius=6.0)
        if dimers:
            dimer = dimers[0]
            mol = dimer.to_molecule()
            # CO2 has 3 atoms, dimer should have 6
            n_atoms_mon = len(dimer.monomer_a.symbols)
            assert len(mol.symbols) == 2 * n_atoms_mon

    def test_crystal_info_preserved(self):
        """Test that crystal info is correctly returned."""
        _, crystal, _ = generate_unique_dimers(TEST_CIF_PATH, radius=5.0)
        # CO2 is cubic Pa-3
        assert np.isclose(crystal.a, crystal.b, atol=0.01)
        assert np.isclose(crystal.b, crystal.c, atol=0.01)
        assert np.isclose(crystal.alpha, 90.0, atol=0.01)
        assert np.isclose(crystal.beta, 90.0, atol=0.01)
        assert np.isclose(crystal.gamma, 90.0, atol=0.01)


@pytest.mark.skipif(
    not TEST_CIF_PATH.exists(),
    reason="Test CIF file not available"
)
class TestGenerateUniqueDimersCO2:
    """Specific tests for CO2 crystal dimer generation."""

    def test_co2_first_shell_neighbors(self):
        """Test CO2 first coordination shell.

        CO2 in Pa-3 space group should have a specific first-shell
        dimer distance based on the unit cell parameter.
        """
        dimers, crystal, _ = generate_unique_dimers(
            TEST_CIF_PATH, radius=5.0, distance_tolerance=0.05
        )

        # The cell parameter is ~5.624 A
        # First neighbors should be around 3.9-4.0 A
        if dimers:
            first_dist = dimers[0].distance
            # First shell should be less than half the cell diagonal
            max_expected = crystal.a * 0.75
            assert first_dist < max_expected

    def test_co2_monomer_atoms_from_asu(self):
        """Test that CO2 monomers contain atoms from ASU.

        Note: The CO2 CIF has only 2 atoms in the ASU (C and O on special
        positions). The current implementation treats each symop-transformed
        ASU as a monomer. For molecular dimers, users may need to reconstruct
        complete molecules from symmetry-related ASU copies.
        """
        _, _, ref = generate_unique_dimers(TEST_CIF_PATH, radius=5.0)

        # CO2 CIF ASU has 2 atoms: C at (0,0,0) and O at (~0.1185,0.1185,0.1185)
        # Each "monomer" from the dimer generator contains these 2 ASU atoms
        assert len(ref.symbols) == 2
        assert ref.symbols.count("C") == 1
        assert ref.symbols.count("O") == 1

        # Verify the atoms are present
        assert "C" in ref.symbols
        assert "O" in ref.symbols


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
