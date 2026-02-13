"""
Space group symmetry operations for crystallographic systems.

This module provides functionality to:
1. Parse CIF files to extract unit cell and symmetry information
2. Parse symmetry operation strings (e.g., "x,y,z", "1/2+x,-y,z+1/2")
3. Apply symmetry operations to generate symmetry-equivalent positions
4. Convert between fractional and Cartesian coordinates
5. Create qcelemental Molecule objects from CIF files

Based on the FFX (Force Field X) crystal module implementation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import qcelemental as qcel
from numpy.typing import NDArray


@dataclass
class SymOp:
    """
    A symmetry operation defined by a 3x3 rotation matrix and a translation vector.

    The symmetry operation transforms fractional coordinates as:
        x' = rot @ x + tr

    Attributes
    ----------
    rot : NDArray[np.float64]
        The 3x3 rotation matrix in fractional coordinates.
    tr : NDArray[np.float64]
        The translation vector in fractional coordinates.
    """

    rot: NDArray[np.float64]
    tr: NDArray[np.float64]

    def __post_init__(self):
        self.rot = np.asarray(self.rot, dtype=np.float64)
        self.tr = np.asarray(self.tr, dtype=np.float64)

    def apply(self, frac_coords: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Apply symmetry operation to fractional coordinates.

        Parameters
        ----------
        frac_coords : NDArray[np.float64]
            Fractional coordinates (3,) or (N, 3).

        Returns
        -------
        NDArray[np.float64]
            Transformed fractional coordinates.
        """
        frac_coords = np.asarray(frac_coords)
        if frac_coords.ndim == 1:
            return self.rot @ frac_coords + self.tr
        return (self.rot @ frac_coords.T).T + self.tr

    def as_matrix(self) -> NDArray[np.float64]:
        """
        Return the SymOp as a 4x4 augmented matrix.

        Returns
        -------
        NDArray[np.float64]
            4x4 matrix representation.
        """
        mat = np.eye(4)
        mat[:3, :3] = self.rot
        mat[:3, 3] = self.tr
        return mat

    def to_xyz_string(self) -> str:
        """
        Convert symmetry operation to xyz string format.

        Returns
        -------
        str
            String like 'x,y,z' or '-x+1/2,y,-z+1/2'.
        """
        axes = ["x", "y", "z"]
        parts = []
        for i in range(3):
            terms = []
            for j, axis in enumerate(axes):
                coef = self.rot[i, j]
                if abs(coef) > 1e-10:
                    if abs(coef - 1.0) < 1e-10:
                        terms.append(f"+{axis}")
                    elif abs(coef + 1.0) < 1e-10:
                        terms.append(f"-{axis}")
                    else:
                        terms.append(f"{coef:+g}*{axis}")
            # Add translation
            tr_val = self.tr[i]
            if abs(tr_val) > 1e-10:
                # Convert to fraction if possible
                tr_str = _float_to_fraction_str(tr_val)
                terms.append(tr_str)
            if not terms:
                terms.append("0")
            part = "".join(terms)
            if part.startswith("+"):
                part = part[1:]
            parts.append(part)
        return ",".join(parts)

    @classmethod
    def identity(cls) -> "SymOp":
        """Return the identity symmetry operation."""
        return cls(rot=np.eye(3), tr=np.zeros(3))


def _float_to_fraction_str(val: float) -> str:
    """Convert a float to a fraction string if it matches common crystallographic fractions."""
    fractions = {
        1 / 2: "+1/2",
        1 / 3: "+1/3",
        2 / 3: "+2/3",
        1 / 4: "+1/4",
        3 / 4: "+3/4",
        1 / 6: "+1/6",
        5 / 6: "+5/6",
        -1 / 2: "-1/2",
        -1 / 3: "-1/3",
        -2 / 3: "-2/3",
        -1 / 4: "-1/4",
        -3 / 4: "-3/4",
        -1 / 6: "-1/6",
        -5 / 6: "-5/6",
    }
    for frac_val, frac_str in fractions.items():
        if abs(val - frac_val) < 1e-10:
            return frac_str
    return f"{val:+g}"


def parse_symop_xyz(xyz_str: str) -> SymOp:
    """
    Parse a symmetry operation from xyz string format.

    Parses strings like:
        - "x,y,z"
        - "-x,y,-z"
        - "1/2+x,1/2-y,z"
        - "x+1/2,-y+1/2,z+1/2"

    Parameters
    ----------
    xyz_str : str
        The symmetry operation in xyz string format.

    Returns
    -------
    SymOp
        The parsed symmetry operation.

    Examples
    --------
    >>> symop = parse_symop_xyz("x,y,z")
    >>> symop = parse_symop_xyz("-x+1/2,y,-z+1/2")
    """
    rot = np.zeros((3, 3))
    tr = np.zeros(3)

    # Clean up the string
    xyz_str = xyz_str.lower().replace(" ", "")
    parts = xyz_str.split(",")

    if len(parts) != 3:
        raise ValueError(f"Invalid symmetry operation string: {xyz_str}")

    for i, part in enumerate(parts):
        # Parse each component (x, y, z translations)
        rot[i], tr[i] = _parse_symop_component(part)

    return SymOp(rot=rot, tr=tr)


def _parse_symop_component(component: str) -> Tuple[NDArray[np.float64], float]:
    """
    Parse a single component of a symmetry operation string.

    Parameters
    ----------
    component : str
        A single component like "x", "-y", "1/2+z", "x-1/2".

    Returns
    -------
    Tuple[NDArray[np.float64], float]
        The rotation row and translation value.
    """
    rot_row = np.zeros(3)
    translation = 0.0

    # Map axis letters to indices
    axis_map = {"x": 0, "y": 1, "z": 2}

    # Normalize the component: ensure it starts with + or -
    if component and component[0] not in "+-":
        component = "+" + component

    pos = 0
    while pos < len(component):
        # Check for fraction followed by axis (e.g., "1/2x" or "+1/2x")
        frac_axis_match = re.match(r"([+-]?)(\d+/\d+)([xyz])", component[pos:])
        if frac_axis_match:
            sign_str, frac, axis = frac_axis_match.groups()
            sign = -1.0 if sign_str == "-" else 1.0
            # This is coefficient * axis (unusual but possible)
            coef = _parse_fraction(frac) * sign
            rot_row[axis_map[axis]] = coef
            pos += frac_axis_match.end()
            continue

        # Check for axis with optional sign (e.g., "x", "-x", "+x")
        axis_match = re.match(r"([+-]?)([xyz])", component[pos:])
        if axis_match:
            sign_str, axis = axis_match.groups()
            sign = -1.0 if sign_str == "-" else 1.0
            rot_row[axis_map[axis]] = sign
            pos += axis_match.end()
            continue

        # Check for standalone fraction (e.g., "+1/2", "-1/3")
        frac_match = re.match(r"([+-]?)(\d+/\d+)", component[pos:])
        if frac_match:
            sign_str, frac = frac_match.groups()
            sign = -1.0 if sign_str == "-" else 1.0
            translation += _parse_fraction(frac) * sign
            pos += frac_match.end()
            continue

        # Check for decimal number
        num_match = re.match(r"([+-]?\d*\.?\d+)", component[pos:])
        if num_match:
            translation += float(num_match.group(1))
            pos += num_match.end()
            continue

        # Skip unrecognized characters (shouldn't happen with valid input)
        pos += 1

    return rot_row, translation


def _parse_fraction(frac_str: str) -> float:
    """
    Parse a fraction string like '1/2' to a float.

    Parameters
    ----------
    frac_str : str
        Fraction string (e.g., "1/2", "2/3").

    Returns
    -------
    float
        The float value.
    """
    if "/" in frac_str:
        num, denom = frac_str.split("/")
        return float(num) / float(denom)
    return float(frac_str)


@dataclass
class Crystal:
    """
    Represents a crystal unit cell with lattice parameters and space group symmetry.

    Attributes
    ----------
    a : float
        Length of the a-axis in Angstroms.
    b : float
        Length of the b-axis in Angstroms.
    c : float
        Length of the c-axis in Angstroms.
    alpha : float
        Angle between b and c axes in degrees.
    beta : float
        Angle between a and c axes in degrees.
    gamma : float
        Angle between a and b axes in degrees.
    space_group_name : str
        Space group name (Hermann-Mauguin notation).
    space_group_number : int
        International Tables space group number.
    symops : List[SymOp]
        List of symmetry operations.
    """

    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    space_group_name: str = "P1"
    space_group_number: int = 1
    symops: List[SymOp] = field(default_factory=list)

    # Computed matrices (set in __post_init__)
    _frac_to_cart: NDArray[np.float64] = field(init=False, repr=False)
    _cart_to_frac: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize transformation matrices."""
        self._update_matrices()
        if not self.symops:
            self.symops = [SymOp.identity()]

    def _update_matrices(self):
        """Compute fractional <-> Cartesian transformation matrices."""
        # Convert angles to radians
        alpha_rad = np.radians(self.alpha)
        beta_rad = np.radians(self.beta)
        gamma_rad = np.radians(self.gamma)

        cos_alpha = np.cos(alpha_rad)
        cos_beta = np.cos(beta_rad)
        cos_gamma = np.cos(gamma_rad)
        sin_gamma = np.sin(gamma_rad)

        # Volume factor
        omega = np.sqrt(
            1
            - cos_alpha**2
            - cos_beta**2
            - cos_gamma**2
            + 2 * cos_alpha * cos_beta * cos_gamma
        )

        # Fractional to Cartesian matrix (column vectors are lattice vectors)
        # Using the standard crystallographic convention:
        # a along x, b in xy plane, c general
        self._frac_to_cart = np.array(
            [
                [self.a, self.b * cos_gamma, self.c * cos_beta],
                [0.0, self.b * sin_gamma, self.c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma],
                [0.0, 0.0, self.c * omega / sin_gamma],
            ]
        )

        # Cartesian to fractional matrix
        self._cart_to_frac = np.linalg.inv(self._frac_to_cart)

    @property
    def volume(self) -> float:
        """Calculate unit cell volume in cubic Angstroms."""
        return abs(np.linalg.det(self._frac_to_cart))

    def to_cartesian(self, frac_coords: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Convert fractional coordinates to Cartesian coordinates.

        Parameters
        ----------
        frac_coords : NDArray[np.float64]
            Fractional coordinates, shape (3,) or (N, 3).

        Returns
        -------
        NDArray[np.float64]
            Cartesian coordinates in Angstroms.
        """
        frac_coords = np.asarray(frac_coords)
        if frac_coords.ndim == 1:
            return self._frac_to_cart @ frac_coords
        return (self._frac_to_cart @ frac_coords.T).T

    def to_fractional(self, cart_coords: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Convert Cartesian coordinates to fractional coordinates.

        Parameters
        ----------
        cart_coords : NDArray[np.float64]
            Cartesian coordinates in Angstroms, shape (3,) or (N, 3).

        Returns
        -------
        NDArray[np.float64]
            Fractional coordinates.
        """
        cart_coords = np.asarray(cart_coords)
        if cart_coords.ndim == 1:
            return self._cart_to_frac @ cart_coords
        return (self._cart_to_frac @ cart_coords.T).T

    def apply_symop(
        self, frac_coords: NDArray[np.float64], symop: SymOp
    ) -> NDArray[np.float64]:
        """
        Apply a symmetry operation to fractional coordinates.

        Parameters
        ----------
        frac_coords : NDArray[np.float64]
            Fractional coordinates.
        symop : SymOp
            Symmetry operation to apply.

        Returns
        -------
        NDArray[np.float64]
            Transformed fractional coordinates.
        """
        return symop.apply(frac_coords)

    def generate_symmetry_equivalents(
        self,
        frac_coords: NDArray[np.float64],
        wrap_to_unit_cell: bool = True,
    ) -> NDArray[np.float64]:
        """
        Generate all symmetry-equivalent positions for given fractional coordinates.

        Parameters
        ----------
        frac_coords : NDArray[np.float64]
            Fractional coordinates, shape (3,) or (N, 3).
        wrap_to_unit_cell : bool
            If True, wrap coordinates to [0, 1).

        Returns
        -------
        NDArray[np.float64]
            All symmetry-equivalent positions, shape (M, 3) or (N*M, 3)
            where M is the number of symmetry operations.
        """
        frac_coords = np.asarray(frac_coords)
        single_atom = frac_coords.ndim == 1

        if single_atom:
            frac_coords = frac_coords.reshape(1, 3)

        all_positions = []
        for symop in self.symops:
            transformed = symop.apply(frac_coords)
            if wrap_to_unit_cell:
                transformed = transformed % 1.0
            all_positions.append(transformed)

        result = np.vstack(all_positions)
        return result


def parse_cif_file(filepath: str | Path) -> Tuple[Crystal, List[str], NDArray[np.float64]]:
    """
    Parse a CIF file to extract crystal and atomic information.

    Parameters
    ----------
    filepath : str or Path
        Path to the CIF file.

    Returns
    -------
    Tuple[Crystal, List[str], NDArray[np.float64]]
        - Crystal object with unit cell and symmetry information
        - List of atom labels/symbols
        - Fractional coordinates array of shape (N, 3)

    Examples
    --------
    >>> crystal, symbols, frac_coords = parse_cif_file("structure.cif")
    """
    filepath = Path(filepath)
    with open(filepath, "r") as f:
        content = f.read()

    # Parse unit cell parameters
    a = _extract_cif_value(content, "_cell_length_a")
    b = _extract_cif_value(content, "_cell_length_b")
    c = _extract_cif_value(content, "_cell_length_c")
    alpha = _extract_cif_value(content, "_cell_angle_alpha")
    beta = _extract_cif_value(content, "_cell_angle_beta")
    gamma = _extract_cif_value(content, "_cell_angle_gamma")

    # Parse space group
    sg_number = _extract_cif_int(content, "_space_group_IT_number")
    sg_name = _extract_cif_string(content, "_symmetry_space_group_name_H-M")
    if sg_name:
        sg_name = sg_name.strip("'\"")

    # Parse symmetry operations
    symops = _parse_cif_symmetry_operations(content)

    # Parse atom sites
    symbols, frac_coords = _parse_cif_atom_sites(content)

    crystal = Crystal(
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        space_group_name=sg_name or "P1",
        space_group_number=sg_number or 1,
        symops=symops if symops else [SymOp.identity()],
    )

    return crystal, symbols, frac_coords


def _extract_cif_value(content: str, key: str) -> float:
    """Extract a numeric value from CIF content, removing uncertainty in parentheses."""
    pattern = rf"{re.escape(key)}\s+(\S+)"
    match = re.search(pattern, content, re.IGNORECASE)
    if match:
        val_str = match.group(1)
        # Remove uncertainty in parentheses, e.g., "5.624(1)" -> "5.624"
        val_str = re.sub(r"\([^)]*\)", "", val_str)
        return float(val_str)
    raise ValueError(f"Could not find {key} in CIF file")


def _extract_cif_int(content: str, key: str) -> Optional[int]:
    """Extract an integer value from CIF content."""
    pattern = rf"{re.escape(key)}\s+(\d+)"
    match = re.search(pattern, content, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _extract_cif_string(content: str, key: str) -> Optional[str]:
    """Extract a string value from CIF content."""
    pattern = rf"{re.escape(key)}\s+(.+)"
    match = re.search(pattern, content, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _parse_cif_symmetry_operations(content: str) -> List[SymOp]:
    """
    Parse symmetry operations from CIF file.

    Looks for the _symmetry_equiv_pos_as_xyz loop.
    """
    symops = []

    # Find the symmetry loop
    # Pattern to match symmetry operations in a loop
    loop_pattern = r"loop_\s*_symmetry_equiv_pos_as_xyz\s*((?:[^\n]+\n)+?)(?=loop_|_\w|$)"
    match = re.search(loop_pattern, content, re.IGNORECASE)

    if match:
        lines = match.group(1).strip().split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith("_"):
                symops.append(parse_symop_xyz(line))
        return symops

    # Alternative format with site_id
    loop_pattern2 = r"loop_\s*(?:_symmetry_equiv_pos_site_id\s*)?_symmetry_equiv_pos_as_xyz\s*((?:[^\n]+\n)+?)(?=loop_|_\w|$)"
    match = re.search(loop_pattern2, content, re.IGNORECASE)

    if not match:
        # Try another common format
        pattern = r"_symmetry_equiv_pos_as_xyz\s*\n((?:.*\n)*?)(?=loop_|_\w)"
        match = re.search(pattern, content, re.IGNORECASE)

    if match:
        lines = match.group(1).strip().split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith("_") and not line.startswith("loop"):
                # Handle format with id number prefix
                parts = line.split()
                if len(parts) >= 1:
                    xyz_str = parts[-1] if len(parts) > 1 else parts[0]
                    # Remove quotes if present
                    xyz_str = xyz_str.strip("'\"")
                    if "," in xyz_str:
                        symops.append(parse_symop_xyz(xyz_str))

    return symops


def _parse_cif_atom_sites(content: str) -> Tuple[List[str], NDArray[np.float64]]:
    """
    Parse atom site information from CIF file.

    Returns atom symbols and fractional coordinates.
    """
    symbols = []
    frac_coords = []

    lines = content.split("\n")

    # Find the atom_site loop with fractional coordinates
    # We need to find a loop_ that contains _atom_site_fract_x
    i = 0
    found_loop = False
    columns = []
    col_map = {}

    while i < len(lines):
        line = lines[i].strip()

        # Look for loop_ followed by _atom_site headers
        if line.lower() == "loop_":
            # Check if this loop has _atom_site_fract_x
            j = i + 1
            temp_columns = []
            while j < len(lines):
                header_line = lines[j].strip()
                if header_line.lower().startswith("_atom_site_"):
                    # Extract the field name
                    field = header_line.lower().replace("_atom_site_", "")
                    temp_columns.append(field)
                    j += 1
                elif header_line.startswith("_") or header_line.lower() == "loop_":
                    # Different type of header or new loop
                    break
                elif header_line == "" or header_line.startswith("#"):
                    j += 1
                else:
                    # Data line - check if we found the right loop
                    break

            # Check if this loop has fractional coordinates
            if "fract_x" in temp_columns and "fract_y" in temp_columns and "fract_z" in temp_columns:
                found_loop = True
                columns = temp_columns
                col_map = {col: idx for idx, col in enumerate(columns)}
                i = j  # Start reading data from here
                break

        i += 1

    if not found_loop:
        raise ValueError("Could not find atom_site loop with fractional coordinates in CIF file")

    # Required columns for coordinates
    fract_x_col = col_map.get("fract_x")
    fract_y_col = col_map.get("fract_y")
    fract_z_col = col_map.get("fract_z")

    # Label column (element symbol)
    label_col = col_map.get("label")
    type_symbol_col = col_map.get("type_symbol")

    # Helper function to parse coordinates
    def parse_coord(s: str) -> float:
        s = re.sub(r"\([^)]*\)", "", s)
        return float(s)

    # Parse data lines
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        # Stop at next loop or new data block or header
        if line.lower().startswith("loop_") or line.startswith("_") or line.lower().startswith("data_"):
            break

        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < len(columns):
            continue

        # Get symbol
        if type_symbol_col is not None:
            symbol = parts[type_symbol_col]
        elif label_col is not None:
            # Extract element from label (e.g., "C1" -> "C", "O2" -> "O")
            label = parts[label_col]
            label_match = re.match(r"([A-Za-z]+)", label)
            if label_match is None:
                continue
            symbol = label_match.group(1)
        else:
            continue

        # Type assertions to satisfy the type checker (we know these are not None from above)
        assert fract_x_col is not None
        assert fract_y_col is not None
        assert fract_z_col is not None

        x = parse_coord(parts[fract_x_col])
        y = parse_coord(parts[fract_y_col])
        z = parse_coord(parts[fract_z_col])

        symbols.append(symbol)
        frac_coords.append([x, y, z])

    return symbols, np.array(frac_coords)


def remove_duplicate_atoms(
    symbols: List[str],
    coords: NDArray[np.float64],
    tolerance: float = 0.01,
) -> Tuple[List[str], NDArray[np.float64]]:
    """
    Remove duplicate atoms based on coordinate proximity.

    Parameters
    ----------
    symbols : List[str]
        List of atom symbols.
    coords : NDArray[np.float64]
        Coordinates array of shape (N, 3).
    tolerance : float
        Distance tolerance for considering atoms as duplicates.

    Returns
    -------
    Tuple[List[str], NDArray[np.float64]]
        Unique atom symbols and coordinates.
    """
    if len(symbols) == 0:
        return symbols, coords

    unique_symbols = [symbols[0]]
    unique_coords = [coords[0]]

    for i in range(1, len(symbols)):
        is_duplicate = False
        for j in range(len(unique_coords)):
            dist = np.linalg.norm(coords[i] - unique_coords[j])
            # Also check for periodic images
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        shifted = coords[i] + np.array([dx, dy, dz])
                        dist = np.linalg.norm(shifted - unique_coords[j])
                        if dist < tolerance:
                            is_duplicate = True
                            break
                    if is_duplicate:
                        break
                if is_duplicate:
                    break
            if is_duplicate:
                break

        if not is_duplicate:
            unique_symbols.append(symbols[i])
            unique_coords.append(coords[i])

    return unique_symbols, np.array(unique_coords)


def cif_to_molecule(
    filepath: str | Path,
    expand_symmetry: bool = True,
    wrap_to_unit_cell: bool = True,
    remove_duplicates: bool = True,
    duplicate_tolerance: float = 0.01,
) -> qcel.models.Molecule:
    """
    Create a QCElemental Molecule object from a CIF file.

    This function parses a CIF file, optionally applies space group symmetry
    operations to expand the asymmetric unit to the full unit cell, and
    returns a QCElemental Molecule object with Cartesian coordinates.

    Parameters
    ----------
    filepath : str or Path
        Path to the CIF file.
    expand_symmetry : bool, default=True
        If True, apply all symmetry operations to generate the full unit cell.
        If False, only the asymmetric unit atoms are returned.
    wrap_to_unit_cell : bool, default=True
        If True, wrap all coordinates to the unit cell [0, 1) in fractional space.
    remove_duplicates : bool, default=True
        If True, remove duplicate atoms that may arise from special positions.
    duplicate_tolerance : float, default=0.01
        Distance tolerance (in fractional coordinates) for detecting duplicates.

    Returns
    -------
    qcel.models.Molecule
        QCElemental Molecule object with Cartesian coordinates.

    Examples
    --------
    >>> mol = cif_to_molecule("carbon_dioxide.cif")
    >>> print(mol.symbols)
    >>> print(mol.geometry)  # Cartesian coordinates in Bohr
    """
    crystal, asu_symbols, asu_frac_coords = parse_cif_file(filepath)

    if expand_symmetry:
        # Apply all symmetry operations
        all_symbols = []
        all_frac_coords = []

        for i, (symbol, frac_coord) in enumerate(zip(asu_symbols, asu_frac_coords)):
            equiv_coords = crystal.generate_symmetry_equivalents(
                frac_coord, wrap_to_unit_cell=wrap_to_unit_cell
            )
            for coord in equiv_coords:
                all_symbols.append(symbol)
                all_frac_coords.append(coord)

        all_frac_coords = np.array(all_frac_coords)

        if remove_duplicates:
            all_symbols, all_frac_coords = remove_duplicate_atoms(
                all_symbols, all_frac_coords, tolerance=duplicate_tolerance
            )
    else:
        all_symbols = asu_symbols
        all_frac_coords = asu_frac_coords

    # Convert to Cartesian coordinates
    cart_coords = crystal.to_cartesian(all_frac_coords)

    # Convert Angstroms to Bohr for QCElemental
    cart_coords_bohr = cart_coords / qcel.constants.bohr2angstroms

    # Create QCElemental Molecule
    mol = qcel.models.Molecule(
        symbols=all_symbols,
        geometry=cart_coords_bohr.flatten(),
    )

    return mol


def get_crystal_info(filepath: str | Path) -> dict:
    """
    Extract crystal information from a CIF file.

    Parameters
    ----------
    filepath : str or Path
        Path to the CIF file.

    Returns
    -------
    dict
        Dictionary containing crystal information:
        - 'a', 'b', 'c': lattice parameters (Angstroms)
        - 'alpha', 'beta', 'gamma': lattice angles (degrees)
        - 'volume': unit cell volume (cubic Angstroms)
        - 'space_group_name': space group name
        - 'space_group_number': IT space group number
        - 'n_symops': number of symmetry operations
        - 'n_atoms_asu': number of atoms in asymmetric unit
    """
    crystal, symbols, frac_coords = parse_cif_file(filepath)

    return {
        "a": crystal.a,
        "b": crystal.b,
        "c": crystal.c,
        "alpha": crystal.alpha,
        "beta": crystal.beta,
        "gamma": crystal.gamma,
        "volume": crystal.volume,
        "space_group_name": crystal.space_group_name,
        "space_group_number": crystal.space_group_number,
        "n_symops": len(crystal.symops),
        "n_atoms_asu": len(symbols),
    }


if __name__ == "__main__":
    # Example usage / test
    import sys

    if len(sys.argv) > 1:
        cif_path = sys.argv[1]
    else:
        # Default test file
        cif_path = "/home/awallace43/projects/x23_dmetcalf_2022_si/cifs/carbon_dioxide.cif"

    print(f"Processing CIF file: {cif_path}")
    print("-" * 60)

    # Get crystal info
    info = get_crystal_info(cif_path)
    print("Crystal Information:")
    print(f"  Lattice parameters: a={info['a']:.3f}, b={info['b']:.3f}, c={info['c']:.3f}")
    print(f"  Lattice angles: alpha={info['alpha']:.1f}, beta={info['beta']:.1f}, gamma={info['gamma']:.1f}")
    print(f"  Volume: {info['volume']:.3f} A^3")
    print(f"  Space group: {info['space_group_name']} (#{info['space_group_number']})")
    print(f"  Symmetry operations: {info['n_symops']}")
    print(f"  Atoms in ASU: {info['n_atoms_asu']}")
    print("-" * 60)

    # Create molecule
    mol = cif_to_molecule(cif_path)
    print(f"\nQCElemental Molecule:")
    print(f"  Number of atoms: {len(mol.symbols)}")
    print(f"  Symbols: {list(mol.symbols)}")
    print(f"  Molecular formula: {mol.get_molecular_formula()}")

    # Print coordinates
    coords_angstrom = mol.geometry.reshape(-1, 3) * qcel.constants.bohr2angstroms
    print("\n  Cartesian coordinates (Angstroms):")
    for i, (sym, coord) in enumerate(zip(mol.symbols, coords_angstrom)):
        print(f"    {sym:2s}  {coord[0]:10.6f}  {coord[1]:10.6f}  {coord[2]:10.6f}")
