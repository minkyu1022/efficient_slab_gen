"""
Build final slab from oriented unit cell.

This module provides functions to generate slab from LMDB index by extracting
information from LMDB and mapping files, utilizing fairchem's slab.py functions.
"""

from __future__ import annotations

import math
import os
import pickle

import lmdb
import numpy as np
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.surface import center_slab

import argparse
from ase import Atoms
from ase.io import write

from fairchem.data.oc.core import Bulk

from fairchem.data.oc.core.slab import (
    compute_slabs, 
    tile_and_tag_atoms_with_info, 
    tile_atoms,
    set_fixed_atom_constraints,
    tag_surface_atoms,
)


def calculate_rmsd_pymatgen(
    struct1: Atoms | Structure,
    struct2: Atoms | Structure,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5,
) -> float | None:
    """
    Calculate RMSD using Pymatgen StructureMatcher.
    
    Arguments
    ---------
    struct1, struct2: Atoms | Structure
        ASE Atoms object or pymatgen Structure object
    ltol: float
        Lattice length tolerance (default: 0.2)
    stol: float
        Site distance tolerance (default: 0.3)
    angle_tol: float
        Angle tolerance in degrees (default: 5)
        
    Returns
    -------
    float | None
        RMSD value (Å) or None if structures don't match
    """
    # Convert ASE Atoms to pymatgen Structure
    if hasattr(struct1, 'get_positions'):
        struct1 = AseAtomsAdaptor.get_structure(struct1)
    if hasattr(struct2, 'get_positions'):
        struct2 = AseAtomsAdaptor.get_structure(struct2)
    
    # Create StructureMatcher
    matcher = StructureMatcher(
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol
    )
    
    # Check if structures match
    if matcher.fit(struct1, struct2):
        # Calculate RMS distance
        rms_dist, max_dist = matcher.get_rms_dist(struct1, struct2)
        return rms_dist
    else:
        return None


def add_adsorbate_to_slab(
    slab_atoms: Atoms,
    adsorbate_atomic_numbers: np.ndarray,
    adsorbate_positions: np.ndarray,
) -> Atoms:
    """
    Add adsorbate to slab structure.
    
    Arguments
    ---------
    slab_atoms: Atoms
        Final tiled slab structure
    adsorbate_atomic_numbers: np.ndarray
        Atomic numbers of adsorbate atoms (shape: (n_adsorbate_atoms,))
    adsorbate_positions: np.ndarray
        True coordinates of adsorbate atoms (shape: (n_adsorbate_atoms, 3))
    
    Returns
    -------
    Atoms
        Slab + adsorbate structure (adsorbate tagged as 2)
    """
    # Copy slab
    slab_with_adsorbate = slab_atoms.copy()
    
    # Get existing atom count
    n_slab_atoms = len(slab_atoms)
    n_adsorbate = len(adsorbate_atomic_numbers)
    
    # Create new atomic numbers and positions arrays
    new_numbers = np.concatenate([
        slab_atoms.numbers,
        adsorbate_atomic_numbers.astype(int)
    ])
    new_positions = np.concatenate([
        slab_atoms.positions,
        adsorbate_positions
    ])
    
    # Create new Atoms object
    result = Atoms(
        numbers=new_numbers,
        positions=new_positions,
        cell=slab_atoms.cell,
        pbc=slab_atoms.pbc
    )
    
    # Set tags: slab atoms keep existing tags, adsorbate gets tag=2
    if slab_atoms.has('tags'):
        slab_tags = slab_atoms.get_tags()
    else:
        slab_tags = np.zeros(n_slab_atoms)
    
    adsorbate_tags = np.full(n_adsorbate, 2)  # tag=2 for adsorbate
    all_tags = np.concatenate([slab_tags, adsorbate_tags])
    result.set_tags(all_tags)
    
    # Copy constraints if present
    if slab_atoms.constraints:
        result.constraints = slab_atoms.constraints.copy()
    
    return result


def get_sid_and_adsorbate_from_lmdb(lmdb_path: str, index: int) -> dict:
    """
    Extract sid (system ID) and adsorbate information from LMDB file.
    
    Arguments
    ---------
    lmdb_path: str
        Path to LMDB file
    index: int
        Data index to extract
    
    Returns
    -------
    dict: {
        'sid': int (system ID),
        'adsorbate_atomic_numbers': np.ndarray (adsorbate atom types, atoms with tags==2),
        'adsorbate_positions': np.ndarray (adsorbate coordinates, shape: (n_adsorbate_atoms, 3))
    }
    """
    # Open LMDB database
    db = lmdb.open(
        str(lmdb_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    
    try:
        # Start transaction
        with db.begin() as txn:
            # Get data for index
            key = f"{index}".encode("ascii")
            value = txn.get(key)
            
            if value is None:
                raise ValueError(f"Index {index} not found in LMDB")
            
            # Deserialize with pickle
            data = pickle.loads(value)
        
        # Convert data to dict format (PyTorch Geometric compatibility)
        data_dict = {}
        
        # Method 1: Try to_dict()
        try:
            data_dict = data.to_dict()
        except (AttributeError, RuntimeError, TypeError):
            pass
        
        # Method 2: Direct access from _store (PyTorch Geometric)
        if not data_dict or 'pos' not in data_dict:
            try:
                store = getattr(data, '_store', None)
                if store is not None:
                    for key in store.keys():
                        try:
                            data_dict[key] = store[key]
                        except (RuntimeError, AttributeError, KeyError):
                            pass
            except (AttributeError, RuntimeError):
                pass
        
        # Method 3: Direct attribute access
        for key in ['tags', 'atomic_numbers', 'pos']:
            if key not in data_dict:
                try:
                    value = getattr(data, key, None)
                    if value is not None:
                        data_dict[key] = value
                except (RuntimeError, AttributeError):
                    pass
        
        # Extract sid (try multiple methods)
        sid = None
        
        # Method 1: Direct attribute access
        try:
            sid = getattr(data, 'sid', None)
        except (RuntimeError, AttributeError):
            pass
        
        # Method 2: Look in __dict__
        if sid is None:
            try:
                if hasattr(data, '__dict__') and 'sid' in data.__dict__:
                    sid = data.__dict__['sid']
            except Exception:
                pass
        
        # Method 3: Use items() method (PyTorch Geometric)
        if sid is None:
            try:
                if hasattr(data, 'items'):
                    for key, value in data.items():
                        if key == 'sid':
                            sid = value
                            break
            except (RuntimeError, AttributeError, TypeError):
                pass
        
        # Method 4: Check keys() then direct access
        if sid is None:
            try:
                if hasattr(data, 'keys'):
                    keys = list(data.keys())
                    if 'sid' in keys:
                        sid = data['sid']
            except (RuntimeError, AttributeError, TypeError, KeyError):
                pass
        
        # Method 5: Direct access from _store
        if sid is None:
            try:
                store = getattr(data, '_store', None)
                if store is not None and 'sid' in store.keys():
                    sid = store['sid']
            except (RuntimeError, AttributeError, KeyError):
                pass
        
        # Extract first element if value is list or tensor
        if sid is not None:
            if isinstance(sid, (list, tuple)) and len(sid) > 0:
                sid = sid[0]
            elif isinstance(sid, torch.Tensor):
                sid = sid.item() if sid.numel() == 1 else sid.tolist()[0] if len(sid) > 0 else None
        
        # Extract tags
        tags = None
        if 'tags' in data_dict:
            tags = data_dict['tags']
        else:
            try:
                tags = getattr(data, 'tags', None)
            except (RuntimeError, AttributeError):
                pass
        
        if tags is None:
            print(f"Warning: No 'tags' attribute found in data at index {index}")
            return {'sid': sid, 'adsorbate_atomic_numbers': None, 'adsorbate_positions': None}
        
        # Convert tags to numpy array
        if isinstance(tags, torch.Tensor):
            tags_np = tags.cpu().numpy()
        else:
            tags_np = np.array(tags)
        
        # Find indices where tags==2 (adsorbate atoms)
        adsorbate_mask = (tags_np == 2)
        
        # Extract adsorbate information
        adsorbate_atomic_numbers = None
        adsorbate_positions = None
        
        if np.any(adsorbate_mask):
            # Extract and filter atomic_numbers
            atomic_numbers = None
            if 'atomic_numbers' in data_dict:
                atomic_numbers = data_dict['atomic_numbers']
            else:
                try:
                    atomic_numbers = getattr(data, 'atomic_numbers', None)
                except (RuntimeError, AttributeError):
                    pass
            
            if atomic_numbers is not None:
                if isinstance(atomic_numbers, torch.Tensor):
                    atomic_numbers_np = atomic_numbers.cpu().numpy()
                else:
                    atomic_numbers_np = np.array(atomic_numbers)
                adsorbate_atomic_numbers = atomic_numbers_np[adsorbate_mask]
            
            # Extract and filter pos information
            pos = None
            if 'pos' in data_dict:
                pos = data_dict['pos']
            else:
                try:
                    pos = getattr(data, 'pos', None)
                except (RuntimeError, AttributeError):
                    pass
                
                # Look in __dict__
                if pos is None:
                    try:
                        if hasattr(data, '__dict__'):
                            if 'pos' in data.__dict__:
                                pos = data.__dict__['pos']
                    except Exception:
                        pass
                
                # Use items() method
                if pos is None:
                    try:
                        if hasattr(data, 'items'):
                            for key, value in data.items():
                                if key == 'pos':
                                    pos = value
                                    break
                    except (RuntimeError, AttributeError, TypeError):
                        pass
                
                # Check keys() then direct access
                if pos is None:
                    try:
                        if hasattr(data, 'keys'):
                            keys = list(data.keys())
                            if 'pos' in keys:
                                pos = data['pos']
                    except (RuntimeError, AttributeError, TypeError, KeyError):
                        pass
            
            if pos is not None:
                if isinstance(pos, torch.Tensor):
                    pos_np = pos.cpu().numpy()
                else:
                    pos_np = np.array(pos)
                adsorbate_positions = pos_np[adsorbate_mask]
        
        return {
            'sid': sid,
            'adsorbate_atomic_numbers': adsorbate_atomic_numbers,
            'adsorbate_positions': adsorbate_positions,
        }
            
    finally:
        db.close()


def get_sid_from_lmdb(lmdb_path: str, index: int) -> int | None:
    """
    Extract sid (system ID) value from LMDB file at specific index.

    Arguments
    ---------
    lmdb_path: str
        Path to LMDB file
    index: int
        Data index to extract

    Returns
    -------
    sid: int | None
        System ID value, or None if not found
    """
    db = lmdb.open(
        str(lmdb_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    try:
        with db.begin() as txn:
            key = f"{index}".encode("ascii")
            value = txn.get(key)

            if value is None:
                raise ValueError(f"Index {index} not found in LMDB")

            data = pickle.loads(value)

        # Extract sid (try multiple methods)
        sid = None

        # Method 1: Direct attribute access
        try:
            sid = getattr(data, "sid", None)
        except (RuntimeError, AttributeError):
            pass

        # Method 2: Look in __dict__
        if sid is None:
            try:
                if hasattr(data, "__dict__") and "sid" in data.__dict__:
                    sid = data.__dict__["sid"]
            except Exception:
                pass

        # Method 3: Use items() method (PyTorch Geometric)
        if sid is None:
            try:
                if hasattr(data, "items"):
                    for key, value in data.items():
                        if key == "sid":
                            sid = value
                            break
            except (RuntimeError, AttributeError, TypeError):
                pass

        # Method 4: Check keys() then direct access
        if sid is None:
            try:
                if hasattr(data, "keys"):
                    keys = list(data.keys())
                    if "sid" in keys:
                        sid = data["sid"]
            except (RuntimeError, AttributeError, TypeError, KeyError):
                pass

        # Method 5: Direct access from _store
        if sid is None:
            try:
                store = getattr(data, "_store", None)
                if store is not None and "sid" in store.keys():
                    sid = store["sid"]
            except (RuntimeError, AttributeError, KeyError):
                pass

        # Extract first element if value is list or tensor
        if sid is not None:
            if isinstance(sid, (list, tuple)) and len(sid) > 0:
                sid = sid[0]
            elif isinstance(sid, torch.Tensor):
                sid = sid.item() if sid.numel() == 1 else sid.tolist()[0] if len(sid) > 0 else None

        return sid

    finally:
        db.close()


def extract_true_system_from_lmdb(lmdb_path: str, index: int) -> Atoms | None:
    """
    Extract data from LMDB file and return full structure (slab + adsorbate) as ASE Atoms.
    Returns atoms with tags==0, 1, 2.
    
    Arguments
    ---------
    lmdb_path: str
        Path to LMDB file
    index: int
        Data index to extract
    
    Returns
    -------
    Atoms | None
        Full structure (all atoms with tags==0, 1, 2), 
        or None if failed
    """
    # Open LMDB database
    db = lmdb.open(
        str(lmdb_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    
    try:
        # Start transaction
        with db.begin() as txn:
            # Get data for index
            key = f"{index}".encode("ascii")
            value = txn.get(key)
            
            if value is None:
                raise ValueError(f"Index {index} not found in LMDB")
            
            # Deserialize with pickle
            data = pickle.loads(value)
        
        # Convert data to dict format (PyTorch Geometric compatibility)
        data_dict = {}
        
        # Method 1: Try to_dict()
        try:
            data_dict = data.to_dict()
        except (AttributeError, RuntimeError, TypeError):
            pass
        
        # Method 2: Direct access from _store (PyTorch Geometric)
        if not data_dict or 'pos' not in data_dict:
            try:
                store = getattr(data, '_store', None)
                if store is not None:
                    for key in store.keys():
                        try:
                            data_dict[key] = store[key]
                        except (RuntimeError, AttributeError, KeyError):
                            pass
            except (AttributeError, RuntimeError):
                pass
        
        # Method 3: Direct attribute access (tags, atomic_numbers, cell)
        for key in ['tags', 'atomic_numbers', 'cell']:
            if key not in data_dict:
                try:
                    value = getattr(data, key, None)
                    if value is not None:
                        data_dict[key] = value
                except (RuntimeError, AttributeError):
                    pass
        
        # Extract tags
        tags = None
        if 'tags' in data_dict:
            tags = data_dict['tags']
        else:
            try:
                tags = getattr(data, 'tags', None)
            except (RuntimeError, AttributeError):
                pass
        
        if tags is None:
            print(f"Warning: No 'tags' attribute found in data at index {index}")
            return None
        
        # Convert tags to numpy array
        if isinstance(tags, torch.Tensor):
            tags_np = tags.cpu().numpy()
        else:
            tags_np = np.array(tags)
        
        # Find indices where tags==0, 1, or 2 (slab + adsorbate)
        atom_mask = (tags_np == 0) | (tags_np == 1) | (tags_np == 2)
        
        if not np.any(atom_mask):
            print(f"Warning: No atoms with tags 0, 1, or 2 found at index {index}")
            return None
        
        # Extract and filter atomic_numbers
        atomic_numbers = None
        if 'atomic_numbers' in data_dict:
            atomic_numbers = data_dict['atomic_numbers']
        else:
            try:
                atomic_numbers = getattr(data, 'atomic_numbers', None)
            except (RuntimeError, AttributeError):
                pass
        
        if atomic_numbers is None:
            print(f"Warning: No 'atomic_numbers' attribute found in data at index {index}")
            return None
        
        if isinstance(atomic_numbers, torch.Tensor):
            atomic_numbers_np = atomic_numbers.cpu().numpy()
        else:
            atomic_numbers_np = np.array(atomic_numbers)
        atomic_numbers_filtered = atomic_numbers_np[atom_mask]
        
        # Cell information (no filtering, cell of full structure)
        cell = None
        if 'cell' in data_dict:
            cell = data_dict['cell']
        else:
            try:
                cell = getattr(data, 'cell', None)
            except (RuntimeError, AttributeError):
                pass
        
        if cell is None:
            print(f"Warning: No 'cell' attribute found in data at index {index}")
            return None
        
        if isinstance(cell, torch.Tensor):
            cell_np = cell.cpu().numpy()
        else:
            cell_np = np.array(cell)
        
        # Convert cell from (1, 3, 3) to (3, 3) if needed
        if cell_np.shape == (1, 3, 3):
            cell_np = cell_np[0]
        elif cell_np.shape != (3, 3):
            print(f"Warning: Unexpected cell shape: {cell_np.shape}")
            return None
        
        # Filter pos information
        pos = None
        
        # Method 1: Get from data_dict
        if 'pos' in data_dict:
            pos = data_dict['pos']
        else:
            # Method 2: Try direct getattr
            try:
                pos = getattr(data, 'pos', None)
            except (RuntimeError, AttributeError):
                pass
            
            # Method 3: Look in __dict__
            if pos is None:
                try:
                    if hasattr(data, '__dict__'):
                        if 'pos' in data.__dict__:
                            pos = data.__dict__['pos']
                except Exception:
                    pass
            
            # Method 4: Use items() method (PyTorch Geometric)
            if pos is None:
                try:
                    if hasattr(data, 'items'):
                        for key, value in data.items():
                            if key == 'pos':
                                pos = value
                                break
                except (RuntimeError, AttributeError, TypeError):
                    pass
            
            # Method 5: Check keys() then direct access
            if pos is None:
                try:
                    if hasattr(data, 'keys'):
                        keys = list(data.keys())
                        if 'pos' in keys:
                            pos = data['pos']
                except (RuntimeError, AttributeError, TypeError, KeyError):
                    pass
        
        if pos is None:
            print(f"Warning: No 'pos' attribute found in data at index {index}")
            return None
        
        if isinstance(pos, torch.Tensor):
            pos_np = pos.cpu().numpy()
        else:
            pos_np = np.array(pos)
        pos_filtered = pos_np[atom_mask]
        
        # Create ASE Atoms object
        atoms = Atoms(
            numbers=atomic_numbers_filtered,
            positions=pos_filtered,
            cell=cell_np,
            pbc=[True, True, True]
        )
        
        # Set tags
        tags_filtered = tags_np[atom_mask]
        atoms.set_tags(tags_filtered)
        
        return atoms
            
    except Exception as e:
        print(f"Error extracting true slab from LMDB: {e}")
        return None
    finally:
        db.close()


def get_slab_params_from_mapping(
    mapping_path: str, sid: int
) -> dict[str, str | tuple | float | bool] | None:
    """
    Extract slab parameters for sid from oc20_data_mapping.pkl file.

    Arguments
    ---------
    mapping_path: str
        Path to oc20_data_mapping.pkl file
    sid: int
        System ID

    Returns
    -------
    params: dict | None
        Dictionary containing:
        - bulk_mpid: str (e.g., "mp-978498")
        - miller_index: tuple (e.g., (1, 1, 0))
        - shift: float (e.g., 0.125)
        - top: bool (e.g., True)
        Returns None if key not found
    """
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    with open(mapping_path, "rb") as f:
        mapping = pickle.load(f)

    target_key = f"random{sid}"

    if target_key not in mapping:
        return None

    value = mapping[target_key]
    if not isinstance(value, dict):
        return None

    return {
        "bulk_mpid": value.get("bulk_mpid"),
        "miller_index": value.get("miller_index"),
        "shift": value.get("shift"),
        "top": value.get("top"),
    }


def create_slab_from_index(
    index: int,
    lmdb_path: str,
    mapping_path: str,
    min_slab_size: float = 7.0,
    min_vacuum_size: float = 20.0,
    min_ab: float = 8.0,
    center_slab: bool = True,
    primitive: bool = True,
    tol: float = 0.3,
    output_dir: str | None = None,
) -> tuple[Structure, bool]:
    """
    Generate oriented unit slab, final slab, repeat info, and vacuum info from LMDB index.

    Arguments
    ---------
    index: int
        LMDB data index
    lmdb_path: str
        Path to LMDB file
    mapping_path: str
        Path to oc20_data_mapping.pkl file
    min_slab_size: float
        Minimum slab thickness in Angstroms
    min_vacuum_size: float
        Minimum vacuum layer size in Angstroms
    min_ab: float
        Minimum distance in x and y directions for the tiled structure
    center_slab: bool
        Whether to center the slab in the cell
    primitive: bool
        Whether to reduce generated slabs to primitive cell
    tol: float
        Tolerance for getting primitive cells
    output_dir: str | None
        Directory to save true structure CIF file. If None, true structure is not saved.

    Returns
    -------
    tuple: (shifted_ouc, is_struct_no_vac)
        shifted_ouc: Structure
            1-layer unit with shift applied (before replication), or replaced by struct_no_vac
        is_struct_no_vac: bool
            Whether shifted_ouc was replaced by struct_no_vac
    """
    
    # Step 0: Extract true structure from LMDB and save if output_dir is provided
    if output_dir is not None:
        true_system_atoms = extract_true_system_from_lmdb(lmdb_path, index)
        if true_system_atoms is not None:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save true_system (slab + adsorbate)
            true_system_path = os.path.join(output_dir, "true_system.cif")
            # write(true_system_path, true_system_atoms)
            
            # Extract and save true_slab (slab only, tags==0 or 1)
            slab_mask = (true_system_atoms.get_tags() == 0) | (true_system_atoms.get_tags() == 1)
            if np.any(slab_mask):
                true_slab_atoms = true_system_atoms[slab_mask].copy()
                true_slab_path = os.path.join(output_dir, "true_slab.cif")
                # write(true_slab_path, true_slab_atoms)
            else:
                print(f"[WARNING] No slab atoms (tags 0 or 1) found in true_system")
        else:
            print(f"[WARNING] Could not extract true structure from LMDB at index {index}")
    
    # Step 1: Extract sid from LMDB
    sid = get_sid_from_lmdb(lmdb_path, index)
    if sid is None:
        raise ValueError(f"Could not extract sid from LMDB at index {index}")

    # Step 2: Extract slab parameters from mapping
    params = get_slab_params_from_mapping(mapping_path, sid)
    if params is None:
        raise ValueError(f"Could not find mapping for sid={sid}")
    if None in params.values():
        raise ValueError(f"Incomplete parameters in mapping for sid={sid}: {params}")

    bulk_mpid = params["bulk_mpid"]
    miller_index = params["miller_index"]
    shift = params["shift"]
    top = params["top"]
    
    print("top:", top)

    # Step 3: Create Bulk from fairchem
    try:
        from fairchem.data.oc.core import Bulk
        from fairchem.data.oc.core.slab import compute_slabs, tile_and_tag_atoms_with_info
    except ImportError:
        raise ImportError("fairchem.data.oc.core is not available")

    bulk = Bulk(bulk_src_id_from_db=bulk_mpid)
    # if output_dir is not None:
    #     write(f'{output_dir}/bulk.cif', bulk.atoms)
    
    # Step 4: Use compute_slabs to get shifted_ouc
    # Note: compute_slabs internally uses SlabGenerator with primitive reduction applied after slab generation
    untiled_slabs = compute_slabs(
        bulk_atoms=bulk.atoms,
        max_miller=max(np.abs(miller_index)),
        specific_millers=[miller_index],
    )
    
    # Filter by shift and top
    matching_untiled_slabs = [
        s for s in untiled_slabs
        if abs(s[2] - shift) < 1e-3 and s[3] == top
    ]
    if not matching_untiled_slabs:
        raise ValueError(
            f"No matching slab found for miller_index={miller_index}, "
            f"shift={shift}, top={top}"
        )
    
    untiled_slab, miller, shift, top, ouc = matching_untiled_slabs[0]
    
    np.save(f"{index}_ouc_c_vector.npy", ouc.lattice.matrix[2])
    
    # write(f'{index}_untiled_slab.cif', AseAtomsAdaptor.get_atoms(struct_with_vac))
    # write(f'{index}_shifted_oriented_unit_cell.cif', AseAtomsAdaptor.get_atoms(shifted_ouc))

    # if output_dir is not None:
    #     write(f'{output_dir}/unit_slab.cif', AseAtomsAdaptor.get_atoms(unit_slab))
    #     write(f'{output_dir}/struct_with_vac.cif', AseAtomsAdaptor.get_atoms(struct_with_vac))
    #     write(f'{output_dir}/struct_no_vac.cif', AseAtomsAdaptor.get_atoms(struct_no_vac))

    # Return chosen base OUC and flag
    
    write(f'{index}_ouc.cif', AseAtomsAdaptor.get_atoms(ouc))
    return ouc


def calculate_surface_normal_from_cell(struct: Structure) -> np.ndarray:
    """Calculate unit surface normal from lattice vectors a, b."""
    a_vec, b_vec, _ = struct.lattice.matrix
    n = np.cross(a_vec, b_vec)
    n /= np.linalg.norm(n)
    return n


def reconstruct_slab_from_shifted_ouc(
    shifted_ouc: Structure,
    min_slab_size: float,
    min_vacuum_size: float,
    min_ab: float,
    tol: float = 0.3,
    center: bool = True,
    primitive: bool = True,
    pre_primitive_ouc: bool = True,
) -> tuple[Structure, Structure, int, int, int, int, float]:
    """
    Reconstruct final slab Structure from shifted_ouc and perform a,b tiling.

    Returns
    -------
    tuple: (pred_slab, pred_struct, n_layers_slab, n_layers_vac, na, nb, height)
        pred_slab: Structure
            Final predicted slab with a,b tiling completed
        pred_struct: Structure
            Predicted slab before tiling (1x1)
        n_layers_slab, n_layers_vac: int
            Number of atom layers in c direction, number of vacuum layers
        na, nb: int
            Tiling coefficients in a,b directions
        height: float
            Height of one layer (Å)
    """
    base_ouc = shifted_ouc
    if pre_primitive_ouc:
        # Pre-reduce OUC to primitive for consistency with final slab a,b reduction
        base_ouc = shifted_ouc.get_primitive_structure(
            tolerance=tol,
            constrain_latt={
                "c": shifted_ouc.lattice.c,
                "alpha": shifted_ouc.lattice.alpha,
                "beta": shifted_ouc.lattice.beta,
            },
        )

    a_vec, b_vec, c_vec = base_ouc.lattice.matrix
    normal = calculate_surface_normal_from_cell(base_ouc)
    height = abs(np.dot(normal, c_vec))

    n_layers_slab = int(math.ceil(min_slab_size / height))
    n_layers_vac = int(math.ceil(min_vacuum_size / height))
    n_layers = n_layers_slab + n_layers_vac

    base_frac_coords = base_ouc.frac_coords
    base_species = base_ouc.species_and_occu

    stacked_frac_coords = base_frac_coords.copy()
    stacked_frac_coords[:, 2] /= n_layers

    all_coords = []
    for idx in range(n_layers_slab):
        _frac_coords = stacked_frac_coords.copy()
        _frac_coords[:, 2] += idx / n_layers
        all_coords.extend(_frac_coords)

    all_species = base_species * n_layers_slab

    new_lattice = [a_vec, b_vec, (n_layers * c_vec)]
    props = base_ouc.site_properties if base_ouc.site_properties else {}
    props = {k: v * n_layers_slab for k, v in props.items()}
    slab_core = Structure(
        new_lattice,
        all_species,
        all_coords,
        site_properties=props,
    )

    if center:
        slab_core = center_slab(slab_core)

    pred_struct = slab_core
    if primitive:
        pred_struct = slab_core.get_primitive_structure(tolerance=tol)

    a_length = float(np.linalg.norm(pred_struct.lattice.matrix[0]))
    b_length = float(np.linalg.norm(pred_struct.lattice.matrix[1]))
    na = int(math.ceil(min_ab / a_length))
    nb = int(math.ceil(min_ab / b_length))

    pred_slab = pred_struct.copy()
    pred_slab.make_supercell([na, nb, 1])

    return pred_slab, pred_struct, n_layers_slab, n_layers_vac, na, nb, height

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--lmdb_path", type=str, default="data/is2re/all/train/data.lmdb")
    parser.add_argument("--mapping_path", type=str, default="oc20_data_mapping.pkl")
    parser.add_argument("--output_dir", type=str, default="unit_slab_recon")
    parser.add_argument("--min_ab", type=float, default=8.0)
    
    # Constants for SlabGenerator
    parser.add_argument("--min_slab_size", type=float, default=7.0)
    parser.add_argument("--min_vacuum_size", type=float, default=20.0)
    
    args = parser.parse_args()
    
    index = args.idx
    lmdb_path = args.lmdb_path
    mapping_path = args.mapping_path
    output_dir = f"{args.output_dir}/{args.idx}"
    
    os.makedirs(output_dir, exist_ok=True)

    # ============================================
    # Step 1: Get shifted_ouc (1-layer unit with shift applied)
    # ============================================
    unit_slab = create_slab_from_index(
        index=index,
        lmdb_path=lmdb_path,
        mapping_path=mapping_path,
        output_dir=output_dir,
        min_slab_size=args.min_slab_size,
        min_vacuum_size=args.min_vacuum_size,
    )
    
    print("\n=== Slab Info from Index ===")
    print(f"unit_slab (1-layer unit): {len(unit_slab)} atoms, (Saved to unit_slab.cif)")


    # ============================================
    # Step 2: Reconstruct slab from shifted_ouc (replicating get_slab logic)
    # ============================================
    
    print("\n=== Starting Final Slab Reconstruction (replicating get_slab logic) ===")

    final_slab_recon_pmg, pred_struct, n_layers_slab, n_layers_vac, na, nb, height = reconstruct_slab_from_shifted_ouc(
        shifted_ouc=unit_slab,
        min_slab_size=args.min_slab_size,
        min_vacuum_size=args.min_vacuum_size,
        min_ab=args.min_ab,
        tol=0.3,
        center=True,
        primitive=True,
        pre_primitive_ouc=True,
    )
    print(f"[DEBUG] height: {height}")
    print(f"[DEBUG] n_layers_slab: {n_layers_slab}, n_layers_vac: {n_layers_vac}")
    print(f"  a,b tiling coefficients (na, nb) calculated: ({na}, {nb})")
    
    # Convert to ASE
    final_slab_recon_ase = AseAtomsAdaptor.get_atoms(final_slab_recon_pmg)
    
    # Apply tags and constraints
    # tag_surface_atoms from fairchem/slab.py requires bulk.atoms
    try:
        # Reload Bulk object
        sid = get_sid_from_lmdb(lmdb_path, index)
        params = get_slab_params_from_mapping(mapping_path, sid)
        bulk = Bulk(bulk_src_id_from_db=params["bulk_mpid"])
        
        print("  Applying surface tags...")
        final_slab_recon_tagged = tag_surface_atoms(final_slab_recon_ase, bulk.atoms)
    except Exception as e:
        print(f"  [Warning] tag_surface_atoms failed: {e}. Proceeding without tags.")
        final_slab_recon_tagged = final_slab_recon_ase

    print("  Applying constraints...")
    final_slab_recon = set_fixed_atom_constraints(final_slab_recon_tagged)
    
    print(f"[Reconstructed] Final slab (atoms only): {len(final_slab_recon)} atoms")
    # write(f'{output_dir}/pred_slab.cif', final_slab_recon)

    # ============================================
    # Step 3: Compare reconstructed structure with True structure (RMSD)
    # ============================================
    
    # Extract "True Slab" (tags 0, 1) from LMDB
    true_system_atoms = extract_true_system_from_lmdb(lmdb_path, index)
    
    if true_system_atoms is not None:
        slab_mask = (true_system_atoms.get_tags() == 0) | (true_system_atoms.get_tags() == 1)
        if np.any(slab_mask):
            true_slab_atoms = true_system_atoms[slab_mask].copy()
            
            print(f"\n=== RMSD Calculation (Reconstructed vs True) ===")
            print(f"Predicted (Recon): {len(final_slab_recon)} atoms")
            print(f"True Slab (LMDB): {len(true_slab_atoms)} atoms")
            
            # RMSD threshold: relaxed when struct_no_vac is used
            rmsd_threshold = 1e-2 if is_struct_no_vac else 1e-3
            
            rmsd = calculate_rmsd_pymatgen(
                struct1=final_slab_recon,
                struct2=true_slab_atoms,
                ltol=0.2, stol=0.3, angle_tol=5,
            )
            
            if rmsd is not None:
                print(f"RMSD (Recon vs True): {rmsd:.4f} Å (threshold: {rmsd_threshold:.0e})")
                if rmsd < rmsd_threshold:
                    print("✅ Reconstruction succeeded!")
                else:
                    print(f"❌ Reconstruction failed: RMSD exceeds threshold({rmsd_threshold:.0e}).")
            else:
                print("❌ Reconstruction failed: Pymatgen StructureMatcher could not match structures.")
        else:
            print(f"❌ Error: True Slab (tags 0, 1) not found in LMDB index {index}.")
    else:
        print(f"❌ Error: Could not extract True System from LMDB index {index}.")

    # ============================================
    # (Adsorbate-related logic is commented out)
    # ============================================
