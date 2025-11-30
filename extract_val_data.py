"""
LMDB 파일과 mapping 파일로부터 모든 정보를 추출하여 val.csv 파일에 저장하는 스크립트

추출하는 정보:
- sid
- bulk_src_id (mapping 파일의 bulk_mpid)
- specific_miller (mapping 파일의 miller_index)
- shift
- top
- true_system_atomic_numbers
- true_system_positions
- true_lattice
- true_tags
- adsorbate_atomic_numbers
- ads_pos_relaxed
"""

from __future__ import annotations

import argparse
import ast
import lmdb
import numpy as np
import os
import pandas as pd
import pickle
import torch
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def get_all_data_from_lmdb(lmdb_path: str, index: int) -> dict | None:
    """
    LMDB 파일에서 모든 필요한 정보를 추출합니다.
    
    Arguments
    ---------
    lmdb_path: str
        Path to LMDB file
    index: int
        Data index to extract
        
    Returns
    -------
    dict | None
        Dictionary containing all extracted data, or None if failed
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
                print(f"Warning: Index {index} not found in LMDB")
                return None
            
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
        for key in ['tags', 'atomic_numbers', 'pos', 'pos_relaxed', 'cell', 'sid']:
            if key not in data_dict:
                try:
                    value = getattr(data, key, None)
                    if value is not None:
                        data_dict[key] = value
                except (RuntimeError, AttributeError):
                    pass
        
        # Extract sid
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
        
        # Method 6: From data_dict
        if sid is None and 'sid' in data_dict:
            sid = data_dict['sid']
        
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
            return None
        
        # Convert tags to numpy array
        if isinstance(tags, torch.Tensor):
            tags_np = tags.cpu().numpy()
        else:
            tags_np = np.array(tags)
        
        # Find indices where tags==0, 1, or 2 (slab + adsorbate)
        atom_mask = (tags_np == 0) | (tags_np == 1) | (tags_np == 2)
        adsorbate_mask = (tags_np == 2)
        
        if not np.any(atom_mask):
            print(f"Warning: No atoms with tags 0, 1, or 2 found at index {index}")
            return None
        
        # Extract atomic_numbers
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
        
        true_system_atomic_numbers = atomic_numbers_np[atom_mask]
        adsorbate_atomic_numbers = atomic_numbers_np[adsorbate_mask] if np.any(adsorbate_mask) else np.array([])
        
        # Extract positions
        pos = None
        if 'pos' in data_dict:
            pos = data_dict['pos']
        else:
            try:
                pos = getattr(data, 'pos', None)
            except (RuntimeError, AttributeError):
                pass
        
        if pos is None:
            # Try additional methods
            try:
                if hasattr(data, '__dict__') and 'pos' in data.__dict__:
                    pos = data.__dict__['pos']
            except Exception:
                pass
            
            if pos is None:
                try:
                    if hasattr(data, 'items'):
                        for key, value in data.items():
                            if key == 'pos':
                                pos = value
                                break
                except (RuntimeError, AttributeError, TypeError):
                    pass
        
        if pos is None:
            print(f"Warning: No 'pos' attribute found in data at index {index}")
            return None
        
        if isinstance(pos, torch.Tensor):
            pos_np = pos.cpu().numpy()
        else:
            pos_np = np.array(pos)
        
        true_system_positions = pos_np[atom_mask]
        
        # Extract cell/lattice
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
        
        true_lattice = cell_np
        
        # Extract pos_relaxed (only for adsorbate)
        ads_pos_relaxed = None
        pos_relaxed = None
        
        if 'pos_relaxed' in data_dict:
            pos_relaxed = data_dict['pos_relaxed']
        else:
            try:
                pos_relaxed = getattr(data, 'pos_relaxed', None)
            except (RuntimeError, AttributeError):
                pass
            
            if pos_relaxed is None:
                try:
                    if hasattr(data, '__dict__') and 'pos_relaxed' in data.__dict__:
                        pos_relaxed = data.__dict__['pos_relaxed']
                except Exception:
                    pass
            
            if pos_relaxed is None:
                try:
                    if hasattr(data, 'items'):
                        for key, value in data.items():
                            if key == 'pos_relaxed':
                                pos_relaxed = value
                                break
                except (RuntimeError, AttributeError, TypeError):
                    pass
        
        if pos_relaxed is not None:
            if isinstance(pos_relaxed, torch.Tensor):
                pos_relaxed_np = pos_relaxed.cpu().numpy()
            else:
                pos_relaxed_np = np.array(pos_relaxed)
            
            if np.any(adsorbate_mask):
                ads_pos_relaxed = pos_relaxed_np[adsorbate_mask]
        
        # Return all extracted data
        return {
            'sid': sid,
            'true_system_atomic_numbers': true_system_atomic_numbers,
            'true_system_positions': true_system_positions,
            'true_lattice': true_lattice,
            'true_tags': tags_np[atom_mask],
            'adsorbate_atomic_numbers': adsorbate_atomic_numbers,
            'ads_pos_relaxed': ads_pos_relaxed,
        }
        
    except Exception as e:
        print(f"Error extracting data from LMDB at index {index}: {e}")
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


def process_row(args):
    """Process a single row - extract all data from LMDB and mapping file"""
    row_idx, lmdb_path, mapping_path = args
    
    try:
        index = row_idx  # Assuming index is the row number or stored elsewhere
        
        # Extract data from LMDB
        lmdb_data = get_all_data_from_lmdb(lmdb_path, index)
        
        if lmdb_data is None:
            return row_idx, False, None, "Failed to extract data from LMDB"
        
        sid = lmdb_data['sid']
        
        # Extract data from mapping file
        slab_params = None
        if sid is not None and mapping_path and os.path.exists(mapping_path):
            slab_params = get_slab_params_from_mapping(mapping_path, sid)
        
        # Prepare result dictionary
        result = {
            'sid': sid,
            'bulk_src_id': slab_params['bulk_mpid'] if slab_params else None,
            'specific_miller': str(slab_params['miller_index']) if slab_params and slab_params.get('miller_index') else None,
            'shift': slab_params['shift'] if slab_params else None,
            'top': slab_params['top'] if slab_params else None,
            'true_system_atomic_numbers': str(lmdb_data['true_system_atomic_numbers'].tolist()),
            'true_system_positions': str(lmdb_data['true_system_positions'].tolist()),
            'true_lattice': str(lmdb_data['true_lattice'].tolist()),
            'true_tags': str(lmdb_data['true_tags'].tolist()),
            'adsorbate_atomic_numbers': str(lmdb_data['adsorbate_atomic_numbers'].tolist()) if len(lmdb_data['adsorbate_atomic_numbers']) > 0 else None,
            'ads_pos_relaxed': str(lmdb_data['ads_pos_relaxed'].tolist()) if lmdb_data['ads_pos_relaxed'] is not None else None,
        }
        
        return row_idx, True, result, None
        
    except Exception as e:
        return row_idx, False, None, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Extract all data from LMDB and mapping file to create val.csv"
    )
    parser.add_argument(
        "--lmdb-path",
        type=str,
        required=True,
        help="Path to LMDB file"
    )
    parser.add_argument(
        "--mapping-path",
        type=str,
        default=None,
        help="Path to oc20_data_mapping.pkl file (optional)"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="val.csv",
        help="Output CSV path (default: val.csv)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index (inclusive)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index (exclusive). If None, process all indices in LMDB."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers. Default: CPU count"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10000,
        help="Save CSV every N rows (0 = save only at end)"
    )
    parser.add_argument(
        "--num-indices",
        type=int,
        default=None,
        help="Number of indices in LMDB (if known). If None, will try to count."
    )
    
    args = parser.parse_args()
    
    lmdb_path = args.lmdb_path
    mapping_path = args.mapping_path
    output_csv = args.output_csv
    start_index = args.start
    end_index = args.end
    num_workers = args.num_workers or cpu_count()
    save_every = args.save_every
    num_indices = args.num_indices
    
    # Count indices in LMDB if not provided
    if num_indices is None or end_index is None:
        print("Counting indices in LMDB...")
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
                cursor = txn.cursor()
                num_indices = sum(1 for _ in cursor)
        finally:
            db.close()
        print(f"Found {num_indices} indices in LMDB")
    
    if end_index is None:
        end_index = num_indices
    
    if end_index <= start_index:
        raise ValueError(f"end index({end_index}) must be greater than start index({start_index})")
    
    # Create list of indices to process
    indices_to_process = list(range(start_index, end_index))
    
    print(f"Processing {len(indices_to_process)} indices (index {start_index} to {end_index-1})")
    print(f"Using {num_workers} workers")
    print(f"LMDB path: {lmdb_path}")
    if mapping_path:
        print(f"Mapping path: {mapping_path}")
    
    # Prepare arguments for parallel processing
    process_args = [
        (idx, lmdb_path, mapping_path)
        for idx in indices_to_process
    ]
    
    # Initialize result dataframe
    columns = [
        'sid', 'bulk_src_id', 'specific_miller', 'shift', 'top',
        'true_system_atomic_numbers', 'true_system_positions', 'true_lattice',
        'true_tags', 'adsorbate_atomic_numbers', 'ads_pos_relaxed'
    ]
    results_df = pd.DataFrame(columns=columns)
    
    # Process rows in parallel
    results_list = []
    with Pool(num_workers) as pool:
        with tqdm(total=len(process_args), desc="Extracting data") as pbar:
            for row_idx, success, result, error in pool.imap(process_row, process_args):
                if success and result is not None:
                    results_list.append(result)
                else:
                    print(f"Error processing index {row_idx}: {error}")
                    # Add row with None values
                    results_list.append({
                        'sid': None,
                        'bulk_src_id': None,
                        'specific_miller': None,
                        'shift': None,
                        'top': None,
                        'true_system_atomic_numbers': None,
                        'true_system_positions': None,
                        'true_lattice': None,
                        'true_tags': None,
                        'adsorbate_atomic_numbers': None,
                        'ads_pos_relaxed': None,
                    })
                
                pbar.update(1)
                
                # Save periodically
                if save_every > 0 and (pbar.n % save_every == 0):
                    print(f"\nSaving progress at {pbar.n} rows...")
                    temp_df = pd.DataFrame(results_list)
                    temp_df.to_csv(output_csv, index=False)
    
    # Create final dataframe
    results_df = pd.DataFrame(results_list)
    
    # Final save
    print(f"\nSaving final results to {output_csv}...")
    results_df.to_csv(output_csv, index=False)
    
    # Print summary
    non_null_sid = results_df['sid'].notna().sum()
    non_null_bulk_src_id = results_df['bulk_src_id'].notna().sum()
    non_null_ads_pos_relaxed = results_df['ads_pos_relaxed'].notna().sum()
    
    print(f"\nSummary:")
    print(f"  Total rows processed: {len(results_list)}")
    print(f"  Successfully extracted sid: {non_null_sid}")
    print(f"  Successfully extracted bulk_src_id: {non_null_bulk_src_id}")
    print(f"  Successfully extracted ads_pos_relaxed: {non_null_ads_pos_relaxed}")


if __name__ == "__main__":
    main()

