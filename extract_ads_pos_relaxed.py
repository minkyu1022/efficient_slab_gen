"""
LMDB 파일로부터 pos_relaxed 정보를 추출하여 metadata.csv에 ads_pos_relaxed 컬럼으로 추가하는 스크립트

get_sid_and_adsorbate_from_lmdb 함수를 참고하여 pos_relaxed에서 tags==2인 부분(adsorbate)만 추출합니다.
"""

from __future__ import annotations

import argparse
import lmdb
import numpy as np
import os
import pandas as pd
import pickle
import torch
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def get_adsorbate_pos_relaxed_from_lmdb(lmdb_path: str, index: int) -> np.ndarray | None:
    """
    LMDB 파일에서 pos_relaxed 정보를 추출하고, tags==2인 부분(adsorbate)의 위치만 반환합니다.
    
    Arguments
    ---------
    lmdb_path: str
        Path to LMDB file
    index: int
        Data index to extract
        
    Returns
    -------
    np.ndarray | None
        Adsorbate relaxed positions (shape: (n_adsorbate_atoms, 3)), or None if not found
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
        if not data_dict or 'pos_relaxed' not in data_dict:
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
        for key in ['tags', 'pos_relaxed']:
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
        
        # Find indices where tags==2 (adsorbate atoms)
        adsorbate_mask = (tags_np == 2)
        
        if not np.any(adsorbate_mask):
            # No adsorbate atoms found
            return None
        
        # Extract pos_relaxed information
        pos_relaxed = None
        
        # Method 1: Get from data_dict
        if 'pos_relaxed' in data_dict:
            pos_relaxed = data_dict['pos_relaxed']
        else:
            # Method 2: Try direct getattr
            try:
                pos_relaxed = getattr(data, 'pos_relaxed', None)
            except (RuntimeError, AttributeError):
                pass
            
            # Method 3: Look in __dict__
            if pos_relaxed is None:
                try:
                    if hasattr(data, '__dict__'):
                        if 'pos_relaxed' in data.__dict__:
                            pos_relaxed = data.__dict__['pos_relaxed']
                except Exception:
                    pass
            
            # Method 4: Use items() method (PyTorch Geometric)
            if pos_relaxed is None:
                try:
                    if hasattr(data, 'items'):
                        for key, value in data.items():
                            if key == 'pos_relaxed':
                                pos_relaxed = value
                                break
                except (RuntimeError, AttributeError, TypeError):
                    pass
            
            # Method 5: Check keys() then direct access
            if pos_relaxed is None:
                try:
                    if hasattr(data, 'keys'):
                        keys = list(data.keys())
                        if 'pos_relaxed' in keys:
                            pos_relaxed = data['pos_relaxed']
                except (RuntimeError, AttributeError, TypeError, KeyError):
                    pass
        
        if pos_relaxed is None:
            print(f"Warning: No 'pos_relaxed' attribute found in data at index {index}")
            return None
        
        # Convert to numpy array
        if isinstance(pos_relaxed, torch.Tensor):
            pos_relaxed_np = pos_relaxed.cpu().numpy()
        else:
            pos_relaxed_np = np.array(pos_relaxed)
        
        # Filter only adsorbate positions (tags==2)
        adsorbate_pos_relaxed = pos_relaxed_np[adsorbate_mask]
        
        return adsorbate_pos_relaxed
        
    except Exception as e:
        print(f"Error extracting pos_relaxed from LMDB at index {index}: {e}")
        return None
    finally:
        db.close()


def process_row(args):
    """Process a single row from metadata CSV"""
    row_idx, row, lmdb_path = args
    
    try:
        index = row['index']
        ads_pos_relaxed = get_adsorbate_pos_relaxed_from_lmdb(lmdb_path, index)
        
        if ads_pos_relaxed is not None:
            # Convert numpy array to list for CSV storage
            ads_pos_relaxed_str = str(ads_pos_relaxed.tolist())
        else:
            ads_pos_relaxed_str = None
            
        return row_idx, True, ads_pos_relaxed_str, None
    except Exception as e:
        return row_idx, False, None, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Extract adsorbate relaxed positions from LMDB and add to metadata.csv"
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default="metadata.csv",
        help="Path to metadata CSV file"
    )
    parser.add_argument(
        "--lmdb-path",
        type=str,
        required=True,
        help="Path to LMDB file"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Output CSV path. Default: overwrite input"
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
        help="End index (exclusive). If None, process all rows."
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
    
    args = parser.parse_args()
    
    metadata_csv = args.metadata_csv
    lmdb_path = args.lmdb_path
    output_csv = args.output_csv or metadata_csv
    start_index = args.start
    end_index = args.end
    num_workers = args.num_workers or cpu_count()
    save_every = args.save_every
    
    # Load metadata CSV
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    
    print(f"Loading metadata CSV: {metadata_csv}")
    metadata_df = pd.read_csv(metadata_csv)
    
    if end_index is None:
        end_index = len(metadata_df)
    
    if end_index <= start_index:
        raise ValueError(f"end index({end_index}) must be greater than start index({start_index})")
    
    # 컬럼 초기화 (없으면 생성)
    if 'ads_pos_relaxed' not in metadata_df.columns:
        metadata_df['ads_pos_relaxed'] = None
    
    rows_to_process = metadata_df.iloc[start_index:end_index]
    
    print(f"Processing {len(rows_to_process)} rows (index {start_index} to {end_index-1})")
    print(f"Using {num_workers} workers")
    print(f"LMDB path: {lmdb_path}")
    
    # Prepare arguments for parallel processing
    process_args = [
        (idx, row, lmdb_path)
        for idx, row in rows_to_process.iterrows()
    ]
    
    # Process rows in parallel
    results = {}
    with Pool(num_workers) as pool:
        with tqdm(total=len(process_args), desc="Extracting pos_relaxed") as pbar:
            for row_idx, success, ads_pos_relaxed_str, error in pool.imap(process_row, process_args):
                if success:
                    metadata_df.loc[row_idx, 'ads_pos_relaxed'] = ads_pos_relaxed_str
                else:
                    print(f"Error processing row {row_idx}: {error}")
                    metadata_df.loc[row_idx, 'ads_pos_relaxed'] = None
                
                results[row_idx] = (success, ads_pos_relaxed_str, error)
                pbar.update(1)
                
                # Save periodically
                if save_every > 0 and (pbar.n % save_every == 0):
                    print(f"\nSaving progress at row {pbar.n}...")
                    metadata_df.to_csv(output_csv, index=False)
    
    # Final save
    print(f"\nSaving final results to {output_csv}...")
    metadata_df.to_csv(output_csv, index=False)
    
    # Print summary
    non_null_count = metadata_df['ads_pos_relaxed'].notna().sum()
    print(f"\nSummary:")
    print(f"  Total rows processed: {len(rows_to_process)}")
    print(f"  Successfully extracted: {non_null_count}")
    print(f"  Failed/Not found: {len(rows_to_process) - non_null_count}")


if __name__ == "__main__":
    main()

