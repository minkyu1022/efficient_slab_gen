from fairchem.data.oc.core import Bulk
from pymatgen.core.surface import SlabGenerator
from fairchem.data.oc.core.slab import standardize_bulk
import math
import argparse
import pandas as pd
import ast
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

min_ab = 8.0
min_slab_size = 7.0
min_vacuum_size = 20.0


def load_single_bulk(bulk_src_id):
    """단일 bulk 로드 함수 (멀티프로세싱용)"""
    try:
        bulk = Bulk(bulk_src_id_from_db=bulk_src_id)
        return bulk_src_id, bulk.atoms.copy(), None
    except Exception as e:
        return bulk_src_id, None, str(e)


def process_row(args):
    """단일 row 처리 함수 (멀티프로세싱용)"""
    row_idx, row, bulk_atoms_dict = args
    
    # CSV에서 읽은 값들의 타입:
    # - bulk_src_id: str
    # - specific_miller: str (예: '(1, 0, 0)')
    bulk_src_id = str(row["bulk_src_id"])
    specific_miller_str = str(row["specific_miller"])
    
    # specific_miller를 tuple[int, ...]로 변환 (SlabGenerator가 기대하는 타입)
    specific_miller = ast.literal_eval(specific_miller_str)
    if isinstance(specific_miller, list):
        specific_miller = tuple(specific_miller)
    
    # bulk atoms 가져오기
    bulk_atoms = bulk_atoms_dict[bulk_src_id]
    
    initial_structure = standardize_bulk(bulk_atoms)
    slab_gen = SlabGenerator(
        initial_structure=initial_structure,
        miller_index=specific_miller,
        min_slab_size=7.0,
        min_vacuum_size=20.0,
        lll_reduce=False,
        center_slab=False,
        primitive=True,
        max_normal_search=1,
    )

    height = slab_gen._proj_height
    n_layers_slab = math.ceil(slab_gen.min_slab_size / height)
    n_layers_vac = math.ceil(slab_gen.min_vac_size / height)

    return row_idx, True, n_layers_slab, n_layers_vac, height, None


def main():
    parser = argparse.ArgumentParser(description="Batch extract slab info and update CSV")
    parser.add_argument("--metadata-csv", type=str, default="info_from_lmdb/metadata.csv")
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive). If None, process all rows.")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of parallel workers. Default: CPU count")
    parser.add_argument("--output-csv", type=str, default=None, help="Output CSV path. Default: overwrite input")
    parser.add_argument("--save-every", type=int, default=10000, help="Save CSV every N rows (0 = save only at end)")
    args = parser.parse_args()

    metadata_csv = args.metadata_csv
    start_index = args.start
    end_index = args.end
    num_workers = args.num_workers or cpu_count()
    output_csv = args.output_csv or metadata_csv
    save_every = args.save_every

    # Load metadata CSV
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    print(f"Loading metadata CSV...")
    metadata_df = pd.read_csv(metadata_csv)

    if end_index is None:
        end_index = len(metadata_df)

    if end_index <= start_index:
        raise ValueError(f"end index({end_index}) must be greater than start index({start_index})")

    # 컬럼 초기화 (없으면 생성)
    if 'n_c' not in metadata_df.columns:
        metadata_df['n_c'] = None
    if 'n_vac' not in metadata_df.columns:
        metadata_df['n_vac'] = None
    if 'height' not in metadata_df.columns:
        metadata_df['height'] = None

    rows_to_process = metadata_df.iloc[start_index:end_index]
    
    print(f"Processing {len(rows_to_process)} rows (index {start_index} to {end_index-1})")
    print(f"Using {num_workers} workers")

    # 필요한 bulk들만 미리 로드 (병렬)
    unique_bulk_ids = list(rows_to_process["bulk_src_id"].unique())
    print(f"Loading {len(unique_bulk_ids)} unique bulks (parallel with {num_workers} workers)...")
    
    bulk_atoms_dict = {}
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(load_single_bulk, unique_bulk_ids, chunksize=10),
            total=len(unique_bulk_ids),
            desc="Loading bulks"
        ))
    
    for bulk_src_id, atoms, error in results:
        if atoms is not None:
            bulk_atoms_dict[bulk_src_id] = atoms
        else:
            print(f"Failed to load bulk {bulk_src_id}: {error}")
    
    print(f"Loaded {len(bulk_atoms_dict)} bulks successfully")

    # 멀티프로세싱으로 처리
    task_args = [
        (start_index + i, row, bulk_atoms_dict) 
        for i, (_, row) in enumerate(rows_to_process.iterrows())
    ]
    
    success_count = 0
    fail_count = 0
    
    print("Processing rows...")
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_row, task_args, chunksize=100),
            total=len(task_args),
            desc="Processing rows"
        ))
    
    # 결과를 DataFrame에 반영
    print("Updating DataFrame...")
    for i, (row_idx, success, n_c, n_vac, height, error) in enumerate(results):
        if success:
            metadata_df.at[row_idx, 'n_c'] = n_c
            metadata_df.at[row_idx, 'n_vac'] = n_vac
            metadata_df.at[row_idx, 'height'] = height
            success_count += 1
        else:
            fail_count += 1
            print(f"Error processing row {row_idx}: {error}")
        
        # 주기적 저장 (체크포인트)
        if save_every > 0 and (i + 1) % save_every == 0:
            print(f"Checkpoint: saving after {i + 1} rows...")
            metadata_df.to_csv(output_csv, index=False)

    # 최종 CSV 저장
    print(f"Final save to {output_csv}...")
    metadata_df.to_csv(output_csv, index=False)

    print(f"\nBatch processing complete!")
    print(f"Success: {success_count}, Failed: {fail_count}")
    print(f"Output saved to: {output_csv}")


if __name__ == "__main__":
    main()
