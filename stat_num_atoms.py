import os
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

from recon_test import (
    create_slab_from_index,
    extract_true_system_from_lmdb,
    reconstruct_slab_from_shifted_ouc,
)


def compute_rmsd(pred_struct, true_struct) -> tuple[bool, float | None]:
    """예측 슬랩과 실제 슬랩의 RMSD를 계산한다."""
    try:
        matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
        if matcher.fit(pred_struct, true_struct):
            rms_dist, _ = matcher.get_rms_dist(pred_struct, true_struct)
            return True, rms_dist
        return False, None
    except Exception as exc:  # pragma: no cover - 분석 단계에서 유연성 확보용
        print(f"Error computing RMSD: {exc}")
        return False, None


ratios: list[float] = []
shifted_sizes: list[int] = []
pred_sizes: list[int] = []
pred_unit_sizes: list[int] = []

change_counter = defaultdict(list)  # increase / decrease / same -> idx 목록
rmsd_success = []
rmsd_fail = []
rmsd_missing = []

increase_count = decrease_count = same_count = 0
rmsd_success_count = rmsd_fail_count = rmsd_missing_count = 0
error_count = 0
error_indices: list[int] = []

parser = argparse.ArgumentParser()
parser.add_argument("--start_idx", type=int, default=0)
parser.add_argument("--end_idx", type=int, default=1000)
parser.add_argument("--lmdb_path", type=str, default="is2res_train_val_test_lmdbs/data/is2re/all/train/data.lmdb")
parser.add_argument("--mapping_path", type=str, default="oc20_data_mapping.pkl")
parser.add_argument("--min_slab_size", type=float, default=7.0)
parser.add_argument("--min_vacuum_size", type=float, default=20.0)
parser.add_argument("--min_ab", type=float, default=8.0)
parser.add_argument("--tolerance", type=float, default=0.3)
parser.add_argument("--pre_primitive_ouc", action="store_true")
args = parser.parse_args()

progress_bar = tqdm(range(args.start_idx, args.end_idx), desc="Analyzing slabs", leave=True)
for idx in progress_bar:
    try:
        shifted_ouc, is_struct_no_vac = create_slab_from_index(
            index=idx,
            lmdb_path=args.lmdb_path,
            mapping_path=args.mapping_path,
            min_slab_size=args.min_slab_size,
            min_vacuum_size=args.min_vacuum_size,
            output_dir=None,
        )
    except Exception as exc:
        print(f"Error creating shifted_ouc for idx={idx}: {exc}")
        error_count += 1
        error_indices.append(idx)
        progress_bar.set_postfix(
            inc=increase_count,
            dec=decrease_count,
            same=same_count,
            rmsd_ok=rmsd_success_count,
            rmsd_fail=rmsd_fail_count,
            err=error_count,
        )
        continue

    try:
        pred_slab_struct, pred_unit_struct, *_ = reconstruct_slab_from_shifted_ouc(
            shifted_ouc=shifted_ouc,
            min_slab_size=args.min_slab_size,
            min_vacuum_size=args.min_vacuum_size,
            min_ab=args.min_ab,
            tol=args.tolerance,
            center=True,
            primitive=True,
            pre_primitive_ouc=args.pre_primitive_ouc,
        )
    except Exception as exc:
        print(f"Error reconstructing slab for idx={idx}: {exc}")
        error_count += 1
        error_indices.append(idx)
        progress_bar.set_postfix(
            inc=increase_count,
            dec=decrease_count,
            same=same_count,
            rmsd_ok=rmsd_success_count,
            rmsd_fail=rmsd_fail_count,
            err=error_count,
        )
        continue

    shifted_len = len(shifted_ouc)
    pred_len = len(pred_slab_struct)
    pred_unit_len = len(pred_unit_struct)

    shifted_sizes.append(shifted_len)
    pred_sizes.append(pred_len)
    pred_unit_sizes.append(pred_unit_len)

    if shifted_len > 0:
        ratios.append(pred_len / shifted_len)

    diff = pred_len - shifted_len
    if diff > 0:
        decrease_count += 1
        change_counter["decrease"].append(idx)
    elif diff < 0:
        increase_count += 1
        change_counter["increase"].append(idx)
    else:
        same_count += 1
        change_counter["same"].append(idx)

    true_system_atoms = extract_true_system_from_lmdb(args.lmdb_path, idx)
    if true_system_atoms is not None and true_system_atoms.has("tags"):
        tags = true_system_atoms.get_tags()
        slab_mask = np.isin(tags, [0, 1])
        if np.any(slab_mask):
            true_slab_atoms = true_system_atoms[slab_mask]
            true_slab_struct = AseAtomsAdaptor.get_structure(true_slab_atoms)
            matched, rmsd = compute_rmsd(pred_slab_struct, true_slab_struct)
            # RMSD threshold: struct_no_vac 사용 시 완화
            rmsd_threshold = 1e-2 if is_struct_no_vac else 1e-3
            if matched and rmsd is not None and rmsd < rmsd_threshold:
                rmsd_success_count += 1
                rmsd_success.append((idx, rmsd))
            else:
                rmsd_fail_count += 1
                rmsd_fail.append((idx, rmsd))
        else:
            rmsd_missing_count += 1
            rmsd_missing.append(idx)
    else:
        rmsd_missing_count += 1
        rmsd_missing.append(idx)

    progress_bar.set_postfix(
        inc=increase_count,
        dec=decrease_count,
        same=same_count,
        rmsd_ok=rmsd_success_count,
        rmsd_fail=rmsd_fail_count,
        err=error_count,
    )

progress_bar.close()


def format_indices(indices: list[int], max_display: int = 20) -> str:
    if len(indices) <= max_display:
        return ", ".join(map(str, indices)) if indices else ""
    head = ", ".join(map(str, indices[:max_display]))
    return f"{head}, ... (total {len(indices)})"


def summary_change(title: str, indices: list[int]) -> None:
    count = len(indices)
    print(f"{title}: {count}")
    if count:
        print(f"  Indices: {format_indices(indices)}")


total_processed = len(pred_sizes)
print(f"Total processed directories: {total_processed}")
print(f"Errors encountered: {error_count}")
if error_indices:
    print(f"  Error indices: {format_indices(error_indices)}")
summary_change("Pred slab atoms increased", change_counter["increase"])
summary_change("Pred slab atoms decreased", change_counter["decrease"])
summary_change("Pred slab atoms unchanged", change_counter["same"])

print("\n=== RMSD Summary ===")
print(f"Success (<1e-3): {len(rmsd_success)}")
if rmsd_success:
    print(f"  Indices: {format_indices([idx for idx, _ in rmsd_success])}")
    example = rmsd_success[0]
    print(f"  Example rmsd={example[1]:.4e}")

print(f"Fail / mismatch: {len(rmsd_fail)}")
if rmsd_fail:
    print(f"  Indices: {format_indices([idx for idx, _ in rmsd_fail])}")
    example = rmsd_fail[0]
    rmsd_value = "None" if example[1] is None else f"{example[1]:.4e}"
    print(f"  Example rmsd={rmsd_value}")

print(f"Missing true slab: {len(rmsd_missing)}")
if rmsd_missing:
    print(f"  Indices: {format_indices(rmsd_missing)}")


# 비율 히스토그램 그리기
if ratios:
    plt.figure(figsize=(10, 6))
    plt.hist(ratios, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('len(pred_slab) / len(shifted_ouc)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Size Ratios (n={len(ratios)})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ratio_histogram.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n=== Ratio Statistics ===")
    print(f"Total processed: {len(ratios)}")
    print(f"Mean ratio: {np.mean(ratios):.2f}")
    print(f"Median ratio: {np.median(ratios):.2f}")
    print(f"Min ratio: {np.min(ratios):.2f}")
    print(f"Max ratio: {np.max(ratios):.2f}")

    # len(shifted_ouc) 히스토그램
    plt.figure(figsize=(10, 6))
    plt.hist(shifted_sizes, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('len(shifted_ouc)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of shifted_ouc Sizes (n={len(shifted_sizes)})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('shifted_ouc_size_histogram.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n=== len(shifted_ouc) Statistics ===")
    print(f"Total processed: {len(shifted_sizes)}")
    print(f"Mean size: {np.mean(shifted_sizes):.2f}")
    print(f"Median size: {np.median(shifted_sizes):.2f}")
    print(f"Min size: {np.min(shifted_sizes):.0f}")
    print(f"Max size: {np.max(shifted_sizes):.0f}")

    # len(pred_unit_slab) 히스토그램
    plt.figure(figsize=(10, 6))
    plt.hist(pred_unit_sizes, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('len(pred_unit_slab)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of pred_unit_slab Sizes (n={len(pred_unit_sizes)})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pred_unit_slab_size_histogram.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n=== len(pred_unit_slab) Statistics ===")
    print(f"Total processed: {len(pred_unit_sizes)}")
    print(f"Mean size: {np.mean(pred_unit_sizes):.2f}")
    print(f"Median size: {np.median(pred_unit_sizes):.2f}")
    print(f"Min size: {np.min(pred_unit_sizes):.0f}")
    print(f"Max size: {np.max(pred_unit_sizes):.0f}")

    # len(pred_slab) 히스토그램
    plt.figure(figsize=(10, 6))
    plt.hist(pred_sizes, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('len(pred_slab)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of pred_slab Sizes (n={len(pred_sizes)})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pred_slab_size_histogram.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n=== len(pred_slab) Statistics ===")
    print(f"Total processed: {len(pred_sizes)}")
    print(f"Mean size: {np.mean(pred_sizes):.2f}")
    print(f"Median size: {np.median(pred_sizes):.2f}")
    print(f"Min size: {np.min(pred_sizes):.0f}")
    print(f"Max size: {np.max(pred_sizes):.0f}")

    print("\nPDF files saved: ratio_histogram.pdf, shifted_ouc_size_histogram.pdf, pred_unit_slab_size_histogram.pdf, pred_slab_size_histogram.pdf")
else:
    print("No valid data found!")