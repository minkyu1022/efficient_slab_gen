from __future__ import annotations

import math
import os
import pickle
import pandas as pd
import ast
import lmdb
import numpy as np
import torch
from collections import Counter
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms
from ase.io import write
from pymatgen.core import Lattice, Structure

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

# -----------------------------------------------------------------------------
# 1. Helper: 비율 검증 함수 (New!)
# -----------------------------------------------------------------------------
def verify_slab_ratio(true_slab, chunk_atoms, repeat):
    print("\n--- Verifying Extraction Ratio ---")
    na, nb, nc = repeat
    total_chunks = na * nb * nc
    
    # 1. True Slab 정보
    # (만약 true_slab에 adsorbate가 없다면 전체가 slab)
    # 안전을 위해 user가 넣은 slab파일은 순수 slab이라고 가정
    n_true = len(true_slab)
    comp_true = Counter(true_slab.get_chemical_symbols())
    
    # 2. Chunk Slab 정보 (Tag!=2 만 필터링)
    slab_indices = [a.index for a in chunk_atoms if a.tag != 2]
    chunk_slab = chunk_atoms[slab_indices]
    n_chunk = len(chunk_slab)
    comp_chunk = Counter(chunk_slab.get_chemical_symbols())
    
    # 3. 개수 검증
    expected_n = n_true / total_chunks
    
    print(f"Input Repeat: {repeat} -> Total Divisions: {total_chunks}")
    print(f"True Slab Total Atoms: {n_true}")
    print(f"Expected Atoms per Chunk: {expected_n:.2f}")
    print(f"Actual Chunk Slab Atoms:  {n_chunk}")
    
    if n_chunk != int(expected_n):
        print(f">> [FAIL] Count Mismatch! Expected {int(expected_n)}, got {n_chunk}")
        print("   Possible reasons: Incorrect N_c (height division) or Ghost atoms.")
        return False
    
    # 4. 조성(Stoichiometry) 검증
    print("Checking Elemental Composition...")
    all_pass = True
    for elem, count in comp_true.items():
        expected_elem_count = count / total_chunks
        actual_elem_count = comp_chunk[elem]
        
        if actual_elem_count != int(expected_elem_count):
            print(f"   - {elem}: Expected {expected_elem_count:.1f}, Got {actual_elem_count} [FAIL]")
            all_pass = False
        else:
            print(f"   - {elem}: Expected {expected_elem_count:.1f}, Got {actual_elem_count} [OK]")
            
    if all_pass:
        print(">> [PASS] The chunk is exactly 1/N of the true slab.")
        return True
    else:
        print(">> [FAIL] Stoichiometry mismatch.")
        return False

def extract_manual_chunk_clean(true_slab, true_system, ads_site_cart, 
                               repeat, user_tags, 
                               output_file="manual_chunk_clean.cif"):
    
    print(f"\n{'='*50}")
    print(f" EXTRACTION (Slab: Strict Grid / Ads: Unconditional)")
    print(f"{'='*50}")
    
    na, nb, nc = repeat
    
    if len(user_tags) != len(true_system):
        print("[ERROR] Tag mismatch.")
        return None
    true_system.set_tags(user_tags)

    # 1. Slab 원자 수 예측 (단원자 Unit 여부 확인용)
    slab_atoms_only = [a for a in true_system if a.tag != 2]
    n_slab_total = len(slab_atoms_only)
    expected = n_slab_total / (na * nb * nc)
    print(f">> Expected Unit Slab Atoms: {expected:.2f}")
    
    # 2. Anchor(기준점) 설정
    target_idx = None
    
    # Case A: Unit Cell이 단원자일 때 (Slab 원자를 기준으로 삼아야 안전함)
    if abs(expected - 1.0) < 0.01: 
        print("   [Logic] Single-atom unit detected. Anchoring on Nearest Slab Atom.")
        slab_indices = [i for i, a in enumerate(true_system) if a.tag != 2]
        slab_pos = true_system.positions[slab_indices]
        dists = np.linalg.norm(slab_pos - ads_site_cart, axis=1)
        target_idx = slab_indices[np.argmin(dists)]
        
    # Case B: 복합 Unit Cell 혹은 다른 경우 (Adsorbate 기준)
    else:
        print("   [Logic] Multi-atom unit detected. Anchoring on Adsorbate.")
        ads_indices = [i for i, a in enumerate(true_system) if a.tag == 2]
        if not ads_indices: return None
        ads_pos = true_system.positions[ads_indices]
        dists = np.linalg.norm(ads_pos - ads_site_cart, axis=1)
        target_idx = ads_indices[np.argmin(dists)]

    # 3. Shift Logic (Anchor를 0번 Grid의 중심 0.5/Na로 이동)
    work_atoms = true_system.copy() 
    frac_coords = work_atoms.get_scaled_positions()
    target_frac = frac_coords[target_idx]

    shift_vec = np.array([0.5/na - target_frac[0], 
                          0.5/nb - target_frac[1], 
                          0.0]) 

    # 전체 시스템 이동 및 [0, 1) 범위로 래핑
    shifted_frac = frac_coords + shift_vec
    shifted_frac[:, :2] %= 1.0 
    
    # 4. Z-Cut 설정
    slab_indices_all = [i for i, a in enumerate(true_system) if a.tag != 2]
    w_slab = shifted_frac[slab_indices_all, 2]
    w_min, w_max = np.min(w_slab), np.max(w_slab)
    w_unit = (w_max - w_min) / nc
    
    k_target = nc - 1 # Top layer
    eps = 1e-8
    w_start = w_min + k_target * w_unit - eps
    w_end = w_min + (k_target + 1) * w_unit + eps
    
    # 5. 원자 추출 (핵심 변경 사항)
    indices = []
    final_scaled_pos = []
    
    for i, atom in enumerate(work_atoms):
        u, v, w = shifted_frac[i]
        
        # -------------------------------------------------------
        # [LOGIC A] Adsorbate (Tag == 2): 무조건 포함 (Unconditional)
        # -------------------------------------------------------
        if atom.tag == 2:
            indices.append(i)
            
            # 좌표 변환: [0, 1] -> [0, Na] (Unit cell 기준으로는 > 1이 될 수 있음)
            # "삐져나가도 되니까 냅둬" -> 별도의 % 1.0 처리 없이 그대로 확장
            new_u = u * na
            new_v = v * nb
            new_w = w
            
            final_scaled_pos.append([new_u, new_v, new_w])

        # -------------------------------------------------------
        # [LOGIC B] Slab Atoms (Tag != 2): 엄격한 Grid 검사
        # -------------------------------------------------------
        else:
            # Z축 검사 (Top Layer 여부)
            if not (w_start <= w <= w_end): continue
            
            # XY축 검사 (0번 Grid 포함 여부)
            # Anchor가 (0.5/na, 0.5/nb)에 있으므로 이 범위 안에 무조건 하나는 들어옴
            if u < 1.0/na - eps and v < 1.0/nb - eps:
                indices.append(i)
                
                new_u = u * na
                new_v = v * nb
                new_w = w
                
                final_scaled_pos.append([new_u, new_v, new_w])

    extracted = true_system[indices].copy()
    print(f">> Extracted Atoms: {len(extracted)}")
    
    # 6. 검증 및 저장
    n_slab_extracted = sum(1 for a in extracted if a.tag != 2)
    n_ads_extracted = sum(1 for a in extracted if a.tag == 2)
    
    print(f">> Slab Count: {n_slab_extracted} (Expected: {int(expected)})")
    print(f">> Ads Count : {n_ads_extracted} (Included All)")

    # 격자 축소
    old_cell = true_system.get_cell()
    new_cell = np.array([old_cell[0]/na, old_cell[1]/nb, old_cell[2]])
    
    patch_atoms = Atoms(
        symbols=extracted.get_chemical_symbols(),
        cell=new_cell,
        scaled_positions=final_scaled_pos,
        pbc=[True, True, True],
        tags=extracted.get_tags()
    )
    
    if output_file:
        write(output_file, patch_atoms)
        
    return patch_atoms

# =============================================================================
# 2. Recon Function (Relative Position Preservation)
# =============================================================================
def reconstruct_oneshot_smart(chunk_input, repeat, height, min_vacuum_size=20.0):
    print(f"\n--- Reconstruction (V2) ---")
    na, nb, nc = repeat
    chunk = chunk_input.copy()
    
    slab_indices = [a.index for a in chunk if a.tag != 2]
    ads_indices = [a.index for a in chunk if a.tag == 2]
    
    if not slab_indices: return None
    
    slab = chunk[slab_indices].copy()
    ads = chunk[ads_indices].copy()
    
    # Lattice Setup
    n_vac = math.ceil(min_vacuum_size / height)
    n_total = nc + n_vac
    
    cell = slab.get_cell()
    a_vec, b_vec, c_vec_orig = cell[0], cell[1], cell[2]
    
    # Tight C calculation
    cross_prod = np.cross(a_vec, b_vec)
    normal_vec = cross_prod / np.linalg.norm(cross_prod)
    current_proj_h = abs(np.dot(c_vec_orig, normal_vec))
    
    scale_factor = height / current_proj_h
    tight_c_vec = c_vec_orig * scale_factor
    super_c_vec = tight_c_vec * n_total
    
    # Slab Stacking
    slab.set_cell([a_vec, b_vec, tight_c_vec], scale_atoms=False)
    slab.center(axis=2)
    
    frac_slab = slab.get_scaled_positions()
    frac_slab[:, 2] /= n_total
    
    all_coords = []
    base_symbols = slab.get_chemical_symbols()
    final_symbols = []
    base_tags = slab.get_tags()
    final_tags = []
    
    for k in range(nc):
        _f = frac_slab.copy()
        _f[:, 2] += k / n_total
        all_coords.append(_f)
        final_symbols.extend(base_symbols)
        final_tags.extend(base_tags)
        
    slab_1x1 = Atoms(symbols=final_symbols, scaled_positions=np.vstack(all_coords), 
                     cell=np.array([a_vec, b_vec, super_c_vec]), pbc=[True, True, True], tags=final_tags)
    
    # XY Tiling
    final_system = slab_1x1.repeat((na, nb, 1))
    
    # [핵심] Adsorbate 배치 (Chunk #1에 배치하되, Slab과의 상대 위치 보존)
    if len(ads) > 0:
        # Chunk 상태에서의 Slab Top Z (Center 정렬 전)
        chunk_slab_z = chunk[slab_indices].positions[:, 2]
        chunk_slab_top = np.max(chunk_slab_z)
        
        # Final System에서의 Top Z
        final_slab_z = final_system.positions[:, 2]
        final_top_z = np.max(final_slab_z)
        
        ads_fracs_unit = ads.get_scaled_positions()
        new_ads_fracs = []
        new_ads_z_cart = []
        
        for i in range(len(ads)):
            # XY: Unit(0~1) -> Super(0~1/N) (0번 타일)
            u_super = ads_fracs_unit[i][0] / na
            v_super = ads_fracs_unit[i][1] / nb
            
            # Z: 높이 차이(dh)를 유지
            dh = ads.positions[i, 2] - chunk_slab_top
            z_super = final_top_z + dh
            
            new_ads_fracs.append([u_super, v_super, 0.0]) # Z temp
            new_ads_z_cart.append(z_super)
            
        ads_recon = Atoms(symbols=ads.get_chemical_symbols(), cell=final_system.get_cell(), tags=[2]*len(ads))
        ads_recon.set_scaled_positions(new_ads_fracs)
        
        # Z 덮어쓰기
        pos = ads_recon.get_positions()
        pos[:, 2] = new_ads_z_cart
        ads_recon.set_positions(pos)
        
        final_system += ads_recon
        
    return final_system

# =============================================================================
# 3. Verify Function (Fixed to Check Counts)
# =============================================================================
def verify_integrity_object(true_system, chunk_atoms, ads_site_cart):
    print("\n--- Structure Integrity Check ---")
    
    # 1. 개수 검증 (가장 중요)
    n_chunk_slab = sum(1 for a in chunk_atoms if a.tag != 2)
    print(f"Chunk Slab Atoms: {n_chunk_slab}")
    
    # 2. 거리 검증
    pos_t = true_system.positions
    dists_t = np.linalg.norm(pos_t - ads_site_cart, axis=1)
    target_idx_t = np.argmin(dists_t)
    
    chunk_frac = chunk_atoms.get_scaled_positions()
    dists_from_center = np.linalg.norm(chunk_frac[:, :2] - np.array([0.5, 0.5]), axis=1)
    
    ads_indices_c = [a.index for a in chunk_atoms if a.tag == 2]
    if not ads_indices_c: return

    target_idx_c = min(ads_indices_c, key=lambda i: dists_from_center[i])
    
    slab_indices_t = [a.index for a in true_system if a.tag != 2]
    dists_true = true_system.get_distances(target_idx_t, slab_indices_t)
    dists_true.sort()
    
    slab_indices_c = [a.index for a in chunk_atoms if a.tag != 2]
    dists_chunk = chunk_atoms.get_distances(target_idx_c, slab_indices_c)
    dists_chunk.sort()
    
    print(f"NN Distances (True vs Chunk):")
    for i in range(min(5, len(dists_chunk))):
        dt = dists_true[i]
        dc = dists_chunk[i]
        diff = abs(dt - dc)
        status = "OK" if diff < 1e-3 else "FAIL"
        print(f" {i+1}: {dt:.4f} | {dc:.4f} | {diff:.6f} [{status}]")
        
def extract_z_slice_clean(true_system, repeat, user_tags, output_file="z_slice_clean.cif"):
    """
    입력받은 true_system이 이미 a,b 방향으로는 Primitive 하다고 가정하고,
    Z 방향으로만 nc 등분하여 Top Layer + Adsorbate를 추출하는 함수.
    """
    
    print(f"\n{'='*50}")
    print(f" EXTRACTION (Z-Slice Only Logic)")
    print(f"{'='*50}")
    
    # a, b는 이미 줄어든 상태이므로 nc만 중요함
    na, nb, nc = repeat 
    
    if len(user_tags) != len(true_system):
        print("[ERROR] Tag mismatch.")
        return None
    
    # 태그 설정 (0: Slab, 1: Slab_surface, 2: Adsorbate 등 사용자가 정의한 대로)
    true_system.set_tags(user_tags)
    
    # 작업용 사본 생성 및 pbc wrapping (안전장치)
    work_atoms = true_system.copy()
    work_atoms.wrap() 
    frac_coords = work_atoms.get_scaled_positions()
    
    # --- 1. Z-Cut 범위 계산 (Slab atoms 기준) ---
    # 태그가 2(Adsorbate)가 아닌 원자들을 Slab으로 간주
    slab_indices = [i for i, a in enumerate(work_atoms) if a.tag != 2]
    
    if not slab_indices:
        print("[ERROR] No slab atoms found.")
        return None
        
    w_slab = frac_coords[slab_indices, 2] # z coordinates of slab atoms
    
    w_min, w_max = np.min(w_slab), np.max(w_slab) # slab의 z 최소/최대 높이
    w_unit = (w_max - w_min) / nc # 슬랩 전체 두께를 nc로 나눈 단위 높이
    
    # 가장 위쪽(Top) 레이어를 타겟으로 설정 (index: nc-1)
    k_target = nc - 1 

    # 부동소수점 오차 허용 범위
    eps = 1e-4
    w_start = w_min + k_target * w_unit - eps
    w_end = w_max + eps # 위쪽 끝은 slab의 max 높이까지 (약간의 여유 포함)
    
    print(f">> Target Z-Block: {k_target+1}/{nc} (Range: {w_start:.4f} ~ {w_end:.4f})")

    # --- 2. 필터링 (Z축만 고려) ---
    indices = []
    
    for i, atom in enumerate(work_atoms):
        w = frac_coords[i, 2]
        
        if atom.tag == 2: 
            # 흡착물(Adsorbate)은 위치 상관없이 무조건 포함
            indices.append(i)
        else:
            # 슬랩 원자는 타겟 Z 범위 안에 있는 경우만 포함
            if w_start <= w <= w_end:
                indices.append(i)
            
    extracted = work_atoms[indices].copy()
    print(f">> Extracted Atoms: {len(extracted)}")
    
    # --- 3. 검증 (Slab 원자 개수 확인) ---
    n_slab_extracted = sum(1 for a in extracted if a.tag != 2)
    n_slab_total = len(slab_indices)
    expected = int(n_slab_total / nc) # 전체 슬랩 원자를 nc로 나눈 값이 되어야 함
    
    print(f">> Slab Count: {n_slab_extracted} (Expected: {expected})")
    
    if n_slab_extracted != expected:
        print(f"   [WARNING] Count mismatch! Check if the slab layers are perfectly even.")
    
    # --- 4. 결과 저장 ---
    # Cell 크기는 변경하지 않음 (이미 a,b는 primitive이고 z는 진공 포함 전체 길이를 유지)
    # 만약 z축 길이를 줄이고 싶다면 vacuum 처리를 별도로 해야 하지만, 보통 slab 추출 시 cell은 유지함.
    
    if output_file:
        write(output_file, extracted)
        print(f">> Saved to: {output_file}")
        
    return extracted

def create_adsorbate_patch(
    primitive_slab: Atoms,
    true_system: Atoms,
    adsorption_site: np.ndarray,
    repeat_info: tuple[int, int, int],
) -> Atoms:
    """
    Creates a patch by placing adsorbate atoms correctly onto a primitive slab.

    This function maps the coordinates of adsorbate atoms from a large supercell
    (`true_system`) to a small, a,b-reduced primitive slab cell.

    Args:
        primitive_slab (Atoms): The 1x1 primitive slab structure (ASE Atoms).
        true_system (Atoms): The full supercell, including slab and adsorbate (ASE Atoms).
                               Must have tags set (2 for adsorbate).
        adsorption_site (np.ndarray): A Cartesian coordinate [x, y, z] indicating
                                      the approximate location of adsorption.
        repeat_info (tuple): Tiling factors (na, nb, nc) used to create the
                             supercell slab from the primitive slab.

    Returns:
        Atoms: An ASE Atoms object containing the primitive_slab atoms and the
               correctly positioned adsorbate atoms.
    """
    print("\n--- Creating Adsorbate Patch ---")
    na, nb, nc = repeat_info

    # 1. Identify adsorbate atoms in the true_system
    true_tags = true_system.get_tags()
    adsorbate_indices_in_true = np.where(true_tags == 2)[0]
    if len(adsorbate_indices_in_true) == 0:
        print("Warning: No adsorbate atoms (tag=2) found in true_system. Returning primitive slab.")
        return primitive_slab.copy()

    adsorbate_atoms_true = true_system[adsorbate_indices_in_true]
    print(f"Found {len(adsorbate_atoms_true)} adsorbate atoms in the supercell.")

    # 2. Find the reference adsorbate atom in the supercell (closest to the adsorption site)
    adsorbate_positions_true = adsorbate_atoms_true.get_positions()
    distances_to_site = np.linalg.norm(adsorbate_positions_true - adsorption_site, axis=1)
    ref_adsorbate_idx_in_true = adsorbate_indices_in_true[np.argmin(distances_to_site)]
    ref_adsorbate_pos_true = true_system.positions[ref_adsorbate_idx_in_true]

    # 3. Find the closest slab atom to this reference adsorbate in the supercell
    slab_indices_in_true = np.where(true_tags != 2)[0]
    slab_positions_true = true_system.positions[slab_indices_in_true]
    distances_to_ref_ads = np.linalg.norm(slab_positions_true - ref_adsorbate_pos_true, axis=1)
    closest_slab_idx_in_true = slab_indices_in_true[np.argmin(distances_to_ref_ads)]
    
    # This is our anchor point in the supercell
    anchor_slab_pos_true = true_system.positions[closest_slab_idx_in_true]

    # 4. Calculate the displacement vector from the anchor slab atom to all adsorbate atoms
    # This vector captures the relative positioning of the entire adsorbate molecule.
    displacement_vectors = adsorbate_atoms_true.get_positions() - anchor_slab_pos_true

    # 5. Find the corresponding anchor atom in the primitive_slab.
    # We assume the primitive_slab is a building block of the true_system's slab.
    # We find the atom in the primitive slab that has the most similar local environment
    # to the anchor slab atom in the true system.
    
    # Get distances to neighbors for the anchor in the true system
    true_distances = sorted(true_system.get_distances(closest_slab_idx_in_true, slab_indices_in_true, mic=True))
    
    best_match_idx = -1
    min_diff = float('inf')

    # Compare with every atom in the primitive slab
    for i in range(len(primitive_slab)):
        prim_distances = sorted(primitive_slab.get_distances(i, range(len(primitive_slab)), mic=True))
        
        # Compare the first few neighbor distances to find the best match
        num_compare = min(len(true_distances), len(prim_distances), 10)
        if num_compare == 0: continue
        
        diff = np.sum(np.abs(np.array(true_distances[1:num_compare]) - np.array(prim_distances[1:num_compare])))
        if diff < min_diff:
            min_diff = diff
            best_match_idx = i
            
    if best_match_idx == -1:
        raise RuntimeError("Could not find a matching anchor atom in the primitive slab.")

    anchor_slab_pos_prim = primitive_slab.positions[best_match_idx]
    print(f"Anchor atom identified in primitive slab at index {best_match_idx}.")

    # 6. Apply the displacement vectors to the primitive anchor to get new adsorbate positions
    new_adsorbate_positions = anchor_slab_pos_prim + displacement_vectors

    # 7. Combine the primitive slab and the new adsorbate atoms
    patch_atoms = primitive_slab.copy()
    
    adsorbate_patch = Atoms(
        symbols=adsorbate_atoms_true.get_chemical_symbols(),
        positions=new_adsorbate_positions,
        tags=[2] * len(adsorbate_atoms_true)
    )
    
    # Combine the two Atoms objects
    patch_atoms.extend(adsorbate_patch)
    
    print(f"Successfully created patch with {len(primitive_slab)} slab atoms and {len(adsorbate_patch)} adsorbate atoms.")
    
    return patch_atoms

def map_adsorbate_to_unit_slab(true_system, untiled_slab, repeat_info):
    """
    true_system: Adsorbate가 포함된 전체 supercell (Image 1)
    untiled_slab: Adsorbate가 없는 1x1 단위 slab (Image 3)
    repeat_info: [na, nb, nc] ex) [2, 2, 1] (c축은 보통 반복하지 않으므로 1로 가정)
    """
    
    # Adsorbate 객체만 추출 (Copy to avoid modifying original)
    adsorbate = true_system[true_system.get_tags()==2].copy()
    
    if len(adsorbate) == 0:
        print("Adsorbate 원자를 찾을 수 없습니다.")
        return untiled_slab.copy()

    # 2. 좌표 변환 준비
    # Adsorbate의 현재 fractional coordinate 가져오기
    frac_coords = adsorbate.get_scaled_positions()
    
    # 반복 팩터 배열 (a, b축만 늘리고 c축은 그대로 두는 경우 [2, 2, 1])
    # 사용자가 c방향 무시라고 했으므로 c 스케일은 1로 둡니다.
    scales = np.array([repeat_info[0], repeat_info[1], 1]) 
    
    # 3. Scaling & Modulo (핵심 로직)
    # 좌표를 확장된 횟수만큼 곱해주고, 1.0으로 나눈 나머지를 취해 0~1 사이로 리셋
    new_frac_coords = (frac_coords * scales) % 1.0
    
    # 4. 새로운 시스템 생성
    final_system = untiled_slab.copy()
    
    # Adsorbate를 새로운 좌표로 설정
    adsorbate.set_cell(untiled_slab.get_cell()) # Cell 정보 맞춤
    adsorbate.set_scaled_positions(new_frac_coords)
    
    # 합치기
    final_system += adsorbate
    
    return final_system

def extract_exact_top_unit(atoms: Atoms, n_c: int, n_vac: int):
    """
    [Debugging Version]
    각 단계별 계산 값과 모든 원자의 z좌표 판정 결과를 상세히 출력합니다.
    """
    
    # ---------------------------------------------------------
    # [Step 0] 초기 정보 출력
    # ---------------------------------------------------------
    original_numbers = atoms.get_atomic_numbers()
    original_total = len(atoms)
    
    print("\n" + "█"*60)
    print("      🔍 DEBUG MODE: Slab Extraction Analysis")
    print("█"*60)
    print(f"INPUTS       : n_c={n_c}, n_vac={n_vac}")
    print(f"TOTAL ATOMS  : {original_total}")
    
    # ---------------------------------------------------------
    # [Step 1] 커트라인(Threshold) 계산 상세
    # ---------------------------------------------------------
    n_total = n_c + n_vac
    h = 1.0 / n_total
    
    # 이론적 계산
    slab_width_frac = n_c * h
    z_center = 0.5
    z_top = z_center + (slab_width_frac / 2)
    # 계산된 cutoff
    z_cutoff_raw = z_top - h 
    # 안전장치 적용된 cutoff
    z_cutoff = z_cutoff_raw - 1e-8
    
    print("-" * 60)
    print(f"CALCULATION  : n_total = {n_total} (Total Layers Equivalent)")
    print(f"             : h (1 layer height) = {h:.6f}")
    print(f"             : Slab Top Z         = {z_top:.6f}")
    print(f"             : Cutoff (Raw)       = {z_cutoff_raw:.6f}")
    print(f"             : Cutoff (Applied)   = {z_cutoff:.8f}")
    print("-" * 60)
    
    # ---------------------------------------------------------
    # [Step 2] 원자별 Z좌표 및 판정 결과 (정렬하여 출력)
    # ---------------------------------------------------------
    scaled_positions = atoms.get_scaled_positions()
    z_coords = scaled_positions[:, 2]
    symbols = atoms.get_chemical_symbols()
    
    # 데이터 수집
    atom_data = []
    for i in range(original_total):
        z = z_coords[i]
        is_kept = z > z_cutoff
        diff = z - z_cutoff # 양수면 통과, 음수면 탈락
        atom_data.append({
            "idx": i,
            "symbol": symbols[i],
            "z": z,
            "diff": diff,
            "status": "✅ KEEP" if is_kept else "❌ DROP"
        })
    
    # Z좌표가 높은 순서대로 정렬 (Top Layer가 맨 위에 오도록)
    atom_data.sort(key=lambda x: x["z"], reverse=True)
    
    print(f"{'Idx':<4} | {'Sym':<4} | {'Z_coord':<10} | {'Dist from Cutoff':<18} | {'Status'}")
    print("-" * 60)
    
    for d in atom_data:
        # 커트라인 근처(±0.05)에 있는 원자는 강조 표시
        highlight = " 👈 CHECK!" if abs(d['diff']) < 0.05 else ""
        print(f"{d['idx']:<4} | {d['symbol']:<4} | {d['z']:.6f}   | {d['diff']:+.6f}           | {d['status']}{highlight}")
        
    print("-" * 60)

    # ---------------------------------------------------------
    # [Step 3] 실제 자르기 및 결과 리턴 (기존 로직 수행)
    # ---------------------------------------------------------
    mask = z_coords > z_cutoff
    unit_slab = atoms[mask]
    
    print(f"RESULT       : Extracted {len(unit_slab)} atoms (Expected {int(original_total/n_c)})")
    print("█"*60 + "\n")
    
    return unit_slab

def extract_top_unit_by_count(atoms: Atoms, n_c: int):
    """
    [Count-Based Version] *사용자 아이디어 적용*
    복잡한 계산 없이, 전체 원자 수와 층 수(n_c)를 기반으로
    '정확히 필요한 개수'만큼 위에서부터 긁어옵니다.
    """
    
    # ---------------------------------------------------------
    # [Step 1] 목표 개수(Target Count) 계산
    # ---------------------------------------------------------
    total_atoms = len(atoms)
    
    # 예: 9개 원자 / 3층 = 3개 (목표)
    if total_atoms % n_c != 0:
        print(f"⚠️ WARNING: Total atoms ({total_atoms}) is not divisible by n_c ({n_c}).")
        # 나눠떨어지지 않아도 일단 정수 몫만큼 진행
    
    target_count = int(total_atoms // n_c)
    
    # ---------------------------------------------------------
    # [Step 2] Z좌표 정렬 및 커트라인 결정 (핵심)
    # ---------------------------------------------------------
    # ASE에서 Z좌표만 가져옴
    z_coords = atoms.get_scaled_positions()[:, 2]
    
    # Z좌표를 내림차순(큰 값이 먼저 오게) 정렬
    # 예: [0.9, 0.8, 0.7, 0.2, 0.1 ...]
    sorted_z = np.sort(z_coords)[::-1]
    
    # 우리가 필요한 건 상위 target_count 개수임.
    # 따라서 커트라인은 '마지막 합격자(target-1)'와 '첫 번째 탈락자(target)' 사이
    last_kept_z = sorted_z[target_count - 1]
    first_dropped_z = sorted_z[target_count]
    
    # 안전한 커트라인 설정 (두 원자의 중간 지점)
    z_cutoff = (last_kept_z + first_dropped_z) / 2
    
    gap_size = last_kept_z - first_dropped_z
    
    print("\n" + "█"*60)
    print(f"      🎯 Count-Based Extraction (Target: {target_count} atoms)")
    print("█"*60)
    print(f"Logic             : Sorting atoms by height and picking top {target_count}")
    print(f"Cutoff Determined : Z = {z_cutoff:.6f}")
    print(f"Separation Gap    : {gap_size:.6f} (Distance between layers)")
    
    # 만약 갭이 너무 작다면(예: 0.05 미만), 층이 겹쳐있거나 구분이 모호한 것임 -> 경고
    if gap_size < 0.05:
        print("⚠️ WARNING: The gap between layers is very small. Are you sure n_c is correct?")

    # ---------------------------------------------------------
    # [Step 3] 추출 및 검증
    # ---------------------------------------------------------
    # 결정된 커트라인으로 마스킹
    mask = z_coords > z_cutoff
    unit_slab = atoms[mask]
    
    # --- 검증 (Validation) ---
    final_numbers = unit_slab.get_atomic_numbers()
    final_counts = Counter(final_numbers)
    original_counts = Counter(atoms.get_atomic_numbers())
    
    print("-" * 60)
    # 실제 추출된 개수 확인
    print(f"Total Atoms: {total_atoms} -> {len(unit_slab)}")
    print(f"Expected   : {target_count} (Result: {'✅ PASS' if len(unit_slab)==target_count else '❌ FAIL'})")
    
    print("-" * 60)
    print(f"{'Z (Atomic No)':<15} | {'Orig':<6} | {'Final':<6} | {'Expected':<8} | {'Status'}")
    print("-" * 60)
    
    all_pass = True
    for z, orig_count in original_counts.items():
        final_count = final_counts[z]
        expected_count = orig_count / n_c
        
        # 개수가 정확히 맞는지 확인
        is_match = abs(final_count - expected_count) < 1e-5
        if not is_match: all_pass = False
        
        status = "✅" if is_match else "❌"
        print(f"Z = {z:<11} | {orig_count:<6} | {final_count:<6} | {expected_count:<8.1f} | {status}")
    
    print("█"*60 + "\n")
    
    return unit_slab

def reconstruct_slab_from_ouc(unit_slab: Atoms, guide_c_vec, n_c: int):
    """
    OUC(1층짜리)의 정보를 이용해, 현재 셀의 변형 여부와 상관없이
    Slab을 100% 완벽하게 복원합니다.
    
    Args:
        unit_slab (Atoms): 추출된 최상단 1개 층 (현재 Lattice를 가짐)
        ouc (Atoms): 1층짜리 원본 OUC 구조 (여기서 c-vector를 추출)
        n_c (int): 쌓을 층 수
        
    Returns:
        Atoms: 완벽하게 복원된 전체 Slab
    """
    # 1. '진짜 이동 벡터' 추출 (Cartesian)
    # OUC가 1층짜리이므로, OUC의 c-vector 자체가 층간 이동 벡터임
    # ASE에서 get_cell()은 [a, b, c] 벡터를 반환하므로 인덱스 2가 c-vector
    true_shift_vector = np.array(guide_c_vec)
    
    print("\n" + "█"*60)
    print("      💎 Perfect Reconstruction Strategy")
    print("█"*60)
    print(f"Ref. Shift Vector (OUC) : {true_shift_vector}")
    print(f"Target Layers (n_c)     : {n_c}")
    print("Logic : Cartesian Shift -> Periodic Wrapping")
    
    # 2. 복원 시작
    # 뼈대는 unit_slab의 셀(Lattice)과 PBC를 따름
    recon_atoms = Atoms(cell=unit_slab.get_cell(), pbc=unit_slab.get_pbc())
    
    # Unit Slab의 Cartesian 좌표
    base_pos = unit_slab.get_positions()
    base_numbers = unit_slab.get_atomic_numbers()
    
    # Top(0) -> Bottom(n_c-1) 방향으로 적층
    for i in range(n_c):
        # 복사본 좌표 생성
        new_pos = base_pos.copy()
        
        # OUC 벡터 방향으로 i칸만큼 '아래로(반대로)' 이동
        # (OUC c-vector는 보통 바닥->천장 방향이므로, 쌓아 내리려면 마이너스)
        displacement = -1 * i * true_shift_vector
        new_pos += displacement
        
        # 임시 Atoms 객체 생성
        layer_atoms = Atoms(numbers=base_numbers, 
                            positions=new_pos, # Cartesian 좌표 입력
                            cell=unit_slab.get_cell(), 
                            pbc=unit_slab.get_pbc())
        
        # 3. [핵심] Wrapping (수학적 보정)
        # 물리적으로 이동한 좌표를 현재의 삐딱한 셀(Lattice) 안으로 접어 넣음
        layer_atoms.wrap()
        
        # 결과 병합
        recon_atoms += layer_atoms
        
    print("-" * 60)
    print(f"Reconstruction Complete! Total Atoms: {len(recon_atoms)}")
    print("█"*60 + "\n")
        
    return recon_atoms

def extract_top_unit_with_direction(atoms: Atoms, n_c: int, direction_vector=None):
    """
    [Direction-Aware Count-Based Extraction]
    특정 벡터 방향(direction_vector)을 '높이'로 간주하여,
    가장 위에 쌓인(해당 벡터 방향으로 값이 가장 큰) 상위 1/n_c 유닛을 추출합니다.
    
    Args:
        atoms (Atoms): 대상 ASE Atoms 객체
        n_c (int): 전체 층 수 (예: 3층이면 3)
        direction_vector (array-like): 적층 방향 벡터 (Cartesian). 
                                       None일 경우 atoms.cell[2] (c축) 사용.
    """
    
    # ---------------------------------------------------------
    # [Step 0] 방향 벡터 설정 및 정규화
    # ---------------------------------------------------------
    if direction_vector is None:
        # 별도 입력이 없으면 현재 cell의 c축 벡터 사용
        target_vec = atoms.cell[2]
        print(f"ℹ️ No direction vector provided. Using Lattice Vector C: {target_vec}")
    else:
        target_vec = np.array(direction_vector)
    
    # 단위 벡터(Unit Vector)로 변환 (크기가 1이어야 투영 길이가 정확함)
    vec_norm = np.linalg.norm(target_vec)
    if vec_norm < 1e-8:
        raise ValueError("Direction vector magnitude is too small (close to zero).")
    unit_vec = target_vec / vec_norm

    # ---------------------------------------------------------
    # [Step 1] 목표 개수(Target Count) 계산
    # ---------------------------------------------------------
    total_atoms = len(atoms)
    target_count = int(total_atoms // n_c)
    
    if total_atoms % n_c != 0:
        print(f"⚠️ WARNING: Total atoms ({total_atoms}) is not divisible by n_c ({n_c}).")

    # ---------------------------------------------------------
    # [Step 2] 벡터 투영을 통한 높이(Height) 계산 및 정렬
    # ---------------------------------------------------------
    # 원자들의 Cartesian 좌표 가져오기
    cart_positions = atoms.positions  # shape: (N, 3)
    
    # [핵심 로직]
    # 모든 원자의 위치를 방향 벡터에 투영(Dot Product)하여 '높이' 스칼라 값 획득
    # h = P · v_unit
    projected_heights = np.dot(cart_positions, unit_vec)
    
    # 투영된 높이를 기준으로 내림차순 정렬 인덱스 확보
    sorted_indices = np.argsort(projected_heights)[::-1]
    
    # 상위 target_count개의 인덱스만 선택
    top_indices = sorted_indices[:target_count]
    
    # 커트라인 분석 (디버깅용)
    last_kept_h = projected_heights[sorted_indices[target_count - 1]]
    first_dropped_h = projected_heights[sorted_indices[target_count]]
    gap_size = last_kept_h - first_dropped_h
    
    print("\n" + "█"*60)
    print(f"      🎯 Directional Extraction (Target: {target_count} atoms)")
    print("█"*60)
    print(f"Direction Vector  : {target_vec}")
    print(f"Projection Logic  : Dot product with unit vector")
    print(f"Separation Gap    : {gap_size:.6f} Å (along the vector)")
    
    if gap_size < 0.1:
        print("⚠️ WARNING: Gap is very small. Are the layers strictly separated along this vector?")

    # ---------------------------------------------------------
    # [Step 3] 추출 및 검증
    # ---------------------------------------------------------
    # 인덱스를 기반으로 원자 추출 (불리언 마스크 대신 인덱스 배열 사용이 더 안전함)
    unit_slab = atoms[top_indices]
    
    # --- 검증 (Validation) ---
    final_numbers = unit_slab.get_atomic_numbers()
    final_counts = Counter(final_numbers)
    original_counts = Counter(atoms.get_atomic_numbers())
    
    print("-" * 60)
    print(f"Total Atoms: {total_atoms} -> {len(unit_slab)}")
    print(f"Expected   : {target_count} (Result: {'✅ PASS' if len(unit_slab)==target_count else '❌ FAIL'})")
    
    print("-" * 60)
    print(f"{'Atomic No':<10} | {'Orig':<6} | {'Final':<6} | {'Expected':<8} | {'Status'}")
    print("-" * 60)
    
    for z_num, orig_count in original_counts.items():
        final_count = final_counts[z_num]
        expected_count = orig_count / n_c
        
        is_match = abs(final_count - expected_count) < 1e-5
        status = "✅" if is_match else "❌"
        print(f"No = {z_num:<6} | {orig_count:<6} | {final_count:<6} | {expected_count:<8.1f} | {status}")
    
    print("█"*60 + "\n")
    
    return unit_slab

def reconstruct_slab_using_vector(unit_slab: Atoms, n_c: int, shift_vector):
    """
    [Vector-Based Reconstruction]
    계산된 무게중심 대신, 알려진 OUC의 벡터(shift_vector)를 직접 사용하여
    정확한 결정학적 위치에 복원합니다.
    """
    if n_c <= 1:
        return unit_slab.copy()

    reconstructed = unit_slab.copy()
    base_positions = unit_slab.positions
    
    # shift_vector가 numpy array인지 확인
    vec = np.array(shift_vector)
    
    # 위에서부터 아래로 쌓아야 하므로, 
    # 만약 guide_c_vec이 '위로 올라가는' 벡터라면 -를 붙여야 할 수도 있습니다.
    # 하지만 보통 untiled_slab이 아래쪽으로 확장된다면, 
    # unit_slab(상단) + vec(하단 방향) 형식이어야 합니다.
    # **중요**: OUC c축은 보통 +z 방향이므로, 아래로 쌓으려면 빼줘야 할 수 있습니다.
    # 아래 로직은 벡터 방향이 "다음 층(아래층)의 위치"를 가리킨다고 가정합니다.
    
    for i in range(1, n_c):
        # 복원할 층 생성
        layer = unit_slab.copy()
        
        # 방향 주의! 
        # extract_top_unit은 가장 '위(Top)'를 가져왔습니다.
        # 따라서 아래로 쌓으려면 (Top - Vector) 가 되어야 할 확률이 높습니다.
        # 사용하시는 guide_c_vec의 방향(위/아래)을 체크해보세요.
        
        # Case A: guide_c_vec이 [0, 0, 10] 처럼 위를 향하는 경우 -> 빼줘야 아래로 쌓임
        # layer.positions = base_positions - (vec * i)
        
        # Case B: guide_c_vec이 층간 변위(Shift) 그 자체인 경우 -> 더함
        # 여기서는 일반적인 OUC c축(위쪽 방향)이라 가정하고 '빼기(-)'를 기본으로 작성합니다.
        layer.positions = base_positions - (vec * i)
        
        reconstructed += layer
        
    return reconstructed

def get_real_stacking_vector(slab: Atoms, n_c: int):
    """
    [PBC Corrected Version]
    단순 좌표 차이가 아니라, Periodic Boundary를 고려하여
    가장 가까운 거리(Minimum Image)를 층간 벡터로 추출합니다.
    """
    # 1. 원자들을 Z축 높이 순으로 정렬 인덱스 확보
    z_coords = slab.positions[:, 2]
    sorted_indices = np.argsort(z_coords)[::-1]
    
    atoms_per_layer = len(slab) // n_c
    
    # 2. Top 층(Layer 0)과 2nd 층(Layer 1)의 인덱스
    idx_L0 = sorted_indices[0 : atoms_per_layer]
    idx_L1 = sorted_indices[atoms_per_layer : 2*atoms_per_layer]
    
    # 3. [핵심] Cartesian이 아닌 'Fractional' 좌표계로 변환
    #    셀의 형태(기울기 등)와 상관없이 0~1 사이 값으로 만듭니다.
    frac_coords = slab.get_scaled_positions()
    
    frac_L0 = frac_coords[idx_L0]
    frac_L1 = frac_coords[idx_L1]
    
    # 4. 각 층의 Fractional Center 계산
    #    (주의: 여기서도 평균 낼 때 PBC 이슈가 있을 수 있으나, 
    #     Slab은 보통 뭉쳐있으므로 일단 mean 사용. 
    #     만약 층 자체가 쪼개져 있다면 이 부분도 보정이 필요하지만, 
    #     보통 Vector 차이 계산에서 보정하면 해결됨)
    center_frac_0 = np.mean(frac_L0, axis=0)
    center_frac_1 = np.mean(frac_L1, axis=0)
    
    # 5. Fractional 차이 계산 (Layer 0 -> Layer 1)
    diff_frac = center_frac_0 - center_frac_1
    
    # 6. [결정적 수정] PBC Wrapping 제거 (Minimum Image Convention)
    #    차이가 0.5보다 크거나 -0.5보다 작으면 정수(1.0)를 더하거나 빼서 보정
    #    예: 차이가 0.9면 -> -0.1로, -0.9면 -> 0.1로 인식해야 함
    diff_frac -= np.round(diff_frac)
    
    # 7. 다시 Cartesian 벡터로 변환
    real_vec = np.dot(diff_frac, slab.cell)
    
    print(f"🔥 Corrected Stacking Vector: {real_vec}")
    return real_vec

def reconstruct_slab(unit_slab: Atoms, n_c: int, shift_vector):
    """
    unit_slab을 shift_vector 방향으로 n_c번 쌓아 원본을 복원합니다.
    """
    if n_c == 1:
        return unit_slab.copy()

    # 복원될 원자들을 담을 리스트
    reconstructed_atoms = unit_slab.copy()
    
    # 원본 unit_slab의 위치
    base_positions = unit_slab.positions.copy()
    
    # 2번째 층부터 n_c번째 층까지 생성하여 추가
    for i in range(1, n_c):
        # i번째 층 = 기본 위치 + (변위 벡터 * i)
        # shift_vector가 '내려가는' 벡터이므로 더해주면 아래로 쌓임
        new_positions = base_positions + (shift_vector * i)
        
        # 새로운 층의 Atoms 객체 생성 (Cell 정보 등은 유지하지 않아도 됨, 나중에 합칠 것임)
        layer = unit_slab.copy()
        layer.positions = new_positions
        
        reconstructed_atoms += layer
        
    return reconstructed_atoms
  
  

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

def center_slab(slab: Structure) -> Structure:
    """Relocate the slab to the center such that its center
    (the slab region) is close to z=0.5.

    This makes it easier to find surface sites and apply
    operations like doping.

    There are two possible cases:
        1. When the slab region is completely positioned between
        two vacuum layers in the cell but is not centered, we simply
        shift the slab to the center along z-axis.
        2. If the slab completely resides outside the cell either
        from the bottom or the top, we iterate through all sites that
        spill over and shift all sites such that it is now
        on the other side. An edge case being, either the top
        of the slab is at z = 0 or the bottom is at z = 1.

    Args:
        slab (Structure): The slab to center.

    Returns:
        Structure: The centered slab.
    """
    # Get all site indices
    all_indices = list(range(len(slab)))

    # Get a reasonable cutoff radius to sample neighbors
    bond_dists = sorted(nn[1] for nn in slab.get_neighbors(slab[0], 10) if nn[1] > 0)
    # TODO (@DanielYang59): magic number for cutoff radius (would 3 be too large?)
    cutoff_radius = bond_dists[0] * 3

    # TODO (@DanielYang59): do we need the following complex method?
    # Why don't we just calculate the center of the Slab and move it to z=0.5?
    # Before moving we need to ensure there is only one Slab layer though

    # If structure is case 2, shift all the sites
    # to the other side until it is case 1
    for site in slab:  # DEBUG (@DanielYang59): Slab position changes during loop?
        # DEBUG (@DanielYang59): sites below z=0 is not considered (only check coord > c)
        if any(nn[1] >= slab.lattice.c for nn in slab.get_neighbors(site, cutoff_radius)):
            # TODO (@DanielYang59): the magic offset "0.05" seems unnecessary,
            # as the Slab would be centered later anyway
            shift = 1 - site.frac_coords[2] + 0.05
            slab.translate_sites(all_indices, [0, 0, shift])

    # Now the slab is case 1, move it to the center
    weights = [site.species.weight for site in slab]
    center_of_mass = np.average(slab.frac_coords, weights=weights, axis=0)
    shift = 0.5 - center_of_mass[2]

    slab.translate_sites(all_indices, [0, 0, shift])

    return slab

def get_primitive_structure_preserve_z(structure: Structure, tolerance: float = 0.25):
    """
    Get primitive structure while preserving z coordinates.
    Only x, y are normalized to [0, 1), z keeps original range.
    
    Returns:
        prim_struct
    """
    def site_label(site):
        return site.species_string
    
    sites = sorted(structure._sites, key=site_label)
    grouped_sites = [list(grp) for _, grp in itertools.groupby(sites, key=site_label)]
    grouped_frac_coords = [np.array([s.frac_coords for s in g]) for g in grouped_sites]
    
    min_frac_coords = min(grouped_frac_coords, key=len)
    min_vecs = min_frac_coords - min_frac_coords[0]
    
    super_ftol = np.divide(tolerance, structure.lattice.abc)
    super_ftol_2 = super_ftol * 2
    
    def pbc_coord_intersection(fc1, fc2, tol):
        dist = fc1[:, None, :] - fc2[None, :, :]
        dist -= np.round(dist)
        return fc1[np.any(np.all(np.abs(dist) < tol, axis=-1), axis=-1)]
    
    for group in sorted(grouped_frac_coords, key=len):
        for frac_coords in group:
            min_vecs = pbc_coord_intersection(min_vecs, group - frac_coords, super_ftol_2)
    
    def factors(n):
        for idx in range(1, n + 1):
            if n % idx == 0:
                yield idx
    
    def get_hnf(form_units):
        for det in factors(form_units):
            if det == 1:
                continue
            for a in factors(det):
                for e in factors(det // a):
                    g = det // a // e
                    supercell_matrices = np.array([
                        [[a, b, c], [0, e, f], [0, 0, g]]
                        for b, c, f in itertools.product(range(a), range(a), range(e))
                    ])
                    yield det, supercell_matrices
    
    grouped_non_nbrs = []
    for gf_coords in grouped_frac_coords:
        fdist = gf_coords[None, :, :] - gf_coords[:, None, :]
        fdist -= np.round(fdist)
        np.abs(fdist, fdist)
        non_nbrs = np.any(fdist > 2 * super_ftol[None, None, :], axis=-1)
        np.fill_diagonal(non_nbrs, val=True)
        grouped_non_nbrs.append(non_nbrs)
    
    num_fu = functools.reduce(math.gcd, map(len, grouped_sites))
    
    for size, ms in get_hnf(num_fu):
        inv_ms = np.linalg.inv(ms)
        
        dist = inv_ms[:, :, None, :] - min_vecs[None, None, :, :]
        dist -= np.round(dist)
        np.abs(dist, dist)
        is_close = np.all(dist < super_ftol, axis=-1)
        any_close = np.any(is_close, axis=-1)
        inds = np.all(any_close, axis=-1)
        
        for inv_m, latt_mat in zip(inv_ms[inds], ms[inds], strict=True):
            new_m = np.dot(inv_m, structure.lattice.matrix)
            ftol = np.divide(tolerance, np.sqrt(np.sum(new_m**2, axis=1)))
            
            valid = True
            new_coords = []
            new_sp = []
            new_props = defaultdict(list)
            new_labels = []
            
            for gsites, gf_coords, non_nbrs in zip(
                grouped_sites, grouped_frac_coords, grouped_non_nbrs, strict=True
            ):
                all_frac = np.dot(gf_coords, latt_mat)
                
                fdist = all_frac[None, :, :] - all_frac[:, None, :]
                fdist = np.abs(fdist - np.round(fdist))
                close_in_prim = np.all(fdist < ftol[None, None, :], axis=-1)
                groups = np.logical_and(close_in_prim, non_nbrs)
                
                if not np.all(np.sum(groups, axis=0) == size):
                    valid = False
                    break
                
                for group in groups:
                    if not np.all(groups[group][:, group]):
                        valid = False
                        break
                if not valid:
                    break
                
                added = np.zeros(len(gsites))
                new_frac_coords_xy = all_frac.copy()
                new_frac_coords_xy[:, :2] = new_frac_coords_xy[:, :2] % 1
                
                for grp_idx, group in enumerate(groups):
                    if not added[grp_idx]:
                        added[group] = True
                        inds_grp = np.where(group)[0]
                        repr_idx = inds_grp[0]
                        coords = new_frac_coords_xy[repr_idx].copy()
                        
                        for inner_idx, ind in enumerate(inds_grp[1:]):
                            offset = new_frac_coords_xy[ind] - coords
                            offset[:2] = offset[:2] - np.round(offset[:2])
                            coords += offset / (inner_idx + 2)
                        
                        z_coords = all_frac[inds_grp, 2]
                        coords[2] = np.mean(z_coords)
                        
                        new_sp.append(gsites[repr_idx].species)
                        for k in gsites[repr_idx].properties:
                            new_props[k].append(gsites[repr_idx].properties[k])
                        new_labels.append(gsites[repr_idx].label)
                        new_coords.append(coords)
            
            if valid:
                inv_m = np.linalg.inv(latt_mat)
                new_latt = Lattice(np.dot(inv_m, structure.lattice.matrix))
                
                prim_struct = Structure(
                    new_latt,
                    new_sp,
                    new_coords,
                    site_properties=dict(new_props),
                    labels=new_labels,
                    coords_are_cartesian=False,
                    to_unit_cell=False
                )
                
                # 재귀적으로 더 작은 primitive 찾기
                return get_primitive_structure_preserve_z(prim_struct, tolerance)
    
    return structure.copy()


def make_supercell_preserve_z(structure: Structure, scaling_matrix):
    """
    Make supercell while preserving z coordinates (no normalization).
    """
    scale_matrix = np.array(scaling_matrix, dtype=int)
    if scale_matrix.shape != (3, 3):
        scale_matrix = (scale_matrix * np.eye(3)).astype(int)
    
    new_lattice = Lattice(np.dot(scale_matrix, structure.lattice.matrix))
    frac_lattice = lattice_points_in_supercell(scale_matrix)
    cart_lattice = new_lattice.get_cartesian_coords(frac_lattice)
    
    new_sites = []
    for site in structure:
        for vec in cart_lattice:
            new_coords = site.coords + vec
            periodic_site = PeriodicSite(
                site.species,
                new_coords,
                new_lattice,
                properties=site.properties,
                coords_are_cartesian=True,
                to_unit_cell=False,
                skip_checks=True,
                label=site.label,
            )
            new_sites.append(periodic_site)
    
    return Structure.from_sites(new_sites, to_unit_cell=False)
