from __future__ import annotations

import math
import os
import pickle
import pandas as pd
import ast
import lmdb
import numpy as np
import torch
import itertools
import functools
from collections import Counter, defaultdict
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from pymatgen.core.sites import PeriodicSite
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.util.coord import lattice_points_in_supercell
from ase import Atoms
from ase.io import write
from pymatgen.core import Lattice, Structure

def calculate_rmsd_pymatgen(
    struct1: Atoms | Structure,
    struct2: Atoms | Structure,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5,
    primitive_cell: bool = False,
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
        primitive_cell=primitive_cell,
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol
    )
    
    # Check if structures match
    if matcher.fit(struct1, struct2):
        # Calculate RMS distance
        rms_dist, max_dist = matcher.get_rms_dist(struct1, struct2)
        return rms_dist, max_dist
    else:
        return None, None

def get_info_from_metadata(df, index):
    row = df.iloc[index]

    def parse(val, dtype=None, np_type=None):
        if isinstance(val, str):
            val = ast.literal_eval(val)
        if dtype == 'array':
            return np.array(val, dtype=np_type)
        elif dtype:
            return dtype(val)
        return val

    data = {
        'meta': {
            'sid': row['sid'],
            'bulk_src_id': row['bulk_src_id'],
            'specific_miller': row['specific_miller'],
            'shift': row['shift'],
            'top': row['top'],
        },
        'config': {
            'n_c': parse(row['n_c'], int),
            'n_vac': parse(row['n_vac'], int),
            'height': parse(row['height'], float),
        },
        'structures': {
            'true_atomic_nums': parse(row['true_system_atomic_numbers'], 'array', int),
            'true_positions': parse(row['true_system_positions'], 'array', float),
            'true_lattice': parse(row['true_lattice'], 'array', float),
            'true_tags': parse(row['true_tags'], 'array'),
            'ads_pos_relaxed': parse(row['ads_pos_relaxed'], 'array', float),
        }
    }
    
    return data

def find_vacuum_axis_ase(atoms: Atoms) -> int:
    # 1. 분수 좌표 가져오기 (wrap=True로 모든 원자를 0~1 사이로 모읍니다)
    scaled_positions = atoms.get_scaled_positions(wrap=True)
    cell_lengths = atoms.cell.lengths() # [a길이, b길이, c길이]
    
    max_gap_size = -1.0
    vacuum_axis = 2 # 기본값
    
    # x(0), y(1), z(2) 축 반복
    for i in range(3):
        # 해당 축의 좌표만 정렬
        coords = np.sort(scaled_positions[:, i])
        
        # 인접 원자 간 간격 계산
        gaps = np.diff(coords)
        
        # PBC 고려: (1.0 + 첫 원자 - 마지막 원자)도 간격에 포함
        boundary_gap = 1.0 + coords[0] - coords[-1]
        
        # 해당 축에서 가장 큰 간격 (fractional)
        current_axis_max_frac_gap = max(np.max(gaps), boundary_gap)
        
        # 실제 거리(Å)로 변환
        real_gap_len = current_axis_max_frac_gap * cell_lengths[i]
        
        # 전체 축 중 으뜸인지 비교
        if real_gap_len > max_gap_size:
            max_gap_size = real_gap_len
            vacuum_axis = i
            
    return vacuum_axis

def align_vacuum_to_z_axis(atoms: Atoms, vac_axis_idx: int) -> Atoms:
    """
    진공 축을 z축(index 2)으로 순서를 바꾸고, 
    공간상에서도 z축 방향으로 서 있게 만듭니다.
    """
    atoms = atoms.copy()
    
    # 1. 축 순서 변경 (Permutation)
    #    진공이 x나 y에 있다면, 이를 z(index 2) 자리로 보냅니다.
    if vac_axis_idx == 0:   # x가 진공 -> (y, z, x) 순서로 변경 (Right-handed 유지)
        new_order = [1, 2, 0]
    elif vac_axis_idx == 1: # y가 진공 -> (z, x, y) 순서로 변경
        new_order = [2, 0, 1]
    else:                   # 이미 z가 진공
        new_order = [0, 1, 2]

    # 축이 변경되어야 한다면 순서 교환 수행
    if vac_axis_idx != 2:
        old_cell = atoms.get_cell()
        new_cell = old_cell[new_order]  # 벡터의 순서만 바꿈 (방향은 그대로)
        
        old_scaled = atoms.get_scaled_positions()
        new_scaled = old_scaled[:, new_order] # 좌표값(fractional)의 열 순서 변경
        
        atoms.set_cell(new_cell)
        atoms.set_scaled_positions(new_scaled)
        
    # ------------------------------------------------------------------------
    # [핵심] 2. 공간상 방향 재설정 (Standardization)
    # 현재 상태: cell[2]가 진공 벡터이지만, 공간상에서는 여전히 옆으로 누워있음.
    # 해결: 격자의 길이와 각도(par)만 뽑아서, "표준 방향"으로 다시 그림.
    #       ASE의 표준 방향 규칙: a축은 x축, b축은 xy평면, c축은 나머지 z방향
    # ------------------------------------------------------------------------
    
    # [a, b, c, alpha, beta, gamma] 추출
    cell_par = atoms.cell.cellpar() 
    
    # 추출한 파라미터로 "새로운 표준 셀"을 만들고 원자들을 그에 맞춰 회전시킴
    # scale_atoms=True가 원자들을 같이 회전시켜주는 역할을 함
    atoms.set_cell(cell_par, scale_atoms=True)
    
    # # 셀 밖으로 나간 원자 정리 (선택사항)
    # atoms.wrap()
    
    return atoms