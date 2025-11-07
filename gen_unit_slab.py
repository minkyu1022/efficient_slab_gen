index = 7

import lmdb
import pickle
import torch
import pickle
import os
import json
from fairchem.data.oc.core import Bulk, Slab

def get_sid_from_lmdb(lmdb_path, index):
    """
    LMDB 파일에서 특정 인덱스의 sid(system ID) 값을 추출합니다.
    
    Args:
        lmdb_path: LMDB 파일 경로
        index: 추출할 데이터 인덱스
    
    Returns:
        sid 값 (system ID)
    """
    # LMDB 데이터베이스 열기
    db = lmdb.open(
        str(lmdb_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    
    try:
        # 트랜잭션 시작
        with db.begin() as txn:
            # 인덱스에 해당하는 데이터 가져오기
            key = f"{index}".encode("ascii")
            value = txn.get(key)
            
            if value is None:
                raise ValueError(f"Index {index} not found in LMDB")
            
            # pickle로 역직렬화
            data = pickle.loads(value)
        
        # sid 추출 (여러 방법 시도)
        sid = None
        
        # 방법 1: 직접 속성 접근
        try:
            sid = getattr(data, 'sid', None)
        except (RuntimeError, AttributeError):
            pass
        
        # 방법 2: __dict__에서 찾기
        if sid is None:
            try:
                if hasattr(data, '__dict__') and 'sid' in data.__dict__:
                    sid = data.__dict__['sid']
            except:
                pass
        
        # 방법 3: items() 메서드 사용 (PyTorch Geometric)
        if sid is None:
            try:
                if hasattr(data, 'items'):
                    for key, value in data.items():
                        if key == 'sid':
                            sid = value
                            break
            except (RuntimeError, AttributeError, TypeError):
                pass
        
        # 방법 4: keys()로 확인 후 직접 접근
        if sid is None:
            try:
                if hasattr(data, 'keys'):
                    keys = list(data.keys())
                    if 'sid' in keys:
                        sid = data['sid']
            except (RuntimeError, AttributeError, TypeError, KeyError):
                pass
        
        # 방법 5: _store에서 직접 접근
        if sid is None:
            try:
                store = getattr(data, '_store', None)
                if store is not None and 'sid' in store.keys():
                    sid = store['sid']
            except (RuntimeError, AttributeError, KeyError):
                pass
        
        # 값이 리스트나 텐서인 경우 첫 번째 요소 추출
        if sid is not None:
            if isinstance(sid, (list, tuple)) and len(sid) > 0:
                sid = sid[0]
            elif isinstance(sid, torch.Tensor):
                sid = sid.item() if sid.numel() == 1 else sid.tolist()[0] if len(sid) > 0 else None
        
        return sid
            
    finally:
        db.close()

# 사용 예시
lmdb_path = "is2res_train_val_test_lmdbs/data/is2re/all/train/data.lmdb"

sid_value = get_sid_from_lmdb(lmdb_path, index)

if sid_value is not None:
    print(f"Index {index}의 sid: {sid_value}")
    print(f"sid 타입: {type(sid_value)}")
else:
    print(f"Index {index}에서 sid를 찾을 수 없습니다.")

# oc20_data_mapping.pkl 파일 경로
mapping_path = "oc20_data_mapping.pkl"

# 찾고 싶은 특정 key 값
target_key = f"random{sid_value}"  # 여기에 원하는 key 값을 입력

# 파일 존재 확인
if os.path.exists(mapping_path):
    # Pickle 파일 로드
    with open(mapping_path, 'rb') as f:
        mapping = pickle.load(f)
    
    print("=" * 80)
    print(f"oc20_data_mapping.pkl - Key: {target_key}")
    print("=" * 80)
    
    # 특정 key 존재 여부 확인
    if target_key in mapping:
        value = mapping[target_key]
        print(f"\n✓ Key를 찾았습니다!")
        
        print(f"\nValue 타입: {type(value)}")
        
        if isinstance(value, dict):
            print(f"\nValue 내용 (딕셔너리):")
            print("-" * 80)
            for k, v in value.items():
                # 값이 복잡한 구조인 경우 예쁘게 출력
                if isinstance(v, (list, tuple)):
                    print(f"  {k}: {v}")
                elif isinstance(v, dict):
                    print(f"  {k}:")
                    for sub_k, sub_v in v.items():
                        print(f"    {sub_k}: {sub_v}")
                else:
                    print(f"  {k}: {v}")
        elif isinstance(value, (list, tuple)):
            print(f"\nValue 내용 (리스트/튜플, 길이: {len(value)}):")
            print("-" * 80)
            print(value)
        else:
            print(f"\nValue 내용:")
            print("-" * 80)
            print(value)
    else:
        print(f"\n✗ Key '{target_key}'를 찾을 수 없습니다.")
        print(f"\n전체 항목 수: {len(mapping)}")
        
        # 비슷한 key가 있는지 확인
        similar_keys = [k for k in mapping.keys() if target_key in str(k)]
        if similar_keys:
            print(f"\n찾으시는 key와 비슷한 항목들 (처음 10개):")
            for i, k in enumerate(similar_keys[:10]):
                print(f"  [{i}] {k}")
    
    print("\n" + "=" * 80)
    
else:
    print(f"파일을 찾을 수 없습니다: {mapping_path}")

# bulks.pkl 파일에서 bulk 선택
bulk = Bulk(bulk_src_id_from_db="mp-978498")

# 특정 miller index로 모든 슬랩 생성
slabs = Slab.from_bulk_get_specific_millers(
    specific_millers=(1, 1, 0),
    bulk=bulk,
)

# 원하는 shift와 top으로 필터링
target_shift = 0.125
target_top = True
matching_slabs = [
    slab for slab in slabs 
    if abs(slab.shift - target_shift) < 1e-3 and slab.top == target_top
]

if matching_slabs:
    slab = matching_slabs[0]  # 유일하다면 첫 번째 것
    
    # Slab 기본 정보 출력
    print(f"=== Slab 기본 정보 ===")
    print(f"Miller indices: {slab.millers}")
    print(f"Shift: {slab.shift}")
    print(f"Top: {slab.top}")
    print(f"Formula: {slab.atoms.get_chemical_formula()}")
    print(f"Number of atoms: {len(slab.atoms)}")
    
    # # Oriented Unit Cell 정보 출력
    # print(f"\n=== Oriented Unit Cell 정보 ===")
    # if slab.oriented_bulk is not None:
    #     print(f"Oriented unit cell formula: {slab.oriented_bulk.formula}")
    #     print(f"Oriented unit cell number of atoms: {len(slab.oriented_bulk)}")
    #     # 필요시 구조 출력
    #     # print(f"Oriented unit cell structure:\n{slab.oriented_bulk}")
    # else:
    #     print("Oriented unit cell 정보: 없음")
    
    # # Repeat 정보 출력
    # print(f"\n=== Repeat 정보 ===")
    # if slab.tile_repeats is not None:
    #     nx, ny, nz = slab.tile_repeats
    #     print(f"Repeat counts (x, y, z): {slab.tile_repeats}")
    #     print(f"  - X 방향 (a'): {nx}번 반복")
    #     print(f"  - Y 방향 (b'): {ny}번 반복")
    #     print(f"  - Z 방향 (c'): {nz}번 반복")
    #     print(f"\n→ Oriented unit cell이 ({nx}, {ny}, {nz})만큼 반복되어 최종 slab이 생성됨")
    # else:
    #     print("Repeat 정보: 없음 (이전 버전의 데이터)")
    
    # # Metadata에서도 확인
    # print(f"\n=== Metadata 정보 ===")
    # metadata = slab.get_metadata_dict()
    # print(f"Metadata에 포함된 정보:")
    # for key, value in metadata['slab_metadata'].items():
    #     if key == 'oriented_bulk':
    #         print(f"  {key}: {type(value).__name__} (Structure 객체)")
    #     else:
    #         print(f"  {key}: {value}")
else:
    print("Matching slab을 찾을 수 없습니다.")