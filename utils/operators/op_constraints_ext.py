from collections import Counter
from typing import Iterable, Set

SCALERS_LINEAR: Set[int] = {7, 8, 9, 10}
SELF_IDEMPOTENT: Set[int] = {7, 8, 9, 10, 13}
EXPANSION: Set[int] = {15, 16, 23}
REDUCTION: Set[int] = {17, 18, 19, 20, 21, 22}
EXPENSIVE_REDUCTION: Set[int] = {19, 21, 22}
TREE_EMBED: int = 23

OUTLIER_FILTER: Set[int] = {25, 26} 
OUTLIER_CLIP: Set[int] = {27, 28}   
OUTLIER_ALL: Set[int] = OUTLIER_FILTER | OUTLIER_CLIP
RESAMPLER: Set[int] = {29, 30, 31}
ROW_CHANGING: Set[int] = OUTLIER_FILTER | RESAMPLER

EFFECTIVE_OPS: Set[int] = set(range(7, 17)) | {23} | OUTLIER_ALL | RESAMPLER


def blocked_pair(a: int, b: int, strict_chain: bool = False) -> bool:
    if a == b and a in SELF_IDEMPOTENT:
        return True                                  
    if a in SCALERS_LINEAR and b in SCALERS_LINEAR:
        return True                                  
    if a == TREE_EMBED and b in {TREE_EMBED, 15, 16}:
        return True                                  
    if a in REDUCTION and b in REDUCTION:
        return True                                  
    if strict_chain and a in EXPANSION and b in EXPENSIVE_REDUCTION:
        return True                                  

    if a in OUTLIER_FILTER and b in OUTLIER_FILTER:
        return True
    if a in OUTLIER_CLIP and b in OUTLIER_CLIP:
        return True
    if a in RESAMPLER and b in RESAMPLER:
        return True
    if a in RESAMPLER and b in OUTLIER_FILTER:
        return True
    return False


def blocked_pair_lease(a: int, b: int, strict_chain: bool = False) -> bool:
    if a == b and a in SELF_IDEMPOTENT:
        return True                                  
    if a in SCALERS_LINEAR and b in SCALERS_LINEAR:
        return True                                  
    # if a == TREE_EMBED and b in {TREE_EMBED, 15, 16}:
    #     return True                                
    if a in REDUCTION and b in REDUCTION:
        return True                                  
    if strict_chain and a in EXPANSION and b in EXPENSIVE_REDUCTION:
        return True                                 

    if a in OUTLIER_FILTER and b in OUTLIER_FILTER:
        return True                                 
    if a in OUTLIER_CLIP and b in OUTLIER_CLIP:
        return True                                 
    if a in RESAMPLER and b in RESAMPLER:
        return True                                 
    if a in RESAMPLER and b in OUTLIER_FILTER:
        return True                                 
    return False


def blocked_chain(chain: Iterable[int], strict_chain: bool = False) -> bool:
    chain = list(chain)
    return any(blocked_pair(chain[i], chain[i + 1], strict_chain)
               for i in range(len(chain) - 1))


def blocked_counter(combo: Iterable[int],
                    n_feature: int,
                    check_reduction_repeat: bool = True) -> bool:
    c = Counter(combo)

    if c.get(TREE_EMBED, 0) >= 2:
        return True
    if n_feature >= 14 and (c.get(23, 0) + c.get(15, 0) + c.get(16, 0)) >= 2:
        return True                                
    if check_reduction_repeat:
        for op in REDUCTION:
            if c.get(op, 0) >= 2:
                return True

    if sum(c.get(op, 0) for op in RESAMPLER) >= 2:
        return True                                
    for op in OUTLIER_FILTER:
        if c.get(op, 0) >= 2:
            return True                       
    if n_feature >= 14 \
            and sum(c.get(op, 0) for op in RESAMPLER) >= 1 \
            and sum(c.get(op, 0) for op in EXPANSION) >= 1:
        return True                                
    return False


def legacy_blocked_pair(a: int, b: int, strict_chain: bool = False) -> bool:
    if a == b and a in {7, 8, 9, 10, 13}:
        return True
    if a in {7, 8, 9, 10} and b in {7, 8, 9, 10}:
        return True
    if a == 23 and b in {23, 15, 16}:
        return True
    if a in {17, 18, 19, 20, 21, 22} and b in {17, 18, 19, 20, 21, 22}:
        return True
    if strict_chain and a in {15, 16, 23} and b in {21, 22, 19}:
        return True
    return False


def legacy_blocked_counter(combo: Iterable[int], n_feature: int,
                           check_reduction_repeat: bool = True) -> bool:
    counter = Counter(combo)
    if counter.get(23, 0) >= 2:
        return True
    if n_feature >= 14 and \
            counter.get(23, 0) + counter.get(15, 0) + counter.get(16, 0) >= 2:
        return True
    if check_reduction_repeat:
        if counter.get(17, 0) >= 2 or counter.get(18, 0) >= 2 or \
                counter.get(19, 0) >= 2 or counter.get(20, 0) >= 2 or \
                counter.get(21, 0) >= 2 or counter.get(22, 0) >= 2:
            return True
    return False
