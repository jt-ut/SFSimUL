from typing import Hashable, List

def match_list(a: List[Hashable], b: List[Hashable]) -> List[int]:
    return [b.index(x) if x in b else None for x in a]

def match(a: List[Hashable], b: List[Hashable]) -> List[int]:
    b_dict = {x: i for i, x in enumerate(b)}
    return [b_dict.get(x, None) for x in a]
