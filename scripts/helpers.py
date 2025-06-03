from collections import defaultdict
from typing import TypedDict, Optional, Any
from dataclasses import dataclass

class ProposedChange(TypedDict):
    original_AID: int
    EID_context: Optional[str]
    column_to_change: str
    original_value: Any
    proposed_value: Any
    rule_name: str

@dataclass
class SplitEvent:
    old_aid: int
    new_aid: int
    column: str
    new_value: str


class UnionFind:
    def __init__(self, elems: list[str]) -> None:
        self.idx  = {e: i for i, e in enumerate(elems)}
        self.par  = list(range(len(elems)))
        self.rank = [0] * len(elems)

    def _find(self, i: int) -> int:
        if self.par[i] != i:
            self.par[i] = self._find(self.par[i])
        return self.par[i]

    def union(self, a: str, b: str) -> None:
        ra, rb = self._find(self.idx[a]), self._find(self.idx[b])
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.par[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.par[rb] = ra
        else:
            self.par[rb] = ra
            self.rank[ra] += 1

    def clusters(self) -> dict[int, list[str]]:
        out: dict[int, list[str]] = defaultdict(list)
        for e, i in self.idx.items():
            out[self._find(i)].append(e)
        return out
