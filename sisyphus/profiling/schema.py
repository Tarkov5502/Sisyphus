"""
sisyphus.profiling.schema — the route log: one record per (call, token, layer).

This is the contract between the rented box that runs the real model and the laptop
that runs the scheduler. Anything that can emit these records — a transformers hook,
a llama.cpp patch, a vLLM plugin — feeds the same analyzer and the same replay.

Record (one JSON object per line, optionally gzip-compressed):
  {"call": "audit-003", "domain": "code", "phase": "decode", "token": 17, "layer": 40,
   "experts": [12, 88, 301, ...], "gates": [0.21, 0.13, 0.09, ...]}

  call     stream id (one prompt = one call)
  domain   the operator's job class — the convoy premise is tested per domain
  phase    "prefill" (prompt tokens) or "decode" (generated tokens)
  token    position within the phase (0-based)
  layer    MoE layer index (0-based over MoE layers only; the dense trunk is not logged)
  experts  LOCAL expert ids at this layer (0..EXPERTS-1), top-k, any order
  gates    the router's normalised weights, aligned with `experts`, summing to ~1

Expert ids are local per layer in the log (that is what the model emits); the replay
converts them to Sisyphus's globally-unique ids (layer * EXPERTS + local).
"""
from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

PHASES = ("prefill", "decode")


@dataclass(frozen=True)
class RouteRecord:
    call: str
    domain: str
    phase: str
    token: int
    layer: int
    experts: tuple[int, ...]
    gates: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}, got {self.phase!r}")
        if len(self.experts) != len(self.gates):
            raise ValueError("experts and gates must align")
        if len(set(self.experts)) != len(self.experts):
            raise ValueError("duplicate expert in one route")

    def to_json(self) -> str:
        d = asdict(self)
        d["experts"] = list(self.experts)
        d["gates"] = [round(g, 6) for g in self.gates]
        return json.dumps(d, separators=(",", ":"))

    @staticmethod
    def from_json(line: str) -> "RouteRecord":
        d = json.loads(line)
        return RouteRecord(d["call"], d.get("domain", ""), d["phase"], int(d["token"]),
                           int(d["layer"]), tuple(int(e) for e in d["experts"]),
                           tuple(float(g) for g in d["gates"]))


def _open(path: str | Path, mode: str):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


class RouteLogWriter:
    """Append-only JSONL writer. Use as a context manager on the rented box."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh = None
        self.count = 0

    def __enter__(self) -> "RouteLogWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = _open(self.path, "a")
        return self

    def write(self, rec: RouteRecord) -> None:
        assert self._fh is not None, "writer not opened"
        self._fh.write(rec.to_json() + "\n")
        self.count += 1

    def write_many(self, recs: Iterable[RouteRecord]) -> None:
        for r in recs:
            self.write(r)

    def __exit__(self, *exc) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def read_log(path: str | Path) -> Iterator[RouteRecord]:
    """Stream records from a .jsonl or .jsonl.gz log."""
    with _open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield RouteRecord.from_json(line)
