from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class OffensiveSliceSpec:
    strategy: str
    ownership_cells: tuple[str, ...]
    direction: str | None = None
    setup_type: str | None = None
    evaluator_mode: str | None = None
    timing_gate: str | None = None
    session_gate: str | None = None
    state_realism_mode: str = "native_trade_proxy"
    feature_gates: dict[str, Any] = field(default_factory=dict)
    size_scale: float = 1.0
    source_mode: str = "native_trade_proxy"
    notes: str = ""
    slice_id: str | None = None

    def resolved_slice_id(self) -> str:
        if self.slice_id:
            return self.slice_id
        parts = [self.strategy]
        if self.setup_type:
            parts.append(f"setup_{self.setup_type}")
        if self.evaluator_mode:
            parts.append(self.evaluator_mode)
        if self.direction:
            parts.append(self.direction)
        if self.timing_gate:
            parts.append(self.timing_gate)
        if self.session_gate:
            parts.append(self.session_gate)
        if self.ownership_cells:
            parts.append("cells_" + "__".join(c.replace("/", "_") for c in self.ownership_cells))
        if self.feature_gates:
            for k in sorted(self.feature_gates):
                v = self.feature_gates[k]
                if isinstance(v, (list, tuple, set)):
                    val = "-".join(str(x) for x in v)
                else:
                    val = str(v)
                parts.append(f"{k}_{val}".replace("/", "_"))
        return "__".join(parts)

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["slice_id"] = self.resolved_slice_id()
        out["ownership_cells"] = list(self.ownership_cells)
        return out
