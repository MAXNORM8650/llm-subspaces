"""Validated whole-tensor energy-tier policies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantizationPolicy:
    name: str
    tiers: tuple[str, ...]
    endpoints: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.tiers) < 2 or len(self.endpoints) != len(self.tiers) - 1:
            raise ValueError("a policy needs one fewer endpoint than precision tiers")
        if any(value <= 0.0 or value >= 1.0 for value in self.endpoints):
            raise ValueError("policy endpoints must be in (0, 1)")
        if any(left >= right for left, right in zip(self.endpoints, self.endpoints[1:])):
            raise ValueError("policy endpoints must be strictly increasing")


DEFAULT_POLICIES = (
    QuantizationPolicy("Q8Q6Q4", ("Q8_0", "Q6_K", "Q4_K"), (0.90, 0.98)),
    QuantizationPolicy("Q6Q5Q4", ("Q6_K", "Q5_K", "Q4_K"), (0.90, 0.98)),
    QuantizationPolicy("Q5Q4Q3", ("Q5_K", "Q4_K", "Q3_K"), (0.90, 0.98)),
    QuantizationPolicy("Q4Q3Q2", ("Q4_K", "Q3_K", "Q2_K"), (0.90, 0.98)),
    QuantizationPolicy("Q3Q2IQ1S", ("Q3_K", "Q2_K", "IQ1_S"), (0.90, 0.98)),
    QuantizationPolicy("Q2IQ1S", ("Q2_K", "IQ1_S"), (0.80,)),
)


def assign_modules(modules: dict, policy: QuantizationPolicy) -> list[dict]:
    total_energy = sum(float(module["energy_total"]) for module in modules.values())
    if total_energy <= 0:
        raise ValueError("analysis has no positive module energy")
    ordered = sorted(
        modules.items(), key=lambda item: float(item[1]["energy_total"]), reverse=True
    )
    cumulative = 0.0
    assignments = []
    for name, module in ordered:
        fraction_before = cumulative / total_energy
        tier_index = next(
            (
                index
                for index, endpoint in enumerate(policy.endpoints)
                if fraction_before < endpoint
            ),
            len(policy.tiers) - 1,
        )
        energy = float(module["energy_total"])
        cumulative += energy
        assignments.append(
            {
                "module": name,
                "precision": policy.tiers[tier_index],
                "energy": energy,
                "energy_fraction": energy / total_energy,
                "cumulative_energy_fraction": cumulative / total_energy,
                "parameters": int(module.get("parameters") or _parameters(module["shape"])),
            }
        )
    return assignments


def _parameters(shape: list[int]) -> int:
    result = 1
    for value in shape:
        result *= int(value)
    return result
