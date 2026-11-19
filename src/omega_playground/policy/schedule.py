from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Schedule:
    trigger: Literal["periodic", "event"]
    period_s: float | None = None
    poll_s: float | None = None

    @classmethod
    def from_json(cls, d: dict) -> "Schedule":
        return cls(trigger=d["trigger"], period_s=d.get("period_s"), poll_s=d.get("poll_s"))

    def interval_s(self) -> float:
        value = self.period_s if self.trigger == "periodic" else self.poll_s
        if value is None or value <= 0:
            raise ValueError(f"{self.trigger} schedule needs a positive interval: {self}")
        return value

    def physics_steps(self, physics_hz: int) -> int:
        """Interval as an exact number of physics steps; rejects non-integer ratios."""
        return physics_steps(self.interval_s(), physics_hz)


def physics_steps(interval_s: float, physics_hz: int) -> int:
    steps = interval_s * physics_hz
    rounded = round(steps)
    if rounded < 1 or abs(steps - rounded) > 1e-6:
        raise ValueError(f"{interval_s} s is not an integer multiple of 1/{physics_hz} s")
    return rounded
