from __future__ import annotations
import re
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Physical input limits. Anything outside is rejected (422); values inside but outside the
# training range are clamped and flagged by the service, never extrapolated.
IRR_MIN, IRR_MAX = -0.05, 1.5          # kW/m2; tiny negatives are a sensor night offset
MOD_MIN, MOD_MAX = -10.0, 100.0        # deg C
AMB_MIN, AMB_MAX = -10.0, 60.0         # deg C
NIGHT_SUN = 0.05                       # irradiance above this is impossible at night
NIGHT_HOURS = lambda h: h >= 20 or h < 5
DATE_ONLY = re.compile(r"^\s*\d{4}-\d{2}-\d{2}\s*$")


def _temp_error(name: str, v: float, lo: float, hi: float) -> str:
    if v > 200:
        return f"{name} {v} looks like Kelvin; send degrees C (subtract 273.15)"
    if v > hi and v <= 200:
        return f"{name} {v} is above {hi} C; if this is Fahrenheit, convert with (F - 32) x 5/9"
    return f"{name} {v} is below the physical limit {lo} C"


class Reading(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "plant_id": 2, "irradiation": 0.72, "module_temperature": 48.5, "ambient_temperature": 31.2, "timestamp": "2020-06-07T10:00:00"}})
    plant_id: Literal[1, 2]
    irradiation: float = Field(strict=True, allow_inf_nan=False, description="Source sensor value, kW/m2")
    module_temperature: float | None = Field(strict=True, allow_inf_nan=False,
                                             description="Degrees C. null (or exactly 0) = sensor missing; inferred in daylight")
    ambient_temperature: float | None = Field(strict=True, allow_inf_nan=False,
                                              description="Degrees C. null (or exactly 0) = sensor missing; inferred in daylight")
    timestamp: datetime = Field(description="Plant local date and time, e.g. 2020-06-07T10:00:00")

    @field_validator("timestamp", mode="before")
    @classmethod
    def needs_time_of_day(cls, v: Any):
        if not isinstance(v, str):
            raise ValueError("timestamp must be an ISO string with date and time, e.g. 2020-06-07T10:00:00")
        if DATE_ONLY.match(v):
            raise ValueError("timestamp has no time of day; the model needs the local time, e.g. 2020-06-07T10:00:00")
        return v

    @field_validator("irradiation")
    @classmethod
    def irr_limits(cls, v: float):
        if v > IRR_MAX:
            hint = " (looks like W/m2; send kW/m2 by dividing by 1000)" if v > 10 else ""
            raise ValueError(f"irradiation {v} is above the physical limit {IRR_MAX} kW/m2{hint}")
        if v < IRR_MIN:
            raise ValueError(f"irradiation {v} cannot be negative (small offsets down to {IRR_MIN} are accepted as dark)")
        return v

    @field_validator("module_temperature")
    @classmethod
    def mod_limits(cls, v):
        if v is not None and not MOD_MIN <= v <= MOD_MAX:
            raise ValueError(_temp_error("module_temperature", v, MOD_MIN, MOD_MAX))
        return v

    @field_validator("ambient_temperature")
    @classmethod
    def amb_limits(cls, v):
        if v is not None and not AMB_MIN <= v <= AMB_MAX:
            raise ValueError(_temp_error("ambient_temperature", v, AMB_MIN, AMB_MAX))
        return v

    @model_validator(mode="after")
    def consistent(self):
        if self.timestamp.tzinfo is not None:
            raise ValueError("timestamp must be plant local time without a UTC offset")
        if self.irradiation > NIGHT_SUN and NIGHT_HOURS(self.timestamp.hour):
            raise ValueError(f"irradiation {self.irradiation} reported at {self.timestamp.strftime('%H:%M')}, "
                             "when there is no sun; check the timestamp (local time?) or the sensor")
        return self


class Prediction(BaseModel):
    expected_ac_power: float
    unit: str
    model: str
    regime: Literal["dark", "daylight"]
    reliability: Literal["ok", "reduced", "low", "outside_envelope"]
    inputs_used: dict[str, float | None]
    within_training_range: bool
    warnings: list[str]
    rules_version: str
    fingerprint: str
    receipt: str


class Batch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    readings: list[Reading] = Field(min_length=1)


class VerifyRequest(Reading):
    expected_ac_power: float = Field(allow_inf_nan=False)
    receipt: str = Field(pattern=r"^[0-9a-f]{64}$")
