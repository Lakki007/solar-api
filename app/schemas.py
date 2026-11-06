from __future__ import annotations
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class Reading(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "plant_id": 2, "irradiation": 0.72, "module_temperature": 48.5, "ambient_temperature": 31.2, "timestamp": "2020-06-07T10:00:00"}})
    plant_id: Literal[1, 2]
    irradiation: float = Field(ge=0, le=1.5, allow_inf_nan=False, description="Source sensor value")
    module_temperature: float = Field(ge=-10, le=100, allow_inf_nan=False, description="Degrees C")
    ambient_temperature: float = Field(ge=-10, le=60, allow_inf_nan=False, description="Degrees C")
    timestamp: datetime = Field(description="Plant local time")

    @model_validator(mode="after")
    def naive_local(self):
        if self.timestamp.tzinfo is not None:
            raise ValueError("timestamp must be plant local time without a UTC offset")
        return self

class Prediction(BaseModel):
    expected_ac_power: float
    unit: str
    model: str
    fingerprint: str
    receipt: str
    within_training_range: bool
    warnings: list[str]

class Batch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    readings: list[Reading] = Field(min_length=1)

class VerifyRequest(Reading):
    expected_ac_power: float = Field(allow_inf_nan=False)
    receipt: str = Field(pattern=r"^[0-9a-f]{64}$")
