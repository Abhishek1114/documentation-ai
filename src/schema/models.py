import datetime as dt
from typing import Literal, Optional
from pydantic import BaseModel, Field

class Equipment(BaseModel):
    id: str
    name: str
    location: str

class Step(BaseModel):
    no: int = Field(ge=1)
    description: str
    result: Literal["pass", "fail", "n/a"]
    initials: str

class Measurement(BaseModel):
    parameter: str
    value: float
    unit: str
    limit_min: Optional[float] = None
    limit_max: Optional[float] = None

class GMPRecord(BaseModel):
    document_id: str
    document_type: Literal["cleaning_record", "maintenance_log", "sop_excerpt"]
    equipment: Equipment
    date: dt.date
    performed_by: str
    reviewed_by: Optional[str] = None
    steps: list[Step]
    measurements: list[Measurement]
    deviations: list[str]