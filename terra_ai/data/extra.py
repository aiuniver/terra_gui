from pydantic import BaseModel


class SizeData(BaseModel):
    value: float
    unit: str
