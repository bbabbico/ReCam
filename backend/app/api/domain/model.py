from pydantic import BaseModel

class BuildingInfoRequest(BaseModel):
    name: str
    category: str
    area: float
    peak: float