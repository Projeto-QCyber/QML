from pydantic import BaseModel
from typing import List, Optional


class Analyst(BaseModel):
    title: str
    content: str

class Specialist(BaseModel):
    predictions: List[int]
    report: Optional[str] = None