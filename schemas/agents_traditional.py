from typing import List, Optional
from pydantic import BaseModel

class PredictionItem(BaseModel):
    id: int
    y_true: int
    baseline_pred: int
    final_pred: Optional[int] = None   
    rationale: Optional[str] = None 

class Analyst(BaseModel):
    title: str
    content: str

class Specialist(BaseModel):
    predictions: List[PredictionItem]
    report: str