from pydantic import BaseModel

class Analyst(BaseModel):
    title: str
    content: str

class Specialist(BaseModel):
    predictions: list
    report: str