"""Typed contract between the LLM stages and the System 1 decision models.

The LLM's only job during decomposition is to translate natural language (the visitor's
question and the retrieved policy text) into these structures. It does NOT decide outcomes:
the System 1 evaluators compute every Boolean / Choice / Score result deterministically.
"""
from typing import Literal

from pydantic import BaseModel, Field

Operator = Literal[">=", ">", "<=", "<", "==", "!=", "in", "not_in", "is_true", "is_false"]


class Fact(BaseModel):
    name: str = Field(description="snake_case fact name, e.g. gpa, year_of_study, has_sports_scholarship")
    value: str = Field(description="Normalized value: plain number (3.6, 2), 'true'/'false', or short text")
    kind: Literal["number", "boolean", "text"] = "text"


class Condition(BaseModel):
    fact: str = Field(description="Name of the fact this condition tests (must match a Fact name if the user gave it)")
    operator: Operator
    value: str = Field("", description="Threshold/comparison value. Comma-separated list for in/not_in; empty for is_true/is_false")
    description: str = Field("", description="Short human-readable rule, e.g. 'GPA of at least 3.5'")
    source: str = Field("", description="Source id of the passage stating this rule, e.g. 'S2'")


class Option(BaseModel):
    label: str
    conditions: list[Condition] = Field(description="All must hold for this option to apply")


class Criterion(BaseModel):
    condition: Condition
    weight: float = Field(1.0, description="Relative importance, 0-1")


class DecisionSpec(BaseModel):
    id: str = Field(description="Short id such as D1")
    type: Literal["boolean", "choice", "score"]
    question: str = Field(description="The single structured question this decision answers")
    match: Literal["all", "any"] = Field("all", description="boolean only: whether all or any conditions must hold")
    conditions: list[Condition] = Field(default_factory=list, description="boolean only; empty for other types")
    options: list[Option] = Field(default_factory=list, description="choice only, in priority order; empty for other types")
    criteria: list[Criterion] = Field(default_factory=list, description="score only; empty for other types")


class Decomposition(BaseModel):
    restated_question: str = ""
    facts: list[Fact] = Field(default_factory=list, description="Facts the visitor stated about themselves or their situation")
    decisions: list[DecisionSpec] = Field(default_factory=list, description="Independent structured decisions; empty if none are needed")
    combine: Literal["all", "any", "none"] = Field(
        "none", description="How boolean decisions combine into the overall outcome; 'none' if purely informational")


class GeneratedAnswer(BaseModel):
    answer: str = Field(description="Markdown answer citing sources inline like [S1]")
    sources_used: list[str] = Field(default_factory=list, description="Source ids actually relied on")
    grounded: bool = Field(description="False if the passages do not contain the information needed")
