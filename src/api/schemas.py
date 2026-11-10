"""API request and response contracts."""

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, StrictInt, StrictStr, field_validator


class ChatRequest(BaseModel):
    message: StrictStr = Field(min_length=1, max_length=4000)
    session_id: UUID

    @field_validator("message")
    @classmethod
    def nonempty_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message must not be blank")
        return value


class Entities(BaseModel):
    menu_item_name: Optional[str] = None
    allergen_name: Optional[str] = None
    requested_reservation_date: Optional[str] = None
    requested_reservation_time: Optional[str] = None
    number_of_guests: Optional[int] = None
    reservation_identifier: Optional[int] = None
    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: UUID
    text: str
    intent: str
    entities: Entities
    data: Optional[Dict[str, Any]] = None
    success: bool
    is_fallback: bool
    needs_clarification: bool
    missing_slots: List[str]


class HealthResponse(BaseModel):
    status: str
    database: str


class MenuItem(BaseModel):
    id: int
    category: str
    name: str
    description: Optional[str]
    price: float
    is_available: bool
    allergen_status: str


class MenuResponse(BaseModel):
    categories: List[str]
    items: List[MenuItem]


class ReservationRequest(BaseModel):
    customer_name: StrictStr = Field(max_length=200)
    customer_phone: StrictStr = Field(max_length=100)
    party_size: StrictInt
    reservation_date: StrictStr = Field(max_length=32)
    reservation_time: StrictStr = Field(max_length=32)


class ReservationResponse(BaseModel):
    reservation_id: int
    table_id: int
    table_number: str
    table_capacity: int
    customer_name: str
    customer_phone: str
    party_size: int
    reservation_date: str
    reservation_time: str
    duration_minutes: int
    status: str
    created_at: str
