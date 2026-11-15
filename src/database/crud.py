"""Deterministic database CRUD services for the Restaurant Voice AI Agent.

Enforces:
1. Strict database-grounded lookups for menus, ingredients, and allergens.
2. Explicit allergen verification status: unverified dishes are NEVER treated as allergen-free.
3. Safe fallback triggers (returns None when an item does not exist).
4. Configurable reservation duration and conflict calculation.
"""

from datetime import datetime, timedelta
import re
import hashlib
import secrets
import sqlite3
from typing import Any, Dict, List, Optional

from src.core.config import config
from src.core.reservation_policy import validate_customer_name, validate_indian_mobile, validate_reservation_slot


def _normalize_name(name: str) -> str:
    """Normalize string for consistent case-insensitive queries."""
    return re.sub(r"\s+", " ", name.strip().lower())


# ---------------------------------------------------------------------------
# Menu, Ingredients & Allergen Services
# ---------------------------------------------------------------------------

def get_menu_categories(conn: sqlite3.Connection) -> List[str]:
    """Retrieve all available menu categories."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM categories ORDER BY id ASC;")
    return [row["name"] for row in cursor.fetchall()]


def get_menu_items(
    conn: sqlite3.Connection,
    category_name: Optional[str] = None,
    available_only: bool = True
) -> List[Dict[str, Any]]:
    """Retrieve menu items, optionally filtered by category name."""
    query = """
        SELECT 
            m.id,
            c.name AS category,
            m.name,
            m.description,
            m.price,
            m.is_available,
            m.allergen_status
        FROM menu_items m
        JOIN categories c ON m.category_id = c.id
        WHERE 1=1
    """
    params: List[Any] = []
    
    if available_only:
        query += " AND m.is_available = 1"
        
    if category_name:
        query += " AND LOWER(c.name) = LOWER(?)"
        params.append(category_name.strip())
        
    query += " ORDER BY c.id ASC, m.price ASC;"
    
    cursor = conn.cursor()
    cursor.execute(query, params)
    return [dict(row) for row in cursor.fetchall()]


def get_menu_item_by_name(
    conn: sqlite3.Connection,
    item_name: str
) -> Optional[Dict[str, Any]]:
    """Retrieve a single menu item with its full ingredients and verified allergens.
    
    Returns None if the item does not exist in the database.
    """
    norm_name = _normalize_name(item_name)
    cursor = conn.cursor()
    
    # 1. Fetch item details
    cursor.execute(
        """
        SELECT 
            m.id,
            c.name AS category,
            m.name,
            m.description,
            m.price,
            m.is_available,
            m.allergen_status
        FROM menu_items m
        JOIN categories c ON m.category_id = c.id
        WHERE LOWER(m.name) = ?;
        """,
        (norm_name,)
    )
    item_row = cursor.fetchone()
    if not item_row:
        return None
        
    item_id = item_row["id"]
    
    # 2. Fetch verified ingredients
    cursor.execute(
        "SELECT ingredient_name FROM menu_item_ingredients WHERE menu_item_id = ? ORDER BY id ASC;",
        (item_id,)
    )
    ingredients = [row["ingredient_name"] for row in cursor.fetchall()]
    
    # 3. Fetch allergens linked in database
    cursor.execute(
        """
        SELECT a.name 
        FROM allergens a
        JOIN menu_item_allergens ma ON a.id = ma.allergen_id
        WHERE ma.menu_item_id = ?
        ORDER BY a.name ASC;
        """,
        (item_id,)
    )
    allergens = [row["name"] for row in cursor.fetchall()]
    
    res = dict(item_row)
    res["ingredients"] = ingredients
    res["allergens"] = allergens
    return res


def get_ingredients_for_item(
    conn: sqlite3.Connection,
    item_name: str
) -> Optional[List[str]]:
    """Retrieve verified ingredients list for a dish.
    
    Returns None if item is not found (enables caller to trigger safe fallback).
    """
    item = get_menu_item_by_name(conn, item_name)
    if item is None:
        return None
    return item["ingredients"]


def check_allergens_for_item(
    conn: sqlite3.Connection,
    item_name: str
) -> Optional[Dict[str, Any]]:
    """Check allergen safety for a dish.
    
    Strict Safety Invariant:
    - If item does not exist -> returns None (triggers safe fallback).
    - If allergen_status is 'unverified' -> verified=False. Never claims absence of allergens.
    - If allergen_status is 'verified' -> verified=True with authoritative allergen list.
    """
    item = get_menu_item_by_name(conn, item_name)
    if item is None:
        return None
        
    is_verified = (item.get("allergen_status") == "verified")
    allergens = item.get("allergens", [])
    
    if not is_verified:
        message = (
            f"Allergen information for '{item['name']}' has not been formally verified by the kitchen. "
            "Please speak directly with restaurant staff."
        )
    elif allergens:
        message = f"'{item['name']}' contains the following verified allergens: {', '.join(allergens)}."
    else:
        message = f"'{item['name']}' has verified allergen records and does not contain tracked major allergens."
        
    return {
        "item_name": item["name"],
        "allergen_status": item.get("allergen_status", "unverified"),
        "is_verified": is_verified,
        "allergens": allergens,
        "message": message
    }


def find_safe_items_for_allergens(
    conn: sqlite3.Connection,
    excluded_allergens: List[str]
) -> List[Dict[str, Any]]:
    """Find menu items that are verified safe and do NOT contain ANY of the excluded allergens.
    
    Strict Safety Invariant:
    - Dishes with allergen_status = 'unverified' are strictly excluded from safe recommendations,
      ensuring unverified information is never treated as confirmed absence of allergens.
    """
    clean_allergens = [a.strip().lower() for a in excluded_allergens if a.strip()]
    tracked = {row["name"].lower() for row in conn.execute("SELECT name FROM allergens")}
    if any(name not in tracked for name in clean_allergens):
        raise ValueError("Allergen is not tracked in the restaurant database.")
    
    if not clean_allergens:
        # If no allergens excluded, return all available verified items
        query = """
            SELECT 
                m.id,
                c.name AS category,
                m.name,
                m.description,
                m.price,
                m.allergen_status
            FROM menu_items m
            JOIN categories c ON m.category_id = c.id
            WHERE m.is_available = 1 AND m.allergen_status = 'verified'
            ORDER BY c.id ASC, m.name ASC;
        """
        cursor = conn.cursor()
        cursor.execute(query)
        return [dict(row) for row in cursor.fetchall()]
        
    placeholders = ",".join("?" for _ in clean_allergens)
    
    query = f"""
        SELECT 
            m.id,
            c.name AS category,
            m.name,
            m.description,
            m.price,
            m.allergen_status
        FROM menu_items m
        JOIN categories c ON m.category_id = c.id
        WHERE m.is_available = 1
        AND m.allergen_status = 'verified'
        AND m.id NOT IN (
            SELECT ma.menu_item_id
            FROM menu_item_allergens ma
            JOIN allergens a ON ma.allergen_id = a.id
            WHERE LOWER(a.name) IN ({placeholders})
        )
        ORDER BY c.id ASC, m.name ASC;
    """
    cursor = conn.cursor()
    cursor.execute(query, clean_allergens)
    return [dict(row) for row in cursor.fetchall()]


# ---------------------------------------------------------------------------
# Table Availability & Reservation Services
# ---------------------------------------------------------------------------

def _parse_time(time_str: str) -> datetime.time:
    """Parse HH:MM string to time object."""
    try:
        return datetime.strptime(time_str.strip(), "%H:%M").time()
    except ValueError:
        raise ValueError(f"Invalid time format '{time_str}'. Expected HH:MM in 24-hour format.")


def _parse_date(date_str: str) -> datetime.date:
    """Parse YYYY-MM-DD string to date object."""
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"Invalid date format '{date_str}'. Expected YYYY-MM-DD.")


def _is_within_operating_hours(res_time: str, duration_minutes: int) -> bool:
    """Check if the reservation slot falls within restaurant operating hours."""
    start_t = _parse_time(res_time)
    open_t = _parse_time(config.opening_time)
    close_t = _parse_time(config.closing_time)
    
    # Calculate reservation end time
    dummy_date = datetime(2000, 1, 1, start_t.hour, start_t.minute)
    end_dt = dummy_date + timedelta(minutes=duration_minutes)
    end_t = end_dt.time()
    
    if start_t < open_t:
        return False
    if end_t > close_t or end_dt.day != 1:  # Check closing time or past midnight
        return False
    return True


def find_available_tables(
    conn: sqlite3.Connection,
    party_size: int,
    reservation_date: str,
    reservation_time: str,
    duration_minutes: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Find active dining tables that can accommodate party_size with no conflicting bookings.
    
    Uses configurable slot duration for conflict detection.
    """
    if type(party_size) is not int or party_size <= 0:
        raise ValueError("Party size must be at least 1 person.")
    if party_size > 9223372036854775807:
        return []
        
    slot_duration = duration_minutes if duration_minutes is not None else config.default_reservation_duration_minutes
    req_date = _parse_date(reservation_date)
    req_time = _parse_time(reservation_time)
    
    req_start = datetime.combine(req_date, req_time)
    req_end = req_start + timedelta(minutes=slot_duration)
    
    # Check operating hours
    if not _is_within_operating_hours(reservation_time, slot_duration):
        return []
        
    cursor = conn.cursor()
    
    # Fetch active tables with sufficient capacity, sorted by best-fit capacity
    cursor.execute(
        """
        SELECT id, table_number, capacity
        FROM tables
        WHERE is_active = 1 AND capacity >= ?
        ORDER BY capacity ASC, table_number ASC;
        """,
        (party_size,)
    )
    candidate_tables = [dict(row) for row in cursor.fetchall()]
    if not candidate_tables:
        return []
        
    # Check conflicting confirmed bookings for candidate tables
    table_ids = [t["id"] for t in candidate_tables]
    placeholders = ",".join("?" for _ in table_ids)
    
    cursor.execute(
        f"""
        SELECT table_id, reservation_time
        FROM reservations
        WHERE reservation_date = ?
        AND status = 'confirmed'
        AND table_id IN ({placeholders});
        """,
        [reservation_date.strip()] + table_ids
    )
    existing_bookings = cursor.fetchall()
    
    # Identify tables with time overlaps using configurable slot duration
    conflicted_table_ids = set()
    for row in existing_bookings:
        tbl_id = row["table_id"]
        res_time = _parse_time(row["reservation_time"])
        res_start = datetime.combine(req_date, res_time)
        res_end = res_start + timedelta(minutes=slot_duration)
        
        # Overlap condition: req_start < res_end and req_end > res_start
        if req_start < res_end and req_end > res_start:
            conflicted_table_ids.add(tbl_id)
            
    # Return available tables ordered by best-fit capacity
    return [t for t in candidate_tables if t["id"] not in conflicted_table_ids]


def create_reservation(
    conn: sqlite3.Connection,
    customer_name: str,
    customer_phone: str,
    party_size: int,
    reservation_date: str,
    reservation_time: str,
    duration_minutes: Optional[int] = None
) -> Dict[str, Any]:
    """Reserve atomically, committing only transactions opened by this function.

    When a transaction is already active, its caller owns commit and rollback;
    the caller should acquire an immediate write transaction before booking and
    must not report confirmation until it has committed.
    """
    owns_transaction = not conn.in_transaction
    if owns_transaction:
        conn.execute("BEGIN IMMEDIATE")
    try:
        result = _create_reservation_in_transaction(
            conn, customer_name, customer_phone, party_size,
            reservation_date, reservation_time, duration_minutes,
        )
        if owns_transaction:
            conn.execute("COMMIT")
        return result
    except Exception:
        if owns_transaction and conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def _create_reservation_in_transaction(
    conn: sqlite3.Connection,
    customer_name: str,
    customer_phone: str,
    party_size: int,
    reservation_date: str,
    reservation_time: str,
    duration_minutes: Optional[int] = None
) -> Dict[str, Any]:
    """Validate, allocate and insert while a transaction is active."""
    # 1. Input sanitization & validation
    name_clean = validate_customer_name(customer_name)
    phone_clean = validate_indian_mobile(customer_phone)
    if type(party_size) is not int or party_size <= 0:
        raise ValueError("Party size must be a positive integer.")
    if party_size > 9223372036854775807:
        raise ValueError("No tables available for that party size.")
        
    slot_duration = duration_minutes if duration_minutes is not None else config.default_reservation_duration_minutes
    
    # Validate date, interval and operating hours against the shared policy.
    validate_reservation_slot(reservation_date, reservation_time, duration_minutes=slot_duration)
        
    # Under the immediate write transaction, reject an exact repeated booking.
    # Normalize the optional country prefix for this comparison only.
    mobile = phone_clean.removeprefix("+91")
    duplicate = conn.execute(
        "SELECT 1 FROM reservations WHERE status='confirmed' AND reservation_date=? "
        "AND reservation_time=? AND (customer_phone=? OR customer_phone=?) "
        "AND customer_name=? COLLATE NOCASE AND party_size=? LIMIT 1",
        (reservation_date.strip(), reservation_time.strip(), mobile, '+91' + mobile, name_clean, party_size),
    ).fetchone()
    if duplicate:
        raise ValueError("A reservation for this phone and time already exists.")

    # 2. Check table availability
    available_tables = find_available_tables(
        conn,
        party_size=party_size,
        reservation_date=reservation_date,
        reservation_time=reservation_time,
        duration_minutes=slot_duration
    )
    
    if not available_tables:
        raise ValueError(
            f"No tables available for party of {party_size} on {reservation_date} at {reservation_time}."
        )
        
    # Best-fit table is first in list (capacity ASC)
    assigned_table = available_tables[0]
    
    # 3. Insert reservation record
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO reservations (
            table_id,
            customer_name,
            customer_phone,
            party_size,
            reservation_date,
            reservation_time,
            status
        ) VALUES (?, ?, ?, ?, ?, ?, 'confirmed')
        RETURNING id, created_at;
        """,
        (
            assigned_table["id"],
            name_clean,
            phone_clean,
            party_size,
            reservation_date.strip(),
            reservation_time.strip()
        )
    )
    res_row = cursor.fetchone()
    # Additive, transactional upgrade: existing reservations and tables are untouched.
    conn.execute("CREATE TABLE IF NOT EXISTS reservation_access (reservation_id INTEGER PRIMARY KEY REFERENCES reservations(id) ON DELETE CASCADE, secret_hash TEXT NOT NULL)")
    access_code = secrets.token_urlsafe(24)
    conn.execute("INSERT INTO reservation_access VALUES (?, ?)", (res_row["id"], hashlib.sha256(access_code.encode('ascii')).hexdigest()))

    return {
        "access_code": access_code,
        "reservation_id": res_row["id"],
        "table_id": assigned_table["id"],
        "table_number": assigned_table["table_number"],
        "table_capacity": assigned_table["capacity"],
        "customer_name": name_clean,
        "customer_phone": phone_clean,
        "party_size": party_size,
        "reservation_date": reservation_date.strip(),
        "reservation_time": reservation_time.strip(),
        "duration_minutes": slot_duration,
        "status": "confirmed",
        "created_at": res_row["created_at"]
    }


def get_reservation_by_id(
    conn: sqlite3.Connection,
    reservation_id: int
) -> Optional[Dict[str, Any]]:
    """Fetch confirmed or cancelled reservation by ID."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT 
            r.id,
            r.table_id,
            t.table_number,
            t.capacity AS table_capacity,
            r.customer_name,
            r.customer_phone,
            r.party_size,
            r.reservation_date,
            r.reservation_time,
            r.status,
            r.created_at
        FROM reservations r
        JOIN tables t ON r.table_id = t.id
        WHERE r.id = ?;
        """,
        (reservation_id,)
    )
    row = cursor.fetchone()
    return dict(row) if row else None


def cancel_reservation(
    conn: sqlite3.Connection,
    reservation_id: int
) -> bool:
    """Cancel an existing reservation."""
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE reservations SET status = 'cancelled' WHERE id = ? AND status = 'confirmed';",
        (reservation_id,)
    )
    return cursor.rowcount > 0


def verified_reservation(conn: sqlite3.Connection, reservation_id: int, access_code: str) -> Optional[Dict[str, Any]]:
    """Return only a reservation whose independently generated secret matches.

    Older bookings have no credential and must be handled by restaurant staff.
    """
    import hmac
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='reservation_access'").fetchone():
        return None
    row = conn.execute("SELECT secret_hash FROM reservation_access WHERE reservation_id=?", (reservation_id,)).fetchone()
    digest = hashlib.sha256(access_code.encode('utf-8')).hexdigest()
    expected = row['secret_hash'] if row else '0' * 64
    if not hmac.compare_digest(digest, expected) or row is None:
        return None
    item = get_reservation_by_id(conn, reservation_id)
    if not item:
        return None
    return {"reservation_id": item['id'], "reservation_date": item['reservation_date'],
            "reservation_time": item['reservation_time'], "party_size": item['party_size'], "status": item['status']}
