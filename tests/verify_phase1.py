"""Verification script for Phase 1 final sign-off requirements.

Verifies:
1. Allergen-free results are returned only for dishes explicitly marked as verified.
2. Reservation duration is read from config (default_reservation_duration_minutes) and not hardcoded.
3. Database initialization idempotency: running init_database twice preserves data and reservations without duplication.
"""

import os
import sys
from pathlib import Path

from src.core.config import config
from src.database.connection import create_connection
from src.database.init_db import init_database
from src.database.crud import (
    find_safe_items_for_allergens,
    check_allergens_for_item,
    find_available_tables,
    create_reservation,
    get_reservation_by_id,
)


def verify_allergen_verification():
    print("--- 1. Testing Allergen Verification Safety ---")
    conn = create_connection("data/restaurant.db")
    
    # 1. Check all items returned when excluding no allergens
    all_safe = find_safe_items_for_allergens(conn, [])
    for item in all_safe:
        assert item["allergen_status"] == "verified", (
            f"SAFETY VIOLATION: Unverified dish '{item['name']}' returned in safe items!"
        )
    print(f"PASS: Baseline safe query returned {len(all_safe)} items, 100% marked 'verified'.")
    
    # 2. Check items when excluding Gluten and Dairy
    safe_dishes = find_safe_items_for_allergens(conn, ["Gluten", "Dairy"])
    safe_names = [d["name"] for d in safe_dishes]
    
    # The unverified daily special has no allergens listed in seed data, but must NOT be in safe items!
    assert "Chef's Daily Rotating Special" not in safe_names, (
        "SAFETY VIOLATION: Unverified dish was assumed allergen-free!"
    )
    for d in safe_dishes:
        assert d["allergen_status"] == "verified", (
            f"SAFETY VIOLATION: Dish '{d['name']}' in safe list is not verified!"
        )
    print(f"PASS: Safe query excluding Gluten & Dairy returned {len(safe_dishes)} items, all verified. Unverified items strictly excluded.")
    
    # 3. Check individual check_allergens_for_item
    item_check = check_allergens_for_item(conn, "Chef's Daily Rotating Special")
    assert item_check is not None
    assert item_check["is_verified"] is False
    assert item_check["allergen_status"] == "unverified"
    assert "not been formally verified" in item_check["message"]
    print("PASS: Individual item check for unverified dish returns is_verified=False with safety disclaimer.")
    
    conn.close()


def verify_configurable_reservation_duration():
    print("\n--- 2. Testing Reservation Duration from Configuration ---")
    conn = create_connection("data/restaurant.db")
    
    # Check what config currently specifies
    original_config_duration = config.default_reservation_duration_minutes
    print(f"Current config.default_reservation_duration_minutes: {original_config_duration}")
    
    # Verify that in crud.py, when duration_minutes is omitted (None), it uses config.default_reservation_duration_minutes
    # Test by temporarily changing config.default_reservation_duration_minutes
    config.default_reservation_duration_minutes = 45
    
    # Book a table without passing duration_minutes
    res = create_reservation(
        conn,
        customer_name="Config Duration Test",
        customer_phone="555-8888",
        party_size=2,
        reservation_date="2026-12-01",
        reservation_time="12:00"
    )
    assert res["duration_minutes"] == 45, (
        f"FAILED: Expected duration_minutes to be 45 from config, got {res['duration_minutes']}"
    )
    print("PASS: create_reservation without duration_minutes correctly inherited 45 from config.")
    
    # Clean up test reservation and restore config
    cursor = conn.cursor()
    cursor.execute("DELETE FROM reservations WHERE id = ?;", (res["reservation_id"],))
    config.default_reservation_duration_minutes = original_config_duration
    print(f"PASS: Restored config.default_reservation_duration_minutes to {original_config_duration}.")
    
    conn.close()


def verify_idempotent_initialization():
    print("\n--- 3. Testing Idempotent Database Initialization & Data Preservation ---")
    test_db = "data/test_idempotent.db"
    if os.path.exists(test_db):
        os.remove(test_db)
        
    # First init
    print("Running initial init_database on test database...")
    init_database(test_db, seed=True)
    conn = create_connection(test_db)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM menu_items;")
    count_items_1 = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM tables;")
    count_tables_1 = c.fetchone()[0]
    
    # Add an active customer reservation
    print("Creating active reservation in test database...")
    res = create_reservation(
        conn,
        customer_name="Preserve Me",
        customer_phone="555-9999",
        party_size=4,
        reservation_date="2026-12-15",
        reservation_time="19:00"
    )
    res_id = res["reservation_id"]
    conn.close()
    
    # Run second init
    print("Running second init_database on test database...")
    init_database(test_db, seed=True)
    
    # Verify data after second init
    conn2 = create_connection(test_db)
    c2 = conn2.cursor()
    c2.execute("SELECT COUNT(*) FROM menu_items;")
    count_items_2 = c2.fetchone()[0]
    c2.execute("SELECT COUNT(*) FROM tables;")
    count_tables_2 = c2.fetchone()[0]
    
    assert count_items_1 == count_items_2, (
        f"FAILED: Menu items duplicated! Before: {count_items_1}, After: {count_items_2}"
    )
    assert count_tables_1 == count_tables_2, (
        f"FAILED: Tables duplicated! Before: {count_tables_1}, After: {count_tables_2}"
    )
    
    # Verify reservation is still intact
    preserved_res = get_reservation_by_id(conn2, res_id)
    assert preserved_res is not None, "FAILED: Customer reservation was lost after second init_database!"
    assert preserved_res["customer_name"] == "Preserve Me"
    print(f"PASS: Menu items count preserved ({count_items_2}), tables count preserved ({count_tables_2}), and reservation '{preserved_res['customer_name']}' preserved intact without duplication.")
    
    conn2.close()
    if os.path.exists(test_db):
        os.remove(test_db)


if __name__ == "__main__":
    verify_allergen_verification()
    verify_configurable_reservation_duration()
    verify_idempotent_initialization()
    print("\nALL VERIFICATION CHECKS COMPLETED SUCCESSFULLY.")
