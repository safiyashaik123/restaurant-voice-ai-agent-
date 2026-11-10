"""Comprehensive unit tests for the Database Layer (Phase 1).

Covers:
1. Schema & Foreign Key constraints enforcement.
2. Verified seed data integrity.
3. Menu and ingredient queries (case-insensitive, trimmed).
4. Explicit allergen verification status (unverified items are never assumed allergen-free).
5. Missing items returning None (safe fallback trigger).
6. Table availability and best-fit capacity matching.
7. Booking conflict detection with configurable reservation duration.
8. Operating hours and input validation.
9. Reservation cancellation.
"""

from datetime import datetime, timedelta
import sqlite3
import unittest
import uuid
from pathlib import Path

from src.core.config import config
from src.database.connection import create_connection
from src.database.init_db import init_database
from src.database.crud import (
    get_menu_categories,
    get_menu_items,
    get_menu_item_by_name,
    get_ingredients_for_item,
    check_allergens_for_item,
    find_safe_items_for_allergens,
    find_available_tables,
    create_reservation,
    get_reservation_by_id,
    cancel_reservation,
)


class TestDatabaseLayer(unittest.TestCase):
    """Test suite executing against an isolated temporary in-memory database."""
    
    def setUp(self):
        """Create fresh in-memory database with schema and seed data for each test."""
        # Use a unique URI memory database per test to guarantee complete test isolation
        self.db_uri = f"file:memdb_{uuid.uuid4().hex}?mode=memory&cache=shared"
        self.conn = create_connection(self.db_uri)
        init_database(db_path=self.db_uri, seed=True)
        
    def tearDown(self):
        """Close connection and clean up."""
        self.conn.close()

    # -----------------------------------------------------------------------
    # 1. Foreign Key & Integrity Tests
    # -----------------------------------------------------------------------
    def test_foreign_keys_enabled(self):
        """Verify that SQLite enforces foreign key constraints."""
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA foreign_keys;")
        self.assertEqual(cursor.fetchone()[0], 1)

    def test_foreign_key_violation_menu_item(self):
        """Inserting a menu item with a non-existent category_id must fail."""
        cursor = self.conn.cursor()
        with self.assertRaises(sqlite3.IntegrityError):
            cursor.execute(
                """
                INSERT INTO menu_items (category_id, name, price, allergen_status)
                VALUES (99999, 'Phantom Dish', 15.0, 'verified');
                """
            )

    def test_foreign_key_violation_reservation(self):
        """Inserting a reservation with an invalid table_id must fail."""
        cursor = self.conn.cursor()
        with self.assertRaises(sqlite3.IntegrityError):
            cursor.execute(
                """
                INSERT INTO reservations (table_id, customer_name, customer_phone, party_size, reservation_date, reservation_time)
                VALUES (99999, 'John Doe', '555-0100', 2, '2026-11-15', '18:00');
                """
            )

    # -----------------------------------------------------------------------
    # 2. Seed Data Integrity Tests
    # -----------------------------------------------------------------------
    def test_seed_categories(self):
        """Verify seed categories are loaded properly."""
        categories = get_menu_categories(self.conn)
        self.assertIn("Appetizers", categories)
        self.assertIn("Mains", categories)
        self.assertIn("Desserts", categories)
        self.assertIn("Beverages", categories)

    def test_seed_tables(self):
        """Verify tables are loaded with varying capacities."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT table_number, capacity FROM tables ORDER BY capacity ASC;")
        tables = cursor.fetchall()
        self.assertTrue(len(tables) >= 6)
        capacities = [t["capacity"] for t in tables]
        self.assertIn(2, capacities)
        self.assertIn(4, capacities)
        self.assertIn(6, capacities)
        self.assertIn(8, capacities)

    # -----------------------------------------------------------------------
    # 3. Menu & Ingredient Query Tests
    # -----------------------------------------------------------------------
    def test_get_menu_items_by_category(self):
        """Verify filtering menu items by category."""
        mains = get_menu_items(self.conn, category_name="Mains")
        self.assertTrue(len(mains) > 0)
        for item in mains:
            self.assertEqual(item["category"], "Mains")

    def test_get_menu_item_by_name_case_insensitive(self):
        """Lookup by dish name should be case-insensitive and handle whitespace."""
        item = get_menu_item_by_name(self.conn, "  margherita PIZZA  ")
        self.assertIsNotNone(item)
        self.assertEqual(item["name"], "Margherita Pizza")
        self.assertIn("Wheat flour", item["ingredients"])
        self.assertIn("Gluten", item["allergens"])
        self.assertIn("Dairy", item["allergens"])

    def test_nonexistent_menu_item_returns_none(self):
        """Non-existent items must return None to trigger safe fallbacks."""
        item = get_menu_item_by_name(self.conn, "Lobster Thermidor")
        self.assertIsNone(item)
        
        ingredients = get_ingredients_for_item(self.conn, "Lobster Thermidor")
        self.assertIsNone(ingredients)

    # -----------------------------------------------------------------------
    # 4. Explicit Allergen Verification Status Tests (Requirement 6)
    # -----------------------------------------------------------------------
    def test_verified_dish_with_allergens(self):
        """Verified dish with allergens returns is_verified=True and authoritative allergen list."""
        result = check_allergens_for_item(self.conn, "Margherita Pizza")
        self.assertIsNotNone(result)
        self.assertTrue(result["is_verified"])
        self.assertEqual(result["allergen_status"], "verified")
        self.assertIn("Gluten", result["allergens"])
        self.assertIn("Dairy", result["allergens"])

    def test_verified_dish_free_of_allergens(self):
        """Verified dish without allergens returns is_verified=True and empty allergen list."""
        result = check_allergens_for_item(self.conn, "Artisan Sorbet Trio")
        self.assertIsNotNone(result)
        self.assertTrue(result["is_verified"])
        self.assertEqual(result["allergen_status"], "verified")
        self.assertEqual(result["allergens"], [])
        self.assertIn("verified allergen records", result["message"])

    def test_unverified_dish_never_treated_as_allergen_free(self):
        """Unverified dish must return is_verified=False and a cautionary disclaimer message."""
        result = check_allergens_for_item(self.conn, "Chef's Daily Rotating Special")
        self.assertIsNotNone(result)
        self.assertFalse(result["is_verified"])
        self.assertEqual(result["allergen_status"], "unverified")
        self.assertIn("not been formally verified", result["message"])

    def test_safe_items_excludes_unverified_dishes(self):
        """find_safe_items_for_allergens must NEVER include unverified dishes."""
        safe_items = find_safe_items_for_allergens(self.conn, ["Gluten", "Dairy"])
        safe_names = [item["name"] for item in safe_items]
        
        # Contains Gluten/Dairy -> must be excluded
        self.assertNotIn("Margherita Pizza", safe_names)
        self.assertNotIn("Bruschetta Classica", safe_names)
        self.assertNotIn("Classic Tiramisu", safe_names)
        
        # Unverified dish -> must strictly NOT be included even if no allergens are explicitly mapped
        self.assertNotIn("Chef's Daily Rotating Special", safe_names)
        
        # Verified allergen-free dish -> must be included
        self.assertIn("Artisan Sorbet Trio", safe_names)
        self.assertIn("Fresh Mint Lemonade", safe_names)

    # -----------------------------------------------------------------------
    # 5. Table Availability & Best-Fit Matching Tests
    # -----------------------------------------------------------------------
    def test_best_fit_table_selection(self):
        """A party of 2 should be assigned a 2-person table first, preserving larger tables."""
        tables = find_available_tables(
            self.conn,
            party_size=2,
            reservation_date="2026-11-20",
            reservation_time="18:00"
        )
        self.assertTrue(len(tables) > 0)
        self.assertEqual(tables[0]["capacity"], 2)

    def test_party_size_exceeding_capacity(self):
        """Party size exceeding maximum table capacity should return empty list."""
        tables = find_available_tables(
            self.conn,
            party_size=20,
            reservation_date="2026-11-20",
            reservation_time="18:00"
        )
        self.assertEqual(len(tables), 0)

    # -----------------------------------------------------------------------
    # 6. Reservation Creation & Conflict Validation Tests
    # -----------------------------------------------------------------------
    def test_create_reservation_success(self):
        """Valid reservation details successfully persist in database."""
        res = create_reservation(
            self.conn,
            customer_name="Alice Smith",
            customer_phone="555-0199",
            party_size=4,
            reservation_date="2026-11-20",
            reservation_time="19:00"
        )
        self.assertIsNotNone(res["reservation_id"])
        self.assertEqual(res["status"], "confirmed")
        self.assertEqual(res["customer_name"], "Alice Smith")
        self.assertGreaterEqual(res["table_capacity"], 4)
        
        # Verify retrieved from DB
        stored = get_reservation_by_id(self.conn, res["reservation_id"])
        self.assertEqual(stored["customer_name"], "Alice Smith")
        self.assertEqual(stored["status"], "confirmed")

    def test_reservation_conflict_detection_same_table(self):
        """Tables of fitting capacity must not be double-booked during overlapping windows."""
        # Book both 2-seat tables (T1 and T2) at 18:00 for default duration (90 min: 18:00 - 19:30)
        res1 = create_reservation(
            self.conn,
            customer_name="Customer One",
            customer_phone="555-0101",
            party_size=2,
            reservation_date="2026-11-20",
            reservation_time="18:00",
            duration_minutes=90
        )
        res2 = create_reservation(
            self.conn,
            customer_name="Customer Two",
            customer_phone="555-0102",
            party_size=2,
            reservation_date="2026-11-20",
            reservation_time="18:00",
            duration_minutes=90
        )
        
        # Attempting a 3rd booking for 2 pax at 18:30 (overlapping) will bump to next available table (4-pax)
        res3 = create_reservation(
            self.conn,
            customer_name="Customer Three",
            customer_phone="555-0103",
            party_size=2,
            reservation_date="2026-11-20",
            reservation_time="18:30",
            duration_minutes=90
        )
        self.assertGreaterEqual(res3["table_capacity"], 4)
        
        # Verify assigned table is distinct
        booked_table_ids = {res1["table_id"], res2["table_id"], res3["table_id"]}
        self.assertEqual(len(booked_table_ids), 3)

    def test_non_overlapping_reservation_succeeds_on_same_table(self):
        """A reservation after the duration window on the same table must succeed."""
        # Table booked at 18:00 with 60 minute duration (ends at 19:00)
        res1 = create_reservation(
            self.conn,
            customer_name="Early Bird",
            customer_phone="555-0201",
            party_size=6,
            reservation_date="2026-11-20",
            reservation_time="18:00",
            duration_minutes=60
        )
        
        # Second booking at 19:00 on the same date for 6 pax must succeed (no overlap)
        res2 = create_reservation(
            self.conn,
            customer_name="Late Diner",
            customer_phone="555-0202",
            party_size=6,
            reservation_date="2026-11-20",
            reservation_time="19:00",
            duration_minutes=60
        )
        
        self.assertEqual(res1["table_id"], res2["table_id"])
        self.assertEqual(res2["status"], "confirmed")

    def test_configurable_duration_shifts_conflict_window(self):
        """Verifying that changing duration_minutes dynamically alters conflict window."""
        # Booking at 18:00
        create_reservation(
            self.conn,
            customer_name="Host",
            customer_phone="555-0301",
            party_size=8,
            reservation_date="2026-11-21",
            reservation_time="18:00",
            duration_minutes=60  # Only booked 18:00 - 19:00
        )
        
        # Under 60 min duration, 19:00 is available for the 8-seat table
        avail_60 = find_available_tables(
            self.conn,
            party_size=8,
            reservation_date="2026-11-21",
            reservation_time="19:00",
            duration_minutes=60
        )
        self.assertEqual(len(avail_60), 1)
        
        # Under 90 min duration, the 18:00 booking would run until 19:30, conflicting with 19:00!
        avail_90 = find_available_tables(
            self.conn,
            party_size=8,
            reservation_date="2026-11-21",
            reservation_time="19:00",
            duration_minutes=90
        )
        self.assertEqual(len(avail_90), 0)

    # -----------------------------------------------------------------------
    # 7. Operating Hours & Input Validation Tests
    # -----------------------------------------------------------------------
    def test_reservation_outside_operating_hours(self):
        """Reservations before opening (11:00) or after closing (22:00) must be rejected."""
        with self.assertRaises(ValueError) as ctx:
            create_reservation(
                self.conn,
                customer_name="Early Bird",
                customer_phone="555-0401",
                party_size=2,
                reservation_date="2026-11-22",
                reservation_time="08:00"
            )
        self.assertIn("operating hours", str(ctx.exception))

    def test_invalid_customer_inputs(self):
        """Empty names, invalid phones, or negative party sizes must be rejected."""
        with self.assertRaises(ValueError):
            create_reservation(self.conn, "", "555-0501", 2, "2026-11-22", "18:00")
            
        with self.assertRaises(ValueError):
            create_reservation(self.conn, "Valid Name", "123", 2, "2026-11-22", "18:00")
            
        with self.assertRaises(ValueError):
            create_reservation(self.conn, "Valid Name", "555-0502", 0, "2026-11-22", "18:00")

    # -----------------------------------------------------------------------
    # 8. Cancellation Tests
    # -----------------------------------------------------------------------
    def test_cancel_reservation_frees_table(self):
        """Cancelling a reservation must free up the table for new bookings."""
        res = create_reservation(
            self.conn,
            customer_name="Cancel Candidate",
            customer_phone="555-0601",
            party_size=8,
            reservation_date="2026-11-23",
            reservation_time="18:00"
        )
        
        # Before cancel: 8-pax table is booked
        avail_before = find_available_tables(
            self.conn,
            party_size=8,
            reservation_date="2026-11-23",
            reservation_time="18:00"
        )
        self.assertEqual(len(avail_before), 0)
        
        # Cancel reservation
        success = cancel_reservation(self.conn, res["reservation_id"])
        self.assertTrue(success)
        
        # After cancel: table is available again
        avail_after = find_available_tables(
            self.conn,
            party_size=8,
            reservation_date="2026-11-23",
            reservation_time="18:00"
        )
        self.assertEqual(len(avail_after), 1)


if __name__ == "__main__":
    unittest.main()
