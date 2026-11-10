"""Focused transaction tests using a temporary SQLite database."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Barrier
import unittest

from src.core.reservation_policy import restaurant_today
from src.database.connection import create_connection, get_db
from src.database.crud import create_reservation
from src.database.init_db import init_database


class TestReservationAtomicity(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db_path = Path(self.temp.name) / "restaurant.db"
        init_database(self.db_path)
        self.booking_date = (restaurant_today() + timedelta(days=1)).isoformat()

    def activate_tables(self, *table_numbers):
        with get_db(self.db_path, immediate=True) as conn:
            conn.execute("UPDATE tables SET is_active = 0")
            conn.executemany("UPDATE tables SET is_active = 1 WHERE table_number = ?", [(number,) for number in table_numbers])

    def assert_closed_and_unlocked(self, connections):
        for conn in connections:
            with self.assertRaises(sqlite3.ProgrammingError):
                conn.execute("SELECT 1")
        # A fresh immediate transaction must be able to acquire the write lock.
        with get_db(self.db_path, immediate=True) as conn:
            self.assertTrue(conn.in_transaction)

    def concurrent_bookings(self):
        barrier = Barrier(2)
        connections = []

        def book(name):
            barrier.wait(timeout=5)
            try:
                with get_db(self.db_path, immediate=True) as conn:
                    connections.append(conn)
                    result = create_reservation(conn, name, "9870001234", 2, self.booking_date, "18:00")
                    self.assertTrue(conn.in_transaction)  # The caller still owns the commit.
                return result  # Only after the context commits.
            except ValueError as exc:
                if not str(exc).startswith("No tables available"):
                    raise
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(book, ("Alice", "Bob")))
        self.assertEqual(len(connections), 2)
        self.assert_closed_and_unlocked(connections)
        return results

    def test_concurrent_bookings_cannot_claim_the_same_table(self):
        self.activate_tables("T1")
        results = self.concurrent_bookings()
        self.assertEqual(sum(result is not None for result in results), 1)
        with get_db(self.db_path) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM reservations WHERE status = 'confirmed'").fetchone()[0], 1)

    def test_concurrent_bookings_use_different_available_tables(self):
        self.activate_tables("T1", "T2")
        results = self.concurrent_bookings()
        self.assertTrue(all(result is not None for result in results))
        self.assertEqual(len({result["table_id"] for result in results}), 2)

    def test_caller_transaction_can_roll_back_reservation(self):
        with self.assertRaisesRegex(RuntimeError, "rollback"):
            with get_db(self.db_path, immediate=True) as conn:
                create_reservation(conn, "Alice", "9870001234", 2, self.booking_date, "18:00")
                raise RuntimeError("rollback")
        self.assert_closed_and_unlocked([conn])
        with get_db(self.db_path) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM reservations").fetchone()[0], 0)

    def test_crud_owns_and_commits_transaction_when_called_without_one(self):
        with closing(create_connection(self.db_path)) as conn:
            self.assertFalse(conn.in_transaction)
            booking = create_reservation(conn, "Alice", "9870001234", 2, self.booking_date, "18:00")
            self.assertFalse(conn.in_transaction)
        self.assert_closed_and_unlocked([conn])
        with get_db(self.db_path) as check:
            self.assertIsNotNone(check.execute("SELECT id FROM reservations WHERE id = ?", (booking["reservation_id"],)).fetchone())

    def test_crud_rolls_back_its_owned_transaction_on_error(self):
        with closing(create_connection(self.db_path)) as conn:
            with self.assertRaises(ValueError):
                create_reservation(conn, "", "9870001234", 2, self.booking_date, "18:00")
            self.assertFalse(conn.in_transaction)
        self.assert_closed_and_unlocked([conn])
        with get_db(self.db_path) as check:
            self.assertEqual(check.execute("SELECT COUNT(*) FROM reservations").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
