"""Database initialisation and seed data loader.

Creates all tables defined in schema.sql and populates verified seed data.
Safe to re-run (idempotent).
"""

import json
from pathlib import Path
from typing import Optional

from src.core.config import config
from src.database.connection import create_connection


SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
SEED_PATH = config.data_dir / "seed_data.json"


def init_database(
    db_path: Optional[Path | str] = None,
    seed: bool = True
) -> None:
    """Initialize SQLite schema and optionally populate verified seed data."""
    conn = create_connection(db_path)
    try:
        # 1. Apply schema
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        conn.executescript(schema_sql)
                # 1.1 Migrate existing databases
        reservation_columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(reservations)"
            ).fetchall()
        }

        if "duration_minutes" not in reservation_columns:
            conn.execute(
                """
                ALTER TABLE reservations
                ADD COLUMN duration_minutes INTEGER NOT NULL DEFAULT 90
                CHECK (duration_minutes > 0)
                """
            )
        
        # 2. Check if already seeded
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM menu_items;")
        item_count = cursor.fetchone()[0]
        
        if seed and item_count == 0 and SEED_PATH.exists():
            with open(SEED_PATH, "r", encoding="utf-8") as f:
                seed_data = json.load(f)
                
            conn.execute("BEGIN;")
            
            # Seed categories
            category_map = {}
            for cat_name in seed_data.get("categories", []):
                cursor.execute(
                    "INSERT INTO categories (name) VALUES (?) ON CONFLICT (name) DO UPDATE SET name=excluded.name RETURNING id;",
                    (cat_name,)
                )
                category_map[cat_name] = cursor.fetchone()[0]
                
            # Seed allergens
            allergen_map = {}
            for allergen_name in seed_data.get("allergens", []):
                cursor.execute(
                    "INSERT INTO allergens (name) VALUES (?) ON CONFLICT (name) DO UPDATE SET name=excluded.name RETURNING id;",
                    (allergen_name,)
                )
                allergen_map[allergen_name] = cursor.fetchone()[0]
                
            # Seed menu items, ingredients, and allergen associations
            for item in seed_data.get("menu_items", []):
                cat_id = category_map[item["category"]]
                allergen_status = item.get("allergen_status", "unverified")
                cursor.execute(
                    """
                    INSERT INTO menu_items (category_id, name, description, price, is_available, allergen_status)
                    VALUES (?, ?, ?, ?, ?, ?)
                    RETURNING id;
                    """,
                    (
                        cat_id,
                        item["name"],
                        item["description"],
                        item["price"],
                        1 if item.get("is_available", True) else 0,
                        allergen_status
                    )
                )
                item_id = cursor.fetchone()[0]
                
                # Insert exact ingredients
                for ing in item.get("ingredients", []):
                    cursor.execute(
                        "INSERT INTO menu_item_ingredients (menu_item_id, ingredient_name) VALUES (?, ?);",
                        (item_id, ing)
                    )
                    
                # Link verified allergens
                for allergen_name in item.get("allergens", []):
                    if allergen_name in allergen_map:
                        cursor.execute(
                            "INSERT INTO menu_item_allergens (menu_item_id, allergen_id) VALUES (?, ?);",
                            (item_id, allergen_map[allergen_name])
                        )
                        
            # Seed tables
            for tbl in seed_data.get("tables", []):
                cursor.execute(
                    """
                    INSERT INTO tables (table_number, capacity, is_active)
                    VALUES (?, ?, ?)
                    ON CONFLICT (table_number) DO UPDATE SET capacity=excluded.capacity, is_active=excluded.is_active;
                    """,
                    (tbl["table_number"], tbl["capacity"], 1 if tbl.get("is_active", True) else 0)
                )
                
            conn.execute("COMMIT;")
    except Exception:
        conn.execute("ROLLBACK;")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    print(f"Initializing database at: {config.db_path}")
    init_database()
    print("Database initialization complete with verified seed data.")
