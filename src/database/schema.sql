-- SQLite Schema for Restaurant Voice AI Agent
-- Strict relational integrity with foreign keys enabled

PRAGMA foreign_keys = ON;

-- Categories (Appetizers, Mains, Desserts, Beverages, etc.)
CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL COLLATE NOCASE
);

-- Menu items with explicit allergen verification status
CREATE TABLE IF NOT EXISTS menu_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_id INTEGER NOT NULL REFERENCES categories(id) ON DELETE RESTRICT,
    name TEXT UNIQUE NOT NULL COLLATE NOCASE,
    description TEXT,
    price REAL NOT NULL CHECK (price >= 0),
    is_available INTEGER NOT NULL DEFAULT 1 CHECK (is_available IN (0, 1)),
    -- 'verified': kitchen/supplier audit completed; 
    -- 'unverified': allergen profile pending review, must NEVER be assumed allergen-free
    allergen_status TEXT NOT NULL DEFAULT 'unverified' CHECK (allergen_status IN ('verified', 'unverified'))
);

-- Master list of allergens (FDA/EU recognized allergens)
CREATE TABLE IF NOT EXISTS allergens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL COLLATE NOCASE
);

-- Exact verified ingredients for each menu item
CREATE TABLE IF NOT EXISTS menu_item_ingredients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    menu_item_id INTEGER NOT NULL REFERENCES menu_items(id) ON DELETE CASCADE,
    ingredient_name TEXT NOT NULL COLLATE NOCASE
);

-- Verified allergen mapping for each menu item (m:n)
CREATE TABLE IF NOT EXISTS menu_item_allergens (
    menu_item_id INTEGER NOT NULL REFERENCES menu_items(id) ON DELETE CASCADE,
    allergen_id INTEGER NOT NULL REFERENCES allergens(id) ON DELETE CASCADE,
    PRIMARY KEY (menu_item_id, allergen_id)
);

-- Dining tables with capacities
CREATE TABLE IF NOT EXISTS tables (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_number TEXT UNIQUE NOT NULL,
    capacity INTEGER NOT NULL CHECK (capacity > 0),
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1))
);

-- Customer reservations
CREATE TABLE IF NOT EXISTS reservations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_id INTEGER NOT NULL REFERENCES tables(id) ON DELETE RESTRICT,
    customer_name TEXT NOT NULL,
    customer_phone TEXT NOT NULL,
    party_size INTEGER NOT NULL CHECK (party_size > 0),
    reservation_date TEXT NOT NULL, -- Format: YYYY-MM-DD
    reservation_time TEXT NOT NULL, -- Format: HH:MM (24-hour)
    status TEXT NOT NULL DEFAULT 'confirmed' CHECK (status IN ('confirmed', 'cancelled')),
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);

-- Indexes for performance & query optimization
CREATE INDEX IF NOT EXISTS idx_menu_items_name ON menu_items(name);
CREATE INDEX IF NOT EXISTS idx_menu_items_category ON menu_items(category_id);
CREATE INDEX IF NOT EXISTS idx_menu_items_allergen_status ON menu_items(allergen_status);
CREATE INDEX IF NOT EXISTS idx_ingredients_item ON menu_item_ingredients(menu_item_id);
CREATE INDEX IF NOT EXISTS idx_allergens_name ON allergens(name);
CREATE INDEX IF NOT EXISTS idx_reservations_date_table ON reservations(reservation_date, table_id, status);
