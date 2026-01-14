"""
Task 3: CSV Data Import to a Database
======================================
This script reads user information from a CSV file and inserts the data
into a SQLite database.

Assumptions:
1. CSV file contains columns: name, email (and optionally: phone, age, city)
2. SQLite database will be created locally if it doesn't exist
3. Duplicate entries will be handled based on email uniqueness
4. The script will validate data before insertion

Author: Divyesh Mutha
Date: January 2026
"""

import csv
import sqlite3
import re
import os
from typing import List, Dict, Tuple, Optional
from datetime import datetime


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_email(email: str) -> bool:
    """
    Validate email format using regex.

    Args:
        email (str): Email address to validate

    Returns:
        bool: True if valid, False otherwise
    """
    if not email:
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))


def validate_name(name: str) -> bool:
    """
    Validate that name is not empty and contains valid characters.

    Args:
        name (str): Name to validate

    Returns:
        bool: True if valid, False otherwise
    """
    if not name or not name.strip():
        return False
    # Allow letters, spaces, hyphens, and apostrophes
    pattern = r"^[a-zA-Z\s\-']+$"
    return bool(re.match(pattern, name.strip()))


def sanitize_string(value: str) -> str:
    """
    Sanitize string input by stripping whitespace and handling None.

    Args:
        value: Input value

    Returns:
        str: Sanitized string
    """
    if value is None:
        return ""
    return str(value).strip()


# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def create_database_connection(db_name: str = "users.db") -> sqlite3.Connection:
    """
    Create a connection to the SQLite database.

    Args:
        db_name (str): Name of the database file

    Returns:
        sqlite3.Connection: Database connection object
    """
    try:
        connection = sqlite3.connect(db_name)
        connection.row_factory = sqlite3.Row  # Enable column access by name
        print(f"Connected to database: {db_name}")
        return connection
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        raise


def create_users_table(connection: sqlite3.Connection) -> None:
    """
    Create the users table with appropriate schema.

    Args:
        connection: SQLite database connection
    """
    create_table_query = """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        phone TEXT,
        age INTEGER,
        city TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    # Create index on email for faster lookups
    create_index_query = """
    CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
    """

    try:
        cursor = connection.cursor()
        cursor.execute(create_table_query)
        cursor.execute(create_index_query)
        connection.commit()
        print("Users table created successfully (or already exists)")
    except sqlite3.Error as e:
        print(f"Error creating table: {e}")
        raise


def insert_user(connection: sqlite3.Connection, user_data: Dict) -> Optional[int]:
    """
    Insert a single user record into the database.

    Args:
        connection: SQLite database connection
        user_data (dict): Dictionary containing user information

    Returns:
        int: The row ID of the inserted record, or None if failed
    """
    insert_query = """
    INSERT INTO users (name, email, phone, age, city)
    VALUES (?, ?, ?, ?, ?);
    """

    try:
        cursor = connection.cursor()
        cursor.execute(insert_query, (
            user_data.get("name"),
            user_data.get("email"),
            user_data.get("phone"),
            user_data.get("age"),
            user_data.get("city")
        ))
        connection.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError as e:
        if "UNIQUE constraint failed" in str(e):
            print(f"  Warning: Duplicate email skipped: {user_data.get('email')}")
            return None
        raise
    except sqlite3.Error as e:
        print(f"Error inserting user: {e}")
        raise


def insert_users_batch(connection: sqlite3.Connection, users: List[Dict]) -> Tuple[int, int]:
    """
    Insert multiple user records, handling duplicates gracefully.

    Args:
        connection: SQLite database connection
        users (list): List of user dictionaries

    Returns:
        tuple: (successful_count, skipped_count)
    """
    successful = 0
    skipped = 0

    for user in users:
        result = insert_user(connection, user)
        if result:
            successful += 1
        else:
            skipped += 1

    return successful, skipped


def fetch_all_users(connection: sqlite3.Connection) -> List[sqlite3.Row]:
    """
    Retrieve all users from the database.

    Args:
        connection: SQLite database connection

    Returns:
        list: List of user records
    """
    select_query = """
    SELECT id, name, email, phone, age, city, created_at
    FROM users
    ORDER BY id;
    """

    try:
        cursor = connection.cursor()
        cursor.execute(select_query)
        return cursor.fetchall()
    except sqlite3.Error as e:
        print(f"Error fetching users: {e}")
        raise


def get_user_count(connection: sqlite3.Connection) -> int:
    """
    Get the total count of users in the database.

    Args:
        connection: SQLite database connection

    Returns:
        int: Number of users
    """
    cursor = connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM users;")
    return cursor.fetchone()[0]


def clear_users_table(connection: sqlite3.Connection) -> None:
    """
    Clear all records from the users table.

    Args:
        connection: SQLite database connection
    """
    try:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM users;")
        connection.commit()
        print("Users table cleared")
    except sqlite3.Error as e:
        print(f"Error clearing table: {e}")
        raise


# =============================================================================
# CSV PROCESSING FUNCTIONS
# =============================================================================

def read_csv_file(file_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Read and parse a CSV file containing user information.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        tuple: (valid_records, invalid_records)
    """
    valid_records = []
    invalid_records = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            # Detect delimiter
            sample = csvfile.read(1024)
            csvfile.seek(0)

            try:
                dialect = csv.Sniffer().sniff(sample)
                reader = csv.DictReader(csvfile, dialect=dialect)
            except csv.Error:
                # Default to comma if detection fails
                reader = csv.DictReader(csvfile)

            # Normalize column names
            if reader.fieldnames:
                reader.fieldnames = [name.lower().strip() for name in reader.fieldnames]

            row_number = 1
            for row in reader:
                row_number += 1

                # Extract and sanitize data
                name = sanitize_string(row.get('name', ''))
                email = sanitize_string(row.get('email', ''))
                phone = sanitize_string(row.get('phone', ''))
                city = sanitize_string(row.get('city', ''))

                # Handle age (might be empty or invalid)
                age_str = sanitize_string(row.get('age', ''))
                try:
                    age = int(age_str) if age_str else None
                except ValueError:
                    age = None

                # Validate required fields
                errors = []
                if not validate_name(name):
                    errors.append("Invalid name")
                if not validate_email(email):
                    errors.append("Invalid email")

                user_data = {
                    "name": name,
                    "email": email.lower(),  # Normalize email to lowercase
                    "phone": phone if phone else None,
                    "age": age,
                    "city": city if city else None,
                    "row_number": row_number
                }

                if errors:
                    user_data["errors"] = errors
                    invalid_records.append(user_data)
                else:
                    valid_records.append(user_data)

        return valid_records, invalid_records

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        raise


def generate_import_report(valid: int, invalid: int, skipped: int) -> str:
    """
    Generate a summary report of the import operation.

    Args:
        valid (int): Number of valid records processed
        invalid (int): Number of invalid records
        skipped (int): Number of skipped duplicates

    Returns:
        str: Formatted report string
    """
    total = valid + invalid + skipped
    report = f"""
+----------------------------------------------------------+
|              CSV IMPORT SUMMARY REPORT                   |
+----------------------------------------------------------+
|  Total Records Processed  : {total:<25} |
|  Successfully Imported    : {valid:<25} |
|  Invalid Records          : {invalid:<25} |
|  Duplicates Skipped       : {skipped:<25} |
|  Import Time             : {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<25} |
+----------------------------------------------------------+
"""
    return report


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def display_users(users: List[sqlite3.Row]) -> None:
    """
    Display users in a formatted table.

    Args:
        users (list): List of user records
    """
    if not users:
        print("\nNo users found in the database.")
        return

    print("\n" + "=" * 100)
    print(f"{'ID':<5} {'Name':<25} {'Email':<30} {'Phone':<15} {'Age':<5} {'City':<15}")
    print("=" * 100)

    for user in users:
        phone = user['phone'] if user['phone'] else 'N/A'
        age = str(user['age']) if user['age'] else 'N/A'
        city = user['city'] if user['city'] else 'N/A'

        # Truncate long values
        name_display = (user['name'][:22] + "...") if len(user['name']) > 25 else user['name']
        email_display = (user['email'][:27] + "...") if len(user['email']) > 30 else user['email']

        print(f"{user['id']:<5} {name_display:<25} {email_display:<30} {phone:<15} {age:<5} {city:<15}")

    print("=" * 100)
    print(f"Total users: {len(users)}")


def display_invalid_records(invalid_records: List[Dict]) -> None:
    """
    Display records that failed validation.

    Args:
        invalid_records (list): List of invalid record dictionaries
    """
    if not invalid_records:
        return

    print("\n" + "-" * 60)
    print("INVALID RECORDS (Not Imported)")
    print("-" * 60)

    for record in invalid_records:
        print(f"  Row {record['row_number']}: {record['name'] or '(empty)'} - "
              f"{record['email'] or '(empty)'}")
        print(f"    Errors: {', '.join(record['errors'])}")

    print("-" * 60)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def import_csv_to_database(csv_path: str, db_name: str = "users.db",
                           clear_existing: bool = False) -> None:
    """
    Main function to import CSV data into SQLite database.

    Args:
        csv_path (str): Path to the CSV file
        db_name (str): Name of the database file
        clear_existing (bool): Whether to clear existing data before import
    """
    print("\n" + "=" * 60)
    print("  TASK 3: CSV Data Import to Database")
    print("  User Data Import System")
    print("=" * 60 + "\n")

    # Step 1: Setup database
    print("STEP 1: Setting up database connection...")
    connection = create_database_connection(db_name)
    create_users_table(connection)

    if clear_existing:
        print("\nClearing existing data...")
        clear_users_table(connection)

    # Step 2: Read and validate CSV
    print(f"\nSTEP 2: Reading CSV file: {csv_path}")
    valid_records, invalid_records = read_csv_file(csv_path)
    print(f"  Found {len(valid_records)} valid records")
    print(f"  Found {len(invalid_records)} invalid records")

    # Step 3: Import valid records
    print("\nSTEP 3: Importing valid records to database...")
    successful, skipped = insert_users_batch(connection, valid_records)

    # Step 4: Display results
    print("\nSTEP 4: Import completed!")
    print(generate_import_report(successful, len(invalid_records), skipped))

    # Display invalid records if any
    display_invalid_records(invalid_records)

    # Step 5: Show database contents
    print("\nSTEP 5: Current database contents:")
    users = fetch_all_users(connection)
    display_users(users)

    # Cleanup
    connection.close()
    print("\nDatabase connection closed.")


def main():
    """
    Main entry point - demonstrates CSV import functionality.
    """
    # Default CSV file path
    csv_file = "sample_users.csv"

    # Check if sample CSV exists, if not create it
    if not os.path.exists(csv_file):
        print(f"Sample CSV file not found. Please create '{csv_file}' first.")
        print("Or run: python create_sample_csv.py")
        print("\nExpected CSV format:")
        print("name,email,phone,age,city")
        print("John Doe,john@example.com,555-1234,30,New York")
        return

    # Run the import
    import_csv_to_database(
        csv_path=csv_file,
        db_name="users.db",
        clear_existing=True  # Set to False to append data
    )


if __name__ == "__main__":
    main()
