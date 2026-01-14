"""
Task 1: API Data Retrieval and Storage
=======================================
This script fetches book data from an external REST API, stores it in a local
SQLite database, and displays the retrieved data.

Assumptions:
1. Using Open Library API (https://openlibrary.org) as the external REST API
2. SQLite database will be created locally if it doesn't exist
3. Books table will store: id, title, author, publication_year

Author: Divyesh Mutha
Date: January 2026
"""

import requests
import sqlite3
import json
from datetime import datetime


# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def create_database_connection(db_name="books.db"):
    """
    Create a connection to the SQLite database.

    Args:
        db_name (str): Name of the database file

    Returns:
        sqlite3.Connection: Database connection object
    """
    try:
        connection = sqlite3.connect(db_name)
        print(f"Successfully connected to database: {db_name}")
        return connection
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        raise


def create_books_table(connection):
    """
    Create the books table if it doesn't exist.

    Args:
        connection: SQLite database connection
    """
    create_table_query = """
    CREATE TABLE IF NOT EXISTS books (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        author TEXT,
        publication_year INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    try:
        cursor = connection.cursor()
        cursor.execute(create_table_query)
        connection.commit()
        print("Books table created successfully (or already exists)")
    except sqlite3.Error as e:
        print(f"Error creating table: {e}")
        raise


def insert_book(connection, title, author, publication_year):
    """
    Insert a single book record into the database.

    Args:
        connection: SQLite database connection
        title (str): Book title
        author (str): Author name
        publication_year (int): Year of publication

    Returns:
        int: The row ID of the inserted record
    """
    insert_query = """
    INSERT INTO books (title, author, publication_year)
    VALUES (?, ?, ?);
    """
    try:
        cursor = connection.cursor()
        cursor.execute(insert_query, (title, author, publication_year))
        connection.commit()
        return cursor.lastrowid
    except sqlite3.Error as e:
        print(f"Error inserting book: {e}")
        raise


def insert_multiple_books(connection, books_list):
    """
    Insert multiple book records into the database.

    Args:
        connection: SQLite database connection
        books_list (list): List of tuples containing (title, author, publication_year)

    Returns:
        int: Number of records inserted
    """
    insert_query = """
    INSERT INTO books (title, author, publication_year)
    VALUES (?, ?, ?);
    """
    try:
        cursor = connection.cursor()
        cursor.executemany(insert_query, books_list)
        connection.commit()
        return cursor.rowcount
    except sqlite3.Error as e:
        print(f"Error inserting books: {e}")
        raise


def fetch_all_books(connection):
    """
    Retrieve all books from the database.

    Args:
        connection: SQLite database connection

    Returns:
        list: List of all book records
    """
    select_query = "SELECT id, title, author, publication_year, created_at FROM books;"
    try:
        cursor = connection.cursor()
        cursor.execute(select_query)
        return cursor.fetchall()
    except sqlite3.Error as e:
        print(f"Error fetching books: {e}")
        raise


def clear_books_table(connection):
    """
    Clear all records from the books table.

    Args:
        connection: SQLite database connection
    """
    try:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM books;")
        connection.commit()
        print("Books table cleared")
    except sqlite3.Error as e:
        print(f"Error clearing table: {e}")
        raise


# =============================================================================
# API FUNCTIONS
# =============================================================================

def fetch_books_from_api(search_query="python programming", limit=10):
    """
    Fetch book data from Open Library API.

    Args:
        search_query (str): Search term for books
        limit (int): Maximum number of books to fetch

    Returns:
        list: List of dictionaries containing book information
    """
    base_url = "https://openlibrary.org/search.json"
    params = {
        "q": search_query,
        "limit": limit
    }

    try:
        print(f"Fetching books from API with query: '{search_query}'...")
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()  # Raise exception for bad status codes

        data = response.json()
        books = []

        for doc in data.get("docs", []):
            # Extract relevant information
            title = doc.get("title", "Unknown Title")

            # Get first author if available
            authors = doc.get("author_name", ["Unknown Author"])
            author = authors[0] if authors else "Unknown Author"

            # Get publication year
            publication_year = doc.get("first_publish_year", None)

            books.append({
                "title": title,
                "author": author,
                "publication_year": publication_year
            })

        print(f"Successfully fetched {len(books)} books from API")
        return books

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        # Return sample data as fallback
        return get_sample_books()


def get_sample_books():
    """
    Return sample book data as fallback when API is unavailable.

    Returns:
        list: List of sample book dictionaries
    """
    print("Using sample book data as fallback...")
    return [
        {"title": "Clean Code", "author": "Robert C. Martin", "publication_year": 2008},
        {"title": "The Pragmatic Programmer", "author": "David Thomas", "publication_year": 1999},
        {"title": "Design Patterns", "author": "Erich Gamma", "publication_year": 1994},
        {"title": "Introduction to Algorithms", "author": "Thomas H. Cormen", "publication_year": 1990},
        {"title": "Python Crash Course", "author": "Eric Matthes", "publication_year": 2015},
        {"title": "Fluent Python", "author": "Luciano Ramalho", "publication_year": 2015},
        {"title": "Learning Python", "author": "Mark Lutz", "publication_year": 2013},
        {"title": "Automate the Boring Stuff", "author": "Al Sweigart", "publication_year": 2015},
    ]


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def display_books(books):
    """
    Display books in a formatted table.

    Args:
        books (list): List of book tuples from database
    """
    if not books:
        print("\nNo books found in the database.")
        return

    print("\n" + "=" * 100)
    print(f"{'ID':<5} {'Title':<40} {'Author':<25} {'Year':<6} {'Added On':<20}")
    print("=" * 100)

    for book in books:
        book_id, title, author, year, created_at = book
        # Truncate long titles and authors for display
        title_display = (title[:37] + "...") if len(title) > 40 else title
        author_display = (author[:22] + "...") if len(author) > 25 else author
        year_display = str(year) if year else "N/A"

        print(f"{book_id:<5} {title_display:<40} {author_display:<25} {year_display:<6} {created_at:<20}")

    print("=" * 100)
    print(f"Total books: {len(books)}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to orchestrate the API fetch, storage, and display operations.
    """
    print("\n" + "=" * 60)
    print("  TASK 1: API Data Retrieval and Storage")
    print("  Book Management System")
    print("=" * 60 + "\n")

    # Step 1: Create database connection and table
    print("STEP 1: Setting up database...")
    connection = create_database_connection("books.db")
    create_books_table(connection)

    # Step 2: Clear existing data (for fresh demonstration)
    print("\nSTEP 2: Clearing existing data for fresh demonstration...")
    clear_books_table(connection)

    # Step 3: Fetch books from API
    print("\nSTEP 3: Fetching books from external API...")
    books_data = fetch_books_from_api(search_query="python", limit=10)

    # Step 4: Store books in database
    print("\nSTEP 4: Storing books in SQLite database...")
    books_to_insert = [
        (book["title"], book["author"], book["publication_year"])
        for book in books_data
    ]
    inserted_count = insert_multiple_books(connection, books_to_insert)
    print(f"Successfully inserted {inserted_count} books into database")

    # Step 5: Retrieve and display books
    print("\nSTEP 5: Retrieving and displaying books from database...")
    stored_books = fetch_all_books(connection)
    display_books(stored_books)

    # Cleanup
    connection.close()
    print("\nDatabase connection closed. Program completed successfully!")


if __name__ == "__main__":
    main()
