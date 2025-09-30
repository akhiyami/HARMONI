"""
Utility functions for managing the SQLite database and creating necessary tables.
==========
Functions:
- create_table: Create a table in the SQLite database if it does not exist.
- empty_database: Erase all data from the whole database.
- empty_table: Empty a specific table in the SQLite database.
- remove_user: Remove a user from the database.
"""

#--------------------------------------- Functions ---------------------------------------#

def create_table(conn):
    """"
    Create a table in the SQLite database if it does not exist.
    This table is used to store user embeddings.
    """
    cursor = conn.cursor()
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS user_embeddings (
            user_id INT PRIMARY KEY,
            embeddings BLOB
        )
    ''')  
    conn.commit()

def empty_database(conn):
    """
    Erase all data from the whole database.
    This function deletes all tables in the database.
    """
    cursor = conn.cursor()

    # Disable foreign key constraints (important for dropping tables in the right order)
    cursor.execute("PRAGMA foreign_keys = OFF;")

    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for (table_name,) in tables:
        cursor.execute(f'DROP TABLE IF EXISTS "{table_name}";')

    # Get all views
    cursor.execute("SELECT name FROM sqlite_master WHERE type='view';")
    views = cursor.fetchall()
    for (view_name,) in views:
        cursor.execute(f'DROP VIEW IF EXISTS "{view_name}";')

    # Get all triggers
    cursor.execute("SELECT name FROM sqlite_master WHERE type='trigger';")
    triggers = cursor.fetchall()
    for (trigger_name,) in triggers:
        cursor.execute(f'DROP TRIGGER IF EXISTS "{trigger_name}";')

    # Get all indexes (excluding auto indexes)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_autoindex%';")
    indexes = cursor.fetchall()
    for (index_name,) in indexes:
        cursor.execute(f'DROP INDEX IF EXISTS "{index_name}";')

    # Vacuum to clean up file size
    conn.commit()
    cursor.execute("VACUUM;")

    # Re-enable foreign keys and close connection
    cursor.execute("PRAGMA foreign_keys = ON;")
    conn.commit()
    conn.close()

def empty_table(table_name, conn):
    """
    Empty a specific table in the SQLite database.
    This function deletes all rows from the specified table.
    """
    cursor = conn.cursor()
    cursor.execute(f'DELETE FROM {table_name}')
    conn.commit()

def remove_user(user_id, conn):
    """
    Remove a user from the database.
    This function deletes the user and their associated data.
    """
    cursor = conn.cursor()
    cursor.execute(f'DROP TABLE IF EXISTS "{user_id}"')
    conn.commit()

    # Also remove the user from the user_embeddings table
    cursor.execute('DELETE FROM user_embeddings WHERE user_id = ?', (user_id,))
    conn.commit()