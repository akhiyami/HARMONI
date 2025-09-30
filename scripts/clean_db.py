"""
Python script to clean the database by removing all entries from the embeddings table, and deleting all user tables.
"""

#--------------------------------------- Imports ---------------------------------------#

import sqlite3
import loguru

import sys
import os
import argparse

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

from conversation.memory.utils import empty_database, create_table

#--------------------------------------- Argument Parsing ---------------------------------------#

parser = argparse.ArgumentParser(description="Clear the database.")
parser.add_argument("-d", "--database", type=str, required=True, help="The database file to clear.")

args = parser.parse_args()

#--------------------------------------- Main Function ---------------------------------------#

def main():
    """
    Main function to clean the database.
    It empties the user_embeddings table and removes all user-specific tables.
    """
    # Connect to the database
    database = args.database
    conn = sqlite3.connect(database)

    # Empty the database
    empty_database(conn)
    # Recreate the embeddings table

    conn = sqlite3.connect(database)  # Reconnect to ensure the table is created in a clean state
    create_table(conn)

    logger = loguru.logger
    logger.success("Database cleaned successfully.")

    # Close the database connection
    conn.close()

#--------------------------------------- Entry Point ---------------------------------------#

if __name__ == "__main__":
    main()