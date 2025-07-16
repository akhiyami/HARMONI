"""
Python script to totally remove a user from the SQLite database, including their associated tables.
"""

import sqlite3
import argparse
import loguru

import sys
import os

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

from memory.utils import remove_user

parser = argparse.ArgumentParser(description="Remove a user from the database.")
parser.add_argument("-u", "--user_id", type=str, required=True, help="The ID of the user to be removed.")

args = parser.parse_args()

def main(user_id):
    """
    Main function to remove a specific user from the database.
    It deletes the user's table and all associated data.
    """
    # Connect to the database
    database = 'memory/users.db'
    conn = sqlite3.connect(database)

    # Remove the user from the database
    remove_user(user_id, conn)

    logger = loguru.logger
    logger.success(f"User {user_id} removed successfully.")

    # Close the database connection
    conn.close()

if __name__ == "__main__":
    main(args.user_id)