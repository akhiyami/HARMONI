"""
Python script to clear the memory of one user by clearing the associated table in the SQLite database.
"""

#--------------------------------------- Imports ---------------------------------------#

import sqlite3
import argparse
import loguru

import sys
import os

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

from memory.utils import empty_table

#--------------------------------------- Argument Parsing ---------------------------------------#

parser = argparse.ArgumentParser(description="Clear user memory from the database.")
parser.add_argument("-d", "--database", type=str, required=True, help="The database file to clear.")
parser.add_argument("-u", "--user_id", type=str, required=True, help="The ID of the user whose memory should be cleared.")

args = parser.parse_args()

#--------------------------------------- Main Function ---------------------------------------#

def main(user_id):
    """
    Main function to clear the memory of a specific user.
    It empties the user's table and removes the user from the database.
    """
    # Connect to the database
    database = args.database
    conn = sqlite3.connect(database)

    # Empty the user's table
    empty_table(user_id, conn)

    logger = loguru.logger
    logger.success(f"Memory for user {user_id} cleared successfully.")

    # Close the database connection
    conn.close()

#--------------------------------------- Entry Point ---------------------------------------#

if __name__ == "__main__":
    main(args.user_id)