from fastapi import FastAPI, HTTPException,Depends, Request
from secret import secret_dict,LocalDB_dict,secret_url,UATDB_dict

from datetime import datetime,date,timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import quote_plus
import json
from langchain_core.tools import tool
import difflib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# UAT Server Database connection details
name = LocalDB_dict["name"] 
username = LocalDB_dict["username"]   # Replace with your actual username
password =  LocalDB_dict["password"] # UATDB_dict["password"] Replace with your actual password
host = LocalDB_dict["host"]  # Corrected host format
port = LocalDB_dict["port"]  # Default MySQL port


# Database connection
def db_conn(db_name: str, username: str, password: str, host: str, port: str = "3306"):
    try:
        # Properly encode the password
        encoded_password = quote_plus(password)
        connection_string = f"mysql+pymysql://{username}:{encoded_password}@{host}:{port}/{db_name}"
        print(f"Connecting to database at {host}:{port}")
        engine = create_engine(connection_string)
        connection = engine.connect()
        print("Database connection successful")
        return connection
    except SQLAlchemyError as err:
        print(f"Database connection error: {err}")
        raise HTTPException(status_code=500, detail="Could not connect to the database")

def get_db_connection(db_name: str, username: str, password: str, host: str, port: str = "3306"):
    conn = db_conn(db_name, username, password, host, port)
    return conn

# conn = get_db_connection(name, username, password, host, port)
# conn = get_db_connection(name, username,password,host,port)
def create_tables():
    try:
        conn = get_db_connection(name, username, password, host, port)

        create_feedback_table_query = """
            CREATE TABLE IF NOT EXISTS feedback_store (
                id INT AUTO_INCREMENT PRIMARY KEY,
                customer_name VARCHAR(255) NOT NULL,
                overall_sentiment VARCHAR(100),
                notes TEXT,
                feedback JSON,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """

        conn.execute(text(create_feedback_table_query))
        conn.commit()
        conn.close()
        logger.info("feedback_store table created successfully.")
    except SQLAlchemyError as err:
        logger.error(f"Error creating feedback_store table: {err}")

# #Execute the function to create the above table
# create_tables()


@tool
def get_customer_info_db(client_id: int = None, client_name: str = None):
    """Fetches customer profile based on the Client_id AND Client_Name."""
    
    print(f"Looking up customer: {client_name and client_id}")  # Debug print
    try:
        conn = get_db_connection(name, username, password, host, port)
        if not conn:
            raise Exception("Database connection is not established.")

        conditions = []
        params = {}

        if client_id:
            conditions.append("CLIENT_ID = :client_id")
            params["client_id"] = client_id
        
        if client_name:
            conditions.append("CLIENT_NAME LIKE :name_like")
            params["name_like"] = f"%{client_name}%"


        if not conditions:
            raise ValueError("At least one of client_id or client_name must be provided.")

        query_str = f"SELECT * FROM tblClient WHERE {' AND '.join(conditions)}"
        query = text(query_str)
        result = conn.execute(query, params)

        records = result.mappings().all()
        print("Record of Customer info:\n", records)
        return records

    except Exception as e:
        print(f"Error fetching customer info: {e}")
        return []



@tool
def store_feedback_data_db(customer_name: str, developer: str, feedback_text: str, sentiment: str, aspect: str, notes: str = "") -> dict:
    """Stores feedback data for a customer in the feedback_store table."""
    try:
        
        conn = get_db_connection(name, username, password, host, port)
        # Check if entry for this customer already exists
        select_query = text("SELECT id, feedback, notes FROM feedback_store WHERE customer_name = :customer_name")
        result = result = conn.execute(select_query, {"customer_name": customer_name}).mappings().fetchone()


        new_feedback_item = {
            "developer": developer,
            "feedback_text": feedback_text,
            "sentiment": sentiment,
            "aspect": aspect
        }

        if result:
            # Customer entry exists – update it
            feedback_list = json.loads(result["feedback"]) if result["feedback"] else []
            feedback_list.append(new_feedback_item)

            combined_notes = result["notes"] + f"\n{notes}" if notes else result["notes"]

            update_query = text("""
                UPDATE feedback_store
                SET feedback = :feedback, notes = :notes, timestamp = :timestamp
                WHERE id = :id
            """)
            conn.execute(update_query, {
                "feedback": json.dumps(feedback_list),
                "notes": combined_notes,
                "timestamp": datetime.now(),
                "id": result["id"]
            })
        else:
            # New customer entry – insert it
            feedback_list = [new_feedback_item]

            insert_query = text("""
                INSERT INTO feedback_store (customer_name, overall_sentiment, notes, feedback, timestamp)
                VALUES (:customer_name, :overall_sentiment, :notes, :feedback, :timestamp)
            """)
            conn.execute(insert_query, {
                "customer_name": customer_name,
                "overall_sentiment": "Pending",
                "notes": notes,
                "feedback": json.dumps(feedback_list),
                "timestamp": datetime.now()
            })

        conn.commit()
        conn.close()
        return {"status": "success", "message": f"Feedback stored for {customer_name}."}

    except SQLAlchemyError as err:
        logger.error(f"Error storing feedback: {err}")
        return {"status": "error", "message": "Failed to store feedback due to a database error."}
    


@tool
def get_feedback_by_customer_and_developer(customer_name: str, developer_names: str) -> dict:
    """Retrieves feedback for a specific developer under a given customer with fuzzy matching."""
    try:
        # Connect to the database
        conn = get_db_connection(name, username, password, host, port)
        print("Database connection successful")

        # Prepare SQL query with case-insensitive, partial match
        select_query = text("""
            SELECT feedback, notes 
            FROM feedback_store 
            WHERE LOWER(customer_name) LIKE :customer_name
            ORDER BY timestamp DESC 
            LIMIT 1;
        """)

        # Apply lower and wildcard for partial matching
        customer_param = f"%{customer_name.lower()}%"
        print("Executing query with customer_name like:", customer_param)

        # Execute query
        result = conn.execute(select_query, {"customer_name": customer_param}).mappings().fetchone()
        # print("Result obtained from feedback:\n", result)

        if not result:
            return {"status": "error", "message": f"No feedback found for customer '{customer_name}'."}

        # Load feedback JSON and normalize developer names
        feedback_list = json.loads(result["feedback"]) if result["feedback"] else []
        requested_devs = [dev.strip().lower() for dev in developer_names.split(",")]

        # Fuzzy developer match
        def is_matching_dev(target_dev, requested_devs):
            target_dev = target_dev.strip().lower()
            return any(difflib.get_close_matches(target_dev, requested_devs, cutoff=0.6))

        matching_feedback = []
        for item in feedback_list:
            devs_in_item = [dev.strip().lower() for dev in item.get("developer", "").split(",")]
            if any(is_matching_dev(dev, requested_devs) for dev in devs_in_item):
                matching_feedback.append(item)

        if not matching_feedback:
            return {
                "status": "error",
                "message": f"No feedback found for developer(s) '{developer_names}' under '{customer_name}'."
            }

        # Return successful result
        feedback_data = {
            "status": "success",
            "customer_name": customer_name,
            "developer_name": developer_names,
            "feedback_entries": matching_feedback,
            "notes": result["notes"]
        }

        print("Feedback data retrieved:\n", feedback_data)
        return feedback_data

    except SQLAlchemyError as err:
        logger.error(f"Error retrieving feedback: {err}")
        return {"status": "error", "message": "Database error occurred during retrieval."}




# def customer_information(client_id: int = None, client_name: str = None):
#     """Fetches customer profile based on the Client_id or Client_Name."""
    
#     # print(f"Looking up customer: {client_name and client_id}")  # Debug print
#     try:
#         conn = get_db_connection(name, username, password, host, port)
#         if not conn:
#             raise Exception("Database connection is not established.")

#         conditions = []
#         params = {}

#         if client_id:
#             conditions.append("CLIENT_ID = :client_id")
#             params["client_id"] = client_id
        
#         if client_name:
#             conditions.append("CLIENT_NAME LIKE :name_like")
#             params["name_like"] = f"%{client_name}%"

#         if not conditions:
#             raise ValueError("At least one of client_id or client_name must be provided.")

#         query_str = f"SELECT * FROM tblClient WHERE {' AND '.join(conditions)}"
#         query = text(query_str)
#         result = conn.execute(query, params)

#         records = result.mappings().all()
#         # print("Record of Customer info:\n", records)
#         return records

#     except Exception as e:
#         print(f"Error fetching customer info: {e}")
#         return []




# customer_information(client_id = 1, client_name = 'mark')




# def customer_information(client_id: int = None, client_name: str = None):
#     """Fetches customer profile based on the Client_id or Client_Name."""
    
#     # print(f"Looking up customer: {client_name and client_id}")  # Debug print
#     try:
#         conn = get_db_connection(name, username, password, host, port)
#         if not conn:
#             raise Exception("Database connection is not established.")

#         conditions = []
#         params = {}

#         if client_id:
#             conditions.append("CLIENT_ID = :client_id")
#             params["client_id"] = client_id
        
#         if client_name:
#             conditions.append("CLIENT_NAME LIKE :name_like")
#             params["name_like"] = f"%{client_name}%"

#         if not conditions:
#             raise ValueError("At least one of client_id or client_name must be provided.")

#         query_str = f"SELECT * FROM tblclient_user WHERE {' AND '.join(conditions)}"
#         query = text(query_str)
#         result = conn.execute(query, params)

#         records = result.mappings().all()
#         # print("Record of Customer info:\n", records)
#         return records

#     except Exception as e:
#         print(f"Error fetching customer info: {e}")
#         return []


def customer_information(client_id: str = None, client_name: str = None):
    """Fetches customer profile based on the client_id or client_name from a dictionary."""

    # Sample in-memory customer data
    customers = [
        {
            "CLIENT_ID": "001",
            "CLIENT_NAME": "Mark Johnson",
            "PROJECT_ASSIGNED": ["Ajay", "Rajesh"]
        },
        # You can add more customers here
    ]

    try:
        if not client_id and not client_name:
            raise ValueError("At least one of client_id or client_name must be provided.")

        results = []

        for customer in customers:
            if client_id and customer["CLIENT_ID"] == client_id:
                results.append(customer)
            elif client_name and client_name.lower() in customer["CLIENT_NAME"].lower():
                results.append(customer)

        return results

    except Exception as e:
        print(f"Error fetching customer info: {e}")
        return []
