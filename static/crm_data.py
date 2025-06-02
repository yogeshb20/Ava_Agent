# crm_data.py

# Sample Data: CRM, Customer Feedback, and Sentiment Analysis

# Customer and Project Information
customers = {
    "Mark": {
        "customer_id": 1,
        "customer_name": "Mark Johnson",
        "client_name": "E-Commerce Platform Co.",
        "project_assigned_to": ["Ajay", "Rajesh"],
        "project_name": "E-Commerce Checkout Flow",
        "project_details": "Development of a new e-commerce platform's checkout flow for higher conversion rates."
    },
    "Sarah": {
        "customer_id": 2,
        "customer_name": "Sarah Patel",
        "client_name": "HealthTech Solutions",
        "project_assigned_to": ["Sandeep", "Priya"],
        "project_name": "Healthcare Data Integration",
        "project_details": "Integrating healthcare data into a unified system with enhanced security and performance."
    },
    "James": {
        "customer_id": 3,
        "customer_name": "James Wilson",
        "client_name": "Retail Corp.",
        "project_assigned_to": ["Tom", "Linda"],
        "project_name": "Retail Inventory System",
        "project_details": "Building a real-time inventory management system for a retail business."
    },
    "Linda": {
        "customer_id": 4,
        "customer_name": "Linda Garcia",
        "client_name": "Financial Services Inc.",
        "project_assigned_to": ["Ajay", "Sandeep"],
        "project_name": "Financial Dashboard",
        "project_details": "Developing a financial analysis and reporting dashboard for business executives."
    },
    "Michael": {
        "customer_id": 5,
        "customer_name": "Michael Brown",
        "client_name": "Logistics Co.",
        "project_assigned_to": ["Rajesh", "Tom"],
        "project_name": "Logistics Tracking System",
        "project_details": "Building an advanced logistics tracking system to improve supply chain visibility."
    }
}

# Clarion Employees (Developers, QA, Managers)
employees = {
    "Ajay": {
        "role": "Developer",
        "skills": ["Backend Development", "API Integration", "Database Management"],
        "projects": ["E-Commerce Checkout Flow", "Financial Dashboard"]
    },
    "Rajesh": {
        "role": "Developer",
        "skills": ["Frontend Development", "UI/UX Design", "Performance Optimization"],
        "projects": ["E-Commerce Checkout Flow", "Logistics Tracking System"]
    },
    "Sandeep": {
        "role": "QA",
        "skills": ["Test Automation", "Manual Testing", "Bug Reporting"],
        "projects": ["Healthcare Data Integration", "Financial Dashboard"]
    },
    "Tom": {
        "role": "Manager",
        "skills": ["Project Management", "Client Communication", "Team Coordination"],
        "projects": ["Retail Inventory System", "Logistics Tracking System"]
    },
    "Linda": {
        "role": "Manager",
        "skills": ["Project Planning", "Stakeholder Communication", "Risk Management"],
        "projects": ["Retail Inventory System", "Financial Dashboard"]
    }
}





client_feedback = {}

# Sample Feedback and Sentiment Analysis
# feedback = {
#     "Mark": {
#         "feedback": [
#             {
#                 "developer": "Ajay",
#                 "feedback_text": "Ajay is very responsive and thorough in his approach, always quick to resolve any concerns.",
#                 "sentiment": "Very Positive",
#                 "aspect": "Developer Performance"
#             },
#             {
#                 "developer": "Rajesh",
#                 "feedback_text": "Rajesh is great at frontend work, but sometimes there could be more proactive communication regarding timelines.",
#                 "sentiment": "Neutral/Constructive",
#                 "aspect": "Communication & Timelines"
#             }
#         ],
#         "overall_sentiment": "Positive",
#         "notes": "Mark is happy with the platform's progress, especially the checkout flow. He suggested better communication around project timelines."
#     },
#     "Sarah": {
#         "feedback": [
#             {
#                 "developer": "Sandeep",
#                 "feedback_text": "Sandeep's testing was thorough, but there were occasional delays in the feedback loop.",
#                 "sentiment": "Neutral",
#                 "aspect": "Testing & Quality Assurance"
#             },
#             {
#                 "developer": "Priya",
#                 "feedback_text": "Priya did a great job with API integrations, ensuring seamless data transfer between systems.",
#                 "sentiment": "Positive",
#                 "aspect": "Developer Performance"
#             }
#         ],
#         "overall_sentiment": "Neutral",
#         "notes": "Sarah noted that the project is progressing well but mentioned some delays in testing feedback."
#     }
# }
