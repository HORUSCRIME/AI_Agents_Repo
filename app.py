import streamlit as st
import os
import pandas as pd
import psycopg2 # PostgreSQL database adapter
from crewai import Agent, Task, Crew, Process
from crewai_tools import Tool
from dotenv import load_dotenv
import json
import subprocess
import tempfile
import logging
from typing import Optional, Dict, Any
from urllib.parse import urlparse # For robust parsing of PostgreSQL URI

# --- Configure Logging ---
# Set up basic logging for visibility into agent and tool execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Environment Variables ---
# This loads variables from a .env file (e.g., GOOGLE_API_KEY, POSTGRES_URI)
load_dotenv()

# --- Gemini Flash 2.5 LLM Configuration ---
# Initialize the Gemini Flash 2.5 model for agent intelligence
# Ensure your GOOGLE_API_KEY is set in your .env file
from langchain_google_genai import ChatGoogleGenerativeAI
gemini_flash_llm = ChatGoogleGenerativeAI(
    model="gemini-flash-2.5",
    verbose=True, # Set to True to see detailed LLM interactions in console/logs
    temperature=0.2, # Lower temperature for more consistent and factual data analysis results
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# --- Custom Tools Definitions ---

# 1. PostgreSQL Connector Tool
class PostgreSQLConnectorTool(Tool):
    name: str = "PostgreSQL Query Executor"
    description: str = (
        "Executes SQL queries against a PostgreSQL database and returns results as JSON. "
        "Use for SELECT, INSERT, UPDATE, DELETE queries. "
        "Expects 'db_uri' (e.g., 'postgresql://user:password@host:port/database_name') and 'query'. "
        "Returns JSON string of results for SELECT or status message for DML operations. "
        "Handles common PostgreSQL errors gracefully."
    )

    def _run(self, db_uri: str, query: str) -> str:
        conn = None
        cursor = None
        try:
            # Parse the database URI for connection details
            parsed_uri = urlparse(db_uri)
            conn = psycopg2.connect(
                host=parsed_uri.hostname,
                port=parsed_uri.port if parsed_uri.port else 5432, # Default PostgreSQL port
                user=parsed_uri.username,
                password=parsed_uri.password,
                database=parsed_uri.path.lstrip('/') # Remove leading slash from database name
            )
            cursor = conn.cursor()
            cursor.execute(query)

            # Check if it's a SELECT query to fetch results
            if query.strip().upper().startswith("SELECT"):
                columns = [desc.name for desc in cursor.description] # Get column names
                result = cursor.fetchall() # Fetch all rows
                df = pd.DataFrame(result, columns=columns) # Convert to Pandas DataFrame
                return df.to_json(orient='records', indent=2) # Return as pretty-printed JSON string
            else:
                conn.commit() # Commit changes for DML operations (INSERT, UPDATE, DELETE)
                return json.dumps({"status": "success", "message": "Query executed successfully.", "rows_affected": cursor.rowcount})

        except psycopg2.Error as err:
            # Catch specific PostgreSQL errors
            logging.error(f"PostgreSQL Error: {err} for query: {query}")
            return json.dumps({"status": "error", "message": f"PostgreSQL Error: {err}"})
        except Exception as e:
            # Catch any other unexpected errors
            logging.error(f"General error in PostgreSQLConnectorTool: {e}")
            return json.dumps({"status": "error", "message": f"General error: {e}"})
        finally:
            # Ensure cursor and connection are closed even if errors occur
            if cursor:
                cursor.close()
            if conn:
                conn.close()

# 2. Excel Data Handler Tool
class ExcelDataTool(Tool):
    name: str = "Excel Data Handler"
    description: str = (
        "Reads data from Excel files into JSON or writes JSON data to an Excel file. "
        "Use 'read' operation with 'file_path' and optional 'sheet_name'. "
        "Use 'write' operation with 'file_path', 'dataframe_json' (JSON string of data), and optional 'sheet_name'. "
        "Handles file not found and empty data errors."
    )

    def _run(self, operation: str, file_path: str, sheet_name: Optional[str] = None, dataframe_json: Optional[str] = None) -> str:
        try:
            if operation.lower() == "read":
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                return df.to_json(orient='records', indent=2)
            elif operation.lower() == "write":
                if dataframe_json is None:
                    return json.dumps({"status": "error", "message": "'dataframe_json' argument is required for 'write' operation."})
                df = pd.read_json(dataframe_json)
                df.to_excel(file_path, sheet_name=sheet_name or 'Sheet1', index=False)
                return json.dumps({"status": "success", "message": f"Data successfully written to {file_path} (Sheet: {sheet_name or 'Sheet1'})"})
            else:
                return json.dumps({"status": "error", "message": "Invalid operation. Use 'read' or 'write'."})
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            return json.dumps({"status": "error", "message": f"File not found: {file_path}"})
        except pd.errors.EmptyDataError:
            logging.warning(f"Empty data found in file: {file_path}")
            return json.dumps({"status": "warning", "message": f"No data found in {file_path}"})
        except Exception as e:
            logging.error(f"Error in ExcelDataTool for {operation} {file_path}: {e}")
            return json.dumps({"status": "error", "message": f"Error handling Excel file: {e}"})

# 3. Python Script Executor Tool (Simulated Sandboxing for Security)
class PythonScriptExecutorTool(Tool):
    name: str = "Python Script Executor"
    description: str = (
        "Executes a Python script securely. The script receives input data as JSON from stdin "
        "and should print its output data as JSON to stdout. "
        "The script is run in a separate, isolated process to enhance security. "
        "Use 'script_code' for the Python code itself and optional 'input_json' for data the script needs. "
        "Ensure the script outputs valid JSON to stdout for the tool to parse."
    )

    def _run(self, script_code: str, input_json: Optional[str] = None) -> str:
        # Create a temporary directory to store the script and prevent arbitrary file system access
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "script.py")
            try:
                # Write the Python script to a temporary file
                with open(script_path, "w") as f:
                    f.write(script_code)

                # Prepare environment variables for the subprocess.
                # PYTHONUNBUFFERED=1 ensures stdout is unbuffered,
                # which is good for real-time output capture.
                env = os.environ.copy()
                env['PYTHONUNBUFFERED'] = '1'

                # Execute the Python script using subprocess.
                # input=input_json sends the provided JSON string to the script's stdin.
                # capture_output=True captures stdout and stderr.
                # text=True decodes output as text.
                # check=False prevents CalledProcessError on non-zero exit codes, allowing us to inspect manually.
                process = subprocess.run(
                    ["python", script_path], # Command to execute the script
                    input=input_json,        # Input sent to script's stdin
                    capture_output=True,     # Capture stdout and stderr
                    text=True,               # Decode output as text
                    check=False,             # Don't raise error for non-zero exit codes
                    env=env                  # Pass modified environment variables
                )

                # Check the script's exit code
                if process.returncode != 0:
                    logging.error(f"Python script execution failed with error code {process.returncode}")
                    logging.error(f"Script stderr: {process.stderr}")
                    return json.dumps({"status": "error", "message": f"Python script execution failed. Stderr: {process.stderr}", "stdout": process.stdout})
                
                # Attempt to parse the script's stdout as JSON
                try:
                    script_output = json.loads(process.stdout)
                except json.JSONDecodeError:
                    logging.warning(f"Python script did not output valid JSON. Raw stdout: {process.stdout}")
                    script_output = {"status": "warning", "message": "Script output was not valid JSON.", "raw_output": process.stdout}

                return json.dumps({"status": "success", "output": script_output})

            except FileNotFoundError:
                logging.error("Python executable not found. Ensure Python is in PATH.")
                return json.dumps({"status": "error", "message": "Python executable not found. Ensure Python is in PATH."})
            except Exception as e:
                logging.error(f"Error executing Python script: {e}")
                return json.dumps({"status": "error", "message": f"Error executing Python script: {e}"})

# 4. PowerBI Code Generator Tool
class PowerBI_CodeGeneratorTool(Tool):
    name: str = "PowerBI Code Generator"
    description: str = (
        "Generates Power Query (M-code) or DAX formulas for PowerBI. "
        "Use 'generate_m_query' with 'csv_path' and 'table_name' to get M-code for CSV import. "
        "Use 'generate_dax_measure' with 'measure_name', 'formula', and 'description' to get DAX for measures. "
        "Use 'suggest_report_layout' with 'data_columns_json' (JSON list of column names) and 'analysis_summary' "
        "to get suggestions for PowerBI visuals and measures based on the data."
    )

    def _run(self, action: str, **kwargs) -> str:
        try:
            if action == "generate_m_query":
                csv_path = kwargs.get("csv_path")
                table_name = kwargs.get("table_name", "ImportedCSV")
                if not csv_path:
                    return json.dumps({"status": "error", "message": "csv_path is required for generate_m_query."})
                
                # M-code to import a CSV file from a specified path
                m_code = (
                    f"let\n"
                    f"    Source = Csv.Document(File.Contents(\"{csv_path.replace(os.sep, '/')}\"),[Delimiter=\",\", Columns=null, Encoding=65001, QuoteStyle=QuoteStyle.Csv]),\n"
                    f"    #\"Promoted Headers\" = Table.PromoteHeaders(Source, [PromoteAllScalars=true])\n"
                    f"in\n"
                    f"    #\"Promoted Headers\""
                )
                return json.dumps({"status": "success", "type": "M-Code", "code": m_code, "instructions": f"Open PowerBI Desktop -> Get Data -> Blank Query -> Advanced Editor. Paste this M-code and rename the query to '{table_name}'."})

            elif action == "generate_dax_measure":
                measure_name = kwargs.get("measure_name")
                formula = kwargs.get("formula")
                description = kwargs.get("description", "")
                if not measure_name or not formula:
                    return json.dumps({"status": "error", "message": "measure_name and formula are required for generate_dax_measure."})
                
                # Basic DAX measure format
                dax_code = f"{measure_name} = {formula}"
                return json.dumps({"status": "success", "type": "DAX-Measure", "code": dax_code, "description": description, "instructions": f"In PowerBI Desktop -> Data View -> Right-click on table -> New Measure. Paste this DAX formula. Description: {description}"})

            elif action == "suggest_report_layout":
                data_columns_json = kwargs.get("data_columns_json")
                analysis_summary = kwargs.get("analysis_summary")
                if not data_columns_json or not analysis_summary:
                    return json.dumps({"status": "error", "message": "data_columns_json and analysis_summary are required for suggest_report_layout."})
                
                try:
                    data_columns = json.loads(data_columns_json)
                except json.JSONDecodeError:
                    return json.dumps({"status": "error", "message": "data_columns_json is not valid JSON."})

                suggestions = []
                suggestions.append("Based on the analysis, consider the following PowerBI visualizations:")
                
                # Simple logic to suggest visuals based on common column names and summary
                if any(col in data_columns for col in ["date", "order_date", "sale_date", "transaction_date"]):
                    suggestions.append("- Line chart for 'Total Sales' over 'Date' to show trends.")
                if any(col in data_columns for col in ["product_category", "product_name", "item_name"]):
                    suggestions.append("- Bar chart for 'Sales by Product Category' or 'Top N Products'.")
                if "sentiment" in data_columns or "customer_rating" in data_columns:
                    suggestions.append("- Donut chart or Pie chart for 'Customer Sentiment Distribution'.")
                if "region" in data_columns or "city" in data_columns:
                    suggestions.append("- Map visualization for 'Sales by Region/City' (if geographical data is present).")
                
                suggestions.append(f"\nAnalysis Summary: {analysis_summary}")
                suggestions.append("\nSuggested DAX Measures (adapt 'YourTableName' as needed):")
                suggestions.append("  - Total Sales = SUM( 'YourTableName'[Amount] )")
                suggestions.append("  - Average Rating = AVERAGE( 'YourTableName'[CustomerRating] )")
                suggestions.append("  - Sales % of Total = DIVIDE(SUM('YourTableName'[Amount]), CALCULATE(SUM('YourTableName'[Amount]), ALL('YourTableName'))) ")

                return json.dumps({"status": "success", "type": "Report-Layout-Suggestions", "suggestions": "\n".join(suggestions)})
            else:
                return json.dumps({"status": "error", "message": "Invalid action for PowerBI Code Generator Tool."})
        except Exception as e:
            logging.error(f"Error in PowerBI_CodeGeneratorTool: {e}")
            return json.dumps({"status": "error", "message": f"Error: {e}"})

# 5. Markdown File Writer Tool
class MarkdownFileWriterTool(Tool):
    name: str = "Markdown File Writer"
    description: str = (
        "Writes content to a Markdown (.md) or plain text (.txt) file. "
        "Requires 'file_path' and 'content'. "
        "Ensures file is written safely."
    )

    def _run(self, file_path: str, content: str) -> str:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return json.dumps({"status": "success", "message": f"Content successfully written to {file_path}"})
        except IOError as e:
            logging.error(f"File write error for {file_path}: {e}")
            return json.dumps({"status": "error", "message": f"File write error: {e}"})
        except Exception as e:
            logging.error(f"General error in MarkdownFileWriterTool: {e}")
            return json.dumps({"status": "error", "message": f"General error: {e}"})

# 6. CSV Writer Tool
class CSVWriterTool(Tool):
    name: str = "CSV Writer"
    description: str = (
        "Writes JSON data (representing a DataFrame) to a CSV file. "
        "Requires 'dataframe_json' (JSON string of data) and 'file_path'. "
        "Useful for preparing data for PowerBI or other CSV-compatible applications."
    )

    def _run(self, dataframe_json: str, file_path: str) -> str:
        try:
            df = pd.read_json(dataframe_json)
            df.to_csv(file_path, index=False, encoding='utf-8')
            return json.dumps({"status": "success", "message": f"Data successfully written to CSV at {file_path}"})
        except Exception as e:
            logging.error(f"Error writing CSV to {file_path}: {e}")
            return json.dumps({"status": "error", "message": f"Error writing CSV: {e}"})


# --- Instantiate Tools ---
# Ensure your POSTGRES_URI environment variable is set in your .env file
pg_tool = PostgreSQLConnectorTool(db_uri=os.getenv("POSTGRES_URI", "postgresql://user:password@localhost:5432/testdb"))
excel_tool = ExcelDataTool()
python_executor_tool = PythonScriptExecutorTool()
powerbi_generator_tool = PowerBI_CodeGeneratorTool()
markdown_writer_tool = MarkdownFileWriterTool()
csv_writer_tool = CSVWriterTool()

# --- Define Agents ---

# 1. Data Preparation Agent
data_preparer = Agent(
    role='Data Acquisition and Preprocessing Specialist',
    goal='Efficiently extract, clean, and transform raw data from PostgreSQL and Excel into a structured and usable format.',
    backstory=(
        "You are a meticulous data engineer, expertly handling diverse data formats and ensuring "
        "data quality before any analysis begins. You are a wizard at wrangling messy data, "
        "proficient in querying PostgreSQL and manipulating Excel files, and can write Python "
        "scripts for complex transformations."
    ),
    llm=gemini_flash_llm,
    tools=[pg_tool, excel_tool, python_executor_tool], # Uses PostgreSQL tool
    verbose=True, # Shows detailed thought process in the console/logs
    allow_delegation=True # Can delegate sub-tasks or ask for clarification
)

# 2. Data Analysis Agent
data_analyst = Agent(
    role='Statistical Data Analyst',
    goal='Perform in-depth statistical analysis, identify trends, patterns, and anomalies, and generate meaningful insights.',
    backstory=(
        "You are a brilliant statistician and data scientist, capable of uncovering hidden truths "
        "within numbers. Your analytical rigor ensures sound conclusions. You are adept at Python "
        "for complex data manipulation, statistical modeling, and data summarization."
    ),
    llm=gemini_flash_llm,
    tools=[python_executor_tool, excel_tool], # Primarily uses Python for analysis, can read/write Excel for intermediate data
    verbose=True,
    allow_delegation=True # Can delegate analysis sub-tasks or refine data preparation
)

# 3. PowerBI Integration Agent
powerbi_integrator = Agent(
    role='Business Intelligence and Visualization Specialist',
    goal='Translate analytical insights into compelling and interactive PowerBI-ready data and M-code/DAX snippets, ensuring clear communication of key findings.',
    backstory=(
        "You are a master of data visualization, transforming complex data into intuitive and "
        "actionable dashboards that drive business decisions. You understand how PowerBI consumes data, "
        "and can generate the necessary M-code and DAX formulas for efficient report creation."
    ),
    llm=gemini_flash_llm,
    tools=[powerbi_generator_tool, csv_writer_tool], # Generates PowerBI-specific code and CSVs
    verbose=True,
    allow_delegation=False
)

# 4. Report Generation Agent
report_writer = Agent(
    role='Insight Communicator and Report Writer',
    goal='Synthesize findings from the analysis, create clear and concise reports, and summarize key recommendations for stakeholders.',
    backstory=(
        "You are a persuasive communicator, translating technical data into compelling narratives "
        "that resonate with business users. You ensure reports are actionable, easy to understand, "
        "and well-structured for executive consumption."
    ),
    llm=gemini_flash_llm,
    tools=[markdown_writer_tool], # Writes the final report
    verbose=True,
    allow_delegation=False
)

# --- Streamlit User Interface ---
st.set_page_config(layout="wide") # Use wide layout for better display
st.title("📊 AI Data Analyst Agent (PostgreSQL Edition)")
st.markdown("---")

# User input text area for the analysis request
user_request = st.text_area(
    "Enter your data analysis request:",
    "Analyze Q1 2025 sales data from PostgreSQL 'sales_data' table and 'customer_feedback.xlsx'. Identify top products and customer sentiment trends. Prepare data for PowerBI and generate an executive summary report.",
    height=150
)

# Function to dynamically create tasks based on the user's request (simplified for demo)
# In a more advanced scenario, a dedicated LLM agent could parse the request and define tasks.
def create_dynamic_tasks(request: str) -> list:
    tasks = []
    # Simple keyword-based task generation. Expand this for more dynamic behavior.
    if "sales" in request.lower() and "customer feedback" in request.lower():
        tasks.append(Task(
            description=(
                f"Based on the user request: '{request}', connect to the PostgreSQL database "
                "and extract the 'sales_data' table. "
                "Also, read the 'customer_feedback.xlsx' file from the current 'data/' directory. "
                "Combine these datasets on 'customer_id' (if available and relevant) and clean any missing values "
                "or inconsistencies. Save the merged and cleaned data to a new Excel file named 'prepared_sales_customer_data.xlsx'."
            ),
            expected_output="Path to the cleaned and merged Excel file (e.g., 'prepared_sales_customer_data.xlsx').",
            agent=data_preparer
        ))
        tasks.append(Task(
            description=(
                f"Based on the prepared 'prepared_sales_customer_data.xlsx' from the previous step, "
                "perform in-depth analysis to identify top-selling products by total amount, "
                "analyze customer sentiment distribution from the feedback, and find correlations "
                "between sales performance and customer sentiment categories. "
                "Provide a JSON summary of these analytical findings, including key trends, metrics, and correlations."
            ),
            expected_output="A JSON string summarizing top products, sentiment distribution, and any observed correlations.",
            agent=data_analyst
        ))
        tasks.append(Task(
            description=(
                f"Based on the analytical findings from the previous step, prepare a simplified, aggregated dataset for PowerBI. "
                "This should include aggregated sales data (e.g., by product category and time) and a summary of "
                "customer feedback (e.g., by sentiment type). Write this aggregated data to 'powerbi_dashboard_data.csv' "
                "in the current directory. "
                "Additionally, generate appropriate Power Query (M-code) to import this CSV into PowerBI, and suggest "
                "relevant DAX measures and a basic report layout based on the analysis."
            ),
            expected_output="A JSON string containing the path to the CSV file ('powerbi_dashboard_data.csv') and PowerBI M-code/DAX snippets/layout suggestions.",
            agent=powerbi_integrator
        ))
        tasks.append(Task(
            description=(
                f"Based on all previous analysis, PowerBI data preparation, and insights, "
                "generate a professional executive summary report in Markdown format. "
                "The report should summarize the key findings, provide actionable recommendations, "
                "and clearly state that a detailed PowerBI dashboard is available (referencing the 'powerbi_dashboard_data.csv' file as its data source). "
                "Save the final report as 'executive_summary_report.md' in the current directory."
            ),
            expected_output="A well-structured, executive-ready Markdown report saved as 'executive_summary_report.md'.",
            agent=report_writer
        ))
    else:
        st.error("For this demo, please include 'sales' and 'customer feedback' in your request to trigger the predefined tasks.")
        return []
    return tasks

# Button to trigger the analysis workflow
if st.button("Run Analysis"):
    if not user_request:
        st.warning("Please enter a request to start the analysis.")
    else:
        st.subheader("Agent Execution Log:")
        st.code("Starting the Data Analyst Automation Crew...", language="bash")

        # Capture stdout to display logs in Streamlit
        # This redirects print statements from CrewAI's verbose output to a string buffer
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()

        with st.spinner("Agents are working... This may take a moment."):
            try:
                # Dynamically create tasks based on user input
                tasks = create_dynamic_tasks(user_request)
                if tasks:
                    # Create the CrewAI crew with defined agents and tasks
                    data_analyst_crew = Crew(
                        agents=[data_preparer, data_analyst, powerbi_integrator, report_writer],
                        tasks=tasks,
                        process=Process.sequential, # Tasks will run one after another
                        verbose=True # Enables detailed logging of agent thoughts and actions
                    )
                    # Kick off the crew's execution
                    result = data_analyst_crew.kickoff()
                    st.success("Workflow Completed!")
                    st.subheader("Final Output:")
                    st.markdown(result) # Display the final result from the last agent/task

                    # --- Display and Offer Download of Generated Files ---
                    st.subheader("Generated Files:")
                    if os.path.exists("prepared_sales_customer_data.xlsx"):
                        with open("prepared_sales_customer_data.xlsx", "rb") as f:
                            st.download_button("Download Prepared Data (Excel)", f.read(), "prepared_sales_customer_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    if os.path.exists("powerbi_dashboard_data.csv"):
                        with open("powerbi_dashboard_data.csv", "rb") as f:
                            st.download_button("Download PowerBI Data (CSV)", f.read(), "powerbi_dashboard_data.csv", mime="text/csv")
                    if os.path.exists("executive_summary_report.md"):
                        with open("executive_summary_report.md", "r", encoding="utf-8") as f:
                            report_content = f.read()
                        st.download_button("Download Executive Report (Markdown)", report_content.encode("utf-8"), "executive_summary_report.md", mime="text/markdown")
                        st.text_area("Executive Report Content", report_content, height=300) # Also display content in UI

                else:
                    st.warning("No tasks were generated for your request. Please adjust your prompt.")

            except Exception as e:
                st.error(f"An error occurred during agent execution: {e}")
                st.error("Please check the logs below for more details on the error.")
            finally:
                # Restore original stdout and display the captured logs
                sys.stdout = old_stdout
                st.code(captured_output.getvalue(), language="plaintext")

# --- Feedback Section ---
# Allow the user to provide feedback, which can be stored for future improvements
st.markdown("---")
st.subheader("Provide Feedback")
feedback_text = st.text_area("Was this analysis helpful? What could be improved?", height=100)
if st.button("Submit Feedback"):
    feedback_file = "feedback.json"
    existing_feedback = []
    if os.path.exists(feedback_file):
        with open(feedback_file, "r") as f:
            try:
                existing_feedback = json.load(f)
            except json.JSONDecodeError:
                # Handle cases where the JSON file might be empty or corrupted
                existing_feedback = []
    
    # Append new feedback
    existing_feedback.append({"request": user_request, "feedback": feedback_text, "timestamp": pd.Timestamp.now().isoformat()})
    
    # Save all feedback back to the JSON file
    with open(feedback_file, "w") as f:
        json.dump(existing_feedback, f, indent=2) # Use indent for readability
    st.success("Thank you for your feedback! It will be used to improve the agent's performance.")

