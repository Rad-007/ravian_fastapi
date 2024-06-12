import autogen
import panel as pn
import openai
import re
import os
import time
import asyncio
from langchain.chat_models.azure_openai import AzureChatOpenAI
from autogen.coding import LocalCommandLineCodeExecutor
import pandas as pd
from autogen.agentchat.agent import Agent
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from sqlalchemy import JSON

config_list = autogen.config_list_from_json(r"new_OAI_CONFIG_LIST.json")




# When using a single openai endpoint, you can use the following:
#config_list = [{"model": "gpt-4", "api_key": os.getenv("OPENAI_API_KEY")}]
#file_path = r"C:\train_data.csv"
file_path = 'curated.csv'  # Replace with your data file path

# Define a function to read and process a CSV file
def read_csv_data(file_path=file_path):
    import pandas as pd
    df = pd.read_csv(file_path)
    df = df.head()
    return df.to_dict()

# Set your LLM config here
llm_config = {
    "timeout": 600,
    "cache_seed": 42,
    "config_list": config_list,
    "temperature": 0.15,
}

input_future = None

class MyConversableAgent(autogen.ConversableAgent):

    async def a_get_human_input(self, prompt: str) -> str:
        global input_future
        input_future=None
        print('AGET!!!!!!')  # or however you wish to display the prompt
        #chat_interface.send(prompt, user="System", respond=False)
        # Create a new Future object for this input operation if none exists
        if input_future is None or input_future.done():
            input_future = asyncio.Future()

        # Wait for the callback to set a result on the future
        await input_future

        # Once the result is set, extract the value and reset the future for the next input operation
        input_value = input_future.result()
        input_future = None
        return input_value

class DataAgent(AssistantAgent):
    def __init__(self, name, file_path, **kwargs):
        super().__init__(name=name, **kwargs)
        self.file_path = file_path
        self.data = None  # Initialize data container

    def load_data(self):
        """Loads data from the file."""
        if self.file_path.endswith(".csv"):
            self.data = pd.read_csv(self.file_path)
            self.head = self.data.head().to_string()  # Convert to string for LLM
            self.columns = list(self.data.columns) 
            self.info = self.data.info()
            self.describe = self.data.describe().to_string()
            self.a_initiate_chat(
                groupchat.agent_by_name("Planner"),
                message=f"""Here are the data summaries:
                **Head:**
                {self.head}
                **Columns:**
                {self.columns}
                **Info:**
                {self.info}
                **Describe:**
                {self.describe}
                """
            )
        elif self.file_path.endswith(".json"):
            import json
            with open(self.file_path, 'r') as f:
                self.data = json.load(f)
        # Add support for other formats (xls, etc.) here
        else:
            raise ValueError(f"Unsupported file type: {self.file_path}")

    def answer_question(self, question):
        """Answers questions based on loaded data."""
        if self.data is None:
            self.load_data()

        # Use pandas to answer questions about data
        if isinstance(self.data, pd.DataFrame):
            try:
                return content_str(self.data.query(question))
            except Exception as e:
                return f"Error executing query: {e}"
        # Add logic for answering questions from JSON and other data formats
        else:
            return "I'm still learning how to answer questions about this data type."

class PlannerAgent(AssistantAgent):
    def __init__(self, name, llm_config, **kwargs):
        super().__init__(name=name, llm_config=llm_config, **kwargs)
        # Attributes to store data summary
        self.data_head = None
        self.columns = None
        self.data_info = None
        self.data_describe = None
        self.unique_values = None

    def store_data_summary(self, head, columns, info, describe, unique_values):
        """Stores the data summary in the agent's attributes."""
        self.data_head = head
        self.columns = columns
        self.data_info = info
        self.data_describe = describe
        self.unique_values = unique_values

    def receive_message(self, message):
        """Processes incoming messages and extracts data summary if present."""
        # Check if the message contains data summary information
        if "Here are the data summaries" in message["content"]:
            # Extract head, columns, info, describe, and unique values from the message
            head_pattern = re.compile(r"\*\*Head:\*\*\n(.*?)\n\*\*Columns:\*\*", re.DOTALL)
            columns_pattern = re.compile(r"\*\*Columns:\*\*\n(.*?)\n\*\*Info:\*\*", re.DOTALL)
            info_pattern = re.compile(r"\*\*Info:\*\*\n(.*?)\n\*\*Describe:\*\*", re.DOTALL)
            describe_pattern = re.compile(r"\*\*Describe:\*\*\n(.*?)\n\*\*Unique Values:\*\*", re.DOTALL)
            unique_values_pattern = re.compile(r"\*\*Unique Values:\*\*\n(.*?)$", re.DOTALL)

            head_match = head_pattern.search(message["content"])
            columns_match = columns_pattern.search(message["content"])
            info_match = info_pattern.search(message["content"])
            describe_match = describe_pattern.search(message["content"])
            unique_values_match = unique_values_pattern.search(message["content"])

            head = head_match.group(1) if head_match else ""
            columns = columns_match.group(1) if columns_match else ""
            info = info_match.group(1) if info_match else ""
            describe = describe_match.group(1) if describe_match else ""
            unique_values = unique_values_match.group(1) if unique_values_match else ""

            # Store the extracted data summary
            self.store_data_summary(head, columns, info, describe, unique_values)
        else:
            # Process other types of messages
            pass

# Planner Agent 
planner = PlannerAgent(
    name="Planner",
    llm_config=llm_config,
    system_message="""You are a Planner Agent. Your job is to understand user requests related to data analysis and create a step-by-step plan to achieve the desired outcome. 
    Always read the data from the location provided.
    The Step by step plan should be exhaustive and would have all the steps which would be needed to solve the problem. If possible try to write an alternative approach to the problem and let other agents decide how they want to solve it.
    Your plan should involve the following agents:
     - DataAgent: To interact with the data
     - ColumnAnalyzer: To analyze the columns of the data, you need this every time you will solve any problem
     - DataSummary: To provide summaries of the data, you can ask for data summary if you face any error from code executor 
     - CodeWriter: To generate code for analysis
     - CodeExecutor: To execute the generated code
     - Debugger: To handle and correct any code errors
     
     Clearly indicate which agent should be responsible for each step. 
     Consider the type of analysis requested (basic, analytics, forecasting, AI/ML) and plan accordingly.
     """
)

# Column Analyzer Agent
column_analyzer = AssistantAgent(
    name="ColumnAnalyzer",
    llm_config=llm_config,
    system_message="""You are a ColumnAnalyzer Agent. You receive instructions from the Planner and analyze the columns of the data provided by the DataAgent.
    - Identify the data type of each column.
    - Identify the name of each column and what is the meaning of the name of these columns
    - Determine if there are any missing values and their percentage, only do this when asked by planner
    - Identify potential outliers or unusual values, only do this when asked by planner
    - Suggest transformations or cleaning steps if needed.
    - Suggest various strategies to solve a given problem by planner, always do this
    """
)

# Data Summary Agent
data_summary = AssistantAgent(
    name="DataSummary",
    llm_config=llm_config,
    system_message="""You are a DataSummary Agent. You receive instructions from the Planner and provide summaries of the data using `df.info()`, `df.describe()`, and other relevant methods.
    - Report the overall shape of the data (number of rows and columns).
    - Provide a summary of the data types of each column.
    - Show descriptive statistics (count, mean, std, min, max, percentiles) for numerical columns.
    - Identify any potential issues or insights based on the summaries.
    """
)

# Code Writer Agent 
code_writer = AssistantAgent(
    name="CodeWriter",
    llm_config=llm_config,
    system_message="""You are a CodeWriter Agent. You receive instructions from the Planner and generate code to perform data analysis tasks.
    - The code should be functional and complete.
    - Specify the programming language (e.g., Python) using code blocks (```python ... ```).
    - Include necessary imports and handle potential errors gracefully.
    - Use appropriate libraries based on the task (pandas for general analysis, scikit-learn for ML, etc.).
    """
)

# Code Executor Agent 
code_executor = UserProxyAgent(
    name="CodeExecutor",
    human_input_mode="NEVER",
    code_execution_config={
        "executor": LocalCommandLineCodeExecutor(work_dir="coding"),
    },
    llm_config=llm_config, 
    system_message="You are the CodeExecutor Agent. Execute the code provided by the CodeWriter and report back the results or errors."
)

# Debugger Agent 
debugger = AssistantAgent(
    name="Debugger",
    llm_config=llm_config,
    system_message="""You are a Debugger Agent. Your role is to analyze code errors reported by the CodeExecutor, suggest fixes to the CodeWriter, and verify if the fixes resolve the issues.  
    - Clearly identify the error, its location, and possible causes.
    - Suggest specific code modifications to the CodeWriter.
    """
)

# User Proxy Agent
user_proxy = MyConversableAgent(
    name="System",
    code_execution_config=False,
    is_termination_msg=lambda msg: "Terminate" in msg["content"],
    human_input_mode="ALWAYS"
)

# Customized speaker selection function for group chat
def custom_speaker_selection_func(last_speaker: Agent, groupchat: GroupChat):
    messages = groupchat.messages
    #default_file_path='curated.csv'
    # Check if file path has been provided
    #file_path_provided = any("file path" in msg["content"].lower() for msg in messages)
    file_path_provided=next((re.search(r"file path: (.+)", msg["content"].lower()).group(1).strip() for msg in messages if re.search(r"file path: (.+)", msg["content"].lower())), default_file_path)

    # Initial interaction
    if len(messages) <= 1:
        return user_proxy  # Start with the user

    # Always select DataAgent to send summaries first
    if last_speaker is user_proxy:
        if file_path_provided:
            return groupchat.agent_by_name("DataAgent")  # Send summaries first
        else:
            return user_proxy  # Keep asking for file path

    # After summaries are sent, proceed based on previous message:
    elif last_speaker is data_agent:
        return planner  # Planner receives summaries and starts planning
    elif last_speaker is planner:
        if "DataAgent load the data" in messages[-1]["content"]:
            return groupchat.agent_by_name("DataAgent")  # Let DataAgent send summaries (shouldn't happen)
        else:
            return code_writer  # Continue with the analysis chain
    elif last_speaker is code_writer:
        return code_executor
    elif last_speaker is code_executor:
        if "exitcode: 1" in messages[-1]["content"]:
            return debugger
        else:
            return user_proxy
    elif last_speaker is debugger:
        return code_writer
    else:
        return user_proxy

# Create the Group Chat
data_agent = DataAgent(name="DataAgent", file_path=file_path, llm_config=llm_config)

groupchat = GroupChat(
    agents=[user_proxy, planner, data_agent, column_analyzer, data_summary, code_writer, code_executor, debugger],
    messages=[],
    max_round=500, 
    speaker_selection_method=custom_speaker_selection_func,
    enable_clear_history=True, # Optional: Allows agents to clear chat history 
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

avatar = {
    user_proxy.name:"👨‍💼", 
    planner.name:"👩‍💻", 
    code_writer.name:"👩‍🔬", 
    code_executor.name:"📝", 
    debugger.name:"🛠", 
    data_agent.name: "🗃️"  # Add DataAgent's avatar
}

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from typing import List
import panel as pn
from new_ton import chat_interface, callback
import shutil

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Store connected websockets
connected_clients: List[WebSocket] = []

@app.on_event("startup")
async def startup_event():
    pn.extension(design="material")
    chat_interface.send("Send a message!", user="System", respond=False)
    chat_interface.servable()

"""
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Process the received message and get the response
            await callback(data, "User", chat_interface)
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

"""
import json
messages = []
@app.websocket("/ws/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await callback(data, "User", chat_interface)
            message = json.loads(data)
            if message['type'] == 'message':
                await websocket.send_json({"user": "User", "message": message['data']})
            # Handle other message types if needed
    except WebSocketDisconnect:
        
        print("WebSocket disconnected")
        connected_clients.remove(websocket)

@app.post("/save_message/")
async def save_message(message: dict):
    messages.append(message)
    with open("messages.json", "w") as file:
        json.dump(messages, file)
    return JSONResponse(content={"message": "Message saved successfully"})



async def broadcast_message(message: str, user: str):
    for client in connected_clients:
        await client.send_text(json.dumps({"user": user, "message": message}))


upload_directory = "uploads"
os.makedirs(upload_directory, exist_ok=True)

# Predefined default file path
default_file_path = 'curated.csv'  # Initial default file path

# Endpoint to upload a file
@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    global default_file_path
    # Save the file to the upload directory
    file_location = os.path.join(upload_directory, file.filename)
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Update the default file path
    default_file_path = file_location
    return JSONResponse(content={"message": "File uploaded successfully", "file_path": default_file_path})



"""
def print_messages(recipient, messages, sender, config):
    #Return answers from here
    print(f"Messages from: {sender.name} sent to: {recipient.name} | num messages: {len(messages)} | message: {messages[-1]}")

    content = messages[-1]['content']

    if all(key in messages[-1] for key in ['name']):
        print(f"Content: {content}, User: {messages[-1]['name']}, Avatar: {avatar[messages[-1]['name']]}") if not False else None

    else:
        print(f"Content: {content}, User: {recipient.name}, Avatar: {avatar[recipient.name]}") if not False else None

    
    return False, None  # required to ensure the agent communication flow continues

"""
def print_messages(recipient, messages, sender, config):
    content = messages[-1]['content']
    user = messages[-1]['name'] if all(key in messages[-1] for key in ['name']) else recipient.name
    avatar1 = avatar[user] if not False else None
    
    # Send message to chat interface
    chat_interface.send(content, user=user, avatar=avatar1, respond=False)
    
    # Broadcast message to WebSocket clients
    asyncio.create_task(broadcast_message(content, user))
    
    return False, None  # Required to ensure the agent communication flow continues

user_proxy.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages, 
    config={"callback": None},
)

planner.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages, 
    config={"callback": None},
) 

# Register replies for the new agents
column_analyzer.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages, 
    config={"callback": None},
)

data_summary.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages, 
    config={"callback": None},
)

code_writer.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages, 
    config={"callback": None},
) 

code_executor.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages, 
    config={"callback": None},
) 

debugger.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages, 
    config={"callback": None},
) 

pn.extension(design="material")

initiate_chat_task_created = False

async def delayed_initiate_chat(agent, recipient, message):
    global initiate_chat_task_created
    # Indicate that the task has been created
    initiate_chat_task_created = True

    # Wait for 2 seconds
    await asyncio.sleep(2)

    # Now initiate the chat
    print(f"Initiating chat with {recipient}: {message}")
    await agent.a_initiate_chat(recipient, message=message)

async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    print("Message from Callback",contents)
    global initiate_chat_task_created
    global input_future

    if not initiate_chat_task_created:
        asyncio.create_task(delayed_initiate_chat(user_proxy, manager, contents))
    else:
        if input_future and not input_future.done():
            input_future.set_result(contents)
        else:
            print("There is currently no input being awaited.")


input_future=None
async def callback2():
    global input_future
    global initiate_chat_task_created
    
    # Asynchronously wait for user input

    loop = asyncio.get_event_loop()
    contents = str(await loop.run_in_executor(None, input, "Enter Message: "))
    print("User Input:", contents)

    
    if not initiate_chat_task_created:
        print("No task created yet, creating one now.")

        task=asyncio.create_task(delayed_initiate_chat(user_proxy, manager, contents))
        await task
        print("Task created and awaited.")
    else:
        if input_future and not input_future.done():
            print("Setting result for input_future.")
            input_future.set_result(contents)
        else:
            print("Failing 3")
            print("There is currently no input being awaited.")

#app.mount("/ravian_backend", StaticFiles(directory="templates"), name="static")


@app.get("/")
async def get():
    html_path = Path(__file__).resolve().parent  / "chat3.html"
    return HTMLResponse(content=html_path.read_text())
    


import uvicorn

if __name__ == "__main__":
    uvicorn.run("chat2:app", host="127.0.0.1", port=8002, reload=True)










import pdb
chat_interface = pn.chat.ChatInterface(callback=callback)

chat_interface.send("Send a message!", user="System", respond=False)

chat_interface.servable()
