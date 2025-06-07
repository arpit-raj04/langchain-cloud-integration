from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

load_dotenv()  # Load env variables from .env

model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"]
)

# response = model.invoke("Hello, world!")
# print(response.content)

# messages = [    SystemMessage(content = "Translate the following from English into Russian"),
#     HumanMessage(content = "Hello this is Arpit Raj! Moving on keeping formalities aside- Empathy is a Leaders Strength. But that doesn't mean thay we cry in Layoffs, otherwise Board meetings would need Tissue Papers instead of Agendas."),
#     ]
# response = model.invoke(    [
#         HumanMessage(content="Hi! I'm Arpit"),
#         AIMessage(content="Hello Arpit! How can I assist you today?"),
#         HumanMessage(content="What's my name?"),
#     ])

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)


#Function to Call Model
def call_ai(state : MessagesState):
    response = model.invoke(state["messages"])
    return {"messages":response}
workflow.add_node("model", call_ai)
workflow.set_entry_point("model")
workflow.add_edge(START, "model")


#MEMORY
memory = MemorySaver()
# memory.save("model", call_model)
app = workflow.compile(checkpointer = memory)

config = {"thread_id": "terminal-chat-001"}

#CHAT LOOP
print("🤖 Chatbot is ready! Type 'exit' to stop.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chat Ends here!")
        break

    #Now Sending user messages to Chatbot
    result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
    response = result["messages"][-1].content

    print("🤖AI:", response)  
    
