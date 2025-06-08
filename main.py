from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

load_dotenv()  # Load env variables from .env

model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    streaming = True
)

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)

#Function to Call Model
def call_ai(state : MessagesState):
    chunks = []
    print("🤖AI: ", end="", flush=True)

    for chunk in model.stream(state["messages"]):
        print(chunk.content, end="", flush=True)
        chunks.append(chunk)

    print()  # for new line after bot finishes
    return {"messages": chunks[-1]}  # return last message to keep history


workflow.add_node("model", call_ai)
workflow.set_entry_point("model")
workflow.add_edge(START, "model")

#MEMORY
memory = MemorySaver()
# memory.save("model", call_model)
app = workflow.compile(checkpointer = memory)

config = {"thread_id": "terminal-chat-001"}

llm = model 
# PDF QA chain: initialize once (when needed)
qa_chain = None

# PDF retriever
def setup_pdf_qa(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    # Extract only the text content from each chunk
    texts = []
    for i, doc in enumerate(chunks):
        content = doc.page_content
        if isinstance(content, str) and content.strip():
            texts.append(content)
        else:
            print(f"Skipping chunk {i} due to invalid content: {content}")

    # Initialize embeddings with the same Azure params as your model
    global embeddings

    # Create FAISS vectorstore from list of strings (texts)
    vectorstore = FAISS.from_texts(texts, embeddings)

    retriever = vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


#CHAT LOOP
print("🤖AI Bot is ready! Type 'exit' or 'quit' to stop.\n")

while True:


    query = input("You: ")

    if query.lower() in ['exit', 'quit']:
        print("🤖AI: Goodbye!")
        break

    # PDF trigger condition
    if "pdf" in query.lower() or "read from" in query.lower():
        if qa_chain is None:
            qa_chain = setup_pdf_qa("Notice.pdf")
        try:
            result = qa_chain.invoke({"query": query})
            answer = result.get("result")  # Extract the answer string
            print("🤖AI:", answer)
            # print(qa_chain.invoke({"query": "What is the subject of the notice?"}))

        except Exception as e:
            print("Error while querying PDF:", e)

    else:
        # Default chat (no PDF)
        app.invoke({"messages": [HumanMessage(content=query)]}, config=config)
        # print("Bot:", result["messages"][-1].content)


