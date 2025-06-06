from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()  # Load env variables from .env

model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"]
)

# response = model.invoke("Hello, world!")
# print(response.content)

messages = [    SystemMessage(content = "Translate the following from English into Russian"),
    HumanMessage(content = "Hello this is Arpit Raj! Moving on keeping formalities aside- Empathy is a Leaders Strength. But that doesn't mean thay we cry in Layoffs, otherwise Board meetings would need Tissue Papers instead of Agendas."),
    ]
response = model.invoke(messages)
print(response.content)