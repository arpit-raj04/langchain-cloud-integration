from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # Load env variables from .env

model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"]
)

response = model.invoke("Hello, world!")
print(response.content)
