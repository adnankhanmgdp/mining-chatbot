from llama_index.llms.gemini import Gemini
from llama_index.core.llms import ChatMessage
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.gemini import GeminiEmbedding
import os
api_key = "AIzaSyBW_AY2BFoIQDbdMxqGdU-1M4hxh-ajdKs"
os.environ["GOOGLE_API_KEY"] = api_key

if __name__ == "__main__":
    llm = Gemini()
    messages = []
    model_name = "models/embedding-001"
    embed_model = GeminiEmbedding(
        model_name=model_name, api_key=api_key, 
    )
    Settings.embed_model = embed_model
    Settings.llm = llm
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    chat_engine = index.as_chat_engine()
    while True:
        query = input("Ask something: ")
        response = chat_engine.chat(message=query, chat_history=messages)
        print(response)
        messages.append(ChatMessage(role="user", content=query))
        messages.append(ChatMessage(role='assistant', content=response))
