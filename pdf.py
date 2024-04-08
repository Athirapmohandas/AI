import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
import qdrant_client
import openai
import os
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from dotenv import load_dotenv


def get_vector_store():
    client = qdrant_client.QdrantClient(
    os.getenv('QDRANT_HOST'),
    api_key = os.getenv('QDRANT_API_KEY')
    )
    embeddings =OpenAIEmbeddings()
    vector_store = Qdrant(
    client=client, 
    collection_name=os.getenv('QDRANT_COLLECTION_NAME'), 
    embeddings=embeddings
    )
    return vector_store

def main():
    load_dotenv()

    st.set_page_config(page_title="Ask Qdrant and OpenAI")
    st.header("Ask your remote database")

    vector_store = get_vector_store()

    llm = OpenAI(temperature=0.6,model="gpt-3.5-turbo-instruct")  # Initialize OpenAI LLM with desired temperature

    # tools = load_tools(["serpapi"],llm=llm)

    openai.api_key = os.getenv("OPENAI_API_KEY")  # Set OpenAI API key

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
    )

    user_question = st.text_input("Ask a question:")
    if user_question:
        
            answer = qa.run(user_question)
            st.write(f"Answer from database: {answer}")

            if "I don't know" in answer:
                #  ##################### Answer from chatgpt
                st.write("Attempting to answer using OpenAI...")
                openai_answer = llm(user_question)
                st.write(f"Answer from OpenAI: {openai_answer}")
                 
            # else:
            #     st.write(f"Answer from Qdrant: {answer}")

if __name__ == "__main__":
    main()
