# fastest_rag_stack

LLama3.3-RAG application
This project build the fastest stack to build a RAG application to chat with your docs. We use:

SambaNova as the inference engine for Llama 3.3.
Llama index for orchestrating the RAG app.
Qdrant VectorDB for storing the embeddings.
Streamlit to build the UI.
Installation and setup
Setup SambaNova:

Get an API key from SambaNova and set it in the .env file as follows:

SAMBANOVA_API_KEY=<YOUR_SAMBANOVA_API_KEY> 
Setup Qdrant VectorDB

docker run -p 6333:6333 -p 6334:6334 \
-v $(pwd)/qdrant_storage:/qdrant/storage:z \
qdrant/qdrant
Install Dependencies: Ensure you have Python 3.11 or later installed.

pip install streamlit llama-index-vector-stores-qdrant llama-index-llms-sambanovasystems sseclient-py
Run the app:

Run the app by running the following command:

streamlit run app.py
