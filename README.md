
# Complete RAG architecture with Open Source LLM and Langchain

Step by step guide to create a RAG pipeline to fetch data from your documents, urls or apis.


## Step 1: Create conda environment

Create a virtual environment to separate your dependencies.
Here we are going to use anaconda virtual environment so you have to install anaconda in your computer.
You can install anaconda from here
https://www.anaconda.com/download/success

Create a new environment by running the following command in your terminal. This command automatically creates a new env in your present directory and installs python. 

```bash
  conda create -p venv python
```

Now check for all environments by running the following command

```bash
  conda list 
```

Now activate the env by running

```bash
  conda activate [path]
```

## Step 2: Install Ollama and download any open-source LLM

Install an open-source llm. Here we are going to use Ollama for this purpose. First download Ollama from https://ollama.com/. You can check all available llms here: https://github.com/ollama/ollama.  
Use this command to download any open-source llm.

```bash
  ollama run llama3.1
```

## Step 3: Installing required dependencies

Install all required packages by running from requirements.txt given in the repo the following command.

```bash
  pip install -r requirements.txt
```

## Step 4: Take a document to create vector store

Take a document to implement RAG pipeline in which we first
create a vector store of document and then by using some llm, we retrieve info from that vector store.
Here we are taking example of a pdf. You can also use csv, url,  etc. Take a look at langchain documentation: https://python.langchain.com/v0.2/docs/introduction/

We have to create vector database to store embeddings. We are going to store database our our disk. Later you can store it on cloud or some server.

## Step 5:

To create a vector store, first create a file say create_database.py and import the following:
```bash
  from langchain_community.document_loaders import PyPDFLoader
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  from langchain.schema import Document
  from langchain_community.embeddings import HuggingFaceBgeEmbeddings
  from langchain_community.vectorstores import FAISS
  import os
  import shutil
```

```bash
  FAISS_PATH = "faiss_index" #path where vector db is to be stored.
  DATA_PATH = "mydocument.pdf" #path to document
```

## Step 6: Load Documents

```bash
  def load_documents():
    loader = PyPDFLoader(DATA_PATH)
    documents = loader.load()
    return documents
```

## Step 7: Split the documents into chunks
We have to split the documents in chunks because of the context window of llm models.
```bash
  def split_text(documents: list[Document]):
      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=1000,
          chunk_overlap=200,
          length_function=len,
          add_start_index=True,
      )
      chunks = text_splitter.split_documents(documents)
      print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

      document = chunks[10]
      print(document.page_content)
      print(document.metadata)

      return chunks
```

## Step 8: Creating and saving db.
Create a new db with FAISS and HuggingFaceBgeEmbeddings and Save the database to disk.

```bash
  def save_to_faiss_index(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(FAISS_PATH):
        shutil.rmtree(FAISS_PATH)

    # Create a new DB from the documents.
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",      #sentence-transformers/all-MiniLM-l6-v2
        model_kwargs={'device':'cpu'},
        encode_kwargs={'normalize_embeddings':True}
    )
    db = FAISS.from_documents(
        chunks, embeddings
    )
    db.save_local("faiss_index")
    print(f"Saved {len(chunks)} chunks to {FAISS_PATH}.")
```
Call all these functions in main:
```bash
  def main():
    generate_data_store()

  def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_faiss_index(chunks)

  if __name__ == "__main__":
    main()
```

It will take some time to create vector store. After it we will call the database and use streamlit to make a chatbot interface. We shall try to have a conversation and check accuracy of results.

## Step 9: Make chatbot UI with streamlit.
Make a new file say chatbot.py and import following:
```bash
  import streamlit as st
  import os
  from langchain_community.llms import Ollama
  from langchain_community.embeddings import HuggingFaceBgeEmbeddings
  from langchain.chains.combine_documents import create_stuff_documents_chain
  from langchain_core.prompts import ChatPromptTemplate
  from langchain.chains import create_retrieval_chain
  from langchain_community.vectorstores import FAISS
```
Then read the database:
```bash
  embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",      #sentence-transformers/all-MiniLM-l6-v2
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}
  )
  vectors = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
```

## Step 10: Using chains to retrieve data.
Create chains to join llm and prompt and retrieve data from vector store. See more about chains in langchain documentation
```bash
  st.title("My help assistant")
  llm = Ollama(model="llama3.1")

  # Tell AI agent what to do
  prompt=ChatPromptTemplate.from_template(
  """
  Answer the questions based on the provided context only.
  Please provide the most accurate response based on the question
  <context>
  {context}
  <context>
  Questions:{input}

  """
  )
  # Create create_stuff_documents_chain and retrieval_chain
  # See more in langchain documentation
  document_chain = create_stuff_documents_chain(llm, prompt)
  retriever = vectors.as_retriever()
  retrieval_chain = create_retrieval_chain(retriever, document_chain)
```
## Step 10: Show the retrieved info in the UI:
```bash
  txt=st.text_input("Input you question here")
  if txt:
      response=retrieval_chain.invoke({"input":txt})
      st.write(response['answer'])

      with st.expander("Document Similarity Search"):
          for i, doc in enumerate(response["context"]):
              st.write(doc.page_content)
              st.write("--------------------------------")
```

## Finally: Run the chatbot.py
Open your terminal and run the following command.
```bash
  streamlit run chatbot.py
```

## Congratulations 🎉🎉
Your have successfully implemented RAG pipeline by using Ollama open-source llms.