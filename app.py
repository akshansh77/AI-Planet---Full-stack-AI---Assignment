import streamlit as st
import io
from streamlit_chat import message
#from htmlTemplates import css,bot_template,user_template
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.llms import Replicate
from langchain.llms import huggingface_hub
from langchain.llms import HuggingFaceHub
from langchain.document_loaders import AmazonTextractPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import PDFPlumberLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
import os
from dotenv import load_dotenv
import tempfile
from langchain.document_loaders.image import UnstructuredImageLoader
from PIL import Image
import easyocr

load_dotenv()

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about your files🤗"]


    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]


def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()
    form_key = f'my_form_{len(st.session_state["generated"])}'
    
    with container:
        with st.form(key=form_key, clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Documents", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                result = conversation_chat(user_input, chain, st.session_state['history'])

            # Concatenate answer and source_documents
            #concatenated_answer = f"{answer}\nSource Documents: {', '.join(source_documents)}"

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(result)
            # Now you can use source_documents as needed

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")


def create_conversational_chain(vector_store):
    load_dotenv()
    # Create llm
    # llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",
    #                     streaming=True, 
    #                     callbacks=[StreamingStdOutCallbackHandler()],
    #                     model_type="llama", config={'max_new_tokens': 500, 'temperature': 0.01,"context_length":2048})
    general_system_template = r""" 
                  Use only the following pieces of context to answer the question at the end. Do not use any other source apart from the provided context. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use five sentences maximum. Speak in a formal tone.Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
                    {context}
                    """
    general_user_template = "Question:```{question}```"
    messages = [
            SystemMessagePromptTemplate.from_template(general_system_template),
            HumanMessagePromptTemplate.from_template(general_user_template)
        ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)
    
    llm = Replicate(
        streaming = True,
        model = "replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781", 
        callbacks=[StreamingStdOutCallbackHandler()],
        model_kwargs = {"temperature": 0.01, "max_length" :500,"top_p":1})
    # # llm=HuggingFaceHub(repo_id="meta-llama/Llama-2-70b-chat-hf",model_kwargs={"temperature":0.01, "max_length":500,"top_p":1,"context_length":2048})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    compressor = CohereRerank()
    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=vector_store.as_retriever(search_kwargs={"k": 5}))                                                       
    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=compression_retriever,#,search_type = "stuff"),
                                                 memory=memory,combine_docs_chain_kwargs={'prompt': qa_prompt})#,return_source_documents=True,chain_type="map_reduce")
    
    return chain

def main():
    load_dotenv()
    # Initialize session state
    initialize_session_state()
    text = []

    for file in os.listdir("docs"):
        file_path = "./docs/" + file

        if file.endswith('.txt'):
            loader = TextLoader(file_path, encoding = 'UTF-8')
            text.extend(loader.load())
        elif file.endswith('.pdf'):
            loader = PDFPlumberLoader(file_path)
            text.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):
            loader = Docx2txtLoader(file_path)
            text.extend(loader.load())
        else:
            # Handle other file types or ignore them
            continue
        # if loader:
        #     text.extend(loader.load())
    st.title("Multi-Doc ChatBot using LLAMA-2 :books:")
    # Initialize Streamlit
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)
    if uploaded_files:
        #text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PDFPlumberLoader(temp_file_path)
                text.extend(loader.load())
            elif file_extension == ".docx" or file_extension == ".doc":
                loader = Docx2txtLoader(temp_file_path)
                text.extend(loader.load())
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path, encoding='UTF-8')
                text.extend(loader.load())
            elif file_extension == ".csv":
                loader = CSVLoader(file_path=temp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
                text.extend(loader.load())
            elif file_extension == '.png' or file_extension == ".jpg" or file_extension == ".jpeg":
                # Load the image
                image = Image.open(temp_file_path)

                # Convert the image to bytes
                image_bytes = io.BytesIO()
                image.save(image_bytes, format=image.format)
                image_bytes = image_bytes.getvalue()

                # Perform OCR using EasyOCR
                reader = easyocr.Reader(['en'])  # Specify the language you want to recognize
                result = reader.readtext(image_bytes)

                # Extract text from the result
                extracted_text = " ".join([text for _, text, _ in result])

                # Create a .txt file in the output directory with the extracted text
                output_file_path = os.path.join("./docs/", f"{os.path.splitext(file.name)[0]}.txt")
                with open(output_file_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(extracted_text)

                # Use the same loader for OCR text
                loader = TextLoader(output_file_path, encoding='utf-8')
                text.extend(loader.load())
                os.remove(output_file_path)

            # Use the same loader for file removal
            if loader:
                os.remove(temp_file_path)


        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
        text_chunks = text_splitter.split_documents(text)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                           model_kwargs={'device': 'cpu'})

        # Create vector store
        #vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
        vector_store=Chroma.from_documents(text_chunks, embedding=embeddings)

        # Create the chain object
        chain = create_conversational_chain(vector_store)

        
        display_chat_history(chain)

if __name__ == "__main__":
    main()