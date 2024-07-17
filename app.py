import streamlit as st
import pickle 
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space 
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS 
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import  get_openai_callback
import os

# Sidebar content
with st.sidebar:
    image = Image.open('eCentrixlogo.png')
    st.image(image, caption = 'eCentrix Solutions')
    # st.markdown('## About')
    # add_vertical_space(5)
    # st.write('eCentrix Solutions')

load_dotenv()

def main():
    image = Image.open('ecentrix.png')
    st.image(image, caption = '')
    st.header('Chat With Your PDF')

    with st.form('Assist ur PDF with ChatGPT'):
    # Upload PDF
        pdf = st.file_uploader("Upload Your PDF", type='pdf')

        # submitted = st.form_submit_button()
    

    # if 'messages' not in st.session_state:
    #     st.session_state.messages = []
    # with st.form(key='chat_form'):
    #     user_input = st.text_input("Masukkan pesan:")
    #     submit_button = st.form_submit_button(label='Kirim')

    # # Jika tombol diklik, simpan pesan ke sesi
    # if submit_button and user_input:
    #     st.session_state.messages.append({"user": user_input, "bot": "Balasan dari chatbot."})
        
    if pdf is not None:
        st.write(f"Uploaded PDF: {pdf.name}")

        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle potential None return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Embeddings
        
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            st.write('Emberddings Loaded From the Disk')
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

            st.write('Emberddings Computation Completed')

        #Accept user question
        query= st.text_input("Ask question about ur pdf file")
        st.write(query)

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm=OpenAI(model_name="gpt-3.5-turbo")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_document=docs, question=query)
                print(cb)
            st.write(response)



if __name__ == '__main__':
    main()
