import streamlit as st
from utills import text, embeddings
def main():
    st.set_page_config(page_title='converse com seus arquivos', page_icon=':books')
    
   
    with st.sidebar:
        st.subheader('seus pdf')
        
        
        

    pdf_docs = st.file_uploader("faça o carregamento de seus pdf", accept_multiple_files= True)
    
   
    if st.button('processar'):
        all_files_text = text.process_files(pdf_docs)

        chunks = text.create_text_chunks(all_files_text)

        vectorstore = embeddings.create_vectorstores(chunks)

        conversation = embeddings.create_conversation_chain(vectorstore)



if __name__ == '_main_':
    main()








