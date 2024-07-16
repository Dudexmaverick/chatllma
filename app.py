import streamlit as st
from utills import text,embeddings
from streamlit_chat import message


def main():


    st.set_page_config(page_title='converse com seus arquivos', page_icon=':books')
    user_question = st.text_input("faça uma pergunta?")
   
    if('conversation' not in st.session_state):
         st.session_state.conversation = None
       


    if(user_question):
        response = st.session_state.conversation(user_question) ['chat_history'][-1]
     
     

        for i, text_message in enumerate(response):
            if i % 2 == 0:
                message(text_message.content, is_user=True, key=str(i) + '_user')
            else:
                message(text_message.content, is_user = False, key=str(i)+ '_bot')
    

    
    
    with st.sidebar:
        st.subheader('seus pdf')
        
        pdf_docs = st.file_uploader("faça o carregamento de seus pdf", accept_multiple_files= True)
    
   
        if st.button('processar'):
            all_files_text = text.process_files(pdf_docs)

            chunks = text.create_text_chunks(all_files_text)

            vectorstore = embeddings.create_vectorstores(chunks)

            st.session_state.conversation = embeddings.create_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()








