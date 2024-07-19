from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import Ollama
from langchain.prompts import ChatPromptTemplate

def create_vectorstores(chunks):

    embeddings = HuggingFaceInstructEmbeddings(model_name = "BAAI/bge-small-en" )
    vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings)

    return vectorstore


def create_conversation_chain(vectorstore):

    llm = Ollama(model="llama3",temperature= 0.1) 
    

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)


    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever= vectorstore.as_retriever(),
        memory=memory
    )
       
    


    return conversation_chain
 
