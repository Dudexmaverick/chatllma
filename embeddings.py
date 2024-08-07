from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import Ollama
from langchain.prompts import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate

def create_vectorstores(chunks):

    embeddings = HuggingFaceInstructEmbeddings(model_name = "BAAI/bge-small-en" )
    vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings)

    return vectorstore


def create_conversation_chain(vectorstore):

    llm = Ollama(model="gemma2",temperature= 0.1) 
    general_system_template = """
    Você é um assistente virtual, que responde exclusivamente em português sobre os conteúdos dos PDFs armazenados.
    Use o contexto fornecido para responder à pergunta de forma clara e concisa.
    Se e pergunta for fora do contexto dos PDFs, responda que não pode responder fora do tópico.:

    {context}

    ---

    Answer the question based on the above context: {question}
    """
    general_user_template = "Question:"""
    messages = [
        SystemMessagePromptTemplate.from_template(general_system_template), 
        HumanMessagePromptTemplate.from_template(general_user_template)
    ]
    qa_prompt = ChatPromptTemplate.from_messages (messages)
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)


    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever= vectorstore.as_retriever(),
        memory=memory, 
        combine_docs_chain_kwargs={'prompt': qa_prompt}
    )
    return conversation_chain
