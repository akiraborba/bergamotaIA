from langchain_community.document_loaders import CSVLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

llm = OllamaLLM(model="literatura_gaucha_qwen")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

loader = CSVLoader(file_path="fact_check.csv", encoding="utf-8")
docs = loader.load()

vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./db_literatura")
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 3
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)

prompt = ChatPromptTemplate.from_template("""
Você é um **especialista em literatura gaúcha**. 
Sua missão é responder às perguntas dos usuários com precisão factual, comentando APENAS obras e autores que estão no contexto fornecido.

<context>
{context}
</context>

INSTRUÇÕES:
REGRAS OBRIGATÓRIAS:
1. Se a pergunta mencionar uma obra ou autor, você deve OBRIGATORIAMENTE buscar esse autor/obra no contexto. NÃO INVENTE INFORMAÇÕES
2. Se o usuário atribuir a obra a um autor errado, corrija usando o contexto.
3. Sob nenhuma circunstância comente obras que não estão no contexto, bem como não invente relações autor-obra.
4. Após elaborar sua resposta, confira se os autores, obras e a relação autor-obra constam no contexto.
5. Caso seja necessário falar de contextos históricos, movimentos literários e questões sociais para além do contexto autor-obra fornecido, SEMPRE revise se todas as obras e autores citados na resposta estão no contexto.                                                    
Pergunta: {input}
""")

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(ensemble_retriever, combine_docs_chain)

bergamota = """
                                                                              
    ▄▄▄                                                      ▄▄▄▄▄▄     ▄▄    
   ██▀▀█▄                                       █▄          █▀ ██     ▄█▀▀█▄  
   ██ ▄█▀       ▄       ▄▄       ▄             ▄██▄            ██     ██  ██  
   ██▀▀█▄ ▄█▀█▄ ████▄▄████ ▄▀▀█▄ ███▄███▄ ▄███▄ ██ ▄▀▀█▄       ██     ██▀▀██  
 ▄ ██  ▄█ ██▄█▀ ██   ██ ██ ▄█▀██ ██ ██ ██ ██ ██ ██ ▄█▀██       ██   ▄ ██  ██  
 ▀██████▀▄▀█▄▄▄▄█▀  ▄▀████▄▀█▄██▄██ ██ ▀█▄▀███▀▄██▄▀█▄██ ██  ▄▄██▄▄ ▀██▀  ▀█▄█
                        ██                                                    
                      ▀▀▀                                                          
                                                                            
"""
print(bergamota)
print("🍊 Bergamota.IA iniciado com sucesso! 🍊")
print("Digite 'sair' para encerrar.\n")

# 4. Loop de Conversa
while True:
    pergunta = input("\nVocê: ")
    if pergunta.lower() == 'sair':
        break
    else:
        resposta = qa_chain.invoke({"input": pergunta})
        print(f"Bergamota 🍊: {resposta['answer']}")
