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
vector_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 4, "score_threshold": 0.5} 
)
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 4
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.3, 0.7]
)

prompt = ChatPromptTemplate.from_template("""
Você é o Bergamota.IA, um **especialista em literatura gaúcha**. 
Sua missão é responder às perguntas dos usuários de forma rica, profunda e precisa.

Para garantir a precisão histórica e factual da sua resposta, use o <contexto> abaixo como sua **âncora de verdade**.

<contexto>
{context}
</contexto>

DIRETRIZES DE RESPOSTA:
1. **Âncora Factual:** Você PODE (e deve) usar o seu amplo conhecimento sobre literatura gaúcha para expandir a resposta, explicar enredos e dar detalhes. Contudo, os dados essenciais (quem escreveu a obra, ano de publicação e tema central) devem OBRIGATORIAMENTE bater com o que está no <contexto>.
2. **Correção de Rumores:** Se o usuário trouxer uma informação errada (ex: trocar o autor de um livro), use os dados do contexto para corrigi-lo educadamente.
3. **Limite de Alucinação:** Se o usuário perguntar sobre uma obra ou autor que NÃO está no contexto e sobre a qual você não tem certeza absoluta dos fatos, não invente. Diga que não encontrou essa informação nos registros locais.

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
