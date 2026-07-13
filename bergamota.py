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
    search_kwargs={"k": 3, "score_threshold": 0.5} 
)
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 3
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.3, 0.7]
)

prompt = ChatPromptTemplate.from_template("""
Você é um **especialista em literatura gaúcha**. 
Sua missão é responder às perguntas dos usuários com precisão factual, comentando APENAS obras e autores que estão no contexto fornecido abaixo.

<context>
{context}
</context>

INSTRUÇÕES E REGRAS OBRIGATÓRIAS:
1. Se a pergunta mencionar uma obra ou autor, você deve OBRIGATORIAMENTE verificar se essa obra e esse autor estão descritos juntos no contexto fornecido.
2. Se o usuário atribuir a obra a um autor errado, corrija usando estritamente o que está no contexto.
3. REGRA DE SEGURANÇA ABSOLUTA: Se o contexto acima NÃO contiver informações sobre a obra ou autor perguntado, ou se o contexto falar de um autor (ex: Martha Medeiros) e a pergunta for sobre outro, NÃO tente adivinhar e NÃO use seu conhecimento externo. Responda exatamente: "Bah, vivente, não encontrei registros exatos sobre essa obra ou autor nos meus arquivos."
4. Jamais invente relações de autoria que não estejam explicitamente escritas dentro das tags <context></context>.

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

        print("\n🔍 [Análise do RAG] Buscando documentos...")
        docs_buscados = ensemble_retriever.invoke(pergunta)
        
        print("--- CONTEXTO INJETADO NO MODELO ---")
        for i, doc in enumerate(docs_buscados):
            print(f"Trecho {i+1}: {doc.page_content[:200]}...") # Mostra os primeiros 200 caracteres de cada trecho retornado
        print("-----------------------------------\n")

        resposta = qa_chain.invoke({"input": pergunta})
        print(f"Bergamota 🍊: {resposta['answer']}")
