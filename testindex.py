import os
import openai
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index import LangchainEmbedding
from llama_index import download_loader
from llama_index import (
    GPTVectorStoreIndex,
    ListIndex,
    SimpleDirectoryReader, 
    LLMPredictor,
    ServiceContext,
    StorageContext,
)
from dotenv import load_dotenv
# Chargement des variables d'environnement depuis le fichier .env
load_dotenv('./.env')

# Configuration de l'API OpenAI
openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"
openai.api_base = os.environ["OPENAI_API_BASE"]
openai.api_key = os.environ["OPENAI_API_KEY"]

#set context window
context_window = 4096

#set number of output tokens
num_output = 512

# Initialisation de l'objet AzureOpenAI
# test1 représente le nom de déployment model sur Azure (le nom du modèle gpt35turbo)
llm = AzureChatOpenAI(deployment_name="default", temperature=0.1, max_tokens=num_output, openai_api_version=openai.api_version, model_kwargs={
    "api_key": openai.api_key,
    "api_base": openai.api_base,
    "api_type": openai.api_type,
    "api_version": openai.api_version,
})
llm_predictor = LLMPredictor(llm=llm)

# Initialisation de l'objet LangchainEmbedding pour l'indexation des documents à partir ici du modèle ada-002 nommé ada-test dans Azureembedding_llm = LangchainEmbedding(
embedding_llm = LangchainEmbedding(
    OpenAIEmbeddings(
        model="text-embedding-ada-002",
        deployment="learning",
        openai_api_key= openai.api_key,
        openai_api_base=openai.api_base,
        openai_api_type=openai.api_type,
        openai_api_version=openai.api_version,
    ),
    embed_batch_size=1,
)

# Chargement des documents à partir du répertoire './data'
documents = SimpleDirectoryReader('./data').load_data()
# #permet de vérifier que les docs ont bien étés chargés
print('Document ID:', documents[0].doc_id)
print((f"Loaded doc n°1 with {len(documents)} pages"))

service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    embed_model=embedding_llm,
    context_window=context_window,
    num_output=num_output,
)

# Création de l'index à partir des documents et du service_context
# index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
# index.storage_context.persist(persist_dir="./storage")
# print((f"Finished building doc n°1 index with {len(index.docstore.docs)} nodes"))

# Chargement de l'index à partir du stockage (commenté pour le moment)
from llama_index import load_index_from_storage
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context, service_context=service_context)
print((f"Finished loading doc n°1 index from storage with {len(index.docstore.docs)} nodes"))

# Prompt de base du chatbot
from llama_index import Prompt
template = (
    "Tu trouveras ci-dessous des informations contextuelles. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Tu es un chatbot assistant technique de la socitété Equans via un chatbot qui ne parle que français. Tu donnes une assistance technique aux questions que te poseras l'utilisateur."
    "D'après le contexte, réponds à la question en français uniquement, même si la question est posée dans une autre langue. Réponds donc à la question:{query_str}\n"

)
qa_template = Prompt(template)
from llama_index.memory import ChatMemoryBuffer
memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
# Requête envoyée au modèle
while True:
    query = input("Query: ")
    if query == "stop":
        break
    chat_engine = index.as_chat_engine(chat_mode="context", memory=memory, similarity_top_k=3, system_prompt="You are a chatbot, able to have normal interactions, as well as talk about a technical document.") #similarity_top_k=3, text_qa_template=qa_template
    answer = chat_engine.chat(query)

    # Affichage des résultats
    #print(answer.get_formatted_sources())
    print('query was:', query)
    print('answer was:', answer)

# Documents source
for node in answer.source_nodes:
    print('-----')
    text_fmt = node.node.text.strip().replace('\n', ' ')[:1000]
    print(f"Text:\t {text_fmt} ...")
    print(f'Metadata:\t {node.node.extra_info}')
    print(f'Score:\t {node.score:.3f}')