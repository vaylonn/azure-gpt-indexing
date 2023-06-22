# azure-gpt-indexing

Cet outil est basé de la libraire llama index.

Outil réalisant le questionnement de documents à partir d'appel d'API OpenAI via Azure.

Se référer à la page portal innoOpenAi dans portal.azure.com

La base du programme se fait à partir de cet exemple: https://gpt-index.readthedocs.io/en/latest/examples/customization/llms/AzureOpenAI.html

Cet outil utilise les libraires llama_index, langchain, openai, json et os.

Si deux documents sont mis dans le dossier data, l'outil va les mettre dans un seul index. Mais on pourra poser des questions spécifiques sur n'importe lequel des deux documents et il pourra nous donner quel est le document source.

# Setup

Avoir d’installé python 10

Avoir un environnement jupyter d'installé si vous voulez lancer le fichier .ipynb (utile pour faire des tests)

Modifier le `.env.sample` avec les bonnes variables puis renommer le fichier en `.env`.
```
OPENAI_API_KEY=api_key (clé api trouvable dans le groupe innoopenai puis "keys and endpoint")
OPENAI_API_BASE=https://xxxx.openai.azure.com/ (trouvable aussi dans "key and endpoints, ici les xxx représente le nom du groupe: innoopenai)
```

# Explications

Placez tous vos fichiers dans le répertoire `data`.

Pour l'instant, il vaut mieux ne rentrer qu'un seul fichier.

Les extensions prises en charge sont les suivantes (dont je suis quasi sûr sont):

   Via SimpleDirectoryReader:
   - `.csv`: CSV,
   - `.docx`: Word Document,
   - `.doc`: Word Document,
   - `.png, .jpg, .jpeg`: Prend le texte de l'image,
   - `.mp3` : Audio,
   - `.mp4`: Video,
   - `.md`: Markdown,
   - `.odt`: Open Document Text,
   - `.pdf`: Portable Document Format (PDF),
   - `.txt`: Text file (UTF-8),

   Via l'implementations de connecteurs:
   - `url`: Page internet via le connecteur BeautifulSoup (ne charge que l'url et pas le site entier)

   Des extensions de fichiers peuvent être rajoutées (même des vidéos youtube) via des connecteurs. Voir: https://gpt-index.readthedocs.io/en/latest/how_to/data_connectors.html

Le programme va créer un dossier pour stocker les données d'une manirère persistente a chaque fois qu'elles sont indéxées une première fois à partir de:
```shell
index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
index.storage_context.persist(persist_dir="./storage")
```

Une fois que les données sont indexées une première fois on peut réutiliser les index déjà créés avec: (permet de gagner du temps)
```shell
from llama_index import load_index_from_storage
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context, service_context=service_context)
```
Le prompt template définit la personnalité et comment va agir l'IA.
Modifier le prompt template pour définir comment on veut qu'il agisse.

# Execution du programme

Le programme s'utilise dans le fichier jupyter ou via le terminal avec le fichier python.

# Note

C'est le programme le plus rapide qu'on a actuellement vu que les modèles tournent sur les serveurs Azure.
