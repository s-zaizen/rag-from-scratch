from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_store = InMemoryVectorStore(embeddings)

bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

assert len(docs) == 1
print(f"Total characters: {len(docs[0].page_content)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200, 
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)

_ = vector_store.add_documents(all_splits)

print(f"Split blog post into {len(all_splits)} sub-documents.")

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. "
     "Answer the user's question based only on the following context.\n\n"
     "{context}"),
    ("human", "{question}"),
])

chain = prompt | model

query = "What is the standard method for Task Decomposition?"

retrieved_docs = vector_store.similarity_search(query, k=2)
context = "\n\n".join(doc.page_content for doc in retrieved_docs)

response = chain.invoke({"context": context, "question": query})
print(response.content)