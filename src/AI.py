import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain_chroma import Chroma
from langchain.schema import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")
os.environ["GOOGLE_API_KEY"] = api_key

loader = CSVLoader(file_path='./test.csv', csv_args={
    'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['City', 'Temperature(Farenheight)']
})
docs = loader.load()
print(docs[0].page_content)
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Replace OpenAIEmbeddings with GooglePalmEmbeddings
vectorstore = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory="./chroma_db")
vectorload = Chroma(persist_directory="./chroma_db", embedding_function=embedding)

query = PromptTemplate.from_template("""What weather patterns do you see in the data? What cities are measured, please name the exact cities? Please keep it very simple.\n Question: {question}\nContext: {context}Response:""")

retrieved_docs = vectorload.as_retriever(search_kwargs={"k": 5})

def generate_context(docs):
    return "\n".join([doc.page_content for doc in docs])

llm = (
    {"context": retrieved_docs | generate_context, "question": RunnablePassthrough()}
    | 
    query
    |
    ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    )
    | StrOutputParser()
)

ai_msg = llm.invoke("What is nevadas temp?")
print(ai_msg)