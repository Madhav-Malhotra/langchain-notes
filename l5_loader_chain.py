from dotenv import load_dotenv
from langchain.document_loaders import NewsURLLoader
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv(".env")

# load data
doc = NewsURLLoader(urls=['https://www.bbc.com/news/world-asia-67879609']).load()
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
docs = splitter.split_documents(doc)

# load chain
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={
    'temperature': 0.9, 'max_length': 256
})
chain = load_summarize_chain(llm=llm, verbose=True, chain_type='map_reduce')

# run chain
chain.run(docs)