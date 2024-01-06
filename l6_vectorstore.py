""" YOUTUBE VIDEO CHATBOT
This script demonstrates how to create a LangChain vector store from a
Youtube video transcript. It uses this to answer questions about the video.
Dependencies: `langchain`, `dotenv`, `faiss-cpu`, HuggingFace API key.
"""

# Library imports
import os
import argparse
from dotenv import load_dotenv

from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import PromptTemplate
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.chains import LLMChain

# Load token
load_dotenv()
KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Setup embeddings
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=KEY, model_name="sentence-transformers/all-MiniLM-l6-v2"
)

# Setup vector store
def vector_store_from_YT(url : str) -> FAISS:
    """Creates a vector store from a Youtube video transcript"""

    print("Loading video transcript...")
    loader = YoutubeLoader.from_youtube_url(url)
    transcript = loader.load()

    print("Extracting data from video...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    sentences = splitter.split_documents(transcript)

    # Create vector store
    db = FAISS.from_documents(sentences, embeddings)
    return db

def get_model() -> LLMChain:
    """Combines an LLM and prompt template to create a LangChain chain"""

    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={
        'temperature': 0.6, 'max_length': 256
    })

    template = """You are a helpful Youtube assistant that answers questions using video transcripts. 
    A user asks you a question that is very important to their career success: {question}
    Answer the question using the following video transcript: {documents}

    Only use factual information from the video to answer. Be detailed and descriptive. If you cannot answer the question, say "I don't know".
    Answer: 
    """
    prompt = PromptTemplate(
        input_variables=['question', 'documents'],
        template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

def get_answer(question : str, url : str) -> str:
    """Gets an answer to a question from a Youtube video transcript"""

    db = vector_store_from_YT(url)
    chain = get_model()
    print("Generating answer...")
    docs = db.similarity_search(question, k=1)
    doc_content = " ".join([d.page_content for d in docs])

    answer = chain.run({'question': question, 'documents': doc_content})
    return answer

def main() -> None:
    global KEY

    # Parse arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question about the video", type=str)
    parser.add_argument("-u", "--url", help="URL of the video", type=str)
    parser.add_argument("-k", "--key", help="HuggingFace API key", type=str, required=False)
    args = parser.parse_args()

    # Handle key errors
    if (args.key):
        KEY = args.key
    if (KEY == None):
        print("HuggingFace API key not found. Please set the HUGGINGFACEHUB_API_TOKEN environment variable or use the -k flag.")
        return

    # Get answer
    answer = get_answer(args.question, args.url)
    print("Answer:")
    print(answer)

if __name__ == "__main__":
    main()