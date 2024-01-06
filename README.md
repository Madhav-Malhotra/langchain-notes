### LangChain Notes

These are a series of scripts that show off various features of LangChain. 
They're useful for me as **a reference to look back on**. If you're learning about LangChain for the first time, I'd instead recommend my [Intro to LangChain](https://docs.google.com/document/d/1D9kfjytOPmmVor2TjIY-LiUrCvEWILbXci5rqbgR84w/edit) instead.

**Description of files**
- `l1_setup.py`: shows how to use the Hugging Face inference API with manual network requests.
- `l2_chains.py`: shows how to use LangChain `PromptTemplate` and `langchain.llms` together.
- `l3_streamlit.py`: shows how to create a quick frontend with Python only to share your models with others.
- `l4_agents.py`: shows how to give models access to external tools like search engines, wikipedia, and calculators.
- `l5_loaders.py`: shows how to load external documents like Youtube transcripts.
- `l5_loader_chain.py`: shows how to use prebuilt load summarise chains with news article loaders. 
- `l6_frontend.py` and `l6_vectorstore.py`: example app with a streamlit frontend and FAISS vectorstore that answers questions about a Youtube video using its transcript. 