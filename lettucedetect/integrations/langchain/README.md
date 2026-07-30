# LettuceDetect + LangChain Integration

Real-time hallucination detection for RAG pipelines.

## Installation

```bash
pip install lettucedetect
pip install langchain langchain-openai langchain-community langchain-chroma
export OPENAI_API_KEY=your_key
```

## Usage

```python
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from lettucedetect.integrations.langchain import stream_with_detection

# Set up your RAG pipeline
documents = ["Your documents here..."]
embeddings = OpenAIEmbeddings()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.create_documents(documents)
vectorstore = Chroma.from_documents(docs, embeddings)

# Create streaming RAG chain
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# Get context and stream with detection
question = "Your question here"
context = [doc.page_content for doc in vectorstore.similarity_search(question, k=3)]

# Stream tokens and hallucination detection in real-time
for event in stream_with_detection(chain, {"query": question}, context, check_every=10):
    if event["type"] == "token":
        print(event["content"], end="", flush=True)  # Stream response
    elif event["type"] == "detection" and event["has_issues"]:
        print(f"\nHallucination detected: {event['issue_count']} issues")
        # Handle detection - log, alert, stop generation, etc.
```

## Direct Callback Usage

For more control, use `LettuceStreamingCallback` directly:

```python
from lettucedetect.integrations.langchain import LettuceStreamingCallback

# Create callback with your settings
callback = LettuceStreamingCallback(
    context=context,
    question=question,
    check_every=10,
    method="transformer",  # or "rag_fact_checker"
)

# Use with any LangChain chain
result = chain.invoke({"query": question}, config={"callbacks": [callback]})

# Stream events as they arrive
for event in callback.stream_events():
    if event["type"] == "token":
        print(event["content"], end="")
    elif event["type"] == "detection":
        handle_detection(event)
```

## What You Get

**Token Events**: Real-time text as it's generated  
**Detection Events**: Hallucination analysis with confidence scores and exact spans

Each detection event includes:
- `has_issues`: Boolean if hallucinations found
- `issue_count`: Number of problematic spans  
- `confidence`: Detection confidence (0-1)
- `spans`: Array of problematic text spans with positions

## Live Demo

See it in action:
```bash
streamlit run lettucedetect/integrations/langchain/examples/streamlit_app.py
python lettucedetect/integrations/langchain/examples/rag_example.py
```

Perfect for building streaming chat apps, real-time APIs, and production RAG systems with automatic quality control.