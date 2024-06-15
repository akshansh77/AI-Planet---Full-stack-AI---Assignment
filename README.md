
Multiple Document Chat Q/A Interface
Overview
This project is an advanced application that combines ChromaDB vector stores, Sentence Transformer embeddings, and the Llama-7 large language model hosted on Replicate to offer a sophisticated question-and-answer experience across multiple documents. ChromaDB vector stores efficiently manage document representations, while Sentence Transformer embeddings enhance the semantic understanding of text. Llama-7, with its state-of-the-art natural language processing capabilities, interprets user queries accurately, providing comprehensive responses. The user-friendly Streamlit deployment ensures a seamless and visually appealing experience, making this interface a powerful tool for precise information retrieval from diverse textual sources.

Features
Multi-format Document Support: Supports PDF, DOC, DOCX, TXT, JPG, JPEG, PNG, CSV files.
Text Extraction: Uses OCR for extracting text from image files.
Semantic Search: Utilizes Sentence Transformer for embedding and semantic search.
Vector Storage: Embeddings stored in ChromaDB for efficient retrieval.
Contextual Compression: Uses Cohere Reranking for contextual compression to return relevant information.
Prevention of Model Hallucination: Prompt engineering to ensure responses stay within the vector database scope.
Diverse Response Generation: Utilizes Maximum Marginal Relevance (MMR) to reduce redundancy and maintain query relevance.
Dataset
The dataset is a collection of text files of Amazon Web Services (AWS) case studies and blog articles related to Generative AI and Large Language Models. The dataset was obtained from Kaggle and is used to train RAG pipelines.

Kaggle Dataset: AWS Case Studies and Blogs

Installation
Clone the Repository

bash
Copy code
git clone <repository-url>
cd <repository-directory>
Install Required Libraries

bash
Copy code
pip install -r requirements.txt
Set Environment Variables

Create a .env file in the root directory of the project and add your environment variables for Replicate, HuggingFace, and Cohere.

plaintext
Copy code
REPLICATE_API_TOKEN=<your-replicate-api-token>
HUGGINGFACE_API_TOKEN=<your-huggingface-api-token>
COHERE_API_TOKEN=<your-cohere-api-token>
Launch the Streamlit Interface

bash
Copy code
streamlit run app.py
Components
Embedding Model
Sentence Transformer: all-MiniLM-L6-v2
Maps sentences & paragraphs to a 384-dimensional dense vector space.
Ideal for tasks like clustering or semantic search.
Based on transformer networks like BERT, RoBERTa, and XLM-RoBERTa.
Vector Database
ChromaDB
Lightweight in-memory DB.
Supports real-time updating of the vector database.
Retrieves metadata along with embeddings for document sources.
Retrieval Process
Loading: Converts various file formats to text.
Splitting: Uses CharacterTextSplitter to break text into 1000-character chunks with 100-character overlap.
Embedding: Uses HuggingFace embedding model to create embeddings from text data.
RAG Techniques
Cohere Reranking: Improves result relevancy by contextual compression.
Prompt Engineering: Prevents LLM hallucination by ensuring responses stay within the provided vector data.
MMR (Maximum Marginal Relevance): Reduces redundancy and maintains query relevance.
Further Improvements
Self-Querying: Enhance accuracy.
LLAMA2 Fine-Tuning: Adapt to specific domains.
MapReduce: Improve performance for large document sets.
Hypothetical Document Embeddings: Test future iterations.
Source Document Return: Utilize ConversationRetrievalChain hyperparameter.
Multimedia Input: Include YouTube videos and other links.
RAGAs Evaluation Scheme: Add to RAG pipeline for improved assessment.
Sources
RAG Optimization
Retrieval with LLAMA2
License
MIT License
