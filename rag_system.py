"""
RAG System - Document processing, vector storage, and agent logic
"""

import os
import datetime
from pathlib import Path
from PyPDF2 import PdfReader
from fuzzywuzzy import fuzz, process
from transformers import pipeline
import gradio as gr

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, Tool

from config import Config


class DocumentProcessor:
    """Handle PDF extraction and document processing"""
    
    def __init__(self):
        self.text_splitter = CharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separator=Config.SEPARATOR
        )
    
    @staticmethod
    def extract_pdf_text(pdf_path):
        """Extract text from PDF file"""
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    
    def process_pdfs_to_txt(self, directory):
        """Convert all PDFs in directory to TXT files"""
        for filename in os.listdir(directory):
            if filename.endswith(".pdf"):
                try:
                    pdf_path = os.path.join(directory, filename)
                    text = self.extract_pdf_text(pdf_path)
                    txt_filename = Path(filename).stem + ".txt"
                    txt_path = os.path.join(directory, txt_filename)
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(text)
                    print(f"  ✓ Extracted: {filename}")
                except Exception as e:
                    print(f"  ✗ Error: {filename} - {str(e)}")
    
    def get_txt_files(self, directory):
        """Get all text files in directory"""
        return [f for f in os.listdir(directory) if f.endswith('.txt')]
    
    def load_and_split_documents(self, directory):
        """Load and split all documents"""
        all_docs = {}
        txt_files = self.get_txt_files(directory)
        
        for txt_file in txt_files:
            try:
                filepath = os.path.join(directory, txt_file)
                loader = TextLoader(filepath, encoding="utf-8")
                docs = loader.load()
                split_docs = self.text_splitter.split_documents(docs)
                
                for doc in split_docs:
                    doc.metadata["source"] = txt_file
                
                all_docs[txt_file] = split_docs
                print(f"  ✓ Loaded: {txt_file} ({len(split_docs)} chunks)")
            except Exception as e:
                print(f"  ✗ Error loading {txt_file}: {str(e)}")
        
        return all_docs


class RAGTools:
    """Tools for ReAct agent"""
    
    def __init__(self, retrievers, txt_files):
        self.retrievers = retrievers
        self.txt_files = txt_files
        self.retrieved_text = ""
        self.current_query = ""
        self.summarizer = self._init_summarizer()
        
        # Person database for age queries
        self.person_db = {
            "Sam": {"Age": 21, "Nationality": "US"},
            "Alice": {"Age": 25, "Nationality": "UK"},
            "Bob": {"Age": 11, "Nationality": "US"}
        }
    
    def _init_summarizer(self):
        """Initialize summarization pipeline"""
        try:
            return pipeline("summarization", model="Falconsai/text_summarization")
        except Exception as e:
            print(f"  ⚠ Warning: Summarizer not available - {str(e)}")
            return None
    
    def get_relevant_document(self, name: str) -> str:
        """Retrieve relevant documents using fuzzy search"""
        best_match = process.extractOne(name, self.txt_files, scorer=fuzz.ratio)
        if not best_match:
            return "\nNo matching documents found.\n"
        
        selected_file = best_match[0]
        retriever = self.retrievers.get(selected_file)
        
        if not retriever:
            return f"\nRetriever not found for {selected_file}.\n"
        
        try:
            results = retriever.get_relevant_documents(self.current_query)
            content = "\n\nRelated document content:\n\n"
            
            for result in results[:Config.MAX_RESULTS]:
                content += result.page_content + "\n"
            
            self.retrieved_text = content
            return content
        except Exception as e:
            return f"\nError retrieving documents: {str(e)}\n"
    
    def get_summarized_text(self, name: str) -> str:
        """Summarize retrieved text"""
        if not self.summarizer:
            return "\nSummarizer not available.\n"
        
        if not self.retrieved_text:
            return "\nNo retrieved text to summarize.\n"
        
        try:
            summary = self.summarizer(
                self.retrieved_text,
                max_length=1000,
                min_length=30,
                do_sample=False
            )[0]['summary_text']
            return summary
        except Exception as e:
            return f"\nError summarizing: {str(e)}\n"
    
    def get_age(self, name: str) -> str:
        """Get person's age from database"""
        if name in self.person_db:
            age = self.person_db[name].get("Age")
            return f"\nAge: {age}\n"
        return f"\nAge information for {name} not found.\n"
    
    def get_date(self, input_str: str) -> str:
        """Get today's date"""
        return f"\n{datetime.date.today()}\n"
    
    def create_tools(self):
        """Create tool objects for agent"""
        return [
            Tool(
                name="Get Relevant document",
                func=self.get_relevant_document,
                description="Retrieve relevant documents. Input should be the document search term."
            ),
            Tool(
                name="Get Summarized Text",
                func=self.get_summarized_text,
                description="Summarize retrieved documents."
            ),
            Tool(
                name="Get Todays Date",
                func=self.get_date,
                description="Get today's date."
            ),
            Tool(
                name="Get Age",
                func=self.get_age,
                description="Get a person's age. Input should be their name."
            )
        ]


class RAGAgent:
    """ReAct-based RAG agent"""
    
    def __init__(self, retrievers, txt_files):
        self.tools_factory = RAGTools(retrievers, txt_files)
        self.model = ChatGroq(
            model_name=Config.MODEL_NAME,
            groq_api_key=Config.GROQ_API_KEY,
            temperature=Config.TEMPERATURE
        )
        self.prompt = hub.pull("hwchase17/react")
        self.tools = self.tools_factory.create_tools()
        self.agent = create_react_agent(self.model, tools=self.tools, prompt=self.prompt)
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=Config.DEBUG,
            handle_parsing_errors=True
        )
    
    def invoke(self, query: str) -> str:
        """Run agent on query"""
        self.tools_factory.current_query = query
        try:
            result = self.executor.invoke({"input": query})
            return result.get("output", "No response generated")
        except Exception as e:
            return f"Error: {str(e)}"


class RAGSystem:
    """Main RAG system that coordinates all components"""
    
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        self.collections = {}
        self.retrievers = {}
        self.txt_files = []
    
    def initialize(self, docs_directory):
        """Initialize the complete RAG system"""
        # Process PDFs
        print("  Processing PDFs...")
        self.doc_processor.process_pdfs_to_txt(docs_directory)
        
        # Load and split documents
        print("  Loading documents...")
        all_docs = self.doc_processor.load_and_split_documents(docs_directory)
        
        # Create vector stores
        print("  Creating vector stores...")
        for txt_file, docs in all_docs.items():
            if docs:
                try:
                    self.collections[txt_file] = Qdrant.from_documents(
                        docs,
                        self.embeddings,
                        location=":memory:",
                        collection_name=txt_file,
                    )
                    self.retrievers[txt_file] = self.collections[txt_file].as_retriever()
                    print(f"  ✓ Created vector store: {txt_file}")
                except Exception as e:
                    print(f"  ✗ Error creating store for {txt_file}: {str(e)}")
        
        # Get available files
        self.txt_files = self.doc_processor.get_txt_files(docs_directory)
    
    def create_agent(self):
        """Create and return the RAG agent"""
        return RAGAgent(self.retrievers, self.txt_files)
    
    def launch_gradio_ui(self, agent, share=False):
        """Launch Gradio web interface"""
        def process_query(question):
            if not question.strip():
                return "Please enter a question."
            return agent.invoke(question)
        
        available_files = ", ".join(self.txt_files) if self.txt_files else "No documents loaded"
        
        interface = gr.Interface(
            fn=process_query,
            inputs=gr.Textbox(
                label="Ask a question",
                placeholder="Enter your question...",
                lines=3
            ),
            outputs=gr.Textbox(label="Response", lines=10),
            title="Intelligent RAG System",
            description=f"""
            **Qdrant + LangChain ReAct + Llama3**
            
            Available Documents: {available_files}
            """,
            examples=[
                ["What age requirement is specified for OpenAI?"],
                ["What resources does Google offer?"],
                ["I am Bob. Will I be eligible in 2027?"],
            ],
            theme=gr.themes.Soft(),
        )
        
        interface.launch(share=share, server_name="0.0.0.0", server_port=7860)