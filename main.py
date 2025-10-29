"""
Main entry point for RAG Application
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

from config import Config
from rag_system import RAGSystem


class RAGApplication:
    """Main application controller"""
    
    def __init__(self):
        """Initialize application"""
        Config.validate()
        self.system = RAGSystem()
        self.agent = None
    
    def setup(self):
        """Setup the RAG system"""
        print("\n" + "=" * 60)
        print("RAG APPLICATION SETUP")
        print("=" * 60)
        
        try:
            print("\n[1/2] Processing and loading documents...")
            self.system.initialize(Config.DOCS_DIRECTORY)
            
            print("\n[2/2] Initializing RAG agent...")
            self.agent = self.system.create_agent()
            
            print("\n✓ Setup complete!\n")
            return True
        except Exception as e:
            print(f"\n✗ Setup failed: {str(e)}\n")
            import traceback
            traceback.print_exc()
            return False
    
    def run_cli(self):
        """Interactive CLI mode"""
        print("\n" + "=" * 60)
        print("INTERACTIVE MODE (type 'quit' to exit)")
        print("=" * 60 + "\n")
        
        while True:
            try:
                query = input("\n> ").strip()
                if query.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                if query:
                    print("\nProcessing...\n")
                    response = self.agent.invoke(query)
                    print(f"Response:\n{response}")
            except KeyboardInterrupt:
                print("\nTerminated by user.")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
    
    def run_ui(self, share=False):
        """Web UI with Gradio"""
        print("\n" + "=" * 60)
        print("LAUNCHING WEB UI")
        print("=" * 60 + "\n")
        
        self.system.launch_gradio_ui(self.agent, share=share)
    
    def run_demo(self):
        """Run demo queries"""
        print("\n" + "=" * 60)
        print("DEMO MODE")
        print("=" * 60 + "\n")
        
        queries = [
            "What age requirement is specified for using OpenAI Services?",
            "What resources does Google offer to users?",
            "I am Bob. Will I be eligible in 2027 for the age requirement?",
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n[Query {i}] {query}")
            print("-" * 60)
            response = self.agent.invoke(query)
            print(f"Response:\n{response}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="RAG Application with Qdrant, LangChain ReAct, Llama3"
    )
    parser.add_argument(
        "--mode",
        choices=["cli", "ui", "demo"],
        default="ui",
        help="Run mode: cli (interactive), ui (web), demo"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Share Gradio UI with public link"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("INTELLIGENT RAG SYSTEM")
    print("Qdrant + LangChain ReAct + Llama3 from Groq")
    print("=" * 60)
    
    app = RAGApplication()
    
    if not app.setup():
        sys.exit(1)
    
    try:
        if args.mode == "cli":
            app.run_cli()
        elif args.mode == "ui":
            app.run_ui(share=args.share)
        elif args.mode == "demo":
            app.run_demo()
    except KeyboardInterrupt:
        print("\n\nApplication terminated.")
        sys.exit(0)


if __name__ == "__main__":
    main()