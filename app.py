# -*- coding: utf-8 -*-
"""
Gradio Frontend for Baby Care Chatbot

This module provides a web-based interface for the baby care chatbot
using Gradio for easy deployment and user interaction.
"""

import os
import gradio as gr

from babycare.integrated_chatbot import IntegratedBabyCareChatbot
from babycare.langsmith_monitor import LangSmithMonitor

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class BabyCareGradioApp:
    """
    Gradio application for the baby care chatbot.
    
    This class manages the web interface and integrates with the
    baby care chatbot backend.
    """
    
    def __init__(self):
        """Initialize the Gradio application."""
        self.chatbot = None
        self.cost_tracker = LangSmithMonitor()
        self.total_documents = 0
        self.current_retrieved_count = 0
        
        # Initialize chatbot
        self._initialize_chatbot()
        
        # Create Gradio interface
        self.interface = self._create_interface()
    
    def _initialize_chatbot(self):
        """Initialize the baby care chatbot."""
        try:
            self.chatbot = IntegratedBabyCareChatbot(enable_monitoring=True)
            
            # Get total document count once at startup
            kb_info = self.chatbot.get_knowledge_base_info()
            self.total_documents = kb_info.get('document_count', 0)
            
            print("✅ Baby Care Chatbot initialized successfully")
            print(f"📚 Total documents in knowledge base: {self.total_documents}")
        except Exception as e:
            print(f"❌ Error initializing chatbot: {e}")
            self.chatbot = None
    
    def _load_css(self):
        """Load CSS from external file."""
        css_file_path = os.path.join(os.path.dirname(__file__), 'styles.css')
        try:
            with open(css_file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Warning: CSS file not found at {css_file_path}")
            return ""
        except Exception as e:
            print(f"Warning: Error loading CSS file: {e}")
            return ""
    
    def _create_interface(self):
        """Create the Gradio interface using gr.Blocks with careful function definitions."""
        
        # Load CSS from external file
        css = self._load_css()
        
        with gr.Blocks(css=css, title="Baby Care Chatbot", theme=gr.themes.Soft()) as interface:
            
            # Header
            gr.HTML("""
            <div class="baby-care-header">
                <h1>👶 Baby Care Chatbot</h1>
                <p>Your expert guide for baby care, nutrition, healthcare, and development</p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Chat interface
                    chatbot_interface = gr.Chatbot(
                        label="Chat with Baby Care Expert",
                        height=500,
                        show_label=True,
                        container=True,
                        elem_classes=["chat-container"]
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Ask me anything about baby care...",
                            label="Your Question",
                            lines=2,
                            scale=4
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    # Clear button
                    clear_btn = gr.Button("Clear Chat", variant="secondary")
                
                with gr.Column(scale=2):
                    # Top row: Cost monitoring and Knowledge base
                    with gr.Row():
                        with gr.Column(scale=1):
                            # Cost monitoring panel
                            with gr.Group():
                                gr.Markdown("### 💰 Cost Monitoring")
                                cost_display = gr.HTML("""
                                <div class="cost-info">
                                    <p><strong>Session Cost:</strong> $0.0000</p>
                                    <p><strong>Total Queries:</strong> 0</p>
                                    <p><strong>Total Tokens:</strong> 0</p>
                                </div>
                                """)
                                
                                refresh_cost_btn = gr.Button("Refresh Costs", size="sm")
                                export_cost_btn = gr.Button("Export Cost Data", size="sm")
                        
                        with gr.Column(scale=1):
                            # Knowledge base info
                            with gr.Group():
                                gr.Markdown("### 📚 Knowledge Base")
                                kb_info = gr.HTML("""
                                <div class="cost-info">
                                    <p><strong>Documents:</strong> Loading...</p>
                                    <p><strong>Categories:</strong> nutrition, sleep, development, safety, health</p>
                                </div>
                                """)
                    
                    # Document cards panel (full width)
                    with gr.Group():
                        gr.Markdown("### 📄 Retrieved Documents")
                        document_cards = gr.HTML("""
                        <div class="cost-info">
                            <p><em>Documents will appear here when used for answers</em></p>
                        </div>
                        """)
                    
                    # Bottom row: Quick actions and PDF upload
                    with gr.Row():
                        with gr.Column(scale=1):
                            # Quick actions
                            with gr.Group():
                                gr.Markdown("### 🚀 Quick Actions")
                                sample_questions = gr.Examples(
                                    examples=[
                                        "How much should a 3-month-old baby eat?",
                                        "What are the signs of colic?",
                                        "When should I start baby-proofing?",
                                        "What are normal sleep patterns for 6-month-olds?",
                                        "Tell me about developmental milestones"
                                    ],
                                    inputs=msg_input,
                                    label="Sample Questions"
                                )
                        
            
            # Event handlers - using simple function definitions without type annotations
            def respond(message, history):
                """Handle user input and generate response."""
                if not message.strip():
                    return "", history, "Please enter a question.", "", ""
                
                if not self.chatbot:
                    return "", history, "❌ Chatbot not initialized. Please check your configuration.", "", ""
                
                try:
                    # Generate response and get retrieval info
                    response, retrieved_count, retrieved_docs = self._get_response_with_retrieval_info(message)
                    self.current_retrieved_count = retrieved_count
                    
                    # Update chat history
                    history.append([message, response])
                    
                    # Update cost display
                    cost_summary = self.chatbot.get_cost_summary()
                    cost_html = f"""
                    <div class="cost-info">
                        <p><strong>Session Cost:</strong> ${cost_summary.get('total_cost', 0):.4f}</p>
                        <p><strong>Total Queries:</strong> {cost_summary.get('total_calls', 0)}</p>
                        <p><strong>Total Tokens:</strong> {cost_summary.get('total_tokens', 0):,}</p>
                    </div>
                    """
                    
                    # Update knowledge base info
                    kb_html = f"""
                    <div class="cost-info">
                        <p><strong>Total Documents:</strong> {self.total_documents}</p>
                        <p><strong>Used for Answer:</strong> {self.current_retrieved_count}</p>
                        <p><strong>Categories:</strong> nutrition, sleep, development, safety, health</p>
                    </div>
                    """
                    
                    # Create document cards
                    if retrieved_docs and len(retrieved_docs) > 0:
                        cards_html = self._create_document_cards(retrieved_docs)
                    else:
                        cards_html = """
                        <div class="cost-info">
                            <p><em>No documents retrieved for this answer</em></p>
                        </div>
                        """
                    
                    return "", history, cost_html, kb_html, cards_html
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    history.append([message, error_msg])
                    return "", history, "Error occurred. Please try again.", "", ""
            
            def clear_chat():
                """Clear the chat history."""
                self.current_retrieved_count = 0
                
                kb_info_html = f"""
                <div class="cost-info">
                    <p><strong>Total Documents:</strong> {self.total_documents}</p>
                    <p><strong>Used for Answer:</strong> 0</p>
                    <p><strong>Categories:</strong> nutrition, sleep, development, safety, health</p>
                </div>
                """
                
                cards_html = """
                <div class="cost-info">
                    <p><em>Documents will appear here when used for answers</em></p>
                </div>
                """
                
                return [], """
                <div class="cost-info">
                    <p><strong>Session Cost:</strong> $0.0000</p>
                    <p><strong>Total Queries:</strong> 0</p>
                    <p><strong>Total Tokens:</strong> 0</p>
                </div>
                """, kb_info_html, cards_html
            
            def refresh_costs():
                """Refresh the cost display."""
                if not self.chatbot:
                    return "❌ Chatbot not available"
                
                cost_summary = self.chatbot.get_cost_summary()
                return f"""
                <div class="cost-info">
                    <p><strong>Session Cost:</strong> ${cost_summary.get('total_cost', 0):.4f}</p>
                    <p><strong>Total Queries:</strong> {cost_summary.get('total_calls', 0)}</p>
                    <p><strong>Total Tokens:</strong> {cost_summary.get('total_tokens', 0):,}</p>
                </div>
                """
            
            def export_costs():
                """Export cost data."""
                if not self.chatbot:
                    return "❌ Chatbot not available"
                
                try:
                    filename = self.chatbot.export_cost_data()
                    if filename:
                        return f"✅ Cost data exported to: {filename}"
                    else:
                        return "❌ Failed to export cost data"
                except Exception as e:
                    return f"❌ Error exporting: {str(e)}"
            
            def update_kb_info():
                """Update knowledge base information."""
                return f"""
                <div class="cost-info">
                    <p><strong>Total Documents:</strong> {self.total_documents}</p>
                    <p><strong>Used for Answer:</strong> 0</p>
                    <p><strong>Categories:</strong> nutrition, sleep, development, safety, health</p>
                </div>
                """
            # Connect event handlers
            msg_input.submit(respond, [msg_input, chatbot_interface], [msg_input, chatbot_interface, cost_display, kb_info, document_cards])
            send_btn.click(respond, [msg_input, chatbot_interface], [msg_input, chatbot_interface, cost_display, kb_info, document_cards])
            clear_btn.click(clear_chat, outputs=[chatbot_interface, cost_display, kb_info, document_cards])
            refresh_cost_btn.click(refresh_costs, outputs=cost_display)
            export_cost_btn.click(export_costs, outputs=gr.Textbox(visible=False))

            
            # Initialize knowledge base info
            interface.load(update_kb_info, outputs=kb_info)
        
        return interface
    
    def _get_response_with_retrieval_info(self, message):
        """Get response from chatbot and track number of retrieved documents."""
        try:
            # Use the new method that properly handles both RAG and conversation flow
            conversation_id = "gradio_session"
            response, retrieved_count, retrieved_docs = self.chatbot.stream_chat_with_retrieval_info(message, conversation_id)
            
            return response, retrieved_count, retrieved_docs
            
        except Exception as e:
            # Fallback to regular chat if the new method fails
            response = self.chatbot.chat(message)
            return response, 0, []
    
    def _create_document_cards(self, documents):
        """Create HTML cards for displaying retrieved documents."""
        if not documents:
            return """
            <div class="cost-info">
                <p><em>No documents retrieved</em></p>
            </div>
            """
        
        cards_html = '<div class="document-cards-container">'
        
        for i, doc in enumerate(documents, 1):
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            
            # Truncate content for display
            display_content = content[:500] + "..." if len(content) > 500 else content
            
            # Get source information
            source = metadata.get('source', 'Unknown')
            category = metadata.get('category', 'General')
            
            # Create card HTML
            card_html = f"""
            <div class="document-card">
                <div class="document-content">
                    <strong>Document {i}:</strong><br>
                    {display_content}
                </div>
                <div class="document-source">
                    📄 Source: {source} | Category: {category}
                </div>
            </div>
            """
            
            cards_html += card_html
        
        cards_html += '</div>'
        return cards_html
    
    def launch(self, share=False, server_name="0.0.0.0", server_port=7860):
        """Launch the Gradio application."""
        if not self.interface:
            print("❌ Interface not created")
            return
        
        print("🚀 Launching Baby Care Chatbot Web Interface...")
        print(f"📱 Local URL: http://{server_name}:{server_port}")
        
        self.interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            show_error=True,
            quiet=False
        )


def main():
    """Main function to run the Gradio application."""
    print("👶 Baby Care Chatbot - Gradio Frontend")
    print("=" * 50)
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenAI API key.")
        return
    
    try:
        # Create and launch the app
        app = BabyCareGradioApp()
        app.launch(share=False)
        
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        raise
        print("Please check your configuration and try again.")


if __name__ == "__main__":
    main()