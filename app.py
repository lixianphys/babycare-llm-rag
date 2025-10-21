"""
Gradio Frontend for Baby Care Chatbot
"""

import os
import gradio as gr

from babycare.integrated_chatbot import IntegratedBabyCareChatbot

from dotenv import load_dotenv
import logging
load_dotenv()

logger = logging.getLogger(__name__)


STORE_CONVERSATION = False

class BabyCareGradioApp:
    """
    Gradio application for the baby care chatbot.
    
    Manages the web interface and integrates with the
    baby care chatbot backend.
    """
    
    def __init__(self):
        self.chatbot = None
        self.total_documents = 0
        self.current_retrieved_count = 0
        
        # Initialize chatbot
        self._initialize_chatbot()
        
        # Create Gradio interface
        self.interface = self._create_interface()
    
    def _initialize_chatbot(self):
        try:
            self.chatbot = IntegratedBabyCareChatbot(
                enable_monitoring=True,
                enable_conversation_storage=STORE_CONVERSATION,
                conversation_db_path="gradio_conversations.json"
            )
            
            # Get total document count once at startup
            kb_info = self.chatbot.get_knowledge_base_info()
            self.total_documents = kb_info.get('document_count', 0)
            
            logger.info("Baby Care Chatbot initialized successfully")
            logger.info(f"Total documents in knowledge base: {self.total_documents}")
            logger.info("Conversation storage enabled")
        except Exception as e:
            logger.error(f"Error initializing chatbot: {e}")
            self.chatbot = None
    
    def _load_css(self):
        css_file_path = os.path.join(os.path.dirname(__file__), 'styles.css')
        try:
            with open(css_file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"CSS file not found at {css_file_path}")
            return ""
        except Exception as e:
            logger.warning(f"Error loading CSS file: {e}")
            return ""
    
    def _create_interface(self):
        """Create the Gradio interface using gr.Blocks."""
        
        # Load CSS from external file
        css = self._load_css()
        
        with gr.Blocks(css=css, title="Baby Care Chatbot", theme=gr.themes.Soft()) as interface:
            
            # Header
            gr.HTML("""
            <div class="baby-care-header">
                <h1>Baby Care Chatbot</h1>
                <p>Your expert guide for pregnancy, baby care, nutrition, healthcare, and development. Data used for RAG system is mainly from https://www.medicinesinpregnancy.org and https://mothertobaby.org. This chatbot is not a medical professional and should not be used as a substitute for professional medical advice.</p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Chat interface
                    chatbot_interface = gr.Chatbot(
                        label="Chat with the baby care expert",
                        type="tuples",
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
                    # Top row: Cost monitoring and knowledge base
                    with gr.Row():                        
                        with gr.Column(scale=1):
                            # Knowledge base info
                            with gr.Group():
                                gr.Markdown("### Knowledge Base")
                                kb_info = gr.HTML("""
                                <div class="cost-info">
                                    <p><strong>Documents:</strong> Loading...</p>
                                    <p><strong>Categories:</strong> nutrition, sleep, development, safety, health</p>
                                </div>
                                """)
                    
                    # Document cards panel (full width)
                    with gr.Group():
                        gr.Markdown("### Retrieved Documents")
                        document_cards = gr.HTML("""
                        <div class="cost-info">
                            <p><em>Documents will appear here when used for answers</em></p>
                        </div>
                        """)
                    
                    # Bottom row: Frequently Asked Questions
                    with gr.Row():  
                        with gr.Column(scale=1):
                            with gr.Group():
                                gr.Markdown("### Frequently Asked Questions")
                                sample_questions = gr.Examples(
                                    examples=[
                                        "should I drink coffee during pregnancy?",
                                        "How will albuterol affect pregnancy and breastfeeding?",
                                        "What are the signs of colic in babies?",
                                        "Does a COVID-19 infection have any negative effect on babies?",
                                    ],
                                    inputs=msg_input,
                                    label="Ask a question"
                                )
                        
            
            # Event handlers
            def respond(message, history):
                """Handle user input and generate response."""
                if not message.strip():
                    return "", history, "Please enter a question.", "", ""
                
                if not self.chatbot:
                    return "", history, " Chatbot not initialized. Please check your configuration.", "", ""
                
                try:
                    # Generate response and get retrieval info
                    response, retrieved_count, retrieved_docs = self._get_response_with_retrieval_info(message)
                    self.current_retrieved_count = retrieved_count
                    
                    # Update chat history
                    history.append([message, response])
                    
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
                    
                    return "", history, kb_html, cards_html
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    history.append([message, error_msg])
                    return "", history, "Error occurred. Please try again.", "", ""
            
            def clear_chat():
                """Clear the chat history and reset the knowledge base info and document cards."""
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
                
                return [], kb_info_html, cards_html
            
            # def refresh_costs():
            #     """Refresh the cost display."""
            #     if not self.chatbot:
            #         return "Chatbot not available"
                
            #     cost_summary = self.chatbot.get_cost_summary()
            #     return f"""
            #     <div class="cost-info">
            #         <p><strong>Session Cost:</strong> ${cost_summary.get('total_cost', 0):.4f}</p>
            #         <p><strong>Total Queries:</strong> {cost_summary.get('total_calls', 0)}</p>
            #         <p><strong>Total Tokens:</strong> {cost_summary.get('total_tokens', 0):,}</p>
            #     </div>
            #     """
            
            # def export_costs():
            #     """Export cost data."""
            #     if not self.chatbot:
            #         return "Chatbot not available"
                
            #     try:
            #         filename = self.chatbot.export_cost_data()
            #         if filename:
            #             return f" Cost data exported to: {filename}"
            #         else:
            #             return "Failed to export cost data"
            #     except Exception as e:
            #         return f"Error exporting: {str(e)}"
            
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
            msg_input.submit(respond, inputs=[msg_input, chatbot_interface], outputs=[msg_input, chatbot_interface, kb_info, document_cards])
            send_btn.click(respond, inputs=[msg_input, chatbot_interface], outputs=[msg_input, chatbot_interface, kb_info, document_cards])
            clear_btn.click(clear_chat, outputs=[chatbot_interface, kb_info, document_cards])
            # refresh_cost_btn.click(refresh_costs, outputs=cost_display)
            # export_cost_btn.click(export_costs, outputs=gr.Textbox(visible=False))

            # Initialize knowledge base info
            interface.load(update_kb_info, outputs=kb_info)
        
        return interface
    
    def _get_response_with_retrieval_info(self, message):
        """Get response from chatbot and track number of retrieved documents."""
        try:
            conversation_id = "gradio_session"
            user_id = "gradio_user"  # You can make this dynamic based on session/user
            
            # Use the integrated method that handles both conversation storage and retrieval info
            response, retrieved_count, retrieved_docs = self.chatbot.stream_chat_with_retrieval_info(
                message, conversation_id, user_id
            )
            
            return response, retrieved_count, retrieved_docs
            
        except Exception as e:
            # Fallback to regular chat if the new method fails
            logger.error(f"Error getting response with retrieval info: {e}. Falling back to regular chat.")
            conversation_id = "gradio_session"
            user_id = "gradio_user"
            response = self.chatbot.chat(message, conversation_id, user_id)
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
            logger.error("Interface not created")
            return
        
        logger.info("Launching Baby Care Chatbot Web Interface...")
        logger.info(f"Local URL: http://{server_name}:{server_port}")
        
        try:
            self.interface.launch(
                share=share,
                server_name=server_name,
                server_port=server_port,
                show_error=True,
                quiet=False
            )
        except KeyboardInterrupt:
            logger.info("Shutting down application...")
            self._cleanup()
        except Exception as e:
            logger.error(f"Error during application launch: {e}")
            self._cleanup()
    
    def _cleanup(self):
        """Clean up resources when the application shuts down."""
        if self.chatbot and self.chatbot.conversation_storage:
            self.chatbot.conversation_storage.close()
            logger.info("Conversation storage closed")


def main():
    """Main function to run the Gradio application."""
    logger.info("Baby Care Chatbot - Gradio Frontend")
    print("=" * 50)
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment variables.")
        logger.info("Please create a .env file with your OpenAI API key.")
        return
    
    try:
        # Create and launch the app
        app = BabyCareGradioApp()
        app.launch(share=False)
        
    except Exception as e:
        logger.error(f"Error launching application: {e}")
        logger.info("Please check your configuration and try again.")


if __name__ == "__main__":
    main()