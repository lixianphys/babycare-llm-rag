"""
Gradio Frontend for Baby Care Chatbot
"""

import os
import gradio as gr

from babycare.integrated_chatbot import IntegratedBabyCareChatbot

from dotenv import load_dotenv
import logging
load_dotenv()


VERSION = "0.1.0"

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
            logger.info("Total documents in knowledge base: {}".format(self.total_documents))
            if STORE_CONVERSATION:
                logger.info("Conversation storage enabled")
            else:
                logger.info("Conversation storage disabled")
        except Exception as e:
            logger.error("Error initializing chatbot: {}".format(e))
            self.chatbot = None
    
    def _load_css(self):
        css_file_path = os.path.join(os.path.dirname(__file__), 'styles.css')
        try:
            with open(css_file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.warning("CSS file not found at {}".format(css_file_path))
            return ""
        except Exception as e:
            logger.warning("Error loading CSS file: {}".format(e))
            return ""
    
    def _create_interface(self):
        """Create the Gradio interface using gr.Blocks."""
        
        # Load CSS from external file
        css = self._load_css()
        
        with gr.Blocks(css=css, title="Baby Care Assistant", theme=gr.themes.Default()) as interface:
            
            # Main layout with information panel and content area
            with gr.Row(elem_classes=["main-container"]):
                # Left information panel
                with gr.Column(scale=1, elem_classes=["info-panel"]):
                    # Title section with icon and name
                    gr.HTML("""
                    <div class="title-section">
                        <div class="title-icon">💬</div>
                        <div class="title-text">
                            <div class="app-name">Baby Care</div>
                            <div class="app-subtitle">Assistant</div>
                        </div>
                        <div class="version-tag">v{}</div>
                    </div>
                    """.format(VERSION))
                    
                    # Description
                    gr.HTML("""
                    <div class="description">
                        Baby Care Assistant is optimized for pregnancy, baby care, nutrition, and child development questions. 
                        Developed with medical literature from MotherToBaby.org and MedicinesInPregnancy.org.
                    </div>
                    """)
                
                # Main content area
                with gr.Column(scale=2, elem_classes=["main-content"]):
                    
                    # Examples section
                    gr.HTML('<div class="examples-header">Examples</div>')
                    
                    # Example buttons grid (reduced to 6 examples)
                    with gr.Row(elem_classes=["examples-grid"]):
                        with gr.Column():
                            example1 = gr.Button("What should I eat during pregnancy?", elem_classes=["example-btn"])
                            example2 = gr.Button("How to soothe a crying baby?", elem_classes=["example-btn"])
                        with gr.Column():
                            example3 = gr.Button("Is it safe to take medication while breastfeeding?", elem_classes=["example-btn"])
                            example4 = gr.Button("What are the signs of colic in babies?", elem_classes=["example-btn"])
                        with gr.Column():
                            example5 = gr.Button("How to establish a sleep routine for newborns?", elem_classes=["example-btn"])
                            example6 = gr.Button("What vaccines are safe during pregnancy?", elem_classes=["example-btn"])
                    
                    # Chat interface (always visible)
                    chatbot_interface = gr.Chatbot(
                        value=[],
                        type="tuples",
                        height=300,
                        show_label=False,
                        container=True,
                        elem_classes=["chat-container"],
                        visible=True
                    )
                    
                    # Input field
                    msg_input = gr.Textbox(
                        placeholder="Ask anything",
                        show_label=False,
                        lines=1,
                        elem_classes=["main-input"]
                    )
                    
                    # Footer
                    gr.HTML("""
                    <div class="footer">
                        <span class="footer-warning">Generated content may be inaccurate or false.</span>
                    </div>
                    """)
                    
                    # Knowledge base info (always visible)
                    with gr.Column(elem_classes=["kb-info"], visible=True) as kb_section:
                        kb_info = gr.HTML("""
                        <div class="kb-display">
                            <p><strong>Knowledge Base:</strong> {self.total_documents} documents</p>
                            <p><strong>Used for Answer:</strong> 0 documents</p>
                        </div>
                        """)
                    
                    # Document cards (always visible)
                    with gr.Column(elem_classes=["doc-cards"], visible=True) as doc_section:
                        document_cards = gr.HTML("""
                        <div class="doc-display">
                            <p><em>Retrieved documents will appear here when you ask a question</em></p>
                        </div>
                        """)
                        
            
            # Event handlers
            def respond(message, history):
                """Handle user input and generate response."""
                if not message.strip():
                    return "", history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                
                if not self.chatbot:
                    error_msg = "❌ Chatbot not initialized. Please check your configuration."
                    history.append([message, error_msg])
                    return "", history, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
                
                try:
                    # Generate response and get retrieval info
                    response, retrieved_count, retrieved_docs = self._get_response_with_retrieval_info(message)
                    self.current_retrieved_count = retrieved_count
                    
                    # Update chat history
                    history.append([message, response])
                    
                    # Update knowledge base info
                    kb_html = """
                    <div class="kb-display">
                        <p><strong>Knowledge Base:</strong> {} documents</p>
                        <p><strong>Used for Answer:</strong> {} documents</p>
                    </div>
                    """.format(self.total_documents, self.current_retrieved_count)
                    
                    # Create document cards
                    if retrieved_docs and len(retrieved_docs) > 0:
                        cards_html = self._create_document_cards(retrieved_docs)
                    else:
                        cards_html = """
                        <div class="doc-display">
                            <p><em>No documents retrieved for this answer</em></p>
                        </div>
                        """
                    
                    return "", history, gr.update(visible=True), gr.update(value=kb_html, visible=True), gr.update(visible=True), gr.update(value=cards_html, visible=True)
                    
                except Exception as e:
                    error_msg = "❌ Error: {}".format(str(e))
                    history.append([message, error_msg])
                    return "", history, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
            
            def clear_chat():
                """Clear the chat history and reset the knowledge base info and document cards."""
                self.current_retrieved_count = 0
                
                kb_info_html = """
                <div class="kb-display">
                    <p><strong>Knowledge Base:</strong> {} documents</p>
                    <p><strong>Used for Answer:</strong> 0 documents</p>
                </div>
                """.format(self.total_documents)
                
                cards_html = """
                <div class="doc-display">
                    <p><em>Retrieved documents will appear here</em></p>
                </div>
                """
                
                return [], gr.update(visible=False), gr.update(value=kb_info_html, visible=False), gr.update(visible=False), gr.update(value=cards_html, visible=False)
            
            # Connect event handlers
            msg_input.submit(respond, inputs=[msg_input, chatbot_interface], outputs=[msg_input, chatbot_interface, kb_section, kb_info, doc_section, document_cards])
            
            # Connect example button handlers
            example1.click(lambda: "What should I eat during pregnancy?", outputs=msg_input).then(respond, inputs=[msg_input, chatbot_interface], outputs=[msg_input, chatbot_interface, kb_section, kb_info, doc_section, document_cards])
            example2.click(lambda: "How to soothe a crying baby?", outputs=msg_input).then(respond, inputs=[msg_input, chatbot_interface], outputs=[msg_input, chatbot_interface, kb_section, kb_info, doc_section, document_cards])
            example3.click(lambda: "Is it safe to take medication while breastfeeding?", outputs=msg_input).then(respond, inputs=[msg_input, chatbot_interface], outputs=[msg_input, chatbot_interface, kb_section, kb_info, doc_section, document_cards])
            example4.click(lambda: "What are the signs of colic in babies?", outputs=msg_input).then(respond, inputs=[msg_input, chatbot_interface], outputs=[msg_input, chatbot_interface, kb_section, kb_info, doc_section, document_cards])
            example5.click(lambda: "How to establish a sleep routine for newborns?", outputs=msg_input).then(respond, inputs=[msg_input, chatbot_interface], outputs=[msg_input, chatbot_interface, kb_section, kb_info, doc_section, document_cards])
            example6.click(lambda: "What vaccines are safe during pregnancy?", outputs=msg_input).then(respond, inputs=[msg_input, chatbot_interface], outputs=[msg_input, chatbot_interface, kb_section, kb_info, doc_section, document_cards])
        
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
            logger.error("Error getting response with retrieval info: {}. Falling back to regular chat.".format(e))
            conversation_id = "gradio_session"
            user_id = "gradio_user"
            response = self.chatbot.chat(message, conversation_id, user_id)
            return response, 0, []
    
    def _create_document_cards(self, documents):
        """Create HTML cards for displaying retrieved documents."""
        if not documents:
            return """
            <div class="doc-display">
                <p><em>No documents retrieved</em></p>
            </div>
            """
        
        cards_html = '<div class="document-cards-container">'
        
        for i, doc in enumerate(documents, 1):
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            
            # Show more content for scrolling (up to 1000 characters)
            display_content = content[:1000] + "..." if len(content) > 1000 else content
            
            # Get source information
            source = metadata.get('source', 'Unknown')
            category = metadata.get('category', 'General')
            
            # Create card HTML
            card_html = """
            <div class="document-card">
                <div class="document-content">
                    <strong>Document {}:</strong><br>
                    {}
                </div>
                <div class="document-source">
                    📄 Source: {} | Category: {}
                </div>
            </div>
            """.format(i, display_content, source, category)
            
            cards_html += card_html
        
        cards_html += '</div>'
        return cards_html
    
    def launch(self, share=False, server_name="0.0.0.0", server_port=7860):
        """Launch the Gradio application."""
        if not self.interface:
            logger.error("Interface not created")
            return
        
        logger.info("Launching Baby Care Chatbot Web Interface...")
        logger.info("Local URL: http://{}:{}".format(server_name, server_port))
        
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
            logger.error("Error during application launch: {}".format(e))
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
        app.launch(share=False)  # Let it find an available port
        
    except Exception as e:
        logger.error("Error launching application: {}".format(e))
        logger.info("Please check your configuration and try again.")


if __name__ == "__main__":
    main()