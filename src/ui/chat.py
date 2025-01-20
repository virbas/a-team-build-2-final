"""Render the chat part of the app"""
import streamlit as st
import logging;

logger = logging.getLogger(__name__)


print ()
    
def render_styles():
    st.markdown("""
    <style>    
    .st-key-messages-container img {
        background-color: #fff
    }
    
    .st-key-user_input {
        position: fixed;
        z-index: 10;
        background-color: rgb(255, 75, 75);
        bottom: 0;
        padding: 10px;
        margin-left: -10px;
        box-sizing: content-box;

        
        border: none;
        border-top: 1px solid #ddd;
    }
    
    .user_message_container {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        margin-bottom:10px;
    }

    .user_message_styles {
        background-color: #303030;
        padding: 10px;
        border-radius: 10px
    }

    .ai_message_container {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        margin-bottom: 10px;
    }

    .ai_message_styles {
        padding: 10px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)


                        
def render():
    from langchain_core.messages import HumanMessage, AIMessage
    from langgraph.errors import GraphRecursionError

    import app
    from agents.document_processor import get_overview
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        
    if "state" not in st.session_state:
        st.session_state["state"] = None
        
    if "documents_summary" not in st.session_state:
        st.session_state["summary"] = get_overview().replace("\n", "<br />");

    def handle_user_input():
        if "user_input" not in st.session_state:
            return
        
        user_input = st.session_state.user_input.strip()
        
        if not user_input:
            return
        
        st.session_state.user_input = ""  # Clear input field after submitting
        st.session_state["messages"].append(HumanMessage(user_input))
        
        state = app.query(user_input, st.session_state["state"])
        
        # TODO: handle recursion error
        if isinstance(state, GraphRecursionError):
            st.session_state["messages"].append(AIMessage("Retrieval failed, please try again in a moment"))
        else:
            # Take one before the last message, since the last message is the vizualizer, where we don't care about the content    
            analyst_message = AIMessage(state["messages"][-1].content, images=state["images_or_error"])
            st.session_state["messages"].append(analyst_message)
            st.session_state["state"] = state
        

    # Description or instructions
    st.html(f"Knowledge base summary: <br />{st.session_state["summary"]}<hr />")
    st.text_input("Ask a question:", placeholder="Type your question here...", key="user_input", on_change=handle_user_input())
     
    with st.container(key="messages-container"):
        for message in st.session_state["messages"]:
            if isinstance(message, HumanMessage):
                st.markdown(f"""
                            <div class="user_message_container">
                                <div class="user_message_styles">
                                    <strong>**You**</strong><br />{message.content}
                                </div>
                            </div>""", unsafe_allow_html=True,
                        )
            else:
                st.markdown(f"""
                            <div class="ai_message_container">
                                <div class="ai_message_styles">
                                    <strong>**Assistant**</strong><br />{message.content}
                                </div>
                            </div>""", unsafe_allow_html=True)
                

            if hasattr(message, "images") and isinstance(message.images, list):
                for image in message.images:
                    st.image(image)
                    
                    
    render_styles()
    
                    
    