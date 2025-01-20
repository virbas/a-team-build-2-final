import streamlit as st
import ui
import ui.chat
import ui.upload;

# Title of the Streamlit app
st.title("iii.org historical data knowledge base")

tabs = st.tabs(["Chat", "Upload"])
# Tab 1: Home Page
with tabs[0]:
    ui.chat.render()

# Tab 2: About Page
with tabs[1]:
   ui.upload.render()