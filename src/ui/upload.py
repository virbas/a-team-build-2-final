"""Render file uploading tab"""
import os
import streamlit as st
import tempfile

_CONST_TMP_FILE_DIR = "data/tmp_files"

def render():
# Set the app title
    from agents import document_processor 

    st.title("Multiple File Upload in Streamlit")

    # File uploader with multiple file support
    uploaded_files = st.file_uploader(
        "Upload your files (you can select multiple)", 
        accept_multiple_files=True
    )

    # print(uploaded_files)
    
    if uploaded_files and len(uploaded_files):
        all_insert_results = [];
        for file in uploaded_files:
            filecontent = file.getvalue().decode("utf-8")

            name, extension = os.path.splitext(file.name)
            
            # Generated file name will be original_name_XXXXX.ext
            with tempfile.NamedTemporaryFile(dir=os.path.join(_CONST_TMP_FILE_DIR), prefix=name + "_", suffix=extension, delete=True, mode="w") as temp_file:
                temp_file.write(filecontent)
                temp_file.flush()
                
                insert_result_or_error = document_processor.insert_file_into_vector(temp_file.name)
                    
                if isinstance(insert_result_or_error, list):
                    all_insert_results.extend(insert_result_or_error)
                else:
                    all_insert_results.append(insert_result_or_error)
                
        document_processor.update_overview()
        
        st.session_state["summary"] = document_processor.get_overview().replace("\n", "<br />");
        
        st.html("<br />".join(all_insert_results))
            
                
            
    
            
    