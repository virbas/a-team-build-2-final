"""
document_preprocessor will read a textual file, llm it's summary, split and index it's embedings
After each insert, a summary of the document will be appended to the file.
After all files are uploaded (or manually called) - the summary files will be send to llm 
make a short summary of it's contents and prepare some example questions.
"""

import os

import re

from typing import List, Dict, Any, Final, Union, Annotated
from pydantic import BaseModel, Field

from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

import shutil

import filecmp

from config.models import llm
from config.db import vector_store

import logging

_logger = logging.getLogger(__name__)

DOCUMENT_STORAGE_PATH: Final[str] = "data/storage/"

_CONST_DOCUMENT_SUMMARY_FILE = "documents_summary.txt"


def _normalize_filename(filename):
    filename = filename.strip()
    name, ext = os.path.splitext(filename)
    name = name.replace(" ", "_")
    name = re.sub(r"[^a-zA-Z0-9._-]", "", name)
    return f"{name}{ext}"

def get_source_contents(source_path: str) -> str:
    """Retrieves the original document contents
        Args:
            source: file name with extension
    """
    
    with open(os.path.join(DOCUMENT_STORAGE_PATH, _normalize_filename(source_path.strip())), encoding="utf-8") as f:
        return f.read()
    
def _check_if_file_exists_in_store(file_path: str):
    """Check if file already exists in the file store.
    File is compared by contents.
    
    Args:
        path: str - full path to the file
        
    """
    
    normalized_file_name=_normalize_filename(os.path.basename(file_path));
    if(os.path.exists(os.path.join(DOCUMENT_STORAGE_PATH, normalized_file_name))):
        return True
    
    existing_files = [os.path.join(DOCUMENT_STORAGE_PATH, file) for file in os.listdir(DOCUMENT_STORAGE_PATH)]
    for existing_file in existing_files:
        if filecmp.cmp(file_path, existing_file):
            return True
    
    return False

class _DataDocument(BaseModel):
    data_rows: str = Field(description="""Extracted rows data from the document""")
    summary: str = Field(description="""A short summary about the file contents""")
    

_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

def insert_file_into_vector(file_path: str) -> Union[List[str], str]:
        """
        Reads file contents, checks if it's not already vectorized (by comparing file contents with the already existing ones),
        Passes it through llm to prepare for the vector store,
        splits the llm result if the result is too long
        and eventually saves it to the vectore store.
        
        It also saves the summary of the file to the tracked summary location
        which maintains a short summary of all the documents together. Check @update_overview
        
        splits it with the
        Args:
            file_path: Path to the textual file

        Returns:
            List of ids inserted into the vectore store or Error string if something failed 
        """
        
        _logger.info(f"Loading Excel file: {file_path}")
        
        if _check_if_file_exists_in_store(file_path):
            return f"File insert error: {file_path} was already indexed."

        normalized_name = _normalize_filename(os.path.basename(file_path))
        shutil.copyfile(file_path, os.path.join(DOCUMENT_STORAGE_PATH, normalized_name))
        
        file_contents = get_source_contents(normalized_name);
        
        llm_for_document_summary = llm.with_structured_output(_DataDocument);
        
        prompt = f"""This file content contains some texts and a data table.
                                            Extract the data in separate rows, where each row starts with the free texts and adds a description of data table row:
                                            At the end, summarize file contents as short as possible. Return summary right away. No need for introductions.

                                            For example:
                                            "
                                                title

                                                data-header-1, data-header-2
                                                data-row-1-col-1, data-row-1-col-2
                                                data-row-2-col-1, data-row-2-col-2

                                                some remarks
                                            "

                                            Should be extracted into separate lines as:
                                            
                                            "
                                            extracted data:
                                                Title, remarks, explanation of data-row-1
                                                Title, remarks, explanation of data-row-2
                                            
                                            summary:
                                                Information about data-header-1, data-header-2  
                                            "

                                            Here are the contents:{file_contents}"""
        
        try:
            response = llm_for_document_summary.invoke(prompt)
        except Exception as e:
            return f"Failed parsing {normalized_name} with with an error: " + str(e)
            
        document = Document(
            page_content=response.data_rows,
            metadata={"source": normalized_name},
        )

        splitted_docs = _text_splitter.split_documents([document])
        try:
            inserted_ids = vector_store.add_documents(splitted_docs)
        except Exception as e:
            return f"Failed indexing {normalized_name} with with an error: " + str(e)
        
        with open(_CONST_DOCUMENT_SUMMARY_FILE, 'a+') as file:
            file.write(response.summary)
        
        return inserted_ids;
    

def update_overview():
    f"""We maintain a short summary (llm summarized) of all documents saved in the vector store.
    Mainly to give the user some overview of what could be the topics to ask.
    
    This function can be called when new files are added for example.
    """
    
    with open(_CONST_DOCUMENT_SUMMARY_FILE, 'r') as file:
        contents = file.read();
        
    result = llm.invoke(f"""Make a summary about the text. No introductions.
                        Just start with "Data about..." and keep it short.
                        
                        Add few example questions on what can be asked about this text.
                        Here are the contents: {contents}""")
    
    with open(_CONST_DOCUMENT_SUMMARY_FILE, 'w') as file:
        file.write(result.content)
        
def get_overview() -> str:
    """retrieves a cached short summary of currently known documents"""
    with open(_CONST_DOCUMENT_SUMMARY_FILE, 'r') as file:
        contents = file.read()
        return contents or "Knowledge base currently is empty"


class DocumentProcessor:
    def __init__(self, llm: BaseChatModel):
        
        """Initialize document processor with a language model.
        Since we are dealing with the structured text, but different formats (.xls, .html)
        we will use the llm to summarize file contents for us.

         Args:
            llm: llm capable of structured output.
        """
        
        self.llm = llm

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def load_file(self, file_path: str) -> List[Document]:
        """
        TODO: Load Excel file using UnstructuredExcelLoader

        Args:
            file_path: Path to the Excel file

        Returns:
            List of Document objects
        """
        _logger.info(f"Loading Excel file: {file_path}")

        file_contents = get_source_contents(file_path);
        response = self.llm.invoke(f"""This file content contains some texts and a data table.
                                            Extract the data in separate rows, where each row starts with the free texts and adds a description of data table row:
                                            Summarize file contents. Just dump the summary right away. No need for introductions.

                                            For example:
                                            "title

                                            data-header-1, data-header-2
                                            data-row-1-col-1, data-row-1-col-2
                                            data-row-2-col-1, data-row-2-col-2

                                            some remarks"

                                            Should be extracted into separate lines as:
                                            Title, remarks, explanation of data-row-1
                                            Title, remarks, explanation of data-row-2

                                            Here are the contents:{file_contents}""")
        document = Document(
            page_content=response.content,
            metadata={"source": os.path.basename(file_path)},
        )

        splitted_docs = self.text_splitter.split_documents([document])
    
        return splitted_docs

    def process_documents(self, documents: List[str]) -> List[Document]:
        """
        Args:
            documents: List of paths to the document

        Returns:
            List of processed Document objects
        """
        _logger.info(f"Processing {len(documents)} documents")
        # TODO: Implement document processing
        
        # data_path = "../data/downloaded_files/"
        # files: List[str] = os.listdir(data_path)[:5]

        documents = [self.load_file(doc) for doc in documents]
        return documents

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main agent function to be called by the supervisor

        Args:
            state: Current state of the system

        Returns:
            Updated state
        """
        # TODO: Implement agent logic
        # Example:
        # 1. Check if there are Excel files to process
        # 2. Load and process documents
        # 3. Update state with processed documents
        return state
