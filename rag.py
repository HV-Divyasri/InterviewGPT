from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

import random


def create_vector_store(pdf_file):

    loader = PyPDFLoader(pdf_file)

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectordb


def generate_question(vectordb):

    docs = vectordb.similarity_search(
        "technical interview questions",
        k=10
    )

    questions = []

    for doc in docs:

        text = doc.page_content

        # Split by question mark
        lines = text.split("?")

        for line in lines:

            line = line.strip()

            if len(line) > 5:
                questions.append(
                    line + "?"
                )

    # Remove duplicates
    questions = list(
        set(questions)
    )

    if len(questions) > 0:

        return random.choice(
            questions
        )

    return "No Question Found"


if __name__ == "__main__":

    pdf_file = r"F:\Projects_Datav\genai_prompt\Interview Questions.pdf"

    vectordb = create_vector_store(
        pdf_file
    )

    for i in range(5):

        print(
            generate_question(
                vectordb
            )
        )

        print("-" * 50)