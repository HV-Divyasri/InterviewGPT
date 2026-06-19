import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from resume_parser import extract_resume_text
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)

def extract_skills(resume_text):
    template = """
    You are an expert HR assistant and automated resume parser.
    Analyze the following resume text and extract all technical skills, programming languages, 
    frameworks, databases, cloud providers, and development tools mentioned.
    
    Return the output ONLY as a clean, raw JSON list of strings. Do not include any introductory sentences, conversational text, or markdown formatting blocks (like ```json).
    
    Example output format:
    ["Python", "SQL", "Flask", "Git", "AWS"]
    
    Resume Text:
    {resume_text}
    """
    
    prompt = PromptTemplate(input_variables=["resume_text"], template=template)
    chain = prompt | llm
    
    try:
        response = chain.invoke({"resume_text": resume_text})
        clean_content = response.content.strip()
        skills_list = json.loads(clean_content)
        return skills_list
        
    except Exception as e:
        print(f"Error extracting skills via AI: {e}")
        return []

if __name__ == "__main__":
    pdf_file = r"C:\Users\Windows 11\Downloads\sample-resume-information-technology.pdf"
    
    print("Reading PDF file...")
    text = extract_resume_text(pdf_file)
    
    if text:
        print("Asking Groq AI to extract skills...")
        ai_skills = extract_skills(text)
        print("\n--- AI Extracted Skills ---")
        print(ai_skills)
    else:
        print("Could not read any text from the PDF path provided.")
