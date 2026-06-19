import streamlit as st
import tempfile
import os

from resume_parser import extract_resume_text
from skill_extractor import extract_skills
from rag import create_vector_store, generate_question
from evaluator import evaluate_answer, client # Reusing the groq client from evaluator

st.set_page_config(
    page_title="InterviewGPT",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 InterviewGPT")
st.subheader("AI Interview Coach using RAG + Groq")

st.markdown("---")

# Initialize session state variables so they don't disappear on page refreshes
if "question" not in st.session_state:
    st.session_state.question = None
if "skills" not in st.session_state:
    st.session_state.skills = None
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

# -----------------------------
# Resume Upload
# -----------------------------
st.header("📄 Upload Resume")
resume = st.file_uploader("Upload Resume PDF", type=["pdf"])

if resume is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(resume.read())
        resume_path = tmp.name

    resume_text = extract_resume_text(resume_path)
    st.success("Resume Uploaded Successfully")

    # Extract skills only once and save to state
    if st.session_state.skills is None:
        st.session_state.skills = extract_skills(resume_text)

    st.subheader("Detected Skills")
    if st.session_state.skills:
        st.write(st.session_state.skills)
    else:
        st.warning("No skills detected.")

    # ---- NEW: INSTANT RESUME IMPROVEMENT RECOMMENDATIONS ----
    st.subheader("💡 Resume Improvement Recommendations")
    
    if st.session_state.recommendations is None:
        with st.spinner("Analyzing resume for improvements..."):
            resume_prompt = f"""
            You are an expert HR Consultant and Tech Recruiter.
            Analyze the following resume text and provide 3-4 highly actionable recommendations 
            to improve it for technical roles (e.g., missing modern tech stacks, formatting tips, or bullet point enhancements).
            
            Resume Text:
            {resume_text}
            """
            try:
                resume_response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": resume_prompt}],
                    temperature=0.3,
                    max_tokens=800
                )
                st.session_state.recommendations = resume_response.choices[0].message.content
            except Exception as e:
                st.error(f"Could not generate recommendations: {e}")

    if st.session_state.recommendations:
        st.markdown(st.session_state.recommendations)
    # ---------------------------------------------------------

    try: os.remove(resume_path) 
    except: pass

st.markdown("---")

# -----------------------------
# Interview PDF Upload
# -----------------------------
st.header("📚 Upload Interview Questions PDF")
interview_pdf = st.file_uploader("Upload Interview Questions PDF", type=["pdf"])

if interview_pdf is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(interview_pdf.read())
        pdf_path = tmp.name

    if st.button("Generate Interview Question"):
        with st.spinner("Generating Question..."):
            vectordb = create_vector_store(pdf_path)
            question = generate_question(vectordb)
            st.session_state.question = question
        st.success("Question Generated Successfully!")
        
    try: os.remove(pdf_path)
    except: pass

# -----------------------------
# Question & Evaluation Display
# -----------------------------
if st.session_state.question:
    st.markdown("---")
    st.header("🎤 Interview Question")
    st.info(st.session_state.question)

    answer = st.text_area("Enter Your Answer", height=200, key="candidate_answer")

    if st.button("Evaluate Answer"):
        if answer.strip() == "":
            st.warning("Please enter an answer.")
        else:
            with st.spinner("Evaluating..."):
                feedback = evaluate_answer(st.session_state.question, answer)

            st.markdown("---")
            st.header("📊 Interview Evaluation")
            st.markdown(feedback)

st.markdown("---")
st.caption("InterviewGPT | Resume Analysis | RAG | Groq")