import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def evaluate_answer(question, answer):
    prompt = f"""
    You are a Senior Technical Interviewer.
    
    Interview Question:
    {question}
    
    Candidate Answer:
    {answer}
    
    Evaluate the answer and provide:
    1. Technical Score (out of 10)
    2. Communication Score (out of 10)
    3. Strengths
    4. Weaknesses
    5. Missing Concepts
    6. Improved Answer
    7. Final Recommendation
    
    Provide detailed feedback.
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert interviewer specializing in Python, SQL, Machine Learning, Deep Learning, NLP, Generative AI, RAG, and LangChain. Provide professional interview feedback."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error during evaluation: {e}"

if __name__ == "__main__":
    print("=" * 70)
    print("INTERVIEW ANSWER EVALUATOR")
    print("=" * 70)

    question = input("\nEnter Interview Question:\n\n")
    answer = input("\nEnter Candidate Answer:\n\n")

    print("\nProcessing evaluation with Groq AI...")
    result = evaluate_answer(question, answer)

    print("\n")
    print("=" * 70)
    print("INTERVIEW EVALUATION REPORT")
    print("=" * 70)
    print(result)
