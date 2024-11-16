import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("API_KEY")
)

# Multi-Agent Workflow for Processing Questions
def process_question_with_agents(question_data):
    # Step 1: Generator Agent produces an initial response
    generator_agent = GeneratorAgent()
    initial_response = generator_agent.generate_initial_response(question_data)

    # Step 2: Verifier Agent probes the initial response with critical questions
    verifier_agent = VerifierAgent()
    probing_results = verifier_agent.probe_response(question_data["question"], initial_response)

    # Step 3: Reasoner Agent analyzes the probing results and produces the final assessment
    reasoner_agent = ReasonerAgent()
    analysis = reasoner_agent.analyze_probing_results(probing_results)
    final_assessment = reasoner_agent.generate_final_assessment(question_data["question"], initial_response, analysis)

    # Format output for each question
    formatted_output = {
        "Question": question_data["question"],
        "Initial Response": initial_response,
        "Probing Questions and Answers": probing_results,
        "Final Assessment": final_assessment,
    }

    return formatted_output

# Generator Agent: Generates initial clinical arguments
class GeneratorAgent:
    def __init__(self):
        pass

    def extract_options(self, options):
        return "\n".join([f"{key}: {value}" for key, value in options.items()])

    def generate_initial_response(self, question_data):
        question = question_data["question"]
        options = self.extract_options(question_data["options"])
        prompt = f"Question:\n{question}\n\nOptions:\n{options}\n\nYou are a medical expert with wide range of knowledge, your each and every decision and answer is crucial and must be accurate and onpoint , refer internet and all the available knowledge resources and analyze each option carefully and select only one best option (A, B, C, D, or E) which you think its right according to the question. Explain your reasoning in very short and crisp points."

        # Generate the initial response using the LLM
        try:
            response = client.chat.completions.create(
                model="nvidia/llama-3.1-nemotron-70b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                top_p=1,
                max_tokens=500,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating content: {e}")
            return "No valid response generated."

# Verifier Agent: Critically evaluates the initial response with probing questions
class VerifierAgent:
    def __init__(self):
        # Probing questions for self-questioning
        self.probing_questions = [
            "What are the key factors that led you to this conclusion by ruling out the other options?",
            "Can you identify any potential weaknesses in your reasoning?",
            "What alternative explanations might exist for this scenario?",
            "How might your answer change if [insert relevant detail] was different?",
            "On a scale of 1-10, how certain are you of this answer, and why?",
            "What additional information would help you be more confident in your answer?",
            "What are the potential consequences if this answer is incorrect?",
            "How does your answer align with standard medical practices or guidelines?"
        ]

    def probe_response(self, question, initial_response):
        probing_results = {}
        for probe in self.probing_questions:
            probing_prompt = f"{question}\n\nInitial response: {initial_response}\n\n{probe}\n\nJust answer the probing questions in short and crisp points. No need for further explanation."
            try:
                probing_response = client.chat.completions.create(
                    model="nvidia/llama-3.1-nemotron-70b-instruct",
                    messages=[{"role": "user", "content": probing_prompt}],
                    temperature=0.5,
                    top_p=1,
                    max_tokens=300,
                    stream=False
                )
                probing_results[probe] = probing_response.choices[0].message.content
            except Exception as e:
                print(f"Error generating probing content: {e}")
                probing_results[probe] = "No valid response generated."
        return probing_results

# Reasoner Agent: Analyzes probing results and provides a final clinical assessment
class ReasonerAgent:
    def __init__(self):
        pass

    def analyze_probing_results(self, probing_results):
        analysis = {}
        for question, response in probing_results.items():
            analysis[question] = f"Analysis of: {response}"
        return analysis

    def generate_final_assessment(self, question, initial_response, analysis):
        assessment_prompt = f"Question: {question}\n\nInitial response: {initial_response}\n\nBased on the following analysis, provide a final assessment and select only one best and the correct option (A, B, C, D, or E):\n{analysis}"
        try:
            final_assessment = client.chat.completions.create(
                model="nvidia/llama-3.1-nemotron-70b-instruct",
                messages=[{"role": "user", "content": assessment_prompt}],
                temperature=0.5,
                top_p=1,
                max_tokens=500,
                stream=False
            )
            return final_assessment.choices[0].message.content
        except Exception as e:
            print(f"Error generating final assessment: {e}")
            return "No valid response generated."

# Streamlit UI
st.title("LLM-Powered Clinical Decision Support System")

# Input fields for the question and options
question = st.text_area("Enter your clinical question:")
options = {}
for i in range(5):
    option_key = chr(65 + i)
    options[option_key] = st.text_input(f"Option {option_key}:")

# Button to trigger the LLM call
if st.button("Generate Response"):
    # Show a spinner while the LLM is generating the response
    with st.spinner("Analyzing the question and options..."):
        # Prepare the question data
        question_data = {
            "question": question,
            "options": options
        }

        # Process the question with the multi-agent framework
        result = process_question_with_agents(question_data)

        # Save the result for display
        st.session_state.result = result

# Display the full response if available
if "result" in st.session_state:
    result = st.session_state.result

    st.subheader("Question")
    st.write(result.get("Question", "Question not available"))

    st.subheader("Initial Response")
    st.write(result.get("Initial Response", "Initial response not available"))

    st.subheader("Probing Questions and Answers")
    for question, answer in result.get("Probing Questions and Answers", {}).items():
        st.html(f"<p><span style='font-size: 20px; font-weight: bold; color: #de6464'>🔶 {question}</span></p>")
        st.write(answer)

    st.subheader("Final Assessment")
    st.write(result.get("Final Assessment", "Final assessment not available"))