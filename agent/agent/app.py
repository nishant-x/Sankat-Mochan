from flask import Flask, request, jsonify
from flask_cors import CORS
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
import PyPDF2

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Initialize the chatbot agent
agent = Agent(
    model=Gemini(
        # api_key="AIzaSyAtzJHUfPM3dy5IGCutmtYTZn49Hst0asc",  # Piyush
        # api_key= "AIzaSyAzN4NIcADJRPjKztikRtCz--JEARybgHQ", # Nishant
        api_key="AIzaSyAtzJHUfPM3dy5IGCutmtYTZn49Hst0asc", # Krish
        id="gemini-2.0-flash-exp"
    ),
    tools=[DuckDuckGoTools()],
    markdown=True
)

# Helper function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
        if not text.strip():
            raise ValueError("The uploaded PDF contains no extractable text.")
        return text
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    
@app.route('/chat', methods=['POST'])
def chat():
    
    pass

@app.route('/analyze-report', methods=['POST'])
def analyze_report():
    try:
        # Ensure a file is uploaded
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        if not file.filename.endswith('.pdf'):
            return jsonify({"error": "Only PDF files are allowed"}), 400

        # Extract text from the PDF
        pdf_text = extract_text_from_pdf(file)

        # Construct the query for the chatbot
        query = """
        You are a highly skilled medical expert with extensive knowledge in 
        clinical diagnostics and textual medical report analysis. Analyze the following patient's medical report and provide detailed insights:
        """
        
        # Run the chatbot agent
        response = agent.run(query + pdf_text)
        return jsonify({"response": response.content})
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(port=3000, debug=True)
