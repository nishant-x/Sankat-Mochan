from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools

agent = Agent(
    model=Gemini(
        api_key= "AIzaSyAtzJHUfPM3dy5IGCutmtYTZn49Hst0asc",
        id="gemini-2.0-flash-exp"
    ),
    tools=[DuckDuckGoTools()],
    markdown=True
)

# Medical Report Analysis Query
query = """
You are a highly skilled medical expert with extensive knowledge in 
clinical diagnostics and textual medical report analysis. Analyze the patient's medical report 
and structure your response as follows:

### 1. Report Overview
- Identify the type of report (Radiology, Pathology, Lab, Clinical Summary, etc.)
- Summarize the main components of the report
- Comment on clarity, completeness, and any missing details

### 2. Key Findings
- Extract primary clinical observations
- Note any significant abnormal results with precise descriptions
- List measurements, lab values, and critical thresholds if applicable
- Identify severity: Normal/Mild/Moderate/Severe

### 3. Diagnostic Assessment
- Provide primary diagnosis with confidence level
- List differential diagnoses in order of likelihood
- Support each diagnosis with evidence from the report
- Note any critical or urgent findings requiring immediate attention

### 4. Patient-Friendly Explanation
- Explain findings in simple, clear language that the patient can understand
- Avoid medical jargon or provide clear definitions
- Use visual analogies where helpful
- Address common patient concerns related to these findings

### 5. Research Context
IMPORTANT: Use the DuckDuckGo search tool to:
- Find recent medical literature about similar cases
- Search for standard treatment protocols
- Provide a list of relevant medical links
- Research any recent technological advances
- Include 2-3 key references to support your analysis

Format your response using clear markdown headers and bullet points. Be concise yet thorough.
"""

# getting the text from pdf

import PyPDF2
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

content = extract_text_from_pdf("lab-report.pdf")
response = agent.run(query+content)
print(response.content)