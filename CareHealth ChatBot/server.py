from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from langchain.prompts import ChatPromptTemplate
from langchain_cohere import ChatCohere
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(
    title="JPL Chatbot",
    version="1.0",
    description="API server for Java Premier League bot",
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cohere API Key
cohere_api_key = "CbMbyOexiMGw6GZ5Jzx99suZx7lkstwnWspnUwRu"

# Ensure 'data' directory exists
os.makedirs("data", exist_ok=True)
BASE_FILE_PATH = "data/base.txt"

# Define Embeddings and Vector Store
embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key, model="embed-english-v3.0")

# Function to load base.txt into FAISS retriever
def load_base_file():
    if not os.path.exists(BASE_FILE_PATH):
        print("⚠️ base.txt not found. Creating an empty file.")
        with open(BASE_FILE_PATH, "w") as f:
            f.write("")

    # Read the file
    with open(BASE_FILE_PATH, "r", encoding="utf-8") as file:
        text_data = file.read()

    # Index data into FAISS
    vectorstore = FAISS.from_texts([text_data], embeddings)
    return vectorstore

# Load base.txt into retriever
vectorstore = load_base_file()
retriever = vectorstore.as_retriever()
retriever.search_kwargs["k"] = 2  # Retrieve top 2 results

# Chatbot Prompt
TEMPLATE = '''
You are a really sympathetic and caring medical assistant for question-answering tasks. 
Your name is 'Sushruta'. You are very good at understanding the user's needs and providing helpful information.


You are a smart medical assistant. Your task is to answer user queries based on the available patient data.  
Always prioritize the provided information before using external knowledge.  

### Response Guidelines:  
 
 # If he user greets yoiu with hi, hello, hey then greet him with *"Hi, [Patient Name]! ,I am sushruta How can i assist you"* (only the name, no medical data , Add greeetings only if the user greets you , If user does,nt greet then provide answer for the asked query ).  
- Otherwise, send only the responce don't tell above part repeatly,  provide **short, clear, and relevant** answers based on patient data.  
- Do not repeat your name in every response . Just say it once with greeting and that's it
- Avoid unnecessary details; keep responses **point to point**.  
- Interact with the user with the languagge they are using during interaction with you 
-for example if user says mera sir dard ho rha he -response must be (पानी पिएं: डिहाइड्रेशन सिरदर्द का एक आम कारण होता है, इसलिए पर्याप्त मात्रा में पानी पिएं। इस विषय पर कुछ और बिंदु बताइए।) or user says- I am having a headache then response must be- Stay Hydrated: Dehydration can cause headaches, so drink plenty of water.
### User Query Context:  
- **Patient Data:** {context}  
- **User Question:** {question}  

### Department routing as per symptoms
- If the user provides you with symptoms and asks for department, provide the mst appropriate department from the following-
- Available Departments:  
- **Cardiology**  
- **Neurology**  
- **Orthopedics**  
- **Pediatrics** 
- **ENT (ear, nose, and throat) department** 
- **General**
- For example, User say : "I am suffering from fever, runny nose, and congestion."
Chatbot responce: "Based on your symptoms, you should consult the ENT (Ear, Nose, and Throat) department."

##Appoinment responce : only responce the related information that the user ask 
for example user say : What was the name of the doctor of my appointment 2
Chatbot responce: "Dr. [Doctor Name] visited the patient on [Date and Time]" 
responce like this for any other information

### Home remidies: provide Home remedies (give only when the user ask about Home remidie otherwise suggest department)
for example : User ask: "I have a sore throat. Any home remedies?"
Chatbot response(should be in bullet points only provide 3-5 remedies, it should be precise and only provide remedy not its explanation , Provide answer in points and change line after each point  ): "For a sore throat, gargling with warm salt water can provide relief. Mix 1/2 teaspoon of salt in a cup of warm water and gargle several times daily."

### Do not suggest any remedies or consultation department or any thing if not asked just give the response according to user interaction consider it for any language user interacts with you , below is the example
for example :  User- mere sir me dard he then response must be -उस स्थिति में मैं आपकी कैसे मदद कर सकता हूँ?

### Handling Missing Data:  
- If patient data is unavailable or insufficient, respond with:  
  *"I couldn't find your Answer."*  
'''


prompt = ChatPromptTemplate.from_template(TEMPLATE)
chat = ChatCohere(cohere_api_key=cohere_api_key)

# Chain that first searches in base.txt before using Cohere
chain = ({'context': retriever, 'question': RunnablePassthrough()} | prompt | chat) 

# Request Model
class QuestionRequest(BaseModel):
    question: str

# Chat Route
@app.post("/chat")
async def chat_endpoint(request: QuestionRequest):
    try:
        response = chain.invoke(request.question).content
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route to Store Text in base.txt
class TextRequest(BaseModel):
    content: str

@app.post("/store-text")
async def store_text(request: TextRequest):
    try:
        with open(BASE_FILE_PATH, "w") as file:  # Overwrites base.txt
            file.write(request.content)
        
        # Reload the retriever with updated data
        global vectorstore
        vectorstore = load_base_file()
        
        return {"message": "Text stored successfully", "file_path": BASE_FILE_PATH}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8080)
