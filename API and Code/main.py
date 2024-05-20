from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pipeline import classify_text  

app = FastAPI()

# CORS configuration
origins = [
    "http://localhost",
    'http://127.0.0.1:5501',
    'http://127.0.0.1:5502',
    'http://127.0.0.1:5500',
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

class TextData(BaseModel):
    text: str

@app.post("/classify/")
async def classify_text_endpoint(data: TextData):
    try:
        text = data.text
        print(text)
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")

        # Classify the text content
        category = classify_text(text)
        print(category)
        return {"category": category}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run the application
# uvicorn main:app --reload
