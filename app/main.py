from fastapi import FastAPI
from app.api import text, speech, logs

app = FastAPI(title="Deaf Inclusion Tool API")

# Include API routes
app.include_router(text.router, prefix="/asl")
app.include_router(speech.router, prefix="/asl")
app.include_router(logs.router)


@app.get("/")
def root():
    return {"message": "Deaf Inclusion Tool API is running"}
