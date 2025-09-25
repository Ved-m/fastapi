# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 💡 FIX 1: Ensure you have a route for the root path ("/") 
# that Vercel tries to access when checking the deployment URL.

@app.get("/")
def home():
    return {"message": "Welcome to FastAPI on Vercel!"} # <-- This handles the root URL

@app.get("/hello")
def read_root():
    return {"message": "Hello !"}

# CORS setup (important for Flutter)
origins = ["*"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


