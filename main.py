from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ⚠️ IMPORTANT: Add CORS middleware for Flutter to access the API.
# In development, we allow all origins.
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/hello")
def read_root():
    # The API returns a JSON object: {"message": "Hello World"}
    return {"message": "Hello World"}