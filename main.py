# main.py - FastAPI Backend with FREE Hugging Face Inference API
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from datetime import datetime
import requests
import pinecone
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from dotenv import load_dotenv
import numpy as np
import time

load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
)
index_name = "internships"
index = pinecone.Index(index_name)

# Hugging Face FREE Inference API
# This model creates 384-dimensional embeddings
HF_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
# Optional: Get a free token from huggingface.co for higher rate limits
HF_TOKEN = os.getenv("HF_TOKEN", "")

# PostgreSQL connection
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        database=os.getenv("POSTGRES_DATABASE"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        port=os.getenv("POSTGRES_PORT", 5432)
    )

# Pydantic models
class UserProfile(BaseModel):
    uid: str
    name: str
    email: str
    phone: str
    domain: str
    skills: str
    location: str
    dob: str
    gender: str
    marks10: str
    marks12: str
    degree: str
    duration: str
    stipend: str

class InternshipResponse(BaseModel):
    id: str
    title: str
    company: str
    location: str
    duration: str
    stipend: str
    domain: str
    skills_required: str
    description: str
    eligibility: str
    match_score: float

# Helper function to create embeddings using Hugging Face API
def create_embedding(text: str, max_retries: int = 3) -> List[float]:
    """Create embedding using Hugging Face's FREE Inference API"""
    headers = {
        "Content-Type": "application/json"
    }
    
    # Add token if available for higher rate limits
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    
    payload = {
        "inputs": text,
        "options": {"wait_for_model": True}
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                HF_API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                embedding = response.json()
                # The API returns embeddings in different formats
                if isinstance(embedding, list):
                    if len(embedding) > 0 and isinstance(embedding[0], list):
                        return embedding[0]  # Nested array format
                    else:
                        return embedding  # Flat array format
                else:
                    print(f"Unexpected embedding format: {type(embedding)}")
                    raise ValueError("Unexpected embedding format")
                    
            elif response.status_code == 503:
                # Model is loading, wait and retry
                wait_time = min(2 ** attempt, 10)
                print(f"Model loading, waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                print(f"HF API error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Embedding API error: {response.status_code}"
                )
                
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"Timeout on attempt {attempt + 1}, retrying...")
                continue
            else:
                raise HTTPException(status_code=500, detail="Embedding API timeout")
        except Exception as e:
            print(f"Error creating embedding: {e}")
            if attempt < max_retries - 1:
                continue
            else:
                raise HTTPException(status_code=500, detail=str(e))
    
    raise HTTPException(status_code=500, detail="Failed to create embedding after retries")

# Helper function to create profile text for embedding
def create_profile_text(profile: UserProfile) -> str:
    """Convert user profile to text for embedding"""
    text = f"""
    Domain: {profile.domain}
    Skills: {profile.skills}
    Location: {profile.location}
    Degree: {profile.degree}
    Duration Preference: {profile.duration}
    Stipend Expectation: {profile.stipend}
    Academic Performance: 10th {profile.marks10}% 12th {profile.marks12}%
    """
    return text.strip()

@app.get("/")
async def root():
    return {
        "message": "PM Internship Matching API is running",
        "embedding_service": "Hugging Face Inference API (FREE)",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dimension": 384
    }

@app.post("/api/match-internships", response_model=List[InternshipResponse])
async def match_internships(profile: UserProfile):
    """
    Match user profile with internships using vector similarity search
    """
    try:
        # Step 1: Create embedding from user profile
        profile_text = create_profile_text(profile)
        user_embedding = create_embedding(profile_text)
        
        # Verify embedding dimension (should be 384 for all-MiniLM-L6-v2)
        if len(user_embedding) != 384:
            print(f"Warning: Unexpected embedding dimension: {len(user_embedding)}")
        
        # Step 2: Query Pinecone for similar internships
        query_results = index.query(
            vector=user_embedding,
            top_k=10,  # Get top 10 matches
            include_metadata=False
        )
        
        if not query_results['matches']:
            return []
        
        # Step 3: Get internship IDs and scores
        internship_matches = []
        for match in query_results['matches']:
            internship_matches.append({
                'id': match['id'],
                'score': match['score']
            })
        
        # Step 4: Fetch detailed internship data from PostgreSQL
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        internship_ids = [m['id'] for m in internship_matches]
        placeholders = ','.join(['%s'] * len(internship_ids))
        
        query = f"""
            SELECT * FROM internships 
            WHERE id IN ({placeholders})
        """
        
        cur.execute(query, internship_ids)
        internships_data = cur.fetchall()
        
        cur.close()
        conn.close()
        
        # Step 5: Combine data with match scores
        internships_dict = {str(i['id']): i for i in internships_data}
        results = []
        
        for match in internship_matches:
            if match['id'] in internships_dict:
                internship = internships_dict[match['id']]
                results.append(InternshipResponse(
                    id=str(internship['id']),
                    title=internship['title'],
                    company=internship['company'],
                    location=internship['location'],
                    duration=internship['duration'],
                    stipend=internship['stipend'],
                    domain=internship['domain'],
                    skills_required=internship['skills_required'],
                    description=internship['description'],
                    eligibility=internship['eligibility'],
                    match_score=round(match['score'], 2)
                ))
        
        # Sort by match score (highest first)
        results.sort(key=lambda x: x.match_score, reverse=True)
        
        return results
        
    except Exception as e:
        print(f"Error in match_internships: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/internship/{internship_id}")
async def get_internship(internship_id: str):
    """Get details of a specific internship"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        query = "SELECT * FROM internships WHERE id = %s"
        cur.execute(query, (internship_id,))
        internship = cur.fetchone()
        
        cur.close()
        conn.close()
        
        if not internship:
            raise HTTPException(status_code=404, detail="Internship not found")
        
        return internship
        
    except Exception as e:
        print(f"Error fetching internship: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/apply-internship")
async def apply_for_internship(application: dict):
    """Record internship application"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Check if already applied
        check_query = """
            SELECT id FROM applications 
            WHERE user_id = %s AND internship_id = %s
        """
        cur.execute(check_query, (application['user_id'], application['internship_id']))
        existing = cur.fetchone()
        
        if existing:
            raise HTTPException(status_code=400, detail="Already applied to this internship")
        
        # Insert new application
        query = """
            INSERT INTO applications (user_id, internship_id, applied_at, status)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """
        
        cur.execute(query, (
            application['user_id'],
            application['internship_id'],
            datetime.now(),
            'pending'
        ))
        
        application_id = cur.fetchone()[0]
        conn.commit()
        
        cur.close()
        conn.close()
        
        return {"success": True, "application_id": application_id}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error applying for internship: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/test-embedding")
async def test_embedding():
    """Test endpoint to verify embedding generation works"""
    try:
        test_text = "Software development intern with Python and React skills"
        embedding = create_embedding(test_text)
        return {
            "success": True,
            "text": test_text,
            "embedding_dimension": len(embedding),
            "first_5_values": embedding[:5] if embedding else [],
            "service": "Hugging Face Inference API"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/user-applications/{user_id}")
async def get_user_applications(user_id: str):
    """Get all applications for a user"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT a.*, i.title, i.company, i.location
            FROM applications a
            JOIN internships i ON a.internship_id = i.id
            WHERE a.user_id = %s
            ORDER BY a.applied_at DESC
        """
        
        cur.execute(query, (user_id,))
        applications = cur.fetchall()
        
        cur.close()
        conn.close()
        
        return applications
        
    except Exception as e:
        print(f"Error fetching applications: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check for Vercel
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "embedding_api": "active"
    }

# For Vercel deployment
handler = app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

