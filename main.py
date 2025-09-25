# main.py - FastAPI Backend with Free Embeddings (Sentence Transformers)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
import pinecone
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from dotenv import load_dotenv
import numpy as np

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

# Initialize Sentence Transformer model (FREE!)
# This model creates 384-dimensional embeddings
print("Loading embedding model... (this may take a minute on first run)")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!")

# Initialize Pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
)
index_name = "internships"
index = pinecone.Index(index_name)

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

# Helper function to create embeddings using Sentence Transformers
def create_embedding(text: str) -> List[float]:
    """Create embedding using Sentence Transformers (FREE)"""
    try:
        # Generate embedding
        embedding = model.encode(text)
        # Convert to list and return
        return embedding.tolist()
    except Exception as e:
        print(f"Error creating embedding: {e}")
        raise HTTPException(status_code=500, detail="Error creating embedding")

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
        "embedding_model": "Sentence Transformers (all-MiniLM-L6-v2)",
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
            "first_5_values": embedding[:5]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# For Vercel deployment
handler = app
