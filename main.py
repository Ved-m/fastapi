# main.py - FastAPI Backend with FREE Hugging Face Inference API
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, validator
from typing import List, Optional
import os
from datetime import datetime
import requests
import pinecone
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool
import json
from dotenv import load_dotenv
import numpy as np
import time
import logging
import re
from functools import lru_cache

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI with proper documentation
app = FastAPI(
    title="PM Internship Matching API",
    description="AI-powered internship matching system using vector similarity",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS with more specific origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001", 
        "https://your-frontend-domain.vercel.app",  # Replace with your actual frontend domain
        "https://*.vercel.app"  # Allow all Vercel preview deployments
    ] if os.getenv("ENVIRONMENT") == "production" else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Database connection pool
connection_pool = None

def init_db_pool():
    """Initialize database connection pool"""
    global connection_pool
    try:
        connection_pool = psycopg2.pool.ThreadedConnectionPool(
            1, 10,  # min and max connections
            host=os.getenv("POSTGRES_HOST"),
            database=os.getenv("POSTGRES_DATABASE"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            port=int(os.getenv("POSTGRES_PORT", 5432))
        )
        logger.info("Database connection pool initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
        raise

def get_db_connection():
    """Get database connection from pool"""
    global connection_pool
    if connection_pool is None:
        init_db_pool()
    
    try:
        return connection_pool.getconn()
    except Exception as e:
        logger.error(f"Failed to get database connection: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")

def return_db_connection(conn):
    """Return database connection to pool"""
    global connection_pool
    if connection_pool and conn:
        connection_pool.putconn(conn)

# Initialize Pinecone
try:
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
    )
    index_name = os.getenv("PINECONE_INDEX_NAME", "internships")
    index = pinecone.Index(index_name)
    logger.info(f"Pinecone initialized with index: {index_name}")
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {e}")
    raise

# Hugging Face FREE Inference API
HF_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Input validation functions
def sanitize_string(text: str) -> str:
    """Sanitize input string to prevent injection attacks"""
    if not text:
        return ""
    # Remove potentially harmful characters but keep useful ones
    return re.sub(r'[<>\"\'%;()&+\x00-\x1f\x7f-\x9f]', '', str(text).strip())

def validate_uid(uid: str) -> str:
    """Validate user ID format"""
    if not uid or not re.match(r'^[a-zA-Z0-9_-]{3,50}$', uid):
        raise ValueError("Invalid user ID format")
    return uid

def validate_phone(phone: str) -> str:
    """Validate phone number format"""
    phone = re.sub(r'[^\d+\-\s()]', '', phone)
    if not re.match(r'^[\d+\-\s()]{10,15}$', phone):
        raise ValueError("Invalid phone number format")
    return phone

# Enhanced Pydantic models with validation
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

    @validator('uid')
    def validate_uid_field(cls, v):
        return validate_uid(v)
    
    @validator('name', 'domain', 'skills', 'location', 'degree', 'duration', 'stipend')
    def validate_text_fields(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Field cannot be empty")
        return sanitize_string(v)
    
    @validator('email')
    def validate_email_field(cls, v):
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, v):
            raise ValueError("Invalid email format")
        return v.lower()
    
    @validator('phone')
    def validate_phone_field(cls, v):
        return validate_phone(v)
    
    @validator('marks10', 'marks12')
    def validate_marks(cls, v):
        try:
            marks = float(v)
            if not 0 <= marks <= 100:
                raise ValueError("Marks must be between 0 and 100")
        except ValueError:
            raise ValueError("Marks must be a valid number")
        return v

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

class ApplicationRequest(BaseModel):
    user_id: str
    internship_id: str
    
    @validator('user_id', 'internship_id')
    def validate_ids(cls, v):
        return validate_uid(v)

# Cache embeddings to avoid repeated API calls
@lru_cache(maxsize=500)
def create_embedding(text: str, max_retries: int = 3) -> List[float]:
    """Create embedding using Hugging Face's FREE Inference API with caching"""
    if not text or len(text.strip()) == 0:
        raise ValueError("Text cannot be empty")
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Add token if available for higher rate limits
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    
    payload = {
        "inputs": text[:1000],  # Limit text length to avoid token limits
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
                        result = embedding[0]  # Nested array format
                    else:
                        result = embedding  # Flat array format
                    
                    # Verify embedding dimension
                    if len(result) != 384:
                        raise ValueError(f"Unexpected embedding dimension: {len(result)}")
                    
                    return result
                else:
                    logger.error(f"Unexpected embedding format: {type(embedding)}")
                    raise ValueError("Unexpected embedding format")
                    
            elif response.status_code == 503:
                # Model is loading, wait and retry
                wait_time = min(2 ** attempt, 20)
                logger.info(f"Model loading, waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            elif response.status_code == 429:
                # Rate limited
                wait_time = min(2 ** attempt, 30)
                logger.info(f"Rate limited, waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"HF API error: {response.status_code} - {response.text}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise HTTPException(
                    status_code=500,
                    detail=f"Embedding API error: {response.status_code}"
                )
                
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                logger.warning(f"Timeout on attempt {attempt + 1}, retrying...")
                time.sleep(2 ** attempt)
                continue
            else:
                raise HTTPException(status_code=500, detail="Embedding API timeout")
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                raise HTTPException(status_code=500, detail="Failed to create embedding")
    
    raise HTTPException(status_code=500, detail="Failed to create embedding after retries")

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

def calculate_weighted_score(profile: UserProfile, internship: dict, base_score: float) -> float:
    """Calculate weighted match score based on multiple factors"""
    try:
        score = base_score
        
        # Domain match boost
        if profile.domain.lower() in internship.get('domain', '').lower():
            score += 0.1
        
        # Skills overlap
        profile_skills = set(skill.strip().lower() for skill in profile.skills.split(',') if skill.strip())
        required_skills = set(skill.strip().lower() for skill in internship.get('skills_required', '').split(',') if skill.strip())
        
        if required_skills:
            skill_overlap = len(profile_skills.intersection(required_skills)) / len(required_skills)
            score += skill_overlap * 0.2
        
        # Location preference
        if profile.location.lower() == internship.get('location', '').lower():
            score += 0.05
        
        return min(score, 1.0)  # Cap at 1.0
    except Exception as e:
        logger.warning(f"Error calculating weighted score: {e}")
        return base_score

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        process_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Response: {request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
        return response
    except Exception as e:
        process_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Error: {request.method} {request.url.path} - {str(e)} - {process_time:.3f}s")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    logger.info("Starting PM Internship Matching API...")
    init_db_pool()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    global connection_pool
    if connection_pool:
        connection_pool.closeall()
        logger.info("Database connection pool closed")

@app.get("/")
async def root():
    return {
        "message": "PM Internship Matching API is running",
        "embedding_service": "Hugging Face Inference API (FREE)",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "version": "1.0.0"
    }

@app.post("/api/match-internships", response_model=List[InternshipResponse])
async def match_internships(profile: UserProfile):
    """
    Match user profile with internships using vector similarity search
    """
    conn = None
    try:
        logger.info(f"Matching internships for user: {profile.uid}")
        
        # Step 1: Create embedding from user profile
        profile_text = create_profile_text(profile)
        user_embedding = create_embedding(profile_text)
        
        # Step 2: Query Pinecone for similar internships
        query_results = index.query(
            vector=user_embedding,
            top_k=20,  # Get more results for better filtering
            include_metadata=False
        )
        
        if not query_results['matches']:
            logger.info("No matches found in vector search")
            return []
        
        # Step 3: Get internship IDs and scores
        internship_matches = [
            {'id': match['id'], 'score': match['score']}
            for match in query_results['matches']
        ]
        
        # Step 4: Fetch detailed internship data from PostgreSQL
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        internship_ids = [m['id'] for m in internship_matches]
        placeholders = ','.join(['%s'] * len(internship_ids))
        
        query = f"""
            SELECT * FROM internships 
            WHERE id IN ({placeholders})
            ORDER BY created_at DESC
        """
        
        cur.execute(query, internship_ids)
        internships_data = cur.fetchall()
        
        cur.close()
        
        # Step 5: Combine data with enhanced match scores
        internships_dict = {str(i['id']): dict(i) for i in internships_data}
        results = []
        
        for match in internship_matches:
            if match['id'] in internships_dict:
                internship = internships_dict[match['id']]
                
                # Calculate weighted score
                weighted_score = calculate_weighted_score(profile, internship, match['score'])
                
                results.append(InternshipResponse(
                    id=str(internship['id']),
                    title=sanitize_string(internship['title']),
                    company=sanitize_string(internship['company']),
                    location=sanitize_string(internship['location']),
                    duration=sanitize_string(internship['duration']),
                    stipend=sanitize_string(internship['stipend']),
                    domain=sanitize_string(internship['domain']),
                    skills_required=sanitize_string(internship['skills_required']),
                    description=sanitize_string(internship['description']),
                    eligibility=sanitize_string(internship['eligibility']),
                    match_score=round(weighted_score, 3)
                ))
        
        # Sort by match score (highest first) and limit results
        results.sort(key=lambda x: x.match_score, reverse=True)
        results = results[:10]  # Return top 10 matches
        
        logger.info(f"Returning {len(results)} matches for user: {profile.uid}")
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in match_internships: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        if conn:
            return_db_connection(conn)

@app.get("/api/internship/{internship_id}")
async def get_internship(internship_id: str):
    """Get details of a specific internship"""
    conn = None
    try:
        # Validate internship ID
        internship_id = validate_uid(internship_id)
        
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        query = "SELECT * FROM internships WHERE id = %s"
        cur.execute(query, (internship_id,))
        internship = cur.fetchone()
        
        cur.close()
        
        if not internship:
            raise HTTPException(status_code=404, detail="Internship not found")
        
        # Sanitize output
        result = dict(internship)
        for key, value in result.items():
            if isinstance(value, str):
                result[key] = sanitize_string(value)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching internship: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        if conn:
            return_db_connection(conn)

@app.post("/api/apply-internship")
async def apply_for_internship(application: ApplicationRequest):
    """Record internship application"""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Check if already applied
        check_query = """
            SELECT id FROM applications 
            WHERE user_id = %s AND internship_id = %s
        """
        cur.execute(check_query, (application.user_id, application.internship_id))
        existing = cur.fetchone()
        
        if existing:
            raise HTTPException(status_code=400, detail="Already applied to this internship")
        
        # Verify internship exists
        verify_query = "SELECT id FROM internships WHERE id = %s"
        cur.execute(verify_query, (application.internship_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="Internship not found")
        
        # Insert new application
        query = """
            INSERT INTO applications (user_id, internship_id, applied_at, status)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """
        
        cur.execute(query, (
            application.user_id,
            application.internship_id,
            datetime.now(),
            'pending'
        ))
        
        application_id = cur.fetchone()[0]
        conn.commit()
        
        cur.close()
        
        logger.info(f"Application created: {application_id} for user: {application.user_id}")
        return {"success": True, "application_id": application_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error applying for internship: {e}")
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        if conn:
            return_db_connection(conn)

@app.get("/api/user-applications/{user_id}")
async def get_user_applications(user_id: str):
    """Get all applications for a user"""
    conn = None
    try:
        # Validate user ID
        user_id = validate_uid(user_id)
        
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT a.id, a.applied_at, a.status, 
                   i.id as internship_id, i.title, i.company, i.location, i.stipend
            FROM applications a
            JOIN internships i ON a.internship_id = i.id
            WHERE a.user_id = %s
            ORDER BY a.applied_at DESC
        """
        
        cur.execute(query, (user_id,))
        applications = cur.fetchall()
        
        cur.close()
        
        # Sanitize output
        results = []
        for app in applications:
            result = dict(app)
            for key, value in result.items():
                if isinstance(value, str):
                    result[key] = sanitize_string(value)
            results.append(result)
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching applications: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        if conn:
            return_db_connection(conn)

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
            "service": "Hugging Face Inference API",
            "model": "sentence-transformers/all-MiniLM-L6-v2"
        }
    except Exception as e:
        logger.error(f"Embedding test failed: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        return_db_connection(conn)
        
        # Test Pinecone connection
        index.describe_index_stats()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "database": "connected",
                "pinecone": "connected",
                "embedding_api": "available"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

# For Vercel deployment
handler = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
