# AI Product Recommendation API
 
This is a simple product recommendation API using machine learning (semantic similarity).

## How It Works
-	Accepts a text query (e.g., "black sneakers")
-	Returns top 3 matching products from a mock database using semantic embeddings
## Tech Stack
-	FastAPI
-	sentence-transformers (all-MiniLM-L6-v2) ## Install Requirements
bash
pip install fastapi uvicorn sentence-transformers
uvicorn main:app --reload
