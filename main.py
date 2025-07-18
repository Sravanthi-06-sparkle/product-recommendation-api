from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer, util
import torch

# Initialize FastAPI app
app = FastAPI()

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Default product data
default_products = [
    {
        "id": 1,
        "name": "Nike Air Max 90",
        "description": "Classic black sneakers with air cushioning",
        "category": "Footwear",
        "tags": ["sneakers", "black", "nike", "shoes"]
    },
    {
        "id": 2,
        "name": "Adidas Ultraboost White",
        "description": "Comfortable white running shoes",
        "category": "Footwear",
        "tags": ["sneakers", "white", "adidas", "running"]
    },
    {
        "id": 3,
        "name": "Puma RS-X Black",
        "description": "Stylish black running sneakers with modern design",
        "category": "Footwear",
        "tags": ["puma", "black", "sneakers", "running"]
    },
    {
        "id": 4,
        "name": "Converse Chuck Taylor",
        "description": "Classic white high-top sneakers",
        "category": "Footwear",
        "tags": ["white", "sneakers", "converse", "casual"]
    },
    {
        "id": 5,
        "name": "New Balance 574",
        "description": "Gray suede sneakers with retro design",
        "category": "Footwear",
        "tags": ["gray", "sneakers", "new balance", "retro"]
    }
]

# Pydantic model for a product
class Product(BaseModel):
    id: int
    name: str
    description: str
    category: str
    tags: List[str]

# Request schema
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    products: Optional[List[Product]] = None

# Recommendation function
def recommend_products(query: str, products: List[dict], top_k: int = 3):
    product_texts = [f"{p['name']} {p['description']}" for p in products]
    product_embeddings = model.encode(product_texts, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, product_embeddings)[0]
    top_results = torch.topk(scores, k=top_k)
    recommended = [products[idx] for idx in top_results.indices]
    return recommended

# Endpoint
@app.post("/recommend")
def get_recommendations(req: QueryRequest):
    products_to_use = req.products if req.products is not None else default_products
    results = recommend_products(req.query, products_to_use, req.top_k)
    return {"recommendations": results}
