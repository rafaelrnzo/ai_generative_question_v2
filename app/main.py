from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.endpoints import query, upload, delete, files, health

app = FastAPI(title="Neo4j PDF RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router)
app.include_router(upload.router)
app.include_router(delete.router)
app.include_router(files.router)
app.include_router(health.router)

@app.get("/")
async def root():
    return {"message": "Welcome to Neo4j PDF RAG API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)