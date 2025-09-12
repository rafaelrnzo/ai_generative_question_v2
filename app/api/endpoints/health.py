import httpx
from fastapi import APIRouter
from neo4j import GraphDatabase
from core.config import OLLAMA_HOST, NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD

router = APIRouter()

@router.get("/health", tags=["Health Check"])
async def health_check():
    health = {
        "ollama": False,
        "neo4j": False
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
            if response.status_code == 200:
                health["ollama"] = True
    except Exception:
        pass

    try:
        driver = GraphDatabase.driver(
            NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run("RETURN 1 AS result")
            if result.single()["result"] == 1:
                health["neo4j"] = True
        driver.close()
    except Exception:
        pass

    status = "ok" if all(health.values()) else "partial" if any(health.values()) else "fail"
    return {
        "status": status,
        "services": health
    }
