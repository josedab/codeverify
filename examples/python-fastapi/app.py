"""Example FastAPI app with intentional issues for CodeVerify to find."""

from fastapi import FastAPI, HTTPException

app = FastAPI(title="User Service")

# In-memory database
users_db: dict[int, dict] = {}


@app.get("/users/{user_id}")
async def get_user(user_id: int):
    """BUG: No null check — returns None if user doesn't exist."""
    user = users_db.get(user_id)
    # CodeVerify finds: potential null dereference on next line
    return {"name": user["name"], "email": user["email"]}


@app.get("/users/{user_id}/score")
async def get_score(user_id: int, total: int, count: int):
    """BUG: Division by zero when count is 0."""
    # CodeVerify finds: division by zero
    average = total / count
    return {"user_id": user_id, "average": average}


@app.post("/users/search")
async def search_users(query: str):
    """BUG: SQL injection via string concatenation."""
    # CodeVerify finds: potential SQL injection
    sql = f"SELECT * FROM users WHERE name = '{query}'"
    return {"query": sql, "results": []}


@app.post("/users/process")
async def process_data(code: str):
    """BUG: Code injection via eval."""
    # CodeVerify finds: eval() is a code injection risk
    result = eval(code)
    return {"result": result}
