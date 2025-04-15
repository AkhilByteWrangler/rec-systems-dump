from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from recommend import load_everything, recommend_top_k, coldstart_recommend
import pandas as pd
import numpy as np
import ast

app = FastAPI()

# === Load model + mappings + recipe data ===
print("\U0001F4E6 Loading model and mappings...")
model, user_map, inv_recipe_map, recipe_id_to_name = load_everything()
recipes_df = pd.read_csv("RAW_recipes.csv").set_index("id")

@app.get("/")
def root():
    return {"message": "Recipe Recommender GNN API is live \U0001F37D️"}

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": True,
        "users": len(user_map)
    }

@app.get("/users")
def get_users(limit: int = 50):
    users = list(user_map.keys())[:limit]
    return [int(uid) for uid in users]

@app.get("/recommend")
def recommend(user_id: int, top_k: int = 5):
    try:
        recs = recommend_top_k(
            model,
            user_id,
            user_map,
            inv_recipe_map,
            recipe_id_to_name,
            top_k=top_k
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return [
        {
            "recipe_id": int(rid),
            "title": str(title),
            "score": float(score)
        }
        for rid, title, score in recs
    ]

@app.get("/recipe/{recipe_id}")
def get_recipe(recipe_id: int):
    if recipe_id not in recipes_df.index:
        raise HTTPException(status_code=404, detail="Recipe ID not found")

    row = recipes_df.loc[recipe_id]

    def parse_field(field, fallback):
        try:
            return ast.literal_eval(field) if isinstance(field, str) else field
        except:
            return fallback

    return {
        "id": int(recipe_id),
        "title": str(row.get("name", "")),
        "description": row.get("description", None),
        "tags": parse_field(row.get("tags", "[]"), []),
        "nutrition": parse_field(row.get("nutrition", "[]"), []),
        "minutes": int(row.get("minutes", 0)),
        "n_ingredients": int(row.get("n_ingredients", 0)),
        "ingredients": parse_field(row.get("ingredients", "[]"), []),
        "steps": parse_field(row.get("steps", "[]"), []),
    }

class ColdStartRequest(BaseModel):
    cuisines: list[str]
    diet: str
    cook_time: int
    favorites: list[str]
    dislikes: list[str]

@app.post("/coldstart")
def coldstart_route(req: ColdStartRequest):
    try:
        recs = coldstart_recommend(
            cuisines=req.cuisines,
            diet=req.diet,
            cook_time=req.cook_time,
            favorites=[i.strip().lower() for i in req.favorites if i.strip()],
            dislikes=[i.strip().lower() for i in req.dislikes if i.strip()]
        )
        return [
            {
                "recipe_id": int(rid),
                "title": str(title),
                "score": float(score)
            } for rid, title, score in recs
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))f