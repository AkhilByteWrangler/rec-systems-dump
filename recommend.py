import torch
import pickle
import pandas as pd
import numpy as np


def load_everything(
    model_path="recipe_gnn_script_100k.pt",
    user_map_path="user_map.pkl",
    recipe_map_path="recipe_map.pkl",
    recipe_csv_path="RAW_recipes.csv"
):
    model = torch.jit.load(model_path, map_location="cpu").eval()

    with open(user_map_path, "rb") as f:
        user_map = pickle.load(f)
    with open(recipe_map_path, "rb") as f:
        recipe_map = pickle.load(f)

    inv_recipe_map = {v: k for k, v in recipe_map.items()}
    recipe_df = pd.read_csv(recipe_csv_path)
    recipe_id_to_name = dict(zip(recipe_df["id"], recipe_df["name"]))

    return model, user_map, inv_recipe_map, recipe_id_to_name


def recommend_top_k(
    model,
    user_id,
    user_map,
    inv_recipe_map,
    recipe_id_to_name,
    top_k=5,
    return_scores=True
):
    if user_id not in user_map:
        raise ValueError(f"❌ User ID {user_id} not found in user_map.")

    user_idx = user_map[user_id]
    with torch.no_grad():
        scores = model(user_idx)
        sorted_indices = torch.argsort(scores, descending=True)

    results = []
    for idx in sorted_indices:
        idx_int = idx.item()
        if idx_int in inv_recipe_map:
            recipe_id = inv_recipe_map[idx_int]
            name = recipe_id_to_name.get(recipe_id, f"Recipe {recipe_id}")
            score = scores[idx_int].item()
            results.append((recipe_id, name, score))
        if len(results) >= top_k:
            break

    return results if return_scores else [r[1] for r in results]


def coldstart_recommend(cuisines, diet, cook_time, favorites, dislikes, top_k=5):
    df = pd.read_csv("RAW_recipes.csv")

    def safe_eval(x):
        try:
            return eval(x) if isinstance(x, str) else x
        except:
            return []

    df["tags"] = df["tags"].apply(safe_eval)
    df["ingredients"] = df["ingredients"].apply(safe_eval)

    df_filtered = df[
        df["minutes"] <= cook_time
    ]

    if diet:
        df_filtered = df_filtered[df_filtered["tags"].apply(lambda t: any(diet.lower() in tag.lower() for tag in t))]

    if cuisines:
        df_filtered = df_filtered[df_filtered["tags"].apply(lambda t: any(c.lower() in tag.lower() for c in cuisines for tag in t))]

    if favorites:
        df_filtered["fav_match"] = df_filtered["ingredients"].apply(lambda ings: sum(1 for ing in ings if ing.lower() in favorites))
    else:
        df_filtered["fav_match"] = 0

    if dislikes:
        df_filtered = df_filtered[df_filtered["ingredients"].apply(lambda ings: not any(ing.lower() in dislikes for ing in ings))]

    df_filtered = df_filtered.sort_values("fav_match", ascending=False)

    results = []
    for _, row in df_filtered.head(top_k).iterrows():
        results.append((row["id"], row["name"], row["fav_match"]))

    return results