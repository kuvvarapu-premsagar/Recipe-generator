import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load recipes
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'recipes.json')
with open(data_path, 'r') as f:
    recipes = json.load(f)

# Prepare ingredient strings for vectorization
ingredient_strings = [' '.join(r['ingredients']) for r in recipes]

# Train TF-IDF vectorizer on all ingredient strings
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1)
ingredient_matrix = vectorizer.fit_transform(ingredient_strings)

# Save model and vectorizer
model_dir = os.path.dirname(__file__)
with open(os.path.join(model_dir, 'vectorizer.pkl'), 'wb') as f:
    pickle.dump(vectorizer, f)

with open(os.path.join(model_dir, 'recipe_model.pkl'), 'wb') as f:
    pickle.dump({
        'recipes': recipes,
        'ingredient_matrix': ingredient_matrix
    }, f)

print(f"[OK] Model trained on {len(recipes)} recipes.")
print(f"[OK] Vectorizer vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"[OK] Saved to {model_dir}")

# Quick test
def predict(user_ingredients, top_n=3):
    user_vec = vectorizer.transform([' '.join(user_ingredients)])
    similarities = cosine_similarity(user_vec, ingredient_matrix).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_n]
    results = []
    for idx in top_indices:
        recipe = recipes[idx]
        score = similarities[idx]
        if score > 0:
            results.append({**recipe, 'score': round(float(score), 3)})
    return results

test = predict(['tomato', 'garlic', 'pasta'])
print(f"\nTest query: tomato, garlic, pasta")
for r in test:
    print(f"  -> {r['name']} (score: {r['score']})")
