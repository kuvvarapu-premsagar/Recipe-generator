import json
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
CORS(app)

# Load trained model
model_dir = os.path.join(os.path.dirname(__file__), 'model')

def load_model():
    with open(os.path.join(model_dir, 'vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(model_dir, 'recipe_model.pkl'), 'rb') as f:
        model_data = pickle.load(f)
    return vectorizer, model_data['recipes'], model_data['ingredient_matrix']

vectorizer, recipes, ingredient_matrix = load_model()

def find_recipes(user_ingredients, top_n=5):
    """Find best matching recipes for given ingredients."""
    # Normalize input
    cleaned = [ing.strip().lower() for ing in user_ingredients if ing.strip()]
    if not cleaned:
        return []
    
    user_text = ' '.join(cleaned)
    user_vec = vectorizer.transform([user_text])
    similarities = cosine_similarity(user_vec, ingredient_matrix).flatten()
    
    top_indices = np.argsort(similarities)[::-1][:top_n]
    results = []
    for idx in top_indices:
        recipe = recipes[idx]
        score = float(similarities[idx])
        if score > 0:
            # Calculate how many ingredients match
            recipe_ings = set(recipe['ingredients'])
            user_ings = set(cleaned)
            matched = recipe_ings & user_ings
            missing = recipe_ings - user_ings
            
            results.append({
                'name': recipe['name'],
                'cuisine': recipe['cuisine'],
                'prep_time': recipe['prep_time'],
                'cook_time': recipe['cook_time'],
                'servings': recipe['servings'],
                'ingredients': recipe['ingredients'],
                'instructions': recipe['instructions'],
                'score': round(score * 100, 1),
                'matched_ingredients': list(matched),
                'missing_ingredients': list(missing)
            })
    
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.get_json()
    if not data or 'ingredients' not in data:
        return jsonify({'error': 'No ingredients provided'}), 400
    
    raw = data['ingredients']
    if isinstance(raw, str):
        ingredients = [i.strip() for i in raw.replace(',', ' ').split() if i.strip()]
    else:
        ingredients = raw
    
    if not ingredients:
        return jsonify({'error': 'Please provide at least one ingredient'}), 400
    
    top_n = data.get('top_n', 5)
    results = find_recipes(ingredients, top_n=top_n)
    
    return jsonify({
        'query_ingredients': ingredients,
        'recipes': results,
        'count': len(results)
    })

@app.route('/api/recipes', methods=['GET'])
def all_recipes():
    return jsonify({'recipes': recipes, 'count': len(recipes)})

if __name__ == '__main__':
    print("AI Chef Assistant starting...")
    print(f"Loaded {len(recipes)} recipes")
    app.run(debug=True, port=5000)
