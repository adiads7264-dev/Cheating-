# Cheating-import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Suppress minor warnings for a cleaner terminal experience
warnings.filterwarnings('ignore')

# ==========================================
# 1. DATASET: Recipes and Ingredients
# ==========================================
# In a full project, you would load this from a CSV. 
# For this mini-project, we use a built-in dictionary.
RECIPES_DB = {
    'recipe_name': [
        'Classic Omelette', 
        'Tomato Basil Pasta', 
        'Chicken Fried Rice',
        'Avocado Toast', 
        'Garlic Butter Shrimp', 
        'Hearty Chicken Salad',
        'Pancakes'
    ],
    'ingredients': [
        ['eggs', 'butter', 'salt', 'pepper', 'cheese'],
        ['pasta', 'tomatoes', 'basil', 'garlic', 'olive oil'],
        ['rice', 'chicken', 'eggs', 'soy sauce', 'onion', 'peas', 'carrots'],
        ['bread', 'avocado', 'lemon', 'salt', 'red pepper flakes'],
        ['shrimp', 'butter', 'garlic', 'parsley', 'lemon'],
        ['chicken', 'mayonnaise', 'celery', 'onion', 'salt', 'pepper'],
        ['flour', 'milk', 'eggs', 'butter', 'sugar', 'baking powder']
    ]
}

# ==========================================
# 2. AI ENGINE INITIALIZATION
# ==========================================
def train_ai_model():
    """Converts ingredient lists into mathematical vectors for comparison."""
    df = pd.DataFrame(RECIPES_DB)
    
    # Machine learning models read strings, not lists. 
    # We join ['eggs', 'butter'] into "eggs butter"
    df['ingredients_str'] = df['ingredients'].apply(lambda x: ' '.join(x))
    
    # Initialize TF-IDF (Term Frequency-Inverse Document Frequency)
    # This weighs unique ingredients higher than common ones (like salt).
    vectorizer = TfidfVectorizer()
    
    # Train the model and create the matrix of our recipe database
    tfidf_matrix = vectorizer.fit_transform(df['ingredients_str'])
    
    return df, vectorizer, tfidf_matrix

# ==========================================
# 3. RECOMMENDATION LOGIC
# ==========================================
def get_recommendations(user_input_list, df, vectorizer, tfidf_matrix, top_n=3):
    """Finds the closest matching recipes based on user ingredients."""
    
    # Convert user input into a single string and vectorize it
    user_str = ' '.join(user_input_list)
    user_vector = vectorizer.transform([user_str])
    
    # Calculate the mathematical distance (similarity) between the user's 
    # ingredients and all recipes in the database.
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    
    # Attach scores to the dataframe
    df['match_score'] = similarity_scores
    
    # Filter out 0% matches and sort by highest score
    matches = df[df['match_score'] > 0].sort_values(by='match_score', ascending=False)
    
    return matches.head(top_n)

# ==========================================
# 4. MAIN APPLICATION LOOP
# ==========================================
def main():
    print("\n" + "="*50)
    print(" ð§âð³ WELCOME TO THE AI PANTRY CHEF ð§âð³ ")
    print("="*50)
    print("Training AI model on recipe database...\n")
    
    df, vectorizer, tfidf_matrix = train_ai_model()
    
    while True:
        print("\nWhat ingredients do you have? (separate with spaces)")
        print("Example: chicken rice eggs")
        print("Type 'quit' or 'exit' to stop.")
        
        raw_input = input("\n> ").strip().lower()
        
        if raw_input in ['quit', 'exit', 'q']:
            print("\nHappy cooking! Goodbye. ð\n")
            break
            
        if not raw_input:
            continue
            
        # Split the input into a list of words
        user_ingredients = raw_input.split()
        
        # Get recommendations
        results = get_recommendations(user_ingredients, df, vectorizer, tfidf_matrix)
        
        # Display results
        if results.empty:
            print("\nâ No matching recipes found. Try different ingredients!")
        else:
            print(f"\nâ Top {len(results)} Recipes Found:\n" + "-"*30)
            
            user_set = set(user_ingredients)
            
            for index, row in results.iterrows():
                recipe_set = set(row['ingredients'])
                
                # Calculate what they have vs what they need
                have = list(recipe_set.intersection(user_set))
                need = list(recipe_set.difference(user_set))
                
                match_pct = int(row['match_score'] * 100)
                
                print(f"ð½ï¸  {row['recipe_name']} ({match_pct}% Match)")
                if have:
                    print(f"   â You have: {', '.join(have)}")
                if need:
                    print(f"   ð You need: {', '.join(need)}")
                print()

if __name__ == "__main__":
    main()
