import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from keybert import KeyBERT
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download necessary data for lemmatizer
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()
kw_model = KeyBERT()

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Predefined list of food items (expand as needed)
FOOD_ITEMS = [
    "raclette", "ceviche", "brownie", "dim sum", "soup", "truffle", "cappuccino", "tapas",
    "beef bourguignon", "ramen", "paella", "escargot", "cheesecake", "sushi", "foie gras",
    "ratatouille", "miso soup", "coq au vin", "steak", "duck confit", "bouillabaisse",
    "cordon bleu", "iced tea", "sashimi", "merlot", "quinoa"
]

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s<>/]', '', str(text))  # Allow < and > for HTML tags, remove other non-alphabetic chars
    text = re.sub(r'<br/br>', '<br>', text)  # Replace <br/br> with <br>
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces
    words = text.strip().lower().split()
    return ' '.join(lemmatizer.lemmatize(word) for word in words)

def get_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if any(keyword in text.lower() for keyword in ['but', 'however', 'though']):
        return 'neutral'  # Force neutral if mixed indicators present
    if score >= 0.2:
        return 'positive'
    elif score <= -0.2:
        return 'negative'
    else:
        return 'neutral'
    
    

def extract_food_item(text):
    text_lower = text.lower()
    for item in FOOD_ITEMS:
        if item in text_lower:
            return item
    return "Other"  # Default cluster for unmatched items

def get_keybert_labels(df, num_clusters):
    cluster_names = {}
    for i in range(num_clusters):
        cluster_reviews = df[df['cluster'] == i]['cleaned']
        if cluster_reviews.empty:
            cluster_names[i] = f"Topic {i+1}"
            continue
        combined_text = " ".join(cluster_reviews.tolist())
        keywords = kw_model.extract_keywords(
            combined_text,
            keyphrase_ngram_range=(1, 2),
            stop_words='english'
        )
        significant_keywords = [kw[0] for kw in keywords if kw[1] > 0.1]
        if not significant_keywords:
            cluster_names[i] = f"Topic {i+1}"
        else:
            keyword_str = significant_keywords[0]
            if len(significant_keywords) > 1:
                keyword_str += f" and {significant_keywords[1]}"
            cluster_names[i] = keyword_str.title() + " Reviews"
    return cluster_names

def process_and_cluster(df, num_clusters=5):
    print("Starting clustering process...")
    results = {"clusters": {}}

    if 'Review' not in df.columns:
        print("Error: No 'Review' column found")
        results["clusters"]["Error"] = {"sentiment_distribution": {}, "reviews": ["No 'Review' column found in uploaded file."]}
        return results, df  # Return original df with error

    df = df.dropna(subset=['Review'])
    print(f"Rows after dropping NaN: {len(df)}")
    df['cleaned'] = df['Review'].apply(clean_text)
    print("Text cleaning with lemmatization completed")

    # Initial grouping by food item
    df['food_item'] = df['Review'].apply(extract_food_item)
    print(f"Identified food items: {df['food_item'].unique()}")

    # Apply sentiment analysis
    df['sentiment'] = df['Review'].apply(get_sentiment)
    print("Sentiment analysis completed")

    # Group by food item and optionally refine with LDA within each group
    for food_item in df['food_item'].unique():
        item_df = df[df['food_item'] == food_item]
        if len(item_df) > 1:  # Only apply LDA if multiple reviews
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=500,
                ngram_range=(1, 2)
            )
            print(f"Vectorizing text for {food_item}...")
            X = vectorizer.fit_transform(item_df['cleaned'])
            print(f"TF-IDF matrix shape for {food_item}: {X.shape}")

            lda = LatentDirichletAllocation(n_components=min(num_clusters, len(item_df)), random_state=42)
            print(f"Running LDA for {food_item}...")
            cluster_labels = lda.fit_transform(X).argmax(axis=1)
            item_df['cluster'] = cluster_labels
        else:
            item_df['cluster'] = 0  # Single review gets its own cluster

        reviews = item_df['Review'].tolist()
        sentiments = item_df['sentiment'].value_counts().to_dict()
        results["clusters"][f"{food_item.capitalize()} Reviews"] = {
            "reviews": reviews,
            "sentiment_distribution": sentiments
        }
    print("Results aggregation completed")
    return results, df  # Return both results and modified df

if __name__ == "__main__":
    df = pd.read_csv('cleaned_food_reviews2.csv')
    results, _ = process_and_cluster(df)
    print(results)