## 🍽️ **Food Review Clustering and Sentiment Analysis**

A Flask-based web application designed to cluster and analyze food reviews based on sentiment and food item mentions.
The app processes CSV files containing reviews, groups them by food items (e.g., ratatouille, ceviche), and applies sentiment analysis using **VADER** to classify reviews as positive, neutral, or negative.

---

### 🔧 Prerequisites

To run this project successfully, make sure you have the following installed:

* Python 3.8 or higher
* Git

---

### ⚙️ Setup

1. **Clone the Repository**
   Run the following in your terminal:

   ```bash
   git clone https://github.com/Faran18/Customer-Food-Reviews-Labelling.git
   cd Customer-Food-Reviews-Labelling
   ```

2. **Create a Virtual Environment**

   * **Windows**:

     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   * **Mac/Linux**:

     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies**
   Make sure `requirements.txt` exists (you can generate one with `pip freeze > requirements.txt` if not).
   Then install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK Data**
   On first run, NLTK will automatically download required data.
   Ensure you are connected to the internet during this step.

---

### 🚀 Usage

1. **Run the Application**

   ```bash
   python main.py
   ```

   This will start the Flask server (usually at `http://127.0.0.1:5000`).

2. **Upload a CSV File**

   * Prepare a CSV with a column that contains review text.
   * Use the web interface to upload it. The app will automatically detect the review column based on content.

3. **View Results**

   * After processing, the app generates a `results.html` page.
   * It displays clustered reviews grouped by food items, with sentiment analysis (positive, neutral, negative).

---

### ✨ Features

* **Automatic Column Detection**: Finds the correct review column by analyzing text length and keywords.
* **Food Item Clustering**: Groups reviews by dishes (e.g., *ratatouille*, *ceviche*) using a predefined food list.
* **Sentiment Analysis**: Uses VADER with custom sentiment boosting for strong opinions.
* **LDA Refinement**: Applies Latent Dirichlet Allocation for sub-topic detection within food items.
* **Web Interface**: Simple upload interface built with Flask to visualize results.

---

### 📦 Dependencies

* `pandas` – for data manipulation
* `nltk` – for text processing
* `vaderSentiment` – for sentiment scoring
* `keybert` – for keyword extraction
* `scikit-learn` – for LDA and vectorization
* `flask` – for the web application


