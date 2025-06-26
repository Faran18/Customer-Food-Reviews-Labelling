from flask import render_template, request
from .clustering import process_and_cluster
import os
import pandas as pd

def init_routes(app):
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/upload', methods=['POST'])
    def upload():
        file = request.files['file']
        if file:
            print(f"Received file: {file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            print(f"Saving to: {filepath}")
            file.save(filepath)

            print("Reading CSV...")
            df = pd.read_csv(filepath)
            print(f"DataFrame shape: {df.shape}, Columns: {df.columns.tolist()}")  # Debug columns
            results, processed_df = process_and_cluster(df)  # Unpack the tuple
            print(f"Clustering results: {results}")

            if "Error" in results["clusters"]:
                return render_template('error.html', message=results["clusters"]["Error"]["reviews"][0])
            print(f"Processed DataFrame columns: {processed_df.columns.tolist()}")  # Debug sentiment column
            return render_template('results.html', clusters=results["clusters"], df=processed_df)  # Use processed_df
        return "No file uploaded", 400