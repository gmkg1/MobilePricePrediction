import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Model Training Function ---
def train_model():
    """
    Trains the Random Forest model and returns the trained model,
    scaler, performance metrics, and the feature order.
    """
    print("--- Loading data and starting model training ---")
    
    # Load the dataset
    df = pd.read_csv('dataset.csv')
    
    # Separate features and target
    X = df.drop('price_range', axis=1)
    y = df['price_range']
    
    # Capture the exact order of columns used for training
    feature_names = X.columns.tolist()
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize and fit the scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the RandomForestClassifier model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print("--- Model training complete ---")
    
    return model, scaler, accuracy, report, feature_names

# --- 2. Flask App Initialization ---
app = Flask(__name__)

# Train the model when the application starts
# The trained objects will be stored in these global variables
model, scaler, accuracy, class_report, feature_names = train_model()

# Define the price range mapping for user-friendly output
price_range_map = {
    '0': 'Low Cost',
    '1': 'Medium Cost',
    '2': 'High Cost',
    '3': 'Very High Cost'
}

# --- 3. Define Flask Routes ---

@app.route('/')
def home():
    """Renders the home page displaying model performance stats."""
    report_items = []
    # Format the classification report for display in the HTML table
    for label, metrics in class_report.items():
        if isinstance(metrics, dict):
            # Use the friendly name for class labels (0, 1, 2, 3)
            display_label = price_range_map.get(label, label)
            report_items.append((display_label, metrics))
        else:
            # Handle summary rows like 'accuracy', 'macro avg', etc.
            report_items.append((label, {'value': metrics}))
            
    return render_template('index.html', accuracy=accuracy, report_items=report_items)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Renders the prediction form and displays prediction results."""
    prediction_result = None
    if request.method == 'POST':
        try:
            # *** FIX: Read features from the form in the correct order ***
            features = []
            for name in feature_names:
                # The form gives us strings, so we cast them to float
                features.append(float(request.form[name]))

            # Create a numpy array, reshape for a single prediction, and scale
            final_features = np.array(features).reshape(1, -1)
            scaled_features = scaler.transform(final_features)
            
            # Get the model's prediction
            prediction = model.predict(scaled_features)
            
            # Map the numeric prediction to its text label
            predicted_class = str(prediction[0])
            prediction_result = price_range_map[predicted_class]

        except (ValueError, KeyError) as e:
            prediction_result = f"Error: Invalid input. Please ensure all fields are filled correctly. Details: {e}"

    return render_template('predict.html', prediction=prediction_result)

# --- 4. Run the Application ---
if __name__ == '__main__':
    print("--- Starting Flask server ---")
    app.run(debug=True)