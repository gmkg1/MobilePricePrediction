# Mobile Price Range Classifier 📱

This is a machine learning project that predicts the price range of a mobile phone based on its hardware specifications. It uses a **Random Forest Classifier** and a **Flask web application** to provide a real-time prediction service.

-----

### Features ✨

  * **Integrated Pipeline**: The application automatically handles the entire machine learning pipeline, from data loading and preprocessing to model training and evaluation.
  * **Random Forest Model**: A robust `RandomForestClassifier` is used to predict the price range, providing high accuracy.
  * **Scalability**: The data is scaled using `StandardScaler` to ensure optimal model performance.
  * **Interactive Web App**: A user-friendly Flask application allows users to input phone specifications and receive an instant prediction.
  * **Performance Metrics**: The home page of the web application displays key model performance statistics, including overall accuracy and a detailed classification report.

-----

### Prerequisites 📋

This project doesn't require a `requirements.txt` file. You can install the necessary Python libraries manually using pip:

```bash
pip install pandas scikit-learn flask numpy
```

-----

### How to Run 🚀

1.  **Place the dataset**: Ensure you have a file named `dataset.csv` in the same directory as `app.py`. This dataset should contain mobile phone specifications and their corresponding `price_range`.

2.  **Run the application**: Open your terminal, navigate to the project directory, and run the following command:

    ```bash
    python app.py
    ```

3.  **Access the web app**: The terminal will display a URL (e.g., `http://127.0.0.1:5000`). Open this link in your web browser.

The model will be trained automatically when the application starts, and the web interface will become available.

-----

### Project Structure 📂

  * `app.py`: The core script containing the model training logic, Flask application, and routing.
  * `templates/`: A folder for HTML templates used by Flask.
      * `index.html`: The main page that displays model performance metrics.
      * `predict.html`: The page with the form for making predictions.
  * `dataset.csv`: The dataset used for training and evaluating the model.
