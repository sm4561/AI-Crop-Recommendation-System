# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# --- Step 1: Load the Dataset ---
# We'll use the Crop Recommendation dataset from Kaggle.
# Make sure you have the 'Crop_recommendation.csv' file in the same directory as this script.
try:
    data = pd.read_csv('Crop_recommendation.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'Crop_recommendation.csv' not found.")
    print("Please download the dataset from: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset")
    exit()

# --- Step 2: Prepare the Data ---
# The dataset is clean, so we just need to separate the features (input) from the target (output).
# Features are the soil/climate conditions.
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
# Target is the crop label.
target = 'label'

X = data[features] # Input variables
y = data[target]   # Output variable (the crop name we want to predict)

print("\nData preparation complete. Features and target are separated.")
print("Features (X):", X.columns.tolist())
print("Target (y):", target)


# --- Step 3: Split Data into Training and Testing Sets ---
# We'll use 80% of the data to train the model and 20% to test its accuracy.
# random_state ensures that we get the same split every time we run the script, which is good for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nData split into training ({len(X_train)} rows) and testing ({len(X_test)} rows) sets.")


# --- Step 4: Build and Train the Machine Learning Model ---
# We are using the RandomForestClassifier, which is an excellent and powerful algorithm for this type of problem.
# It's an ensemble of decision trees, which makes it more accurate than a single tree.
model = RandomForestClassifier(n_estimators=100, random_state=42)

print("\nTraining the Random Forest model...")
# The 'fit' method is where the model learns from the training data.
model.fit(X_train, y_train)
print("Model training completed.")


# --- Step 5: Evaluate the Model ---
# Now we use the 20% of data that the model has never seen (X_test) to make predictions.
print("\nEvaluating model performance...")
y_pred = model.predict(X_test)

# We then compare the model's predictions (y_pred) with the actual correct answers (y_test).
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%") # Aims for 90%+


# --- Step 6: Save the Trained Model ---
# We save the trained model to a file so we can load and use it in our backend API
# without having to retrain it every time. 'crop_model.joblib' is the file that will be created.
model_filename = 'crop_model.joblib'
joblib.dump(model, model_filename)

print(f"\nModel saved successfully as '{model_filename}'.")
print("\nStep 1 is complete! You now have a trained AI model ready to be used in a backend.")
