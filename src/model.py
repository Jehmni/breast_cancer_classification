{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5aeff80c-999b-4af1-9b90-356cd891ef8e",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import classification_report, accuracy_score\n",
    "import joblib\n",
    "\n",
    "# Load the preprocessed data\n",
    "X_train, X_test, y_train, y_test = joblib.load('../data/preprocessed_data.pkl')\n",
    "\n",
    "# Initialize the model\n",
    "model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "\n",
    "# Train the model\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Make predictions\n",
    "y_pred = model.predict(X_test)\n",
    "\n",
    "# Evaluate the model\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "report = classification_report(y_test, y_pred)\n",
    "print(f'Accuracy: {accuracy}')\n",
    "print(report)\n",
    "\n",
    "# Save the trained model\n",
    "joblib.dump(model, '../models/breast_cancer_model.pkl')\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
