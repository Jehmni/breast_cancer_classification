{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dd0e297a-1fc3-4708-90b5-b9af9e5ad5c4",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "import joblib\n",
    "\n",
    "# Define column names as per the dataset description\n",
    "column_names = [\n",
    "    'id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',\n",
    "    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',\n",
    "    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',\n",
    "    'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',\n",
    "    'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst',\n",
    "    'fractal_dimension_worst'\n",
    "]\n",
    "\n",
    "# Load the dataset\n",
    "data = pd.read_csv('../data/wdbc.data', header=None, names=column_names)\n",
    "\n",
    "# Encode the target variable\n",
    "data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})\n",
    "\n",
    "# Drop the 'id' column as it's not useful for prediction\n",
    "data = data.drop(columns=['id'])\n",
    "\n",
    "# Split the data into features and target\n",
    "X = data.drop(columns=['diagnosis'])\n",
    "y = data['diagnosis']\n",
    "\n",
    "# Split the data into training and testing sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Standardize the features\n",
    "scaler = StandardScaler()\n",
    "X_train = scaler.fit_transform(X_train)\n",
    "X_test = scaler.transform(X_test)\n",
    "\n",
    "# Save the preprocessed data and scaler\n",
    "joblib.dump((X_train, X_test, y_train, y_test), '../data/preprocessed_data.pkl')\n",
    "joblib.dump(scaler, '../data/scaler.pkl')\n"
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
