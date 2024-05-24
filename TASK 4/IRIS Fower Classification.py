import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
iris = pd.read_csv("IRIS.csv")

# Split features and target variable
X = iris.drop(columns=['species'])
y = iris['species']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Mapping of species labels for display
species_map = {
    'Iris-setosa': 'setosa',
    'Iris-versicolor': 'versicolor',
    'Iris-virginica': 'virginica'
}

def predict_species():
    """
    Predicts the species of an Iris flower based on user input.
    """
    print("Enter the sepal length, sepal width, petal length, and petal width of the Iris flower:")
    try:
        sepal_length = float(input("Sepal Length (cm): "))
        sepal_width = float(input("Sepal Width (cm): "))
        petal_length = float(input("Petal Length (cm): "))
        petal_width = float(input("Petal Width (cm): "))

        user_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = model.predict(user_data)
        predicted_species = species_map[prediction[0]]

        print(f'The predicted species is: {predicted_species}')
    except ValueError:
        print("Please enter valid numeric values for sepal length, sepal width, petal length, and petal width.")

if __name__ == "__main__":
    predict_species()
