import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

def load_data(file_path):
    """Load the IMDb Movies dataset"""
    return pd.read_csv(file_path, encoding='latin1')

def preprocess_data(data):
    """Preprocess the dataset"""
    # Check for missing values
    print("Missing values:\n", data.isnull().sum())

    # Define features and target variable
    features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
    target = 'Rating'
    X = data[features]
    Y = data[target]

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Define categorical features and create a preprocessing pipeline
    categorical_features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('cat', categorical_transformer, categorical_features)
    ])
    
    return X_train, X_test, Y_train, Y_test, preprocessor

def train_model(X_train, Y_train, preprocessor):
    """Train the linear regression model"""
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    model.fit(X_train, Y_train)
    return model

def evaluate_model(model, X_test, Y_test):
    """Evaluate the model on the testing set"""
    Y_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    print(f"Mean Squared Error: {mse}")

def main():
    # Load the dataset
    data = load_data("IMDb-Movies-India.csv")

    # Preprocess the dataset
    X_train, X_test, Y_train, Y_test, preprocessor = preprocess_data(data)

    # Train the model
    model = train_model(X_train, Y_train, preprocessor)

    # Evaluate the model
    evaluate_model(model, X_test, Y_test)

    # Take input from user for movie details
    genre = input("Enter the genre of the movie: ")
    director = input("Enter the director of the movie: ")
    actor1 = input("Enter the first actor of the movie: ")
    actor2 = input("Enter the second actor of the movie: ")
    actor3 = input("Enter the third actor of the movie: ")

    # Create a DataFrame with user input
    input_data = pd.DataFrame({
        'Genre': [genre],
        'Director': [director],
        'Actor 1': [actor1],
        'Actor 2': [actor2],
        'Actor 3': [actor3]
    })

    # Predict the rating for the input movie
    predicted_rating = model.predict(preprocessor.transform(input_data))
    print("Predicted Rating:", predicted_rating[0])

if __name__ == "__main__":
    main()
