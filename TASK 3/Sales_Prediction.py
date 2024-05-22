import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_data(filepath):
    """Load the dataset from a CSV file."""
    return pd.read_csv(filepath)

def explore_data(data):
    """Generate descriptive statistics and pairplot for the dataset."""
    print(data.describe())
    sns.pairplot(data)
    plt.show()

def plot_heatmap(data):
    """Plot the heatmap of the correlation matrix."""
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.show()

def train_model(X_train, Y_train):
    """Train a linear regression model."""
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model

def evaluate_model(model, X_test, Y_test):
    """Evaluate the model and print metrics."""
    Y_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")
    return Y_pred

def plot_results(Y_test, Y_pred):
    """Plot actual vs predicted sales."""
    plt.scatter(Y_test, Y_pred)
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs Predicted Sales")
    plt.show()

def main():
    # Load data
    data = load_data("C:/Users/SUN/Downloads/advertising.csv")
    
    # Explore data
    explore_data(data)
    
    # Plot heatmap
    plot_heatmap(data)
    
    # Prepare data for modeling
    X = data[["TV", "Radio", "Newspaper"]]
    Y = data["Sales"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Train model
    model = train_model(X_train, Y_train)
    
    # Evaluate model
    Y_pred = evaluate_model(model, X_test, Y_test)
    
    # Plot results
    plot_results(Y_test, Y_pred)

if __name__ == "__main__":
    main()
