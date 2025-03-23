
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset."""
    try:
        df = pd.read_csv(file_path)

        df.rename(columns={
            "Land Size (acres)": "LandSize",
            "Soil pH": "SoilPH",
            "Past Yield (kg per acre)": "PastYield",
            "Past Rainfall (mm in last season)": "PastRainfall",
            "Avg Temperature (°C in last season)": "AvgTemperature"
        }, inplace=True)

        categorical_cols = ["Country", "Region", "Soil Type", "Crop Type"]

        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        df = df.dropna()

        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        return None


def train_model(df):
    """Train the Random Forest model and evaluate its performance."""
    try:
        X = df.drop(columns=["Credit Score"])
        y = df["Credit Score"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print("\nModel Performance:")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"R² Score: {r2}")

        return model, X.columns.tolist()
    except Exception as e:
        print(f"Error during model training: {e}")
        return None, None


def save_model_and_columns(model, columns, model_file="random_forest_credit_score.pkl",
                           columns_file="trained_columns.pkl"):
    """Save the trained model and column names."""
    try:
        joblib.dump(model, model_file)
        joblib.dump(columns, columns_file)
        print("\nModel and trained column names saved.")
    except Exception as e:
        print(f"Error saving model or columns: {e}")


def main():
    """Main function to run the script."""
    file_path = "Crop_Details.csv"

    # Load and preprocess data
    df = load_and_preprocess_data(file_path)
    if df is None:
        return

    # Train the model
    model, trained_columns = train_model(df)
    if model is None or trained_columns is None:
        return

    # Save the model and column names
    save_model_and_columns(model, trained_columns)


if __name__ == "__main__":
    main()