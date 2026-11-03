
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load dataset
data = pd.read_csv("student-mat.csv", sep=";")

# 2. Select input features and target
features = [
    "age",
    "studytime",
    "failures",
    "absences",
    "Medu",
    "Fedu",
    "traveltime",
    "health"
]

X = data[features]
y = data["G3"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Define models
models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=4, random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=4,
        random_state=42
    )
}

# 5. Cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

results = []

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_absolute_error"
    )

    mean_mae = -scores.mean()

    results.append({
        "Model": name,
        "CV MAE": round(mean_mae, 2)
    })

    print(name, "Average CV MAE:", round(mean_mae, 2))

# 6. Select the best model
comparison = pd.DataFrame(results)

best_model_name = comparison.loc[
    comparison["CV MAE"].idxmin(), "Model"
]

best_model = models[best_model_name]

# 7. Train selected model
best_model.fit(X_train, y_train)

# 8. Evaluate on unseen test data
predictions = best_model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nMODEL COMPARISON")
print(comparison.to_string(index=False))

print("\nSELECTED MODEL:", best_model_name)
print("Test MAE:", round(mae, 2))
print("Test R2 Score:", round(r2, 2))

# 9. Save trained model and feature names
joblib.dump(
    {
        "model": best_model,
        "features": features,
        "model_name": best_model_name
    },
    "best_model.pkl"
)

print("\nModel saved successfully!")
