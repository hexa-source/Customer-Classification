


from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, classification_report

import time

from src.data_prep import tiers_cat, cat_features, num_features, load_data
from src.data_prep import load_data, split_features_target
from src.preprocessor import Pipeline, feature_names



data = "data/processed/bank_customer(1).csv"

def train(data_path: str, test_size: float = 0.30, random_state: int = 42):

    df = load_data(data_path)    
    X, Y = split_features_target(df)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    transformer = ColumnTransformer(transformers = [
        ("cat", OneHotEncoder(handle_unknown = "ignore"), cat_features),
        ("tier", OrdinalEncoder(), tiers_cat),
        ("num", StandardScaler(), num_features)
        ])
    pipeline = Pipeline(steps = [
        ("transformer", transformer),
        ("classifier", MLPClassifier())
        ])
    t0 = time.time()
    pipeline.fit(X_train, Y_train)
    train_time = time.time() - t0
    print(f"Model trained in {train_time:.3f} seconds")

    print("\n-------Detailed Classification Report-------\n")
    print(classification_report(Y_test, pipeline.predict(X_test)))

    return {
        "Model": pipeline,
        "X_train": X_train,
        "X_test": X_test,
        "Y_train": Y_train,
        "Y_test": Y_test,
    }




if __name__ == "__main__":
    train(data)
