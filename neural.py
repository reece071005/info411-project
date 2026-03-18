import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix ,roc_curve, auc

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

# Read the preprocessed dataset from Excel file
df = pd.read_excel("INFO411_master_preprocessed.xlsx")

# Display first few rows to understand structure
print("Dataset Preview:")
print(df.head())


# --------------------------------------------------
# Feature selection
# --------------------------------------------------

# Select important features related to customer behavior
features = [
    'RecencyDays',        # Days since last purchase
    'AvgBasketValue',    # Average spending per transaction
    'NumItems'           # Total number of items purchased
]

# Input variables (features)
X = df[features]

# Target variable (label)
y = df['IsChurned']   # 1 = churned, 0 = not churned


# --------------------------------------------------
# Train-test split
# --------------------------------------------------

# Split data into training (80%) and testing (20%)
# random_state ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# --------------------------------------------------
# Feature scaling (for Neural Network only)
# --------------------------------------------------

# Initialize scaler
scaler = StandardScaler()

# Fit scaler on training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Apply same transformation to test data
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# Decision Tree Model (Baseline Model)
# --------------------------------------------------

# Initialize Decision Tree with limited depth to prevent overfitting
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)

# Train model on unscaled data (trees do not require scaling)
dt_model.fit(X_train, y_train)

# Make predictions on test set
dt_predictions = dt_model.predict(X_test)


# --------------------------------------------------
# Neural Network Model (MLP Classifier)
# --------------------------------------------------

# Initialize Neural Network with 2 hidden layers
nn_model = MLPClassifier(
    hidden_layer_sizes=(16, 8),  # two layers: 16 neurons and 8 neurons
    activation='relu',           # non-linear activation function
    solver='adam',               # optimization algorithm
    max_iter=2000,               # increase iterations for convergence
    random_state=42
)

# Train model on scaled data (important for NN performance)
nn_model.fit(X_train_scaled, y_train)

# Make predictions
nn_predictions = nn_model.predict(X_test_scaled)


# --------------------------------------------------
# Evaluation Function
# --------------------------------------------------

def evaluate_model(name, y_test, predictions):
    """
    Calculate and print key classification metrics.
    Helps compare model performance.
    """

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc_score = roc_auc_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    print("\n", name)
    print("------------------")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)
    print("ROC-AUC:", auc_score)
    print("Confusion Matrix:\n", cm)


# --------------------------------------------------
# Confusion Matrix Visualization
# --------------------------------------------------

def plot_confusion_matrix(y_test, predictions, title):
    """
    Plot confusion matrix as heatmap.
    Shows model performance in terms of correct/incorrect predictions.
    """

    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()


# --------------------------------------------------
# ROC Curve Visualization
# --------------------------------------------------

def plot_roc_curve(model, X_test, y_test, title):
    """
    Plot ROC curve to evaluate classification performance.
    Shows trade-off between True Positive Rate and False Positive Rate.
    """

    # Get probability of positive class (churn)
    probs = model.predict_proba(X_test)[:, 1]

    # Compute ROC curve values
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    # Plot curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")  # baseline

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.show()


# --------------------------------------------------
# Model Evaluation
# --------------------------------------------------

# Print evaluation metrics
evaluate_model("Decision Tree", y_test, dt_predictions)
evaluate_model("Neural Network", y_test, nn_predictions)

# Plot confusion matrices
plot_confusion_matrix(y_test, dt_predictions, "Decision Tree Confusion Matrix")
plot_confusion_matrix(y_test, nn_predictions, "Neural Network Confusion Matrix")

# Plot ROC curves
plot_roc_curve(dt_model, X_test, y_test, "Decision Tree ROC Curve")
plot_roc_curve(nn_model, X_test_scaled, y_test, "Neural Network ROC Curve")