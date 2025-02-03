import calendar
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

df = pd.read_csv("/Users/daisydoyle/Documents/gitlab-projects/study/src/bank-direct-marketing-2024.csv", sep=';', quoting=3, quotechar='"')
df = df.applymap(lambda x: x.strip('"') if isinstance(x, str) else x)
df.columns = df.columns.str.strip('"')
df.drop_duplicates()

df['default'] = df['default'].map({'no':0,'yes':1,'unknown':0})
df['y'] = df['y'].map({'no':0,'yes':1})
df['month'] = df['month'].str.title()
df['month'] = df['month'].apply(lambda x: list(calendar.month_abbr).index(x))
day_of_week_map = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5}
df['day_of_week'] = df['day_of_week'].map(day_of_week_map)
X = df.drop(["y"], axis=1)
y = df['y'] 

numeric_features = ["age", 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_features = ["job", "marital", 'education', "contact", "poutcome"]
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


def train_and_execute_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    confusion_matrix_data = confusion_matrix(y_test, y_pred)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix_data.ravel()
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_data).plot()
    accuracy = accuracy_score(y_test, y_pred)

    if '1' in report:
        precision = report['1']['precision']
        recall = report['1']['recall']
        f1_score = report['1']['f1-score']
    else:
        precision = recall = f1_score = 0.0

    fpr = ((false_positive)/(false_positive+true_negative))
    fnr = (false_negative/(true_positive + false_negative))
    tpr = true_positive/(true_positive + false_negative)
    tnr = (true_negative/(true_negative+false_positive))

    print(f"""
    Model: {model_name}
    Precision: {precision:.2f}
    Recall: {recall:.2f}
    Accuracy: {accuracy:.2f}
    F1 Score: {f1_score:.2f}

    FPR: {fpr:.2f}
    TPR: {tpr:.2f}
    TNR: {tnr:.2f}
    FNR: {fnr:.2f}
        """)
    

random_forest = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(class_weight='balanced'))])
train_and_execute_model(random_forest, 'Random Forest')

logistic_regression = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LogisticRegression(class_weight='balanced'))])
train_and_execute_model(logistic_regression, 'Logistic Regression')

