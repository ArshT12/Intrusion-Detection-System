
import pandas as pd

file_path = 'RT_IOT2022.csv'
dataset = pd.read_csv(file_path)

print("Dataset loaded successfully")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler


labelencoder = LabelEncoder()
dataset['proto'] = labelencoder.fit_transform(dataset['proto'])
dataset['service'] = labelencoder.fit_transform(dataset['service'])


dataset.ffill(inplace=True)


X = dataset.drop(['Attack_type', 'Unnamed: 0'], axis=1)
y = dataset['Attack_type']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(max_iter=5000)  
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)



model_no_reg = LogisticRegression(C=1e10, solver='saga', max_iter=5000)

X_train_scaled_small = X_train_scaled[:10000]  # Using only 1000 samples for speed
y_train_small = y_train[:10000]

model_no_reg.fit(X_train_scaled_small, y_train_small)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


predictions_no_reg = model_no_reg.predict(X_test_scaled)
print("no Regularization Metrics:")

accuracy = accuracy_score(y_test, predictions_no_reg)
print("Accuracy:", accuracy)


precision = precision_score(y_test, predictions_no_reg, average='weighted',zero_division=0)
print("Precision:", precision)


recall = recall_score(y_test, predictions_no_reg, average='weighted')
print("Recall:", recall)


f1 = f1_score(y_test, predictions_no_reg, average='weighted',zero_division=0)
print("F1 Score:", f1)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


X_train_scaled_small = X_train_scaled[:1000]  
y_train_small = y_train[:1000]


model_l1 = LogisticRegression(penalty='l1', C=0.15, solver='saga', max_iter=5000, tol=0.1, verbose=1)


model_l1.fit(X_train_scaled_small, y_train_small)


predictions_l1 = model_l1.predict(X_test_scaled)


accuracy_l1 = accuracy_score(y_test, predictions_l1)


precision_l1 = precision_score(y_test, predictions_l1, average='weighted', zero_division=0)


recall_l1 = recall_score(y_test, predictions_l1, average='weighted')


f1_l1 = f1_score(y_test, predictions_l1, average='weighted', zero_division=0)


print("L1 Regularization Metrics:")
print("Accuracy:", accuracy_l1)
print("Precision:", precision_l1)
print("Recall:", recall_l1)
print("F1 Score:", f1_l1)



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


model_l2 = LogisticRegression(C=0.15, solver='saga', max_iter=5000)


model_l2.fit(X_train_scaled, y_train)


predictions_l2 = model_l2.predict(X_test_scaled)



accuracy_l2 = accuracy_score(y_test, predictions_l2)
precision_l2 = precision_score(y_test, predictions_l2, average='weighted', zero_division=0)
recall_l2 = recall_score(y_test, predictions_l2, average='weighted')
f1_l2 = f1_score(y_test, predictions_l2, average='weighted', zero_division=0)


print("L2 Regularization Metrics:")
print("Accuracy:", accuracy_l2)
print("Precision:", precision_l2)
print("Recall:", recall_l2)
print("F1 Score:", f1_l2)


X_train_scaled_small = X_train_scaled[:2000]  
y_train_small = y_train[:2000]

model_elastic = LogisticRegression(penalty='elasticnet', C=0.2, l1_ratio=0.4, solver='saga', max_iter=5000)


model_elastic.fit(X_train_scaled_small, y_train_small)


predictions_elastic = model_elastic.predict(X_test_scaled)





accuracy_elastic = accuracy_score(y_test, predictions_elastic)
precision_elastic = precision_score(y_test, predictions_elastic, average='weighted', zero_division=0)
recall_elastic = recall_score(y_test, predictions_elastic, average='weighted')
f1_elastic = f1_score(y_test, predictions_elastic, average='weighted', zero_division=0)


print("Elastic Net Regularization Metrics:")
print("Accuracy:", accuracy_elastic)
print("Precision:", precision_elastic)
print("Recall:", recall_elastic)
print("F1 Score:", f1_elastic)

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


pca = PCA(n_components=0.95)  


pipeline = make_pipeline(StandardScaler(), pca, LogisticRegression(solver='saga', max_iter=5000))


pipeline.fit(X_train, y_train)  


predictions_pca = pipeline.predict(X_test)





accuracy_pca = accuracy_score(y_test, predictions_pca)
precision_pca = precision_score(y_test, predictions_pca, average='weighted', zero_division=0)
recall_pca = recall_score(y_test, predictions_pca, average='weighted')
f1_pca = f1_score(y_test, predictions_pca, average='weighted', zero_division=0)


print("PCA + Logistic Regression Metrics:")
print("Accuracy:", accuracy_pca)
print("Precision:", precision_pca)
print("Recall:", recall_pca)
print("F1 Score:", f1_pca)


pipeline_no_reg_with_pca = make_pipeline(
    PCA(n_components=0.95),  
    LogisticRegression(C=1e10, solver='saga',max_iter=5000)
)


pipeline_no_reg_with_pca.fit(X_train_scaled, y_train)


predictions_no_reg_with_pca = pipeline_no_reg_with_pca.predict(X_test_scaled)


accuracy_no_reg_with_pca = accuracy_score(y_test, predictions_no_reg_with_pca)
precision_no_reg_with_pca = precision_score(y_test, predictions_no_reg_with_pca, average='weighted', zero_division=0)
recall_no_reg_with_pca = recall_score(y_test, predictions_no_reg_with_pca, average='weighted')
f1_no_reg_with_pca = f1_score(y_test, predictions_no_reg_with_pca, average='weighted', zero_division=0)

print("No Regularization with PCA Metrics:")
print("Accuracy:", accuracy_no_reg_with_pca)
print("Precision:", precision_no_reg_with_pca)
print("Recall:", recall_no_reg_with_pca)
print("F1 Score:", f1_no_reg_with_pca)



pipeline_l1_with_pca = make_pipeline(
    PCA(n_components=0.95),
    LogisticRegression(penalty='l1', C=0.15, solver='saga', max_iter=5000)
)


pipeline_l1_with_pca.fit(X_train_scaled, y_train)


predictions_l1_with_pca = pipeline_l1_with_pca.predict(X_test_scaled)


accuracy_l1_with_pca = accuracy_score(y_test, predictions_l1_with_pca)
precision_l1_with_pca = precision_score(y_test, predictions_l1_with_pca, average='weighted', zero_division=0)
recall_l1_with_pca = recall_score(y_test, predictions_l1_with_pca, average='weighted')
f1_l1_with_pca = f1_score(y_test, predictions_l1_with_pca, average='weighted', zero_division=0)

print("L1 Regularization with PCA Metrics:")
print("Accuracy:", accuracy_l1_with_pca)
print("Precision:", precision_l1_with_pca)
print("Recall:", recall_l1_with_pca)
print("F1 Score:", f1_l1_with_pca)



pca = PCA(n_components=0.95)  
model_l2_with_pca = make_pipeline(PCA(n_components=0.95), LogisticRegression(C=0.15, penalty='l2', solver='saga', max_iter=5000))


model_l2_with_pca.fit(X_train_scaled, y_train)


predictions_l2_with_pca = model_l2_with_pca.predict(X_test_scaled)


accuracy_l2_with_pca = accuracy_score(y_test, predictions_l2_with_pca)
precision_l2_with_pca = precision_score(y_test, predictions_l2_with_pca, average='weighted', zero_division=0)
recall_l2_with_pca = recall_score(y_test, predictions_l2_with_pca, average='weighted')
f1_l2_with_pca = f1_score(y_test, predictions_l2_with_pca, average='weighted', zero_division=0)

print("L2 Regularization with PCA Metrics:")
print("Accuracy:", accuracy_l2_with_pca)
print("Precision:", precision_l2_with_pca)
print("Recall:", recall_l2_with_pca)
print("F1 Score:", f1_l2_with_pca)


pipeline_elastic_with_pca = make_pipeline(
    PCA(n_components=0.95),
    LogisticRegression(penalty='elasticnet', C=0.15, l1_ratio=0.5, solver='saga', max_iter=5000)
)


pipeline_elastic_with_pca.fit(X_train_scaled, y_train)


predictions_elastic_with_pca = pipeline_elastic_with_pca.predict(X_test_scaled)


accuracy_elastic_with_pca = accuracy_score(y_test, predictions_elastic_with_pca)
precision_elastic_with_pca = precision_score(y_test, predictions_elastic_with_pca, average='weighted', zero_division=0)
recall_elastic_with_pca = recall_score(y_test, predictions_elastic_with_pca, average='weighted')
f1_elastic_with_pca = f1_score(y_test, predictions_elastic_with_pca, average='weighted', zero_division=0)

print("Elastic Net with PCA Metrics:")
print("Accuracy:", accuracy_elastic_with_pca)
print("Precision:", precision_elastic_with_pca)
print("Recall:", recall_elastic_with_pca)
print("F1 Score:", f1_elastic_with_pca)



