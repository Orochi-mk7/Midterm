import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# load the ARFF
with open('KDDTest+.arff', 'r') as f:
    data, meta = arff.loadarff(f)

# convert the ARFF data to a Pandas DataFrame
df = pd.DataFrame(data)

# print unique values in the 'protocol_type' column
print("Unique values in 'protocol_type' column:", df['protocol_type'].unique())

# handle unexpected values in 'protocol_type'
unexpected_values = ['unexpected_value', 'another_value']
df['protocol_type'] = df['protocol_type'].replace(unexpected_values, 'unknown')

# manually encode 'protocol_type' to numerical values
protocol_type_mapping = {'tcp': 0, 'udp': 1, 'icmp': 2, 'unknown': -1}
df['protocol_type'] = df['protocol_type'].map(protocol_type_mapping)

# convert other categorical columns to numerical representation
label_encoder = LabelEncoder()
df['service'] = label_encoder.fit_transform(df['service'].astype(str))
df['flag'] = label_encoder.fit_transform(df['flag'].astype(str))
df['label'] = label_encoder.fit_transform(df['label'].astype(str))

# assuming the labels are now in a column named 'label'
X = df.drop('label', axis=1)  # Features
y = df['label']  # Labels

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create a RandomForest classifier
classifier = RandomForestClassifier()

# train the classifier using the training data
classifier.fit(X_train, y_train)

# use the trained classifier to predict labels for the test set
predictions = classifier.predict(X_test)

# evaluate the classifier's performance
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# display a detailed report on the classifier's performance
print("Classification Report:")
print(classification_report(y_test, predictions))