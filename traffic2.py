import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = 'traffic.csv'
df = pd.read_csv(file_path)

# Convert format
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Extract  'DateTime' column
df['hour'] = df['DateTime'].dt.hour
df['day_of_week'] = df['DateTime'].dt.dayofweek
df['month'] = df['DateTime'].dt.month

# Define the traffic light state based on the number of vehicles
def determine_traffic_light_state(vehicles):
    if vehicles < 10:
        return 'Green'
    elif 10 <= vehicles < 20:
        return 'Yellow'
    else:
        return 'Red'

# Apply the function to create a new column 'traffic_light_state'
df['traffic_light_state'] = df['Vehicles'].apply(determine_traffic_light_state)

# Mapping
state_mapping = {'Green': 0, 'Yellow': 1, 'Red': 2}
df['traffic_light_state'] = df['traffic_light_state'].map(state_mapping)

# Define the features and target
X = df[['Junction', 'Vehicles', 'hour', 'day_of_week', 'month']]
y = df['traffic_light_state']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Green', 'Yellow', 'Red'])

print(f'Accuracy: {accuracy * 100:.2f}%')
print(report)

# Example of predicting for new input data
firstNumber = input("Junction: ")
secondNumber = input("Vehicles: ")
thirdNumber = input("Hour: ")
fourthNumber = input("Day of Week: ")
month = input("Month: ")

new_data = {
    'Junction': [int(firstNumber)],
    'Vehicles': [int(secondNumber)],
    'hour': [int(thirdNumber)],
    'day_of_week': [int(fourthNumber)],
    'month': [int(month)]
}
new_df = pd.DataFrame(new_data)
predicted_state = model.predict(new_df)
state_mapping_inv = {0: 'Green', 1: 'Yellow', 2: 'Red'}
print(f'Predicted traffic light state: {state_mapping_inv[predicted_state[0]]}')
