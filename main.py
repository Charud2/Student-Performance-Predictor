from model import train_model
import pandas as pd

model = train_model()

print("Enter student details:")

hours = float(input("Hours studied: "))
attendance = float(input("Attendance (%): "))
previous = float(input("Previous score: "))

input_data = pd.DataFrame([[hours, attendance, previous]], columns = ["hours_studied", "attendance", "previous_score"])

prediction = model.predict(input_data)

print("\nPredicted Final Score:", round(prediction[0], 2))