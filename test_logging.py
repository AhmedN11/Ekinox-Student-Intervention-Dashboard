"""
Test script to verify logging functionality.
"""
import pandas as pd
from data_processor import process_student_data, logger as dp_logger

# Test data processor logging
print("Testing Data Processor logging...")
print("=" * 80)

# Create a small test dataset
test_data = {
    'StudentID': ['TEST001', 'TEST002', 'TEST003'],
    'FinalGrade': [8, 15, 12],
    'absences': [10, 2, 5],
    'studytime': [1, 3, 2],
    'Dalc': [3, 1, 2],
    'Walc': [4, 1, 2],
    'failures': [1, 0, 0],
    'schoolsup': ['no', 'yes', 'no'],
    'famsup': ['no', 'yes', 'yes'],
    'famrel': [2, 4, 3],
    'health': [2, 5, 4]
}

df = pd.DataFrame(test_data)
print(f"Created test dataset with {len(df)} students")

# Process the data (this will trigger logging)
print("\nProcessing student data...")
processed_df = process_student_data(df)

print(f"\nProcessing complete!")
print(f"Processed {len(processed_df)} students")
print(f"Columns: {list(processed_df.columns)}")
print(f"\nUrgency zones:")
print(processed_df['urgency_zone'].value_counts())

print("\n" + "=" * 80)
print("Check the Logs folder for log files:")
print("  - Logs/dashboard.log (from app.py)")
print("  - Logs/data_processor.log (from data_processor.py)")
print("=" * 80)

# Display sample log entries
print("\nSample processed data:")
print(processed_df[['StudentID', 'FinalGrade', 'intervention_score', 'priority_score', 'urgency_zone']])
