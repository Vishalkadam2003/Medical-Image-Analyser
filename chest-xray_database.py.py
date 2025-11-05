import sqlite3

conn = sqlite3.connect("patients.db")
cursor = conn.cursor()

# Add missing columns if they don't exist
try:
    cursor.execute("ALTER TABLE patient_data ADD COLUMN timestamp TEXT")
except:
    pass

try:
    cursor.execute("ALTER TABLE patient_data ADD COLUMN top_disease TEXT")
except:
    pass

try:
    cursor.execute("ALTER TABLE patient_data ADD COLUMN probabilities TEXT")
except:
    pass

conn.commit()
conn.close()

print("Table updated successfully.")
