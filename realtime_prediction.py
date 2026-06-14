import serial
import time
import joblib
import pandas as pd

# Load trained model
model = joblib.load("motor_model.pkl")
print("✅ AI Model loaded successfully!\n")

# Connect to Arduino
ser = serial.Serial('COM3', 9600)  # change COM port as needed
time.sleep(2)
print("🔌 Connected to Arduino. Monitoring started...\n")

# Thresholds
VIB_THRESHOLD = 30    # Based on your SW-420 data
TEMP_THRESHOLD = 40.0   # °C

print("time_ms | vib_count | temp_c | humidity | STATUS")
print("--------------------------------------------------")

try:
    while True:
        line = ser.readline().decode().strip()
        if not line or "time_ms" in line:
            continue

        parts = line.split(",")
        if len(parts) != 4:
            continue

        time_ms = int(parts[0])
        vib = int(parts[1])
        temp = float(parts[2])
        hum = float(parts[3])

        # === Threshold-based Fault Condition ===
        fault_condition = (vib >= VIB_THRESHOLD) or (temp >= TEMP_THRESHOLD)

        # === AI Prediction ===
        X = pd.DataFrame([[vib, temp, hum]], columns=["vib_count", "temp_c", "humidity"])
        pred = model.predict(X)[0]

        # === Final decision ===
        if fault_condition or pred == 1:
            print(f"{time_ms} | {vib} | {temp:.2f} | {hum:.2f} | 🔴 FAULT")
        else:
            print(f"{time_ms} | {vib} | {temp:.2f} | {hum:.2f} | 🟢 NORMAL")

except KeyboardInterrupt:
    print("\n🛑 Monitoring stopped.")
    ser.close()
