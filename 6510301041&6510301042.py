import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. โหลดข้อมูล
data = pd.read_csv("D:\Semester 2\Artificial Intelligence\classwork\cleaned_vaccine.csv")
print(data.head())

# 2. ตรวจสอบข้อมูลเบื้องต้น
print(data.info())
print(data.isnull().sum())

# 3. เตรียมข้อมูล
# ใช้คอลัมน์เข็ม 1 และเข็ม 2 ของทุกกลุ่มเพื่อทำนายเข็ม 3
feature_columns = [
    'บุคลากร_เข็ม1', 'บุคลากร_เข็ม2', 'ผู้สูงอายุ_เข็ม1', 'ผู้สูงอายุ_เข็ม2',
    'ผู้ป่วยเรื้อรัง_เข็ม1', 'ผู้ป่วยเรื้อรัง_เข็ม2', 'หญิงตั้งครรภ์_เข็ม1', 'หญิงตั้งครรภ์_เข็ม2',
    'ประชาชนทั่วไป_เข็ม1', 'ประชาชนทั่วไป_เข็ม2', 'นักเรียน_เข็ม1', 'นักเรียน_เข็ม2'
]
X = data[feature_columns]
y = data[['บุคลากร_เข็ม3', 'ผู้สูงอายุ_เข็ม3', 'ผู้ป่วยเรื้อรัง_เข็ม3', 'หญิงตั้งครรภ์_เข็ม3',
          'ประชาชนทั่วไป_เข็ม3', 'นักเรียน_เข็ม3']]

# แปลงค่า y ให้เป็นค่าตัวเลข (Regression Problem)
y = y.values

# แบ่งข้อมูล Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ปรับค่าข้อมูลให้เป็นสเกลเดียวกัน (Normalization)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. สร้างโมเดล MLP สำหรับ Regression
model = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],), 
                       kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dropout(0.4),  # ลด Overfitting
    keras.layers.Dense(128, activation='relu', 
                       kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu', 
                       kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dense(y_train.shape[1], activation='linear')  # ใช้ Linear Activation สำหรับ Regression
])

# คอมไพล์โมเดล
optimizer = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mae'])

# Early Stopping เพื่อลด Overfitting
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                               patience=15, restore_best_weights=True)

# 5. ฝึกโมเดล
history = model.fit(X_train, y_train, epochs=300, batch_size=8, 
                    validation_data=(X_test, y_test), callbacks=[early_stopping])

# 6. ทดสอบโมเดล
y_pred = model.predict(X_test)

# 7. ประเมินผล Regression
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# คำนวณเปอร์เซ็นต์ error
mae_percentage = (mae / np.mean(y_test)) * 100
mse_percentage = (np.sqrt(mse) / np.mean(y_test)) * 100

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)
print(f"Model Accuracy: {r2 * 100:.2f}%")
print(f"MAE as Percentage: {mae_percentage:.2f}%")
print(f"RMSE as Percentage: {mse_percentage:.2f}%")

# 8. ตรวจสอบ Overfitting และ Underfitting
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()

plt.figure(figsize=(10,5))
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.title('Training vs Validation MAE')
plt.show()


# 9. ทดสอบโมเดลด้วยตัวอย่างใหม่
sample_input = np.array([[37677, 35535, 162644, 136021, 59284, 48706, 826, 577, 414258,
                          293742, 39840, 34258]])
sample_input = scaler.transform(sample_input)
prediction = model.predict(sample_input)
print("Prediction for new sample:", prediction)
