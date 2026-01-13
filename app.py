import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 1. Modeli yükle (Hata alırsan uzantıyı .h5 yap)
model = load_model('atik_siniflandirma_modeli.h5')

# 2. Sınıf isimlerini Kaggle klasör sırasına göre belirle
# (cardboard, glass, metal, paper, plastic, trash)
siniflar = ['Karton', 'Cam', 'Metal', 'Kagit', 'Plastik', 'Cop'] 

cap = cv2.VideoCapture(0)

print("Kamera açılıyor... Kapatmak için 'q' tuşuna basın.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Görüntü ön işleme
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR'den RGB'ye çevir
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0) / 255.0
    
    # Tahmin yap
    predictions = model.predict(img_array, verbose=0)
    score = np.max(predictions)
    label = siniflar[np.argmax(predictions)]

    # Ekranda göster
    color = (0, 255, 0) if score > 0.5 else (0, 0, 255)
    text = f"{label} ({int(score*100)}%)"
    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow('Akilli Atik Ayrirma Sistemi', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()