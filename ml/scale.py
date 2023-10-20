import numpy as np
from sklearn.preprocessing import StandardScaler

# 0 ile 70 arasındaki tam sayı verilerini oluştur
veri = np.arange(71).reshape(-1, 1)  # 0'dan 70'e kadar olan tam sayılar

# Standart ölçekleme işlemi
scaler = StandardScaler()
standart_olcekli_veri = scaler.fit_transform(veri)

# Sonucu görüntüle
print("Önceki veri:\n", veri)
print("Standart ölçekli veri:\n", standart_olcekli_veri[67])