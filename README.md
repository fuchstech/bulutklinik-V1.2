# 🌐 Bulutklinik Hackathon Projesi

Bu repo, **2023 yılında düzenlenmiş olan Bulutklinik Hackathon** için 24 saat içerisinde hazırlanmış ve yarışmada **birincilik ödülüne layık görülmüştür**. Proje, açık kaynaklı veri setlerini kullanarak eğitilmiş **pnömoni**, **cilt hastalıkları** ve **akciğer kanseri tespiti** programlarını içermektedir. Ayrıca, projeye ait web tasarım dosyalarına [buradan ulaşabilirsiniz](https://matiricie.com/bulutklinik/).

<img src="https://github.com/fuchstech/bulutklinik-V1.2/blob/main/images/odul.jpg" alt="Birincilik Ödülü" width="500" />

---

## 🚀 Proje Özeti
> **Proje özeti ilerleyen günlerde eklenecektir.**

---

## 📑 Proje Sunumu
Projeye ait detaylı sunumu incelemek için [Canva üzerinden erişebilirsiniz](https://www.canva.com/design/DAFx4PAW4Xo/sA0Ed5eXwhsNhKg3nNYDNw/edit?utm_content=DAFx4PAW4Xo&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton).

---
## 📁 Kodların Çalıştırılması

Projeyi çalıştırmak için aşağıdaki adımları takip edebilirsiniz:

### 1. Ortamı Oluşturma ve Aktifleştirme
Öncelikle bir Conda ortamı oluşturun ve aktifleştirin:
```bash
conda create -n bk
conda activate bk
pip install -r requirements.txt
```
```bash

cd pnomenia_predict
python3 uipno.py
```
```bash
cd skin_diseases
python3 predict_skin.py #you need to change file paths
```
```bash
cd chest ct scan
python3 uipredict.py
```
