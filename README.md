Hangi zaman dilimlerinde daha iyi satışlar yapıldığını ve hangi fiyat aralıklarının daha iyi satıldığını belirlemek için satış verilerinin analizi

### 1. Veri Hazırlığı:

* Zaman bilgisi (tarih/saat)
* Satış fiyatı
* Satış miktarı
* Ürün ID

Sales data csv:

* ```date```: Satış tarihi
* ```price```: Ürün fiyatı
* ```quantity```: Satış miktarı
* ```product_id```: Ürün ID

### 2. Kullanılacak Kütüphaneler:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
```

### Veri Yükleme ve İnceleme:

```python
# Veriyi yükleme
data = pd.read_csv('sales_data.csv', parse_dates=['date'])

# İlk 5 satırı görüntüleme
print(data.head())

# Veri yapısını inceleme
print(data.info())
```

### 3. Veri Ön İşleme ve Analiz:

```python
# Tarihi yıl, ay, ve haftanın günü olarak ayırma
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day_of_week'] = data['date'].dt.dayofweek

# Fiyat aralıkları oluşturma
data['price_range'] = pd.cut(data['price'], bins=[0, 50, 100, 150, 200, np.inf], labels=['0-50', '51-100', '101-150', '151-200', '200+'])

# Ürün bazında gruplama ve satış analizleri
sales_by_product = data.groupby(['product_id', 'year', 'month']).agg({'quantity': 'sum', 'price': 'mean'}).reset_index()

# Fiyat aralıklarına göre satış miktarı analizi
sales_by_price_range = data.groupby(['product_id', 'price_range']).agg({'quantity': 'sum'}).reset_index()

print(sales_by_product)
print(sales_by_price_range)
```

### 4. Model Eğitimi (Random Forest Regressor):

```python
# Girdi ve çıktı değişkenlerini belirleme
X = data[['product_id', 'price', 'year', 'month', 'day_of_week']]
y = data['quantity']

# Eğitim ve test verilerini bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluşturma
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Tahmin yapma
y_pred = model.predict(X_test)

# Modelin performansını değerlendirme
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')
```

### 5. En Uygun Fiyatı Bulma:
```python
# Fiyat tahmini yapma ve en uygun fiyatı bulma
optimal_prices = {}

for product in data['product_id'].unique():
    #print(product)
    product_data = X_test[X_test['product_id'] == product]
    #print(product_data)
    if not product_data.empty:
        # Modelin tahminlerini yapma
        predictions = model.predict(product_data)
        
        # En yüksek tahmin değerinin indeksini bulma
        best_index = np.argmax(predictions)
        
        # İlgili fiyatı bulma
        best_price = product_data.iloc[best_index]['price']
        optimal_prices[product] = best_price

# Her ürün için en uygun fiyatları yazdırma
for product, price in optimal_prices.items():
    print(f'Product ID: {product}, En Uygun Fiyat: {price}')
```

### 6. Sonuçların Görselleştirilmesi:
```python
# Ay bazında ürün satış grafiği
plt.figure(figsize=(10,6))
sns.lineplot(x='month', y='quantity', hue='product_id', data=sales_by_product, marker='o')
plt.title('Ürün Bazında Aylık Satış Miktarı')
plt.show()

# Fiyat aralığına göre ürün satış grafiği
plt.figure(figsize=(10,6))
sns.barplot(x='price_range', y='quantity', hue='product_id', data=sales_by_price_range)
plt.title('Ürün Bazında Fiyat Aralığına Göre Satış Miktarı')
plt.show()
```
