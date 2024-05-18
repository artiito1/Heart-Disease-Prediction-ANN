
import numpy as np                                                                                  
import pandas as pd                                                                                                                                                   # type: ignore
from sklearn import preprocessing                                                                          
from sklearn.model_selection import train_test_split                                                 
import seaborn as sns                                                                                           
from keras.layers import Dense, Dropout                                                                                           
from keras.models import Sequential                                                                                           
from keras import callbacks                                                                                                                                                                               # type: ignore

#veiler çekimi 
data = pd.read_csv("Heart Disease dataset.csv")

#Mevcut verilere genel bakış
genelBilgiler = data.describe() 

#öncelikle hedefi değerlendirelim ve verilerimizin dengesiz olup olmadığını öğrenelim
cols= ["#6daa9f","#774571"]
sns.countplot(x= data["target"], palette= cols)

#özelliklere X ve hedefe y olarak değer atama
features=data.drop(["target"],axis=1)
target=data["target"]

#Özellikler için standart bir ölçekleyici kurmak
col_names = list(features.columns)
s_scaler = preprocessing.StandardScaler()
features_df = s_scaler.fit_transform(features)
features_df = pd.DataFrame(features_df, columns=col_names)   
features_df.describe()

#bölme testi ve eğitim setleri
X_train, X_test, y_train,y_test = train_test_split(features_df,target,test_size=0.25,random_state=0)

early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, # iyileştirme olarak sayılacak minimum değişiklik miktarı
    patience=30, # durmadan önce kaç dönem beklenmeli
    restore_best_weights=True)

# sinir ağının başlatılması
model = Sequential()

# katmanlar(layers)
model.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 13))
model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# YSA'nın derlenmesi
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# YSA'yı eğitin
history = model.fit(X_train, y_train, batch_size = 32, epochs = 500,callbacks=[early_stopping], validation_split=0.2)

val_accuracy = np.mean(history.history['val_accuracy'])
print("\n%s: %.2f%%" % ('val_accuracy', val_accuracy*100))

# Test seti sonuçlarını tahmin etme
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
np.set_printoptions()

#model kaydetmesi 
import pickle
pickle.dump(model, open('model0.pkl','wb'))




