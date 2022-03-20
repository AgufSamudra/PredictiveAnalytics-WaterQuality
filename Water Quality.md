# Laporan Proyek Machine Learning - Gufranaka Samudra
## _Water Quality_

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

## Domain Proyek

Akses ke air minum yang aman sangat penting untuk kesehatan, hak asasi manusia dan komponen kebijakan yang efektif untuk perlindungan kesehatan. Hal ini penting sebagai masalah kesehatan dan pembangunan di tingkat nasional, regional dan lokal. Di beberapa daerah, telah terbukti bahwa investasi dalam penyediaan air dan sanitasi dapat menghasilkan keuntungan ekonomi bersih, karena pengurangan efek kesehatan yang merugikan dan biaya perawatan kesehatan lebih besar daripada biaya melakukan intervensi.

* Masalah air minum bersih ini sangat penting untuk kesehatan manusia, dengan kita bisa menyelesaikan masalah ini maka kita telah menyelematkan banyak orang dengan air minum yang bersih dan layak.
* Banyak sebagian orang sulit untuk mendapati air bersih dan mengetahui apakah air tersebut layak untuk di minum. Namun dengan teknologi ML kita bisa membuat sebuah model yang bisa memberi tahu apakah air tersebut layak untuk di minum atau tidak.
* Sumber referensi
[Kaggle](https://www.kaggle.com/adityakadiwal/water-potability)

## Business Understanding
Seperti yang di bahas sebelumnya, kita akan membangun model Machine Learning untuk mendeteksi apakah air tersebut layak minum atau tidak.
### Problem Statements
* Bagaimana melihat apa saja fitur kandungan air yang mempengaruhi kualitas air tersebut?
* Seberapa akurat model Machine Learning yang akan di buat?
* Beberapa data masih belum sepenugnya bersih, serta bagaimana cara membersihkannya?

### Goals
* Mengetahui semua isi dari kandungan tersebut
* Kita akan mencoba 3 model untuk mencari yang terbaik
* Melakukan pembersihan pada data yang kita miliki sebelum melakukan tahap pemodelan.

## Data Understanding

Dataset yang kita gunakan berasal dari website Kaggle, dataset ini memiliki 3276 baris dengan 10 kolom. Tools yang kita gunakan adalah Google Colab sebagai code editor.
[Link Dataset](https://www.kaggle.com/adityakadiwal/water-potability)

Hampir semua tipe data pada setiap atribute adalah bertipe `Float` hanya atribute target kita saja yang memiliki tipe data `Int`. Data pada atribute target hanya bernilai `0` dan `1`, untuk membedakan kategori dari data tersebut.

Setelah kita melihat dataset kita, ada beberapa problem pada dataset kita seperti,
* Data dengan bernilai 0
* Adanya Missing Value
* Dataset yang mengandung Outliers

### Identifikasi data yang bernilai 0
Dalam pemroresan kita melihat deskripsi dari data kita menggunakan fungsi `describe()` setelah kita jalankan, kita akan mendapat deskripsi dari dataset yang kita gunakan. Kita akan melihat nilai `count` `mean` `std` `min` `25%` `50%` `75%` `max` jika memperhatikan nilai-nilai itu dan fokus kepada nilai `min` di dapatkan data bernilai 0.

### Missing Value
Selanjutnya kita akan melihat apakah data kita memiliki missing value, dengan fungsi `isnull().sum()` kita bisa melihat apakah data kita memiliki missing value atau tidak.

| Variable  | missing value |
| ----- | --- |
| ph   | 491 |
| Hardness | 0  |
| Solids   | 0  |
| Chloramines | 0  |
| Sulfate   | 781  |
| Conductivity | 0  |
| Organic_carbon   | 0  |
| Trihalomethanes | 162  |
| Turbidity   | 0  |
| Potability | 0  |

Dari hasil data di atas kita bisa menyimpulkan bahwa variable `ph` memiliki missing value sebanyak **492**, kemudian variable `sulfate` memiliki missing value sebanyak **781**, dan terakhir variable `Trihalomethanes` memiliki missing value sebanyak **162**.

### Outliers
Outliers adalah nilai pada dataset kita yang memiliki nilai sangat jauh pada data yang lain. Jika kita visualisasikan data dengan fungsi `boxplot()`. Terlihat ada data yang berada di luar jangkauan data lain.


![Build Status](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAWAAAAEGCAYAAABbzE8LAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAOCklEQVR4nO3df2zcdR3H8dd7vZpt4K9tOrVDDi0BCVGBxqAYE+RHuiLgn/5kBmEJaFfJjIASlyWNIdGoUBVD/EEXCMYgRgJddaCJ0aixRWVjA/bNqLDxa3QIjCJbu7d/3LXebverx/d77972fCRk7fd797n3Ld/v845v19bcXQCA1lsUPQAAHKsIMAAEIcAAEIQAA0AQAgwAQXLzufGKFSs8n89nNAoAHJ3Gx8efd/e3lW+fV4Dz+bzGxsbSmwoAjgFm9u9K27kEAQBBCDAABCHAABCEAANAEAIMAEEIMAAEIcAAEIQAA0AQAgwAQQgwAAQhwAAQhAADQBACDABBCDAABCHAABCEAANAEAIMAEEIMAAEIcAAEGRevxMOaNbQ0JCSJEl1zT179kiSurq6Ul13Vnd3t/r7+zNZG5AIMFokSRL9c9sOzSxdltqaHVMvSpKeeS39w7hjal/qawLlCDBaZmbpMr16al9q6y15ZESSUl2zfG0gS1wDBoAgBBgAghBgAAhCgAEgCAEGgCAEGACCEGAACEKAASAIAQaAIAQYAIIQYAAIQoABIAgBBoAgBBgAghBgAAhCgAEgCAEGgCAEGACCEGAACEKAASAIAQaAIAQYAIIQYAAIQoABIAgBBoAgBBgAghBgAAhCgAEgCAEGgCAEGACCEGAACEKAASAIAQaAIAQYAIIQYAAIQoABIAgBBoAgBBgAghBgAAhCgAEgCAEGgCAEuAWGhoY0NDQUPQbQEI7X1slFD3AsSJIkegSgYRyvrcM7YAAIQoABIAgBBoAgBBgAghBgAAhCgAEgCAEGgCAEGACCEGAACEKAASAIAQaAIAQYAIIQYAAIQoABIAgBBoAgBBgAghBgAAhCgAEgCAEGgCAEGACCEGAACEKAASAIAQaAIAQYAIIQYAAIQoABIAgBBoAgBBgAghBgAAhCgAEgCAEGgCAEGACCEGAACEKAASAIAQaAIAQYAIIQYAAIQoABIEhLAjw5Oal169ZpcnIy8zVq3a7SvtJt9fZLUpIk6uvr05VXXqkkSSred3JyUldffbWuuuoqJUmiJEl08ODBpp87EKX8uL7iiivU19en8fHxw7ZffvnlOvfcc3XhhRdq7dq1DZ9/89mf5nNpxf0a0ZIADw8Pa+vWrdq0aVPma9S6XaV9pdvq7ZekwcFBTU1NaefOnRocHKx43+HhYW3fvl07duzQ4OCgXnnlFT377LNNP3cgSvlxnSSJpqamtGHDhsO279q1S+6uAwcO6LHHHmv4/JvP/jSfSyvu14jMAzw5OanR0VG5u0ZHR5t6FWl0jVq3q7SvdNvmzZu1efPmqvtHR0c1Pj6uiYmJuTUnJibm7lu+TultJGnfvn2ZvbIDWSg/P+677765ffv376+4fdbIyEjd86/aYzXbiUafy3zWz3quXKqrVTA8PKxDhw5JkmZmZrRp0yZdc801maxR63aV9rn73LbSSwSV9s/MzGjDhg0V5yu978GDB+XuR9zG3bV27VqtWrVqXs/9aJEkiRYdOPLvZaFa9N+XlCQva2BgIHqUlkuSREuWLDnsnKl2XNfaXu/8Kz2H0+hELc2un/Vcdd8Bm9laMxszs7G9e/fO+wHuv/9+TU9PS5Kmp6e1ZcuWzNaodbtK+0q3ufvcgVRp//T0tPbv31/xcUvvW+lgnPXCCy80/JyBaOXnRyW1jvd651+1x2q2E7U0u37Wc9V9B+zut0q6VZJ6enrm/Rbm/PPP18jIiKanp5XL5XTBBRfMe8hG16h1u0r73H1um5nNPt+K+3O5nBYvXlwxwqX3NbOqB+XFF1+c6qtnOxkYGND4rva5Dn5o8ZvU/Z6Vuummm6JHabnZd/0nnnjiYedHpeO61vFe7/wrlUYnaml2/aznyvwa8Jo1a7RoUeFhOjo6dNlll2W2Rq3bVdpXuq2zs1O5XK7q/o6ODm3cuLHi43Z2dqqzs/OIj0uZWVPPHYhSfn50dHQccZta2+udf9Ueq9lO1NLs+lnPlXmAly9frt7eXpmZent7tXz58szWqHW7SvtKt61evVqrV6+uur+3t1dnnXWW8vn83Jr5fH7uvuXrlN5GkpYtW9bUcweilJ8fF1100dy+448/vuL2WX19fXXPv2qP1WwnGn0u81k/67ky/yKcVHgVmZiYeF2vHo2uUet2lfaVb6u3/4YbbtC6devU1dWla6+9VjfffHPF+yZJInfX+vXrNTAwoJUrVzb93IEo5cf/9u3b9dRTT2njxo0aHh6e275t2zY9/vjj6uzsVD6fb/j8m8/+tJ9L1vdrhNW6iF6up6fHx8bGUh/iaDd7Te1YvJ44a/Ya8Kun9qW25pJHRiQp1TVL1z7rGL8GfCw+96yY2bi795Rv51uRASAIAQaAIAQYAIIQYAAIQoABIAgBBoAgBBgAghBgAAhCgAEgCAEGgCAEGACCEGAACEKAASAIAQaAIAQYAIIQYAAIQoABIAgBBoAgBBgAghBgAAhCgAEgCAEGgCAEGACCEGAACEKAASAIAQaAIAQYAIIQYAAIQoABIAgBBoAgBBgAghBgAAhCgAEgCAEGgCAEGACCEGAACEKAASAIAQaAILnoAY4F3d3d0SMADeN4bR0C3AL9/f3RIwAN43htHS5BAEAQAgwAQQgwAAQhwAAQhAADQBACDABBCDAABCHAABCEAANAEAIMAEEIMAAEIcAAEIQAA0AQAgwAQQgwAAQhwAAQhAADQBACDABBCDAABCHAABCEAANAEAIMAEEIMAAEIcAAEIQAA0AQAgwAQQgwAAQhwAAQhAADQBACDABBCDAABCHAABCEAANAEAIMAEEIMAAEIcAAEIQAA0AQAgwAQQgwAAQhwAAQJBc9AI4dHVP7tOSRkRTXm5SkVNf8/9r7JK1MfV2gFAFGS3R3d6e+5p4905Kkrq4sQrkyk5mBUgQYLdHf3x89ArDgcA0YAIIQYAAIQoABIAgBBoAgBBgAghBgAAhCgAEgCAEGgCAEGACCEGAACEKAASAIAQaAIAQYAIIQYAAIQoABIAgBBoAgBBgAghBgAAhCgAEgCAEGgCDm7o3f2GyvpH83cNMVkp5vdqgAzJutdptXar+ZmTdbr3feE939beUb5xXgRpnZmLv3pL5wRpg3W+02r9R+MzNvtrKal0sQABCEAANAkKwCfGtG62aFebPVbvNK7Tcz82Yrk3kzuQYMAKiPSxAAEIQAA0CQVANsZr1m9qiZJWZ2XZprZ8HMTjCzP5jZdjN72MwGomdqhJl1mNk/zOze6FnqMbO3mNldZvaIme0wsw9Hz1SLmV1TPBa2mdmdZrY4eqZyZvYzM3vOzLaVbFtmZlvMbGfxz7dGzliqyrzfLh4TD5nZr83sLZEzlqo0b8m+9WbmZrYijcdKLcBm1iHph5JWSzpN0qfN7LS01s/ItKT17n6apLMlfakNZpakAUk7oodo0E2SRt39VEkf0AKe28y6JK2T1OPup0vqkPSp2Kkquk1Sb9m26yQ94O4nS3qg+PlCcZuOnHeLpNPd/f2SHpN0fauHquE2HTmvzOwESRdKeiKtB0rzHfCHJCXuvsvdD0j6haRLU1w/de7+tLs/WPz4ZRXi0BU7VW1mtkrSRZJ+Ej1LPWb2Zkkfk/RTSXL3A+7+n9ip6spJWmJmOUlLJT0VPM8R3P2PkvaVbb5U0nDx42FJn2zpUDVUmtfdf+fu08VP/yppVcsHq6LK368kfU/S1ySl9i8X0gxwl6QnSz7frQUes1Jmlpd0hqS/xU5S1/dVOAgORQ/SgJMk7ZX08+Ilk5+Y2XHRQ1Xj7nskfUeFdzhPS3rR3X8XO1XDVrr708WPn5G0MnKYebpc0uboIWoxs0sl7XH3f6W5Ll+Ek2Rmx0v6laSvuPtL0fNUY2afkPScu49Hz9KgnKQzJd3i7mdIekUL63+ND1O8bnqpCi8c75J0nJl9Lnaq+fPCvy1ti39fambfUOFS4B3Rs1RjZkslfV3SN9NeO80A75F0Qsnnq4rbFjQz61Qhvne4+93R89RxjqRLzGxChUs8Hzez22NHqmm3pN3uPvt/FXepEOSF6nxJj7v7Xnc/KOluSR8JnqlRz5rZOyWp+OdzwfPUZWZfkPQJSZ/1hf0NCe9V4UX5X8Vzb5WkB83sHa934TQD/HdJJ5vZSWb2BhW+eHFPiuunzsxMheuTO9z9u9Hz1OPu17v7KnfPq/D3+3t3X7Dv0Nz9GUlPmtkpxU3nSdoeOFI9T0g628yWFo+N87SAv2hY5h5Ja4ofr5H0m8BZ6jKzXhUupV3i7lPR89Ti7lvd/e3uni+ee7slnVk8vl+X1AJcvKD+ZUm/VeGg/aW7P5zW+hk5R9LnVXgn+c/if33RQx1l+iXdYWYPSfqgpG8Fz1NV8Z36XZIelLRVhfNjwX3LrJndKekvkk4xs91m9kVJN0q6wMx2qvBO/sbIGUtVmfcHkt4oaUvxvPtx6JAlqsybzWMt7Hf+AHD04otwABCEAANAEAIMAEEIMAAEIcAAEIQA46hhZhNp/ZQqoBUIMAAEIcBoO2aWL/4s2TuKP2P4ruL360tSv5k9aGZbzezU0EGBOggw2tUpkn7k7u+T9JKkq4vbn3f3MyXdIumrUcMBjSDAaFdPuvufix/fLumjxY9nf6DSuKR8q4cC5oMAo12Vfw/97OevFf+cUeHHYQILFgFGu3p3ye+X+4ykP0UOAzSDAKNdParC7/DbIemtKlzzBdoKPw0Nbaf466PuLf7iTKBt8Q4YAILwDhgAgvAOGACCEGAACEKAASAIAQaAIAQYAIL8D/kSATtqVFwGAAAAAElFTkSuQmCC)


### Variabel-variabel pada Water Quality dataset adalah sebagai berikut:

* ph : PH merupakan parameter penting dalam mengevaluasi keseimbangan asam-basa air. Ini juga merupakan indikator kondisi asam atau basa status air.
* Hardness : Kekerasan terutama disebabkan oleh garam kalsium dan magnesium. Garam-garam ini larut dari endapan geologis yang dilalui air.
* Solids : Air memiliki kemampuan untuk melarutkan berbagai anorganik dan beberapa mineral atau garam organik seperti kalium, kalsium, natrium, bikarbonat, klorida, magnesium, sulfat, dll. Mineral ini menghasilkan rasa yang tidak diinginkan dan warna yang encer dalam penampilan air.
* Chloramines : disinfektan utama yang digunakan dalam sistem air publik. Kloramin paling sering terbentuk ketika amonia ditambahkan ke klorin untuk mengolah air minum.
* Sulfate : zat alami yang ditemukan di mineral, tanah, dan batuan.
* Conductivity : Air murni bukanlah penghantar arus listrik yang baik, melainkan isolator yang baik. Peningkatan konsentrasi ion meningkatkan konduktivitas listrik air. Umumnya, jumlah padatan terlarut dalam air menentukan konduktivitas listrik.
* Organic_carbon : Total Organic Carbon (TOC) di perairan sumber berasal dari bahan organik alami (NOM) yang membusuk serta sumber sintetis.
* Trihalomethanes : THM adalah bahan kimia yang dapat ditemukan dalam air yang diolah dengan klorin.
* Turbidity : ukuran sifat pemancar cahaya air dan tes ini digunakan untuk menunjukkan kualitas pembuangan limbah sehubungan dengan materi koloid.
* Potability : Menunjukkan jika air aman untuk konsumsi manusia di mana 1 berarti Dapat Diminum dan 0 berarti Tidak dapat diminum.


## Data Preparation

Ada beberapa tahap yang akan kita lakukan terhadap dataset yang kita miliki, seperti sebagai berikut,

* Mengatasi Missing Value dan Data bernilai 0 pada dataset.
* Menghapus Outliers pada beberapa atribut di dalam dataset.
* Pembagian dataset dengan fungsi train_test_split dari library sklearn.
* Melakukan Standarisasi pada data yang kita miliki.

### Identifikasi data yang bernilai 0 dan Missing Value
Pada data sebelumnya kita mendapati Missing Value dan data bernilai 0, jadi kami akan menghapus data tersebut. Dengan menggunakan fungsi `dropna()`, Missing Value dan Data bernilai 0 akan di hapus dari dataset.

Kita sengaja menghapus data tersebut karena akan mengganggu pada saat proses pelatihan, dan kita juga akan tetap menjaga nilai asli dari dataset kita. Oleh karena itu kita tidak mengisi data kosong dengan nilai ==mean==/==rata-rata==.

### Outliers
Dari visualisasi saat Data Understanding kita menemui adanya outliers, oleh karena itu kita juga akan mengatasi masalah tersebut dengan metode `IQR`

Sama seperti Missing Value, Outliers juga mempengaruhi pada saat proses pelatihan di karenakan nilai pada dataset yang memiliki rentang yang sangat jauh atau berbeda dengan data yang lainnya.

### Spliting Dataset ke Train set dan Test set
Kita akan membagi data kita menjadi data latihan dan data uji, Sklearn sudah menyediakan untuk membagi dataset kita menjadi train dan test set menggunakan kelas `train_test_split`. Di gunakan agar memudahkan kita pada saat proses latihan dan proses tes.


### Standarisasi
Kita juga akan melakukan standarisasi dengan tujuan membuat scale pada atribute data menjadi rentang yang sama, Library Sklearn `StandardScaler` memudahkan kita untuk melakukan standarisasi.


## Modeling

Tahap ini kita akan melakukan pemodelan serta pelatihan pada dataset kita yang sudah kita bersihan sebelumnya, dan pada tahap ini kita akan menggunakan 3 Algoritma Machine Learning serta memilih algoritma mana yang terbaik. Antara lain,
* Random Forest
* SVM
* XGBoost

### Random Forest

 Proses klasifikasi pada random forest berawal dari memecah data sampel yang ada kedalam decision tree secara acak. Setelah pohon terbentuk,maka akan dilakukan voting pada setiap kelas dari data sampel. Kemudian, mengkombinasikan vote dari setiap kelas kemudian diambil vote yang paling banyak.Dengan menggunakan random forest pada klasifikasi data maka, akan menghasilkan vote yang paling baik.
 
 Kita juga mengisi beberapa parameter yaitu,
 * criterion : untuk mengukur kualitas sebuah split.
 * max_depth : Kedalaman maksimum pohon.
 * max_features : Jumlah fitur yang perlu dipertimbangkan saat mencari split terbaik
 * n_estimators : Jumlah pohon.
 * random_state : untuk mengatasi data acak.

 
 Ada juga kelebihan dan kekurangan dari algoritma Random Forest, yaitu,
 
 **Kelebihan**
 
 * Dapat menghasilkan error yang lebih rendah.
 * Memberikan hasil yang bagus dalam klasifikasi.
 * Dapat mengatasi data training dalam jumlah sangat besar secara efisien.
 
**Kekurangan**

Kekurangan dari algoritma ini adalah pembelajaran bisa berjalan lambat, tergantung pada parameter yang digunakan dan tidak bisa memperbaiki model yang dihasilkan secara berulang.

 Kalian bisa cek untuk penjelasan Random Forest lebih lengkap serta Parameter yang di gunakan di link bawah.
 [Random Forest](https://id.wikipedia.org/wiki/Random_forest#:~:text=Proses%20klasifikasi%20pada%20random%20forest,diambil%20vote%20yang%20paling%20banyak)
 
### SVM

Cara kerja dari metode Support Vector Machine khususnya pada masalah non-linear adalah dengan memasukkan konsep kernel ke dalam ruang berdimensi tinggi. Tujuannya adalah untuk mencari hyperplane atau pemisah yang dapat memaksimalkan jarak (margin) antar kelas data.

Kita juga mengisi beberapa parameter yaitu,
* kernel : digunakan untuk menghitung matriks kernel dari matriks data.
* C : Parameter regularisasi.
* gamma : Koefisien kernel untuk `rbf`, `poli` dan `sigmoid`.
* random_state : untuk mengatasi data acak.


Tidak hanya Random Forest, SVM juga memiliki kelebihan serta kekurangan pada algoritmanya.

**Kelebihan**
Pengklasifikasi SVM menawarkan akurasi tinggi dan bekerja dengan baik dengan ruang dimensi tinggi. SVM pengklasifikasi pada dasarnya menggunakan subset dari poin pelatihan sehingga hasilnya menggunakan memori yang sangat sedikit.

**Kekurangan**
Mereka memiliki waktu pelatihan yang tinggi sehingga dalam praktiknya tidak cocok untuk kumpulan data yang besar. Lain kerugiannya adalah pengklasifikasi SVM tidak berfungsi dengan baik dengan kelas yang tumpang tindih.

Untuk lebih lengkap kalian bisa kunjungi tautan di bawah.
[SVM Klasifikasi](https://www.dqlab.id/kenali-tentang-algoritma-support-vector-machine#:~:text=Cara%20kerja%20dari%20metode%20Support,(margin)%20antar%20kelas%20data.)

### XGBoost

XGBoost adalah implementasi open-source yang populer dan efisien dari algoritma pohon gradien didorong. Gradient boosting adalah algoritma pembelajaran yang diawasi, yang mencoba untuk secara akurat memprediksi variabel target dengan menggabungkan perkiraan satu set model yang lebih sederhana dan lebih lemah.

Untuk parameter yang kita gunakan hanya satu yaitu, `booster` Penguat mana yang digunakan adalah **gbtree** menggunakan model berbasis pohon sementara dan menggunakan fungsi linier.


**Kelebihan**
* Bagus situasi dimana terdapat banyak variabel kategorikal.
* Memiliki jumlah obserbasi yang banyak pada training data. 
* Untuk masalah klasifikasi, terutama untuk yang terkait dengan masalah bisnis dunia nyata.
 
**Kekurangan**
* Jika jumlah observasi pada data training sangat jauh lebih kecil daripada jumlah fiturnya.
* Task regresi yang melibatkan prediksi output kontinyu.
* Bidang pemrosesan bahasa alami atau NLP. 

Kalian bisa melihat dokumen nya langsung di bawah,
[XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html)

## Evalution

Pada tahap evaluasi kita akan menggunakan Confusion matrix, Confusion Matrix adalah pengukuran performa untuk masalah klasifikasi machine learning dimana keluaran dapat berupa dua kelas atau lebih.  Confusion Matrix adalah tabel dengan 4 kombinasi berbeda dari nilai prediksi dan nilai aktual. Ada empat istilah yang merupakan representasi hasil proses klasifikasi pada confusion matrix yaitu True Positif, True Negatif, False Positif, dan False Negatif.

* True Positive (TP) : Program memprediksi positif dan itu benar.
* True Negative (TN) : Program memprediksi negatif dan itu benar.
* False Positive (FP) : Program memprediksi positif dan itu salah.
* False Negative (FN) : Program memprediksi negatif dan itu salah. 

Nilai Prediksi adalah keluaran dari program dimana nilainya Positif dan Negatif.
Nilai Aktual adalah nilai sebenarnya dimana nilainya True dan False.

Berikut adalah contoh tabel Confusion Matrik,

![TabelCM](https://ichi.pro/assets/images/max/724/0*l30v6Id3wZrw8FAO)


### Evalution Random Forest
Setelah kita menerapakan Confusion Matrik pada model Random Forest kita mendapat score 66%.

Maka Hasil nya akan seperti ini, 
[[313 | 10]
 [170 | 46]]
0.6660482374768089

Kalo kita lihat dari tabel di atas, maka cara bacanya,
Program prediksi benar untuk kategori `1`, sebanyak **313** dan salah **10**.
Program prediksi salah untuk kategori `0`, sebanyak **170** dan benar **46**.
Dengan score total **66%**

Dan jika kita jumlahkan total semua angka maka akan keluar hasil **539** dan itulah angka dari data **test set** kita.

### Evalution SVM
Setelah kita menerapakan Confusion Matrik pada model Random Forest kita mendapat score 71%.

Maka Hasil nya akan seperti ini, 
[[306  17]
 [135  81]]
0.7179962894248608

Kalo kita lihat dari tabel di atas, maka cara bacanya,
Program prediksi benar untuk kategori `1`, sebanyak **306** dan salah **17**.
Program prediksi salah untuk kategori `0`, sebanyak **135** dan benar **81**.
Dengan score total **71%**

Dan jika kita jumlahkan total semua angka maka akan keluar hasil **539** dan itulah angka dari data **test set** kita.

### Evalution XGBoost
Setelah kita menerapakan Confusion Matrik pada model Random Forest kita mendapat score 66%.

Maka Hasil nya akan seperti ini, 
[[296  27]
 [154  62]]
0.6641929499072357

Kalo kita lihat dari tabel di atas, maka cara bacanya,
Program prediksi benar untuk kategori `1`, sebanyak **296** dan salah **27**.
Program prediksi salah untuk kategori `0`, sebanyak **154** dan benar **62**.
Dengan score total **66%**

Dan jika kita jumlahkan total semua angka maka akan keluar hasil **539** dan itulah angka dari data **test set** kita.

## Kesimpulan
Dari hasil evaluasi di atas kita bisa melihat bahwa Algoritma SVM memiliki score di atas yang lain, oleh karena itu kita akan memilih Algoritma `SVM` sebagai Algoritma yang kita gunakan untuk model Machine Learning Water Quality.

Referensi : https://ichi.pro/id/metrik-evaluasi-untuk-model-klasifikasi-249524043947369
https://socs.binus.ac.id/2020/11/01/confusion-matrix/
