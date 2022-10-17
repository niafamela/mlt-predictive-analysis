# Laporan Proyek Machine Learning - Nia Famela Simanjuntak M302Y0795
## Domain Proyek
Tempat tinggal atau rumah merupakan suatu kebutuhan utama bagi setiap manusia untuk dapat berlindung dan mempertahankan hidup. Rumah atau tempat tinggal pada dasarnya memiliki nilai yang bergantung pada luas, lokasi, dapur, ruang tamu, ruang makan, jumlah tempat tidur, jumlah kamar mandi, dan fitur-fitur lainnya. 

Suatu rumah memiliki harga yang diukur berdasarkan nilai rumah tersebut. Namun, harga setiap rumah pada dasarnya berbeda-beda dan tidak selalu pasti atau tidak dapat diprediksi manual secara akurat, yang dimana Harga dari setiap rumah diukur dari nilai yang dimiliki oleh rumah tersebut. Namun, hal tersebut perlu diatasi oleh seseorang yang menyewanya dengan meningkatkan prediksi mereka dalam menentukan harga sewa yang pantas dengan kriteria rumahnya masing.

Untuk mengatasi hal tersebut dalam memprediksi harga sewa rumah, maka diperlukan suatu penelitian dengan menggunakan model Machine Learning. Dimana model ini diharapkan dapat memprediksi harga sewa rumahnya sesuai dengan harga saat ini atau harga pasar. Prediksi dengan model ini selanjutnya dapat dijadikan pedoman atau referensi bagi suatu seorang penyewa atau perusahaan dalam menyewa rumah agar mendapatkan profit yang baik. 


## Business Understanding
Proyek ini dibangun untuk bisnis perusahaan dengan ciri-ciri sebagai berikut :
- Perusahaan membeli atau mempunyai sebuah rumah atau apartemen kemudian menyewanya kepada konsumen.
- Perusahaan membuka konsultasi harga sewa rumah dan apartemen ke konsumen apabila ingin bertanya mengenai harganya 

### Problem Statement
- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap harga sewa rumah atau apartemen?
- Bagaimana cara memproses data agar mendapat model yang baik dengan data yang sudah dilatih?
- Berapa harga pasar rumah atau apartemen dengan karakteristik atau fitur tertentu?

### Goals
- Mengetahui fitur yang paling berkorelasi dengan harga sewa rumah atau apartemen
- Melakukan preprocessing data dan pelatihan data agar mendapatkan model yang baik dan sesuai
- Membuat model machine learning yang dapat memprediksi harga sewa rumah atau apartemen seakurat mungkin berdasarkan fitur-fitur yang ada.

### Solution Statements
- Melakukan Exploratory Data Analysis pada dataset dengan melakukan deskripsi variabel, menangani missing value dan outliers, univariate analysis, multivariate analysis serta melakukan visualisasi data.
- Melakukan preparation data dengan melakukab train test split dan standarisasi data agar bisa digunakan dalam membangun model.
- Melakukan model machine learning dengan tiga algoritma yaitu K-Nearest Neighbour, Random Forest, dan AdaBoost serta mengevaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik.

# Data Understanding
Dataset yang digunakan merupakan data harga sewa rumah dengan berbagai fitur yang terdapat di negara India. Dataset ini dapat diunduh pada link berikut https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset. 
Dataset tersebut memiliki format csv dengan 4746 sample dengan 12 fitur, dimana terdapat 4 fitur bertipe int64 dan 8 fitur bertipe object. Dan pada dataset tersebut tidak terdapat missing value.

##### Variabel-variabel pada House Rent Prediction yaitu sebagai berikut:
- BHK: Jumlah Kamar Tidur, Aula, Dapur.
- Rent: Sewa Rumah/Apartemen/Rumah Susun.
- Size: Ukuran Rumah/Apartemen/Rumah Susun dalam Kaki Persegi.
- Floor: Rumah/Apartemen/Rumah Susun yang terletak di mana Lantai dan Jumlah Lantai (Contoh: Ground dari 2, 3 dari 5, dll.)
- Area Type: Ukuran Rumah/Apartemen/Rumah Susun yang dihitung pada Area Super atau Area Karpet atau Area Bangun.
- Area Locality:  Lokalitas Rumah/Apartemen/Rumah Susun.
- City: Kota dimana Rumah/Apartemen/Rumah Susun berada.
- Furnishing Status: Status Perabotan Rumah/Apartemen/Rumah Susun, baik Furnished atau Semi-Furnished atau Unfurnished.
- Tenant Preferred: Jenis Tenant Preferred oleh Pemilik atau Agen.
- Bathroom: Jumlah Kamar Mandi.
- Point of Contact: Siapa yang harus Anda hubungi untuk informasi lebih lanjut mengenai Rumah/Apartemen/Rumah Susun.

Dari ke 12 fitur tersebut dilakukan drop coloumn pada fitur Point of Contract dan Posted On karena tidak mempengaruhi harga sewa rumah dalam membangun model prediksi harga sewa rumah/apartemen/rumah susun.

# Menangani Outliers
Terdapat Outliers pada fitur, BHK, Size, dan Bathroom. Sehingga dilakukan penanganan Outliers dengan teknik IQR method.  IQR adalah singkatan dari Inter Quartile Range. Untuk memahami apa itu IQR, mari kita ingat lagi konsep kuartil. Kuartil dari suatu populasi adalah tiga nilai yang membagi distribusi data menjadi empat sebaran. Seperempat dari data berada di bawah kuartil pertama (Q1), setengah dari data berada di bawah kuartil kedua (Q2), dan tiga perempat dari data berada di kuartil ketiga (Q3). Dengan demikian interquartile range atau IQR = Q3 - Q1.

Rumus IQR yaitu:
```sh
Batas bawah = Q1 - 1.5 * IQR
Batas atas = Q3 + 1.5 * IQR
```
Sehingga Dataset tersebut sekarang telah bersih dan memiliki 4131 sampel.

# Univariate Analysis
Pertama, fitur pada dataset dibagi menjadi categorical features dan numerical features. 
#### Categorical Features
Pada fitur categorical terdiri dari Area Type, City, Furnishing Status, Tenant Preferred. 
- Fitur Area Type
```sh
             jumlah sampel   persentase 
Super Area             2330    56.402808
Carpet Area            1799    43.548778
Built Area                2     0.048414
```
- Fitur City
```sh
           jumlah sampel   persentase 
Bangalore             840    20.334060
Chennai               831    20.116195
Hyderabad             804    19.462600
Mumbai                602    14.572743
Delhi                 538    13.023481
Kolkata               516    12.490922
```
- Fitur Furnishing Status
```sh
                jumlah sampel   persentase 
Semi-Furnished            1931    46.744130
Unfurnished               1698    41.103849
Furnished                  502    12.152021
```
- Fitur Tenant Preferred
```sh
                  jumlah sampel   persentase 
Bachelors/Family            3055    73.953038
Bachelors                    707    17.114500
Family                       369     8.932462
```
Pada fitur Area Locality dan Floor terdapat persebaran data yang tidak merata sehingga fitur tersebut tidak dimasukkan kedalam fitur categorical dan didrop column.

#### Numerical Features
Pada numerical features terdapat fitur BHK, Rent, Size, Bathroom.
Persebaran data pada numericak features sebagai berikut:
![2](https://user-images.githubusercontent.com/92345291/195993847-77dda151-e37f-4304-b4dc-e8f22e67ab32.png)

Berdasarkan histogram  diagram diatas bisa diperoleh beberapa informasi, antara lain:
- pada Fitur BHK dan Fitur Bathroom, terdapat sekitar 1 sampai 3 BHK dan Bathroom dan paling banyak jumlahnya adalah 2
- pada Fitur Size, sebagian besar rumah memiliki luas di bawah 1500 sqft
- pada Fitur Rent, rentang harga sewa cukup tinggi, yaitu dari 1200 hingga 3500000. Pembagian harga sewa tersebut tidak merata sehingga dampaknya terhadap model tidak bagus

# Multivariate Analysis
Multivariate Analysis menunjukkan hubungan antara dua atau lebih variabel sehingga pada data akan dilakukan analisis data pada fitur kategori dan numerik.
#### Categorical Features
- Fitur Area Type
![3](https://user-images.githubusercontent.com/92345291/196147424-758796b7-351b-4110-8718-79093cc7a8a9.png)
Fitur Area Type memiliki pengaruh yang kecil terhadap rata-rata harga sewa (Rent)

- Fitur City
![4](https://user-images.githubusercontent.com/92345291/196147810-a58ce0ed-159e-4a17-b074-900c96393434.png)
Fitur City memiliki pengaruh cukup besar terhadap rata-rata harga sewa (Rent). Harga sewa (Rent) terbesar terletak di Mumbai. Hal tersebut membuktikan bahwa kota Mumbai memiliki harga termahal untuk ditinggali

- Fitur Furnishing Status
![5](https://user-images.githubusercontent.com/92345291/196148715-58a545d8-5ffd-41a8-823c-f46f7d47aab3.png)
Fitur Furnishing Status memiliki pengaruh cukup besar terhadap rata-rata harga sewa (Rent). Hal tersebut merupakan hal yang biasa dilakukan apabila membeli rumah atau aparatemen yang sudah memiliki furniture atau perabotan didalamnya yang mengakibatkan harga sewa (Rent) juga bertambah dibandingkan dengan harga sewa (Rent) yang tidak emmiliki furniture sama sekali didalamnya. 

- Fitur Tenant Preffered
![6](https://user-images.githubusercontent.com/92345291/196149769-9b122045-b2d6-425e-a9fc-c128850f3e3c.png)
Fitur Tenant Preferred memiliki pengaruh yang cukup terhadap rata-rata harga sewa (Rent). Dimana harga tertinggi berada pada Family, karena akan membutuhkan rumah yang cukup luas atau nyaman untuk keluarga, terutama jika memiliki keluarga yang banyak didalamnya, sehingga harga sewa (Rent) juga akan lebih tinggi juga. Hal tersebut dapat kita lihat perbandingannya pada grafik diatas. 

### Numerical Features
Untuk mengamati hubungan antara fitur numerik, kita akan menggunakan fungsi pairplot()
![7](https://user-images.githubusercontent.com/92345291/196151518-fd5fceaf-081f-4393-a906-149dc8666113.png)
Pada pola sebaran data grafik pairplot diatas, terlihat bahwa ‘BHK’, ‘Size’, dan ‘Bathroom’ memiliki korelasi yang tinggi dengan fitur "Rent"

Untuk mengevaluasi skor korelasinya, gunakan fungsi corr().
![8](https://user-images.githubusercontent.com/92345291/196152492-6a5ecea5-7e90-4861-bb87-28602372ffa5.png)
Jika kita amati, fitur ‘BHK’, ‘Size’, dan ‘Bathroom’ memiliki skor korelasi yang besar (di atas 0.6) dengan fitur target ‘Rent’. Artinya, fitur 'Rent' berkorelasi tinggi dengan ketiga fitur tersebut.

# Data Preparation
### One Hot Encoding
One hot encoding merupakan teknik mengubah data kategorik menjadi data numerik, dimana setiap data yang merupakan kategori diubah menjadi kolom baru dengan nilai 0 atau 1. Fitur yang akan diubah pada dataset yaitu Area Type, City, Furnishing Status, dan Tenant Preferred.

### Train Test Split
Train test split merupakan proses membagi data menjadi data latih dan data uji. Data latih akan digunakan untuk membangun model, sedangkan data uji akan digunakan untuk menguji performa model. Pada dataset ini akan dilakukan proporsi pembagian sebesar 90:10, dimana jumlah dataset seluruhnya adalah 4131 dibagi menjadi 3717 untuk data latih dan 414 untuk data uji.

### Standarisasi
Algoritma machine learning memiliki performa lebih baik dan konvergen lebih cepat ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal. Proses scaling dan standarisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. Standardisasi adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan. Pada proyek ini, akan digunakan teknik StandarScaler dari library Scikitlearn. StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi. 
Sehingga dihasilkan:
![9](https://user-images.githubusercontent.com/92345291/196155636-f36cb204-1c21-4654-a8df-f1d15d42c65d.png)

Untuk mengecek nilai mean dan standar deviasi pada setelah proses standarisasi yaitu:
![10](https://user-images.githubusercontent.com/92345291/196155890-d9acfbec-3980-4049-96af-cd36dba6b069.png)
Sekarang nilai mean = 0 dan standar deviasi = 1.

# Model
Untuk mengevaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik akan dilakukan pemodelan machine learning dengan tiga algoritma. Ketiga algoritma yang digunakan, antara lain:
### K-Nearest Neighbor
Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif). 
Pada proyek ini digunakan sklearn.neighbors.KNeighborsRegressor dengan memasukkan X_train dan y_train dalam membangun model. Pada tahap ini menggunakan k = 10 tetangga (n_neighbors = 10)  dan metric Euclidean untuk mengukur jarak antara titik yang dimana hanya melatih data training dan menyimpan data testing untuk tahap evaluasi.

### Random Forest
Random forest merupakan salah satu model machine learning yang termasuk ke dalam kategori ensemble (group) learning. Dimana ia merupakan model prediksi yang terdiri dari beberapa model dan bekerja secara bersama-sama. Pada model ensemble, setiap model harus membuat prediksi secara independen. Kemudian, prediksi dari setiap model ensemble ini digabungkan untuk membuat prediksi akhir. Proyek ini menggunakan sklearn.ensemble.RandomForestRegressor dengan memasukkan X_train dan y_train dalam membangun model. Berikut adalah parameter-parameter yang digunakan:
```sh
n_estimator
```
jumlah trees (pohon) di forest, set n_estimator=50
```sh
max_depth
``` 
kedalaman atau panjang pohon, set max_depth=16
```sh
random_state
``` 
digunakan untuk mengontrol random number generator yang digunakan, set random_state=55
```sh
n_jobs
``` 
jumlah job (pekerjaan) yang digunakan secara paralel, set n_jobs=-1, artinya semua proses berjalan secara paralel

### Boosting
Algoritma ini bekerja dengan membangun model dari data latih. Kemudian ia membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan.  algoritma ini bertujuan untuk meningkatkan performa atau akurasi prediksi. Caranya adalah dengan menggabungkan beberapa model sederhana dan dianggap lemah (weak learners) sehingga membentuk suatu model yang kuat (strong ensemble learner). Proyek ini menggunakan sklearn.ensemble.AdaBoostRegressor dengan memasukkan X_train dan y_train dalam membangun model. 
Berikut merupakan parameter-parameter yang digunakan:
```sh
learning_rate
``` 
bobot yang diterapkan pada setiap regressor di masing-masing proses iterasi boosting, set learning_rate=0.05
```sh
random_state
```
digunakan untuk mengontrol random number generator yang digunakan, set random_state=55

# Evaluasi 
Metrik yang akan digunakan pada prediksi ini adalah MSE atau Mean Squared Error yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. Rumusnya adalah sebagai berikut:
![image](https://user-images.githubusercontent.com/92345291/196162775-2b298fbc-ec90-4e24-8953-c1c66429b27e.png)
Keterangan :
N = jumlah dataset
yi = nilai sebenarnya
y_pred = nilai prediksi

Namun, sebelum menghitung nilai MSE dalam model, perlu melakukan proses scaling fitur numerik pada data uji. Selanjutnya, evaluasi ketiga model dengan metrik MSE. Saat menghitung nilai Mean Squared Error pada data train dan test, membaginya dengan nilai 1e3. Hal ini bertujuan agar nilai mse berada dalam skala yang tidak terlalu besar. 
- Hasil evaluasi pada data latih dan data test adalah sebagai berikut
![11](https://user-images.githubusercontent.com/92345291/196163521-77852db3-0055-4e21-b0b1-7fd506197ffa.png)

- Hasil plot metrik dengan bar chart
![12](https://user-images.githubusercontent.com/92345291/196164014-af7a2d11-c760-439a-ab53-1eea7c33fbfa.png)
Dari gambar di atas, terlihat bahwa, model Random Forest (RF) memberikan nilai eror yang paling kecil dibandingkan dengan model yang lainnya.

- Untuk mengujinya, buat prediksi menggunakan beberapa harga dari data test, dan hasilnya sebagai berikut:
![13](https://user-images.githubusercontent.com/92345291/196165116-1948b6c4-f852-4553-b17f-edab47080529.png)
Terlihat bahwa prediksi dengan Random Forest (RF) memberikan hasil yang paling mendekati. Selain itu, untuk menguji hasil prediksi pada data lain, dapat dilakukan dengan cara mengubah indeks pada X_test.

