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

