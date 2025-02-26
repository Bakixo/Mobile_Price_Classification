Mobile Price Classification
Bu proje, cep telefonlarının fiyat kategorilerini tahmin etmek amacıyla çeşitli makine öğrenimi algoritmalarını kullanarak bir sınıflandırma modeli oluşturur. Farklı modellerin doğruluğunu değerlendirip, en iyi performansı gösteren modeli seçerek test verileri üzerinde tahminler yapar. Ayrıca, modelin özellik önemini belirleyip görselleştirir.

Proje Açıklaması
Bu projede, cep telefonlarıyla ilgili çeşitli özellikler (örneğin, batarya gücü, RAM, kamera özellikleri, vb.) kullanılarak, telefonların hangi fiyat aralığında olduğunu sınıflandırmak amaçlanmıştır. Proje, aşağıdaki makine öğrenimi algoritmalarını kullanarak eğitilmiş üç farklı model içerir:

Rastgele Orman (Random Forest Classifier)
Gradient Boosting Classifier
Lojistik Regresyon (Logistic Regression)
Projede ayrıca, en iyi performansı gösteren modeli seçmek ve hiperparametre optimizasyonu yapmak için Grid Search kullanılmıştır.

Kurulum
Proje, Python 3.x ve aşağıdaki kütüphaneleri gerektirir:

pandas
numpy
scikit-learn
matplotlib
seaborn
Aşağıdaki komutları kullanarak gerekli kütüphaneleri yükleyebilirsiniz:

bash
Copy
Edit
pip install pandas numpy scikit-learn matplotlib seaborn
Kullanım
Veri Seti: Proje, train.csv ve test.csv dosyalarını kullanır. Eğitim verisi train.csv dosyasından alınır, test verisi ise test.csv dosyasından alınır.

Model Eğitimi ve Değerlendirme:

Eğitim verileri üzerinde üç farklı model (Random Forest, Gradient Boosting, Logistic Regression) eğitilir.
Modellerin doğruluğu hesaplanır ve karşılaştırılır.
Grid Search ile en iyi Rastgele Orman modeli bulunur.
Sonuçlar:

En iyi model ile test verisi üzerinde tahminler yapılır ve sonuçlar predictions.csv dosyasına kaydedilir.
Modelin özelliklerinin önem dereceleri görselleştirilir.
Görseller
Projenin çeşitli aşamalarında elde edilen bazı görseller:

Özellikler Arası Korelasyon Matrisi:

En Önemli 10 Özellik:

Confusion Matrix (Doğrulama Seti):



Katkıda Bulunma
Bu projeye katkıda bulunmak için aşağıdaki adımları takip edebilirsiniz:

Bu repo'yu fork edin.
Yeni bir branch oluşturun (git checkout -b feature/yenilik).
Değişikliklerinizi yapın ve commit edin.
Yeni branch'ınızı repo'ya push edin (git push origin feature/yenilik).
Pull request oluşturun.
Lisans
Bu proje MIT Lisansı altında lisanslanmıştır.
