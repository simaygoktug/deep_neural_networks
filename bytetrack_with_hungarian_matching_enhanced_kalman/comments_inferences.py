import numpy as np

#git clone https://github.com/ifzhang/ByteTrack.git
#cd ByteTrack
#pip3 install -r requirements.txt
#python3 setup.py develop

#cmd ile bu aşama tamamlandı ve proje klonlandı.
#Ana proje kodları C++ ile yazılmış buna göre .cpp dosyaları da oluşturarak projeyi bilgisayarına entegre et.
#Ama zaten klonlama yapıldığı için tüm proje bilgisayara işlenmiş ve teste hazır olacak.

#Youtube videosunu saniye saniye izleyerek futbol maçı dataseti üzerinde ByteTrack test et.
#Sadece optimizasyon aşamasına geçildiğinde bir üst paragraftaki yapılması gerekenleri yap.

#Youtube Tutorial videosuna başlandı. 3.00 dk 
#Kaggle üzerinden dataset indirilecek.

#Youtube Tutorial videosunda ilerlenildi. 3.34 dk 
#Google Colab üzerindeki notebookta Kaggle API isteği için kod satırları yazıldı.
#Kaggle API alındı.
#Kaggle_Username = gktucanimay
#Kaggle_Key = 6b82cafce38b1c5e8c24500ccbe9979b

#Youtube Tutorial videosunda ilerlenildi. 4.23 dk 
#kaggle.json hatası yarım saatte düzeltildi ve ihtiyacımız olan 30 saniyelik videolardan oluşan dataseti indirecek kod parçası hatalıydı. 
#Düzeltilerek Google Colab üzerinde çalışır hale getirildi.

#%cd {HOME}
#!kaggle competitions files -c dfl-bundesliga-data-shootout | \
#grep clips | head -20 | \
#awk '{print $1}' | \
#while read -r line; \
#  do kaggle competitions download -c dfl-bundesliga-data-shootout -f line -p clips-quiet; unzip line.zip -d clips && rm line.zip \
#done

#Yeni tutorial bulundu. Bu Türkçe kaynaktan ve yukardaki yabancı kaynaktan ilerlenecek. --> https://www.youtube.com/watch?v=doJ3vp_9wK4
#4.54 dk
#Bu notebook'taki kodlar takip edilecek. --> https://github.com/roboflow/notebooks/blob/main/notebooks/how-to-track-and-count-vehicles-with-yolov8.ipynb

#YOLO ve Bytetrack'ı aynı anda kullanmanın amacı: 
#İlk yol, iki algoritmanın çıktılarını birleştirmektir. Bu, daha yüksek doğruluk ve performans elde etmek için yapılabilir.
#Başka bir yol, iki algoritmayı farklı görevler için kullanmaktır. 
#Örneğin, YOLO, nesneleri tespit etmek için kullanılabilirken, ByteTrack, nesnelerin konumunu ve yönünü izlemek için kullanılabilir. 

#12.05 dk
#Performans testi için tüm kodlar tamamlandı, tutorial devamında başka örnekler üzerinde de test yapacağım.
#Bu da bitince kaynak kodları üzerinde değişikliğe başlayabiliriz.

#15.49 dk
#Detection box'lar çok büyük olduğu için çıktı sonuç videosunda çok yer kaplıyordu. Düzenleme yapıldı ve test edildi.
#Tamamen doğru ve sıkıntısız çalışıyor.
#in: --> belirlediğimiz çizgiden -x kordinatında giren nesneleri sayarken; out: --> belirlediğimiz çizgiden +x kordinatında çıkan nesneleri saymaktadır.

#Video tamamlandı.
#Havaalanındaki insanları saydırma da tamamlandı.
#Eğer YOLO'nun kendi kütüphanesinde bulunmayan bir nesne takibi yapmak istersek yapmamız gereken şeyler:
#1.) Örneğin takip etmek istediğimiz nesne çikolata ise modeli onun için eğitip yeni yolo8x.pt dosyası elde etmeliyiz.
#2.) Bu dosyayı home directory'mize koymalıyız.
#3.) settings part'ında yeni .pt dosyamızın adı farklıysa yeni ismini yazmalıyız.
#4.) Directory ID'lerini 0 olarak değiştirmeliyiz. Eğer birden fazla eğittiğimiz ve takibini yapacağımız nesne varsa [0,1,....] şeklinde düzeltmeliyiz.

#Optimizasyon ve hızlandırma için araştırmalara başlandı:
#Kalman Filtresi'nden yararlanılmış, değişiklik yapılabilir.
#Bytetracker, videolarda yüzleri takip etmek için kullanılan bir açık kaynaklı yazılımdır. 
#Kalman filtresi, Bytetracker'ın yüzleri takip etmesini sağlayan önemli bir bileşendir.
#Bu, gürültülü veya eksik verilerden bile doğru tahminler yapmak için kullanılan bir istatistiksel filtredir.

#YAPILDI#
#kalman_filter.py üzerinde yapılabilecek değişiklikler:
#Matrislerin yeniden kullanımı: predict() ve update() yöntemlerinde, _motion_mat ve _update_mat matrisleri her seferinde yeniden hesaplanıyor. 
#Bu, gereksiz hesaplamalara neden oluyor. Bu matrisleri bir kez hesaplayıp saklayarak optimizasyon yapılabilir.
#Kovaryans matrisinin hesaplanmasında verimlilik: predict() ve update() yöntemlerinde, kovaryans matrisinin hesaplanması;
#np.linalg.multi_dot() işlevi kullanılarak gerçekleştiriliyor. 
#Bu işlev, matris çarpımlarını verimli bir şekilde hesaplamak için tasarlanmıştır. Ancak, bu işlev hala bazı gereksiz hesaplamalar içerebilir. 
#Bu hesaplamaları optimize etmek için np.linalg.block_diag() işlevi kullanılabilir.
#Bu değişiklikler, kodun performansını yaklaşık %30-40 oranında artırabilir:

class KalmanFilter:

    def __init__(self):
        self._motion_mat = np.eye(2 * self.ndim, 2 * self.ndim)
        self._update_mat = np.eye(self.ndim, 2 * self.ndim)

    def predict(self, mean, covariance):
        #Matrisleri yeniden kullanmak için
        mean = mean @ self._motion_mat
        covariance = (
            self._motion_mat @ covariance @ self._motion_mat.T
            + self._motion_noise
        )
        return mean, covariance

    def update(self, mean, covariance, measurement):
        #Matrisleri yeniden kullanmak için
        projected_mean = self._update_mat @ mean
        projected_cov = self._update_mat @ covariance @ self._update_mat.T
        return self._update(projected_mean, projected_cov, measurement)
#YAPILDI#

#Yüz takibi uygulaması, yüzlerin çok hızlı veya çok yavaş hareket ettiği bir ortamda çalışıyorsa: 
#İvme modelini kullanan bir Kalman Filtresi daha iyi performans gösterebilir. 
#Ayrıca, gating işlemi için başka bir mesafe metriği kullanmak, daha iyi sonuçlar verebilir.

#byte_tracker.py incelemesi:

#STrack sınıfı, bir cismin izini tutmak için kullanılan bir sınıftır.

#STrack sınıfının aşağıdaki özelliklere sahiptir:
#tlwh: Cismin sınırlandırma kutusunun koordinatları.
#score: Cismin güven skoru.
#track_id: Cismin izini tanımlamak için kullanılan benzersiz bir kimlik.
#mean: Cismin Kalman filtresi ortalaması.
#covariance: Cismin Kalman filtresi kovaryansı.
#state: Cismin izinin durumunu gösteren bir enum (Tracked, Lost, Removed).

#STrack sınıfının aşağıdaki metotları vardır:
#predict(): Cismin Kalman filtresi ile tahmin edilen yeni konumunu hesaplar.
#multi_predict(): Birden fazla cismin Kalman filtresini tahmin edilen yeni konumlarını hesaplar.
#activate(): Cismi izleme listesine ekler.
#re_activate(): Cismi izleme listesine yeniden ekler.
#update(): Cismin izini günceller.
#tlwh_to_xyah(): Cismin sınırlandırma kutusunu merkez koordinatları, en boy oranı ve yükseklik şeklinde dönüştürür.
#to_xyah(): Cismin sınırlandırma kutusunu merkez koordinatları, en boy oranı ve yükseklik şeklinde döndürür.

#BYTETracker sınıfı, çoklu cisim izleme için kullanılan bir sınıftır.

#BYTETracker sınıfı, aşağıdaki adımları izleyerek çoklu cisim izleme işlemini gerçekleştirir:
#Cisimleri algılar.
#İzlenen cisimlerin yeni konumlarını tahmin eder.
#İzlenen cisimleri algılanan cisimlerle eşleştirir.
#Eşleşmeyen cisimleri yeni izler olarak başlatır.
#Eşleşmeyen izleri kaybeder.

#Kodun adım adım açıklaması:

#STrack sınıfının tanımlanması.
#BYTETracker sınıfının tanımlanması.
#update() metodunun tanımlanması.

#update() metodunda aşağıdaki adımlar gerçekleştirilir:
#Algılanan cisimler tespit edilir.
#İzlenen cisimlerin yeni konumları tahmin edilir.
#İzlenen cisimler algılanan cisimlerle eşleştirilir.
#Eşleşmeyen cisimler yeni izler olarak başlatılır.
#Eşleşmeyen izleri kaybeder.
#update() metodunun çağrılması.

#byte_tracker.py üzerinde yapılabilecek değişiklikler:

#YAPILDI#
#Kalman filtresi tahminini optimize etmek: Kalman filtresi tahmini, BYTETracker'ın en zaman alıcı adımlarından biridir. 
#Bu tahmini optimize etmek için, daha verimli bir Kalman filtresi algoritması kullanılabilir. 
#Örneğin, Unscented Kalman filtresi veya Particle filter gibi algoritmalar daha verimli olabilir:

from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import ParticleFilter

class STrack(BaseTrack):
    ...

    def predict(self):
        # Gaussian Kalman filter
        self.mean, self.covariance = self.kalman_filter.predict()

        # Unscented Kalman filter
        self.mean, self.covariance = UnscentedKalmanFilter(
            self.mean, self.covariance, self.transition_matrix,
            self.measurement_matrix, self.process_noise_cov,
            self.measurement_noise_cov
        ).predict()

        # Particle filter
        self.mean, self.covariance = ParticleFilter(
        self.mean, self.covariance, self.transition_matrix,
        self.measurement_matrix, self.process_noise_cov,
        self.measurement_noise_cov
        ).predict()
#YAPILDI#

#Eşleştirme algoritmasını optimize etmek: Eşleştirme algoritması, BYTETracker'ın bir diğer zaman alıcı adımıdır. 
#Bu algoritmayı optimize etmek için, daha verimli bir eşleştirme algoritması kullanılabilir. 
#Örneğin, Hungarian algorithm veya Delaunay triangulation gibi algoritmalar daha verimli olabilir:

#YAPILDI#
from delaunay import delaunay_triangulation

def update(self, output_results, img_info, img_size):
    ...

    strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
    # Predict the current location with KF
    STrack.multi_predict(strack_pool)

    # Hungarian algorithm
    matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

    # Delaunay triangulation
    points = np.asarray([s.tlwh for s in strack_pool])
    triangles = delaunay_triangulation(points)
    matches = []
    for i in range(len(strack_pool)):
        for j in triangles[i]:
            if j < len(strack_pool):
                matches.append((i, j))
    u_track = [i for i in range(len(strack_pool)) if i not in matches]
    u_detection = [i for i in range(len(detections)) if i not in matches]
#YAPILDI#
    
#Fonksiyonlar ve makalenin kendisi de incelenecek.

#Makaleden bulduğum Pseudo-Code üzerinde yapılabilecek optimizasyonlar hakkında fikir alındı:

#1. Kalman filtresi tahminlerini önbelleklemek: -- Bizim güncel olarak kullandığımız algoritma: kalman_filter.py
#Her bir çerçeve için, tüm izleyiciler için Kalman filtresi tahminlerini hesaplamak zaman alıcıdır. 
#Bu tahminleri önbelleğe almak, bu işlemi önemli ölçüde hızlandırabilir.

#2. Herhangi bir ilişkilendirmeyi gerçekleştirmeden önce yüksek puanlı tespitleri filtrelemek:
#Herhangi bir ilişkilendirmeyi gerçekleştirmeden önce yüksek puanlı tespitleri filtrelemek, ilişkilendirme süresini önemli ölçüde azaltabilir. 
#Bu, düşük puanlı tespitlerin ilişkilendirmede kullanılmamasını sağlar.

#3. Daha verimli bir İLİŞKİLENDİRME algoritması kullanmak: -- Bizim güncel olarak kullandığımız algoritma: matching.py
#ByteTrack, iki tespiti eşleştirmek için iki benzerlik ölçümü kullanır. Bu ölçümler, her bir tespitin konumunu, boyutunu ve yönünü karşılaştırır. 
#Ancak, bu ölçümler, her bir tespitin tam geometrisini hesaba katmaz. 
#Daha verimli bir ilişkilendirme algoritması kullanmak, daha doğru eşleştirmeler sağlayabilir ve bu da toplam izleme süresini azaltabilir.

#4. Daha verimli bir İZLEYİCİ algoritması kullanmak: -- Bizim güncel olarak kullandığımız algoritma: basetrack.py ve byte_tracker.py 
#ByteTrack, her bir izleyiciyi bir Kalman filtresi kullanarak izler. Kalman filtreleri, izleyicinin konumunu ve hızını tahmin etmek için kullanılır. 
#Ancak, Kalman filtreleri, her bir izleyicinin yeni bir tespitle ilişkilendirildiğinde güncellenmelidir. 
#Daha verimli bir izleyici algoritması kullanmak, bu güncellemeyi daha hızlı hale getirebilir.

#Kalman Filtresi üzerinde yapılabilecek optimizasyonlar için detaylı araştırma yapıldı: 
#Bu linkteki yazı kesinlikle okunmalı temel prensip hakkında güzel ve kısa bilg veriyor: https://medium.com/@syndrome/kalman-filter-nedir-51c38a12c423 
#Bu filtre hakkında çok detaylı formüllerle desteklenmiş matematiksel ve teorik bilgi için: https://burakbayramli.github.io/dersblog/tser/tser_083_kf/kalman_filtreleri.html 

#Aslında "BAŞKA BiR TRACKING PREDICTION ALGORiTMASI KULLANMAK YERİNE" sadece kalman_filter.py ve byte_tracker.py üzerinde efektif iyileştirmeler yapmak daha mantıklı gibi.
#Bu konuyla ilgili güzel kısa biz yazı: 

#Because it's provably optimal for a large class of common problems (linear estimators with quadratic cost functions). 
#It's one of those solutions where the math just works out beautifully, making it highly implementable. 
#It has very nice interpretations in both stochastic systems and linear algebra. 
#And there are some very specific, very important problems which fit this category— navigation, control of linear systems, etc.
#And many of the alternatives are much more computationally intensive to implement. 
#For example, HMMs and Particle Filters are much more general. 
#But they aren't a nice closed form solution, they are computationally expensive heuristics with much weaker provable performance guarantees. 
#We're seeing more of these now, since computation is so much cheaper. 

#Ancak "Ensemble Random Forest Filter" makalesi göz önüne alınabilir. 
#Sanırım sonuca varabilmek için bir kere değişiklik yapıp veri seti üzerinde performans testi yapmak gerekecek:

#Bytetracker gibi MOT projelerinde nesne tespiti ve takibi için "ensemble random forest filter" mi yoksa "kalman filter" mı kullanılmalı? 
#Hangisi daha iyi performans gösterir?

#Bu sorunun cevabı, MOT projesinin spesifik gereksinimlerine bağlıdır. Genel olarak, ensemble random forest filter (ERF) ve Kalman filter, MOT için etkili olan iki farklı nesne tespiti ve takip yöntemidir.

#ERF, bir dizi rastgele orman sınıflandırıcısını birleştirerek çalışan bir yöntemdir. 
#Bu, nesne tespiti ve takibi için güçlü bir performans sağlar, çünkü sınıflandırıcılar birbirinin hatalarını dengeleyerek daha doğru sonuçlar verir.
#ERF, özellikle çok sayıda nesne ve arka plan karmaşası olan durumlarda etkilidir.

#Kalman filter, nesnelerin hareketini tahmin etmek için kullanılan bir yöntemdir. 
#Bu, nesne takibi için güçlü bir performans sağlar, çünkü nesnelerin hareketini doğru bir şekilde tahmin ederek nesnelerin konumlarını daha doğru bir şekilde takip eder.
#Kalman filter, özellikle nesnelerin hareketinin öngörülebildiği durumlarda etkilidir.

#Bytetracker gibi MOT projelerinde, ERF ve Kalman filter'ın her ikisi de etkili bir şekilde kullanılabilir. 
#ERF, çok sayıda nesne ve arka plan karmaşası olan durumlarda daha iyi performans gösterirken. 
#Kalman filter, nesnelerin hareketinin öngörülebildiği durumlarda daha iyi performans gösterir.

#ByteTrack diğer muadillerine göre NEYİ FARKLI YAPIYOR. Bunu tam anlamıyla öğrendikten sonra kodlarda oluşturduğun fikirlere göre iyileştirmelere başlayabilirsin.
#ByteTrack vs DeepSort üzerinden bazı çıkarımlar yakalayabilirsin.
#Bu yazıyı okuduğunda tamamen anlayacaksın: https://medium.com/mlearning-ai/all-you-need-to-know-about-bytetrack-tracker-5cda1c039fa4 

#Çoğu çok nesneli izleyici, izleme için algılamayı kullanır. 
#Ancak nesne algılama kutularının güven puanı, algılamada gerçek pozitif/yanlış pozitif değiş tokuşuna neden olur. 
#Algoritmaların çoğu, gerçek pozitifleri artırmak için bir eşik kullanarak algılama kutularını ortadan kaldırır. 
#Ancak bu aynı zamanda yanlış pozitif sonuçlara da neden olur.

#Bu çalışmanın yazarları, düşük puanlı olanları göz ardı etmek yerine hemen hemen her tespit kutusunu ilişkilendirerek bu soruna bir çözüm yöntemi önermektedir. 
#Yöntemleri aslında tespit ve veri ilişkilendirme kavşak alanına dayanmaktadır. 
#Bu veri ilişkilendirme yöntemine BYTE adı verilir. Bu ilişkilendirme, tracklet'lerle benzerliklerin incelenmesi yoluyla yapılan bir eşleştirme işlemidir.

#Çoğu izleyicinin aksine neredeyse her tespit kutusunu saklıyorlar. 
#Daha sonra bu kutular düşük ve yüksek puanlı kutulara ayrılır. 
#Daha sonra mevcut çerçevenin yeni konumlarını tahmin etmek için Kalman Filtresi kullanılır.

#Kalman VS ERF 

#Kalman filtresi, bir dizi ölçümden sistemin gizli parametrelerini tahmin etmek için kullanılan bir tahmin yöntemidir. 
#Random forest, bir dizi özellikten bir hedef değişkeni tahmin etmek için kullanılan bir karar ağacı kümesidir.

#İki yöntemi birleştirerek, bir dizi görüntüden bir nesnenin gizli parametrelerini tahmin etmek için bir model oluşturabilirsiniz. 
#Bu model, görüntülerden nesnenin konumunu, boyutunu ve yönünü tahmin edebilir.

#Bu model, nesne takibi, nesne tanıma ve nesne davranışı analizi gibi çeşitli uygulamalarda kullanılabilir.
#Özellikle, amacınız doğrultusunda bu modeli kullanmanın aşağıdaki faydaları olabilir:

#Daha doğru tahminler: Random forest, Kalman filtresinden daha doğru tahminler üretebilir. 
#Bunun nedeni, Random Forest'in birden fazla karar ağacından gelen tahminleri birleştirmesidir.
#Daha esnek model: Random forest, Kalman filtresinden daha esnek bir modeldir. 
#Bunun nedeni, Random Forest'in birden fazla özelliği dikkate alabilmesidir.
#Daha hızlı tahminler: Random forest, Kalman filtresinden daha hızlı tahminler üretebilir. 
#Bunun nedeni, Random Forest'in Kalman filtresinden daha az hesaplama gerektirmesidir.
    
import numpy as np

class RandomForestEnsembleKalmanFilter(object):
    def __init__(self, n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=1, bootstrap=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap

        self.estimators = []
        for _ in range(self.n_estimators):
            estimator = RandomForestRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, bootstrap=self.bootstrap)
            self.estimators.append(estimator)

    def fit(self, X, y):
        for estimator in self.estimators:
            estimator.fit(X, y)

    def predict(self, X):
        predictions = []
        for estimator in self.estimators:
            predictions.append(estimator.predict(X))
        predictions = np.array(predictions)
        return np.mean(predictions, axis=0)

class RandomForestRegressor(object):
    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1, bootstrap=True):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap

        self.tree_ = None

    def fit(self, X, y):
        self.tree_ = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf)
        self.tree_.fit(X, y)

    def predict(self, X):
        return self.tree_.predict(X)

class DecisionTreeRegressor(object):
    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.root_ = None

    def fit(self, X, y):
        self.root_ = Node(X, y, max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf)

    def predict(self, X):
        return self.root_.predict(X)

class Node(object):
    def __init__(self, X, y, max_depth, min_samples_split, min_samples_leaf):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.X = X
        self.y = y

        self.is_leaf_ = False
        self.feature_ = None
        self.threshold_ = None
        self.left_ = None
        self.right_ = None

    def fit(self):
        if self.y.shape[0] < self.min_samples_split:
            self.is_leaf_ = True
            return

        feature, threshold = self.find_best_split()
        if feature is None:
            self.is_leaf_ = True
            return

        self.feature_ = feature
        self.threshold_ = threshold

        left_X, left_y = self.X[self.X[:, feature] <= threshold], self.y[self.X[:, feature] <= threshold]
        right_X, right_y = self.X[self.X[:, feature] > threshold], self.y[self.X[:, feature] > threshold]

        self.left_ = Node(left_X, left_y, max_depth=self.max_depth - 1, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf)
        self.right_ = Node(right_X, right_y, max_depth=self.max_depth - 1, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf)

    def find_best_split(self):
        best_gain = float("-inf")
        best_feature = None
        best_threshold = None

        for feature in range(self.X.shape[1]):
            thresholds = np.unique(self.X[:, feature])
            for threshold in thresholds:
                gain = self.information_gain(threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def information_gain(self, threshold):
        left_X, left_y = self.X[self.X[:, self.feature_] <= threshold], self.y[self.X[:, self.feature_] <= threshold]
        right_X, right_y = self.X[self.X[:, self.feature_] > threshold], self.y[self.X[:, self.feature_] > threshold]

        left_entropy = entropy(left_y)
        right_entropy = entropy(right_y)

        split_entropy = (left_X.shape[0] / self.X.shape[0]) * left_entropy + (right_X.shape[0] / self.X.shape[0]) * right_entropy

        info_gain = entropy(self.y) - split_entropy
        return info_gain

    def entropy(self, y):
        p = np.sum(y == 1) / y.shape[0]
        if p == 0 or p == 1:
            return 0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

#Node sınıfı, bir karar ağacı nodunu temsil eder. fit() işlevi, nod için en iyi bölmeyi bulur. 
#find_best_split() işlevi, bir bölme için bilgi kazancını hesaplar. 
#entropy() işlevi, bir dizi etiketin entropisiyi hesaplar.
#DecisionTreeRegressor sınıfı, bir karar ağacı sınıflandırıcısını temsil eder. 
#fit() işlevi, sınıflandırıcıyı eğitmek için verileri kullanır. 
#predict() işlevi, yeni verileri sınıflandırmak için sınıflandırıcıyı kullanır.
#RandomForestRegressor sınıfı, bir rastgele orman sınıflandırıcısını temsil eder. 
#fit() işlevi, sınıflandırıcıyı eğitmek için verileri kullanır.
#predict() işlevi, yeni verileri sınıflandırmak için sınıflandırıcıyı kullanır.

#Artık tüm bu çıkarımlara göre hocanın istediği şekilde proposal yazmaya başlayabilirsin.

#Kalan Yapılacak Listesi:
# 1) byte_tracker_improved_hungarian.py dosyasında U_Kalman kısmını doğru düzelttiğinden emin ol, üstünden geç. +
# 2) kalman_filter.py dosyasında Google Bard'ın matris tavsiyesini de uygula. +  
# 3) byte_tracker_improved_delaunay.py yaz. - YAPILAMADI!
# 4) ERF kodu yazdır ve nasıl entegre edeceğini çöz. +
# 5) Performans metrikleri için kod yazdır ve nasıl entegre edeceğini çöz. +
# 6) Tüm performans metriklerinin kodunu tamamla. +
# 7) Google Colab'e bu metrikleri entegre et. Kendi değişikliklerinin öncesinde nasıl performans verdiğini raporla. BUNU HALLET!

