import java.io.*;
import java.util.*;

public class proje {
    public static void main(String[] args) throws IOException {
        Csv csv = new Csv();
        csv.csvOkuma("spambase.csv");
        csv.veriyiAyir(csv.datalar, csv.siniflar, 0.66);

        LinearRegression lineerModel = new LinearRegression();
        lineerModel.egit(csv.egitimVerisi, csv.egitimSiniflari);
        double lineerDogruluk = lineerModel.dogrulukHesapla(csv.testVerisi, csv.testSiniflari);
        System.out.println("Lineer Regresyon Test Doğruluk Oranı (Accuracy): %" + (lineerDogruluk * 100));

        LogisticRegression lojistikModel = new LogisticRegression();
        lojistikModel.egit(csv.egitimVerisi, csv.egitimSiniflari);
        double lojistikDogruluk = lojistikModel.dogrulukHesapla(csv.testVerisi, csv.testSiniflari);
        System.out.println("Lojistik Regresyon Test Doğruluk Oranı (Accuracy): %" + (lojistikDogruluk * 100));
    }
}

class Csv {
    double[][] datalar;
    double[] siniflar;
    double[][] egitimVerisi;
    double[] egitimSiniflari;
    double[][] testVerisi;
    double[] testSiniflari;

    void csvOkuma(String dosyaYolu) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(dosyaYolu));
        List<double[]> satirListesi = new ArrayList<>();
        List<Double> etiketListesi = new ArrayList<>();

        String satir;
        while ((satir = br.readLine()) != null) {
            if (satir.trim().isEmpty())
                continue;
            String[] degerler = satir.split(",");

            // İlk sütun sayı değilse (örneğin deauth_test.csv'deki başlık satırıysa) bu satırı atla
            try {
                Double.parseDouble(degerler[0].trim());
            } catch (NumberFormatException e) {
                continue;
            }

            double[] satirVerisi = new double[degerler.length - 1];
            for (int j = 0; j < degerler.length - 1; j++) {
                satirVerisi[j] = Double.parseDouble(degerler[j].trim());
            }
            satirListesi.add(satirVerisi);

            // Son sütun kontrolü: Sayıysa (2. durum) doğrudan al, String ise (1. durum) dönüştür
            String etiket = degerler[degerler.length - 1].trim();
            try {
                // 2. Durum: Son sütun zaten sayısal (örn: "0" veya "1")
                double sayisalEtiket = Double.parseDouble(etiket);
                etiketListesi.add(sayisalEtiket);
            } catch (NumberFormatException e) {
                // 1. Durum: Son sütun String (örn: "M" veya "R")
                etiketListesi.add(etiket.equals("M") ? 1.0 : 0.0);
            }
        }
        br.close();

        datalar = satirListesi.toArray(new double[0][]);
        siniflar = new double[etiketListesi.size()];
        for (int i = 0; i < etiketListesi.size(); i++) {
            siniflar[i] = etiketListesi.get(i);
        }

        // Tüm özellikleri (sütunları) Min-Max normalizasyonu ile 0.0 - 1.0 aralığına çek
        normalizeEt();
    }

    void normalizeEt() {
        int satirSayisi = datalar.length;
        int sutunSayisi = datalar[0].length;

        for (int j = 0; j < sutunSayisi; j++) {
            double min = datalar[0][j];
            double max = datalar[0][j];

            for (int i = 1; i < satirSayisi; i++) {
                if (datalar[i][j] < min) min = datalar[i][j];
                if (datalar[i][j] > max) max = datalar[i][j];
            }

            double aralik = max - min;
            if (aralik > 0) {
                for (int i = 0; i < satirSayisi; i++) {
                    datalar[i][j] = (datalar[i][j] - min) / aralik;
                }
            }
        }
    }

    void veriyiAyir(double[][] veriler, double[] siniflar, double egitimOrani) {
        int ornekSayisi = veriler.length;
        int egitimSayisi = (int) (ornekSayisi * egitimOrani);

        // Satır indekslerini oluşturup Fisher-Yates algoritması ile saf döngüyle karıştırıyoruz
        int[] indeksler = new int[ornekSayisi];
        for (int i = 0; i < ornekSayisi; i++) {
            indeksler[i] = i;
        }

        Random rastgele = new Random();
        for (int i = ornekSayisi - 1; i > 0; i--) {
            int j = rastgele.nextInt(i + 1);
            int gecici = indeksler[i];
            indeksler[i] = indeksler[j];
            indeksler[j] = gecici;
        }

        egitimVerisi = new double[egitimSayisi][];
        egitimSiniflari = new double[egitimSayisi];

        testVerisi = new double[ornekSayisi - egitimSayisi][];
        testSiniflari = new double[ornekSayisi - egitimSayisi];

        // İlk %66'lık kısmı Eğitim setine aktar
        for (int i = 0; i < egitimSayisi; i++) {
            int rastgeleIndeks = indeksler[i];
            egitimVerisi[i] = veriler[rastgeleIndeks];
            egitimSiniflari[i] = siniflar[rastgeleIndeks];
        }

        // Kalan %34'lük kısmı Test setine aktar
        for (int i = egitimSayisi; i < ornekSayisi; i++) {
            int rastgeleIndeks = indeksler[i];
            testVerisi[i - egitimSayisi] = veriler[rastgeleIndeks];
            testSiniflari[i - egitimSayisi] = siniflar[rastgeleIndeks];
        }
    }
}

class LinearRegression {
    double[] agirliklar;
    double b = 0.0;
    double ogrenmeOrani = 0.003;
    int iterasyon = 10000;
    double tolerans = 1e-6;

    void egit(double[][] X, double[] y) {
        int satirSayisi = X.length;
        int ozellikSayisi = X[0].length;
        agirliklar = new double[ozellikSayisi];
        double oncekiMse = Double.MAX_VALUE;

        for (int epoch = 0; epoch < iterasyon; epoch++) {
            double[] gradyanW = new double[ozellikSayisi];
            double gradyanB = 0.0;
            double toplamKareHata = 0.0;

            for (int i = 0; i < satirSayisi; i++) {
                double tahmin = tahminEtHam(X[i]);
                double hata = tahmin - y[i];

                toplamKareHata += Math.pow(hata, 2);

                for (int j = 0; j < ozellikSayisi; j++) {
                    gradyanW[j] += hata * X[i][j];
                }
                gradyanB += hata;
            }

            // Ağırlıkları (w) ve sabiti (b) güncelle
            for (int j = 0; j < ozellikSayisi; j++) {
                agirliklar[j] -= ogrenmeOrani * (2.0 / satirSayisi) * gradyanW[j];
            }
            b -= ogrenmeOrani * (2.0 / satirSayisi) * gradyanB;

            double mse = toplamKareHata / satirSayisi;

            if (Math.abs(oncekiMse - mse) < tolerans) {
                System.out.println("Lineer Regresyon Erken Durdurma -> Epoch: " + epoch + " | Son MSE: " + mse);
                break;
            }
            oncekiMse = mse;
        }
    }

    double tahminEtHam(double[] x) {
        double toplam = b;
        for (int j = 0; j < x.length; j++) {
            toplam += agirliklar[j] * x[j];
        }
        return toplam;
    }

    double sinifTahminEt(double[] x) {
        return tahminEtHam(x) >= 0.5 ? 1.0 : 0.0;
    }

    double dogrulukHesapla(double[][] testX, double[] testY) {
        int dogruSayisi = 0;
        for (int i = 0; i < testX.length; i++) {
            if (sinifTahminEt(testX[i]) == testY[i]) {
                dogruSayisi++;
            }
        }
        return (double) dogruSayisi / testX.length;
    }
}

class LogisticRegression {
    double[] agirliklar;
    double b = 0.0;
    double ogrenmeOrani = 0.05;
    int iterasyon = 10000;
    double tolerans = 1e-6;

    // Sigmoid aktivasyon fonksiyonu: 1 / (1 + e^(-z))
    double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    void egit(double[][] X, double[] y) {
        int satirSayisi = X.length;
        int ozellikSayisi = X[0].length;
        agirliklar = new double[ozellikSayisi];
        double oncekiLoss = Double.MAX_VALUE;

        for (int epoch = 0; epoch < iterasyon; epoch++) {
            double[] gradyanW = new double[ozellikSayisi];
            double gradyanB = 0.0;
            double toplamLoss = 0.0;

            for (int i = 0; i < satirSayisi; i++) {
                double olasilik = olasilikTahminEt(X[i]);
                double hata = olasilik - y[i];

                // Log-Loss (Cross-Entropy) hesabı (log(0) hatasını önlemek için 1e-15 ile sınırlandırılır)
                double p = Math.max(1e-15, Math.min(1.0 - 1e-15, olasilik));
                toplamLoss += -(y[i] * Math.log(p) + (1.0 - y[i]) * Math.log(1.0 - p));

                for (int j = 0; j < ozellikSayisi; j++) {
                    gradyanW[j] += hata * X[i][j];
                }
                gradyanB += hata;
            }

            // Ağırlıkları (w) ve sabiti (b) güncelle
            for (int j = 0; j < ozellikSayisi; j++) {
                agirliklar[j] -= ogrenmeOrani * (1.0 / satirSayisi) * gradyanW[j];
            }
            b -= ogrenmeOrani * (1.0 / satirSayisi) * gradyanB;

            double ortalamaLoss = toplamLoss / satirSayisi;

            if (Math.abs(oncekiLoss - ortalamaLoss) < tolerans) {
                System.out.println("Lojistik Regresyon Erken Durdurma -> Epoch: " + epoch + " | Son Loss: " + ortalamaLoss);
                break;
            }
            oncekiLoss = ortalamaLoss;
        }
    }

    double olasilikTahminEt(double[] x) {
        double z = b;
        for (int j = 0; j < x.length; j++) {
            z += agirliklar[j] * x[j];
        }
        return sigmoid(z);
    }

    double sinifTahminEt(double[] x) {
        return olasilikTahminEt(x) >= 0.5 ? 1.0 : 0.0;
    }

    double dogrulukHesapla(double[][] testX, double[] testY) {
        int dogruSayisi = 0;
        for (int i = 0; i < testX.length; i++) {
            if (sinifTahminEt(testX[i]) == testY[i]) {
                dogruSayisi++;
            }
        }
        return (double) dogruSayisi / testX.length;
    }
}