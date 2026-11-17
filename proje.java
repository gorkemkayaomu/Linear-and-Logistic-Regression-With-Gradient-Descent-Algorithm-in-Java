import java.awt.*;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import javax.swing.*;

public class proje {
    public static void main(String[] args) throws IOException {
        Csv csv = new Csv();
        csv.csvOkuma("spambase.csv");
        csv.veriyiAyir(csv.datalar, csv.siniflar, 0.66);

        LinearRegression lineerModel = new LinearRegression();
        lineerModel.egit(csv.egitimVerisi, csv.egitimSiniflari);
        double[] lineerMetrikler = lineerModel.metrikleriHesaplaVeYazdir(csv.testVerisi, csv.testSiniflari);

        LogisticRegression lojistikModel = new LogisticRegression();
        lojistikModel.egit(csv.egitimVerisi, csv.egitimSiniflari);
        double[] lojistikMetrikler = lojistikModel.metrikleriHesaplaVeYazdir(csv.testVerisi, csv.testSiniflari);

        GrafikPenceresi.goster(lineerModel.hataGecmisi, lojistikModel.hataGecmisi, lineerMetrikler, lojistikMetrikler);
    }
}

class Csv {
    String dosyaAdi;
    double[][] datalar;
    double[] siniflar;
    double[][] egitimVerisi;
    double[] egitimSiniflari;
    double[][] testVerisi;
    double[] testSiniflari;

    void csvOkuma(String dosyaYolu) throws IOException {
        this.dosyaAdi = dosyaYolu.replace(".csv", "");
        BufferedReader br = new BufferedReader(new FileReader(dosyaYolu));
        List<double[]> satirListesi = new ArrayList<>();
        List<Double> etiketListesi = new ArrayList<>();
        String referansEtiket = null;

        String satir;
        while ((satir = br.readLine()) != null) {
            if (satir.trim().isEmpty())
                continue;
            String[] degerler = satir.split(",");

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

            String etiket = degerler[degerler.length - 1].trim();
            try {
                double sayisalEtiket = Double.parseDouble(etiket);
                etiketListesi.add(sayisalEtiket);
            } catch (NumberFormatException e) {
                if (referansEtiket == null) {
                    referansEtiket = etiket;
                }
                etiketListesi.add(etiket.equals(referansEtiket) ? 1.0 : 0.0);
            }
        }
        br.close();

        datalar = satirListesi.toArray(new double[0][]);
        siniflar = new double[etiketListesi.size()];
        for (int i = 0; i < etiketListesi.size(); i++) {
            siniflar[i] = etiketListesi.get(i);
        }

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

    void veriyiAyir(double[][] veriler, double[] siniflar, double egitimOrani) throws IOException {
        int ornekSayisi = veriler.length;
        int egitimSayisi = (int) (ornekSayisi * egitimOrani);

        int[] indeksler = new int[ornekSayisi];
        for (int i = 0; i < ornekSayisi; i++) {
            indeksler[i] = i;
        }

        Random rastgele = new Random(42);
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

        for (int i = 0; i < egitimSayisi; i++) {
            int rastgeleIndeks = indeksler[i];
            egitimVerisi[i] = veriler[rastgeleIndeks];
            egitimSiniflari[i] = siniflar[rastgeleIndeks];
        }

        for (int i = egitimSayisi; i < ornekSayisi; i++) {
            int rastgeleIndeks = indeksler[i];
            testVerisi[i - egitimSayisi] = veriler[rastgeleIndeks];
            testSiniflari[i - egitimSayisi] = siniflar[rastgeleIndeks];
        }

        csvKaydet(dosyaAdi + "_egitim.csv", egitimVerisi, egitimSiniflari);
        csvKaydet(dosyaAdi + "_test.csv", testVerisi, testSiniflari);
    }

    void csvKaydet(String kayitYolu, double[][] veriler, double[] etiketler) throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter(kayitYolu));
        for (int i = 0; i < veriler.length; i++) {
            StringBuilder satir = new StringBuilder();
            for (int j = 0; j < veriler[i].length; j++) {
                satir.append(veriler[i][j]).append(",");
            }
            satir.append(etiketler[i]);
            bw.write(satir.toString());
            bw.newLine();
        }
        bw.close();
    }
}

class LinearRegression {
    double[] agirliklar;
    double b = 0.0;
    double[] gradyanLineerW;
    double gradyanLineerB = 0.0;
    double ogrenmeOrani = 0.003;
    int iterasyon = 100000;
    double tolerans = 1e-6;
    List<Double> hataGecmisi = new ArrayList<>();

    void egit(double[][] X, double[] y) {
        int satirSayisi = X.length;
        int ozellikSayisi = X[0].length;
        agirliklar = new double[ozellikSayisi];
        double oncekiJw = Double.MAX_VALUE;
        hataGecmisi.clear();

        for (int epoch = 0; epoch < iterasyon; epoch++) {
            gradyanLineerW = new double[ozellikSayisi];
            gradyanLineerB = 0.0;
            double toplamKareHata = 0.0;

            for (int i = 0; i < satirSayisi; i++) {
                double tahmin = tahminEtHam(X[i]);
                double hata = tahmin - y[i];

                toplamKareHata += Math.pow(hata, 2);

                for (int j = 0; j < ozellikSayisi; j++) {
                    gradyanLineerW[j] += hata * X[i][j];
                }
                gradyanLineerB += hata;
            }

            for (int j = 0; j < ozellikSayisi; j++) {
                gradyanLineerW[j] = (1.0 / satirSayisi) * gradyanLineerW[j];
                agirliklar[j] -= ogrenmeOrani * gradyanLineerW[j];
            }
            gradyanLineerB = (1.0 / satirSayisi) * gradyanLineerB;
            b -= ogrenmeOrani * gradyanLineerB;

            double jw = toplamKareHata / (2.0 * satirSayisi);
            hataGecmisi.add(jw);

            if (Math.abs(oncekiJw - jw) < tolerans) {
                System.out.println("Lineer Regresyon Erken Durdurma -> Epoch: " + epoch + " | Son J(w): " + jw);
                break;
            }
            oncekiJw = jw;
        }

        System.out.println("Lineer Regresyon Son Gradyan W: " + Arrays.toString(gradyanLineerW));
        System.out.println("Lineer Regresyon Son Gradyan B: " + gradyanLineerB);
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

    double[] metrikleriHesaplaVeYazdir(double[][] testX, double[] testY) {
        int tp = 0, tn = 0, fp = 0, fn = 0;
        for (int i = 0; i < testX.length; i++) {
            double tahmin = sinifTahminEt(testX[i]);
            double gercek = testY[i];
            if (tahmin == 1.0 && gercek == 1.0) tp++;
            else if (tahmin == 0.0 && gercek == 0.0) tn++;
            else if (tahmin == 1.0 && gercek == 0.0) fp++;
            else if (tahmin == 0.0 && gercek == 1.0) fn++;
        }

        double accuracy = (double) (tp + tn) / testX.length;
        double precision = (tp + fp) > 0 ? (double) tp / (tp + fp) : 0.0;
        double recall = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0.0;
        double f1 = (precision + recall) > 0 ? (2.0 * precision * recall) / (precision + recall) : 0.0;

        System.out.println("\n--- LINEER REGRESYON PERFORMANS METRİKLERİ ---");
        System.out.println("Confusion Matrix -> TP: " + tp + " | TN: " + tn + " | FP: " + fp + " | FN: " + fn);
        System.out.printf("Accuracy (Doğruluk) : %%%.2f\n", accuracy * 100);
        System.out.printf("Precision (Kesinlik): %%%.2f\n", precision * 100);
        System.out.printf("Recall (Duyarlılık) : %%%.2f\n", recall * 100);
        System.out.printf("F1-Score            : %%%.2f\n\n", f1 * 100);

        return new double[]{accuracy, precision, recall, f1, tp, tn, fp, fn};
    }
}

class LogisticRegression {
    double[] agirliklar;
    double b = 0.0;
    double[] gradyanW;
    double gradyanB = 0.0;
    double ogrenmeOrani = 0.003;
    int iterasyon = 100000;
    double tolerans = 1e-6;
    List<Double> hataGecmisi = new ArrayList<>();

    double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    void egit(double[][] X, double[] y) {
        int satirSayisi = X.length;
        int ozellikSayisi = X[0].length;
        agirliklar = new double[ozellikSayisi];
        double oncekiJw = Double.MAX_VALUE;
        hataGecmisi.clear();

        for (int epoch = 0; epoch < iterasyon; epoch++) {
            gradyanW = new double[ozellikSayisi];
            gradyanB = 0.0;
            double toplamLoss = 0.0;

            for (int i = 0; i < satirSayisi; i++) {
                double olasilik = olasilikTahminEt(X[i]);
                double hata = olasilik - y[i];

                double p = Math.max(1e-15, Math.min(1.0 - 1e-15, olasilik));
                toplamLoss += -(y[i] * Math.log(p) + (1.0 - y[i]) * Math.log(1.0 - p));

                for (int j = 0; j < ozellikSayisi; j++) {
                    gradyanW[j] += hata * X[i][j];
                }
                gradyanB += hata;
            }

            for (int j = 0; j < ozellikSayisi; j++) {
                gradyanW[j] = (1.0 / satirSayisi) * gradyanW[j];
                agirliklar[j] -= ogrenmeOrani * gradyanW[j];
            }
            gradyanB = (1.0 / satirSayisi) * gradyanB;
            b -= ogrenmeOrani * gradyanB;

            double jw = toplamLoss / satirSayisi;
            hataGecmisi.add(jw);

            if (Math.abs(oncekiJw - jw) < tolerans) {
                System.out.println("Lojistik Regresyon Erken Durdurma -> Epoch: " + epoch + " | Son J(w): " + jw);
                break;
            }
            oncekiJw = jw;
        }

        System.out.println("Lojistik Regresyon Son Gradyan W: " + Arrays.toString(gradyanW));
        System.out.println("Lojistik Regresyon Son Gradyan B: " + gradyanB);
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

    double[] metrikleriHesaplaVeYazdir(double[][] testX, double[] testY) {
        int tp = 0, tn = 0, fp = 0, fn = 0;
        for (int i = 0; i < testX.length; i++) {
            double tahmin = sinifTahminEt(testX[i]);
            double gercek = testY[i];
            if (tahmin == 1.0 && gercek == 1.0) tp++;
            else if (tahmin == 0.0 && gercek == 0.0) tn++;
            else if (tahmin == 1.0 && gercek == 0.0) fp++;
            else if (tahmin == 0.0 && gercek == 1.0) fn++;
        }

        double accuracy = (double) (tp + tn) / testX.length;
        double precision = (tp + fp) > 0 ? (double) tp / (tp + fp) : 0.0;
        double recall = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0.0;
        double f1 = (precision + recall) > 0 ? (2.0 * precision * recall) / (precision + recall) : 0.0;

        System.out.println("\n--- LOJİSTİK REGRESYON PERFORMANS METRİKLERİ ---");
        System.out.println("Confusion Matrix -> TP: " + tp + " | TN: " + tn + " | FP: " + fp + " | FN: " + fn);
        System.out.printf("Accuracy (Doğruluk) : %%%.2f\n", accuracy * 100);
        System.out.printf("Precision (Kesinlik): %%%.2f\n", precision * 100);
        System.out.printf("Recall (Duyarlılık) : %%%.2f\n", recall * 100);
        System.out.printf("F1-Score            : %%%.2f\n\n", f1 * 100);

        return new double[]{accuracy, precision, recall, f1, tp, tn, fp, fn};
    }
}

class GrafikPenceresi extends JPanel {
    List<Double> lineerHata;
    List<Double> lojistikHata;
    double[] lineerMetrikler;
    double[] lojistikMetrikler;

    GrafikPenceresi(List<Double> lineerHata, List<Double> lojistikHata, double[] lineerMetrikler, double[] lojistikMetrikler) {
        this.lineerHata = lineerHata;
        this.lojistikHata = lojistikHata;
        this.lineerMetrikler = lineerMetrikler;
        this.lojistikMetrikler = lojistikMetrikler;
        setBackground(Color.WHITE);
    }

    static void goster(List<Double> lineerHata, List<Double> lojistikHata, double[] lineerMetrikler, double[] lojistikMetrikler) {
        JFrame pencere = new JFrame("Lineer vs Lojistik Regresyon Performans ve Karmaşıklık Matrisi");
        pencere.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        pencere.setSize(1080, 760);
        pencere.add(new GrafikPenceresi(lineerHata, lojistikHata, lineerMetrikler, lojistikMetrikler));
        pencere.setLocationRelativeTo(null);
        pencere.setVisible(true);
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g;
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        int genislik = getWidth();
        int yukseklik = getHeight();
        int ustH = (int) (yukseklik * 0.52);

        int solX = 60, solY = 45, solW = (genislik / 2) - 90, solH = ustH - 95;
        g2.setColor(Color.BLACK);
        g2.setFont(new Font("SansSerif", Font.BOLD, 14));
        g2.drawString("Epoch - Maliyet J(w) Eğrisi", solX + 40, solY - 15);

        g2.drawLine(solX, solY, solX, solY + solH);
        g2.drawLine(solX, solY + solH, solX + solW, solY + solH);

        double maxHata = 0.0;
        for (double d : lineerHata) if (d > maxHata) maxHata = d;
        for (double d : lojistikHata) if (d > maxHata) maxHata = d;
        if (maxHata == 0) maxHata = 1.0;

        int maxEpoch = Math.max(lineerHata.size(), lojistikHata.size());

        g2.setFont(new Font("SansSerif", Font.PLAIN, 11));
        for (int i = 0; i <= 4; i++) {
            int yPos = solY + solH - (i * solH / 4);
            double deger = (maxHata * i) / 4.0;
            g2.setColor(new Color(230, 230, 230));
            g2.drawLine(solX + 1, yPos, solX + solW, yPos);
            g2.setColor(Color.BLACK);
            g2.drawString(String.format("%.2f", deger), solX - 35, yPos + 4);
        }
        g2.drawString("0", solX, solY + solH + 18);
        g2.drawString(String.valueOf(maxEpoch), solX + solW - 25, solY + solH + 18);
        g2.drawString("Epoch", solX + solW / 2 - 15, solY + solH + 32);

        g2.setStroke(new BasicStroke(2.2f));
        cizgiCiz(g2, lineerHata, maxEpoch, maxHata, solX, solY, solW, solH, new Color(30, 100, 220));
        cizgiCiz(g2, lojistikHata, maxEpoch, maxHata, solX, solY, solW, solH, new Color(220, 50, 50));

        g2.setColor(new Color(30, 100, 220));
        g2.fillRect(solX + solW - 150, solY + 10, 12, 12);
        g2.setColor(Color.BLACK);
        g2.drawString("Lineer J(w)", solX + solW - 132, solY + 21);

        g2.setColor(new Color(220, 50, 50));
        g2.fillRect(solX + solW - 150, solY + 30, 12, 12);
        g2.setColor(Color.BLACK);
        g2.drawString("Lojistik J(w)", solX + solW - 132, solY + 41);

        int sagX = (genislik / 2) + 40, sagY = 45, sagW = (genislik / 2) - 80, sagH = ustH - 95;
        g2.setFont(new Font("SansSerif", Font.BOLD, 14));
        g2.drawString("Performans Değerlendirme Metrikleri (%)", sagX + 40, sagY - 15);

        g2.drawLine(sagX, sagY, sagX, sagY + sagH);
        g2.drawLine(sagX, sagY + sagH, sagX + sagW, sagY + sagH);

        g2.setFont(new Font("SansSerif", Font.PLAIN, 11));
        for (int i = 0; i <= 5; i++) {
            int yPos = sagY + sagH - (i * sagH / 5);
            g2.setColor(new Color(230, 230, 230));
            g2.drawLine(sagX + 1, yPos, sagX + sagW, yPos);
            g2.setColor(Color.BLACK);
            g2.drawString("%" + (i * 20), sagX - 35, yPos + 4);
        }

        String[] metrikAdlari = {"Accuracy", "Precision", "Recall", "F1-Score"};
        int grupGenislik = sagW / 4;
        int barW = 28;

        for (int m = 0; m < 4; m++) {
            int merkezX = sagX + (m * grupGenislik) + (grupGenislik / 2);

            int b1H = (int) (lineerMetrikler[m] * sagH);
            int b1X = merkezX - barW - 2;
            g2.setColor(new Color(30, 100, 220));
            g2.fillRect(b1X, sagY + sagH - b1H, barW, b1H);
            g2.setColor(Color.BLACK);
            g2.setFont(new Font("SansSerif", Font.PLAIN, 10));
            g2.drawString(String.format("%.1f", lineerMetrikler[m] * 100), b1X + 2, sagY + sagH - b1H - 5);

            int b2H = (int) (lojistikMetrikler[m] * sagH);
            int b2X = merkezX + 2;
            g2.setColor(new Color(220, 50, 50));
            g2.fillRect(b2X, sagY + sagH - b2H, barW, b2H);
            g2.setColor(Color.BLACK);
            g2.drawString(String.format("%.1f", lojistikMetrikler[m] * 100), b2X + 2, sagY + sagH - b2H - 5);

            g2.setFont(new Font("SansSerif", Font.BOLD, 11));
            g2.drawString(metrikAdlari[m], merkezX - 24, sagY + sagH + 20);
        }

        int altY = ustH + 45;
        karmasiklikMatrisiCiz(g2, "Lineer Regresyon Karmaşıklık Matrisi", lineerMetrikler, solX + 40, altY);
        karmasiklikMatrisiCiz(g2, "Lojistik Regresyon Karmaşıklık Matrisi", lojistikMetrikler, sagX + 40, altY);
    }

    void karmasiklikMatrisiCiz(Graphics2D g2, String baslik, double[] m, int x, int y) {
        int tp = (int) m[4];
        int tn = (int) m[5];
        int fp = (int) m[6];
        int fn = (int) m[7];

        g2.setColor(Color.BLACK);
        g2.setFont(new Font("SansSerif", Font.BOLD, 14));
        g2.drawString(baslik, x + 30, y - 15);

        int kutuW = 140, kutuH = 85;
        int basX = x + 70, basY = y + 25;

        g2.setFont(new Font("SansSerif", Font.BOLD, 12));
        g2.drawString("Tahmin: 1", basX + 40, basY - 8);
        g2.drawString("Tahmin: 0", basX + kutuW + 40, basY - 8);
        g2.drawString("Gerçek: 1", basX - 65, basY + 48);
        g2.drawString("Gerçek: 0", basX - 65, basY + kutuH + 48);

        kutuCiz(g2, basX, basY, kutuW, kutuH, new Color(200, 240, 205), "TP (Doğru Pozitif)", tp);
        kutuCiz(g2, basX + kutuW, basY, kutuW, kutuH, new Color(250, 210, 210), "FN (Yanlış Negatif)", fn);
        kutuCiz(g2, basX, basY + kutuH, kutuW, kutuH, new Color(250, 210, 210), "FP (Yanlış Pozitif)", fp);
        kutuCiz(g2, basX + kutuW, basY + kutuH, kutuW, kutuH, new Color(200, 240, 205), "TN (Doğru Negatif)", tn);
    }

    void kutuCiz(Graphics2D g2, int x, int y, int w, int h, Color arkaPlan, String etiket, int deger) {
        g2.setColor(arkaPlan);
        g2.fillRect(x, y, w, h);
        g2.setColor(Color.DARK_GRAY);
        g2.setStroke(new BasicStroke(1.5f));
        g2.drawRect(x, y, w, h);

        g2.setColor(Color.BLACK);
        g2.setFont(new Font("SansSerif", Font.PLAIN, 11));
        g2.drawString(etiket, x + 18, y + 28);

        g2.setFont(new Font("SansSerif", Font.BOLD, 20));
        g2.drawString(String.valueOf(deger), x + (w / 2) - 12, y + 60);
    }

    void cizgiCiz(Graphics2D g2, List<Double> veriler, int maxEpoch, double maxHata, int x, int y, int w, int h, Color renk) {
        if (veriler.size() < 2) return;
        g2.setColor(renk);
        for (int i = 0; i < veriler.size() - 1; i++) {
            int x1 = x + (int) ((double) i / (maxEpoch - 1) * w);
            int y1 = y + h - (int) ((veriler.get(i) / maxHata) * h);
            int x2 = x + (int) ((double) (i + 1) / (maxEpoch - 1) * w);
            int y2 = y + h - (int) ((veriler.get(i + 1) / maxHata) * h);
            g2.drawLine(x1, y1, x2, y2);
        }
    }
}
