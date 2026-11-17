import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import javax.swing.*;
import java.awt.*;

public class proje {
    public static void main(String[] args) throws IOException {
        Csv csv = new Csv();
        csv.csvOkuma("sonar data.csv");
        csv.veriyiAyir(csv.datalar, csv.siniflar, 0.66);

        LinearRegression lineerModel = new LinearRegression();
        lineerModel.egit(csv.egitimVerisi, csv.egitimSiniflari);
        double lineerDogruluk = lineerModel.dogrulukHesapla(csv.testVerisi, csv.testSiniflari);
        System.out.println("Lineer Regresyon Test Doğruluk Oranı (Accuracy): %" + (lineerDogruluk * 100));

        LogisticRegression lojistikModel = new LogisticRegression();
        lojistikModel.egit(csv.egitimVerisi, csv.egitimSiniflari);
        double lojistikDogruluk = lojistikModel.dogrulukHesapla(csv.testVerisi, csv.testSiniflari);
        System.out.println("Lojistik Regresyon Test Doğruluk Oranı (Accuracy): %" + (lojistikDogruluk * 100));

        GrafikPenceresi.goster(lineerModel.hataGecmisi, lojistikModel.hataGecmisi, lineerDogruluk, lojistikDogruluk);
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
    int iterasyon = 10000;
    double tolerans = 1e-6;
    List<Double> hataGecmisi = new ArrayList<>();

    void egit(double[][] X, double[] y) {
        int satirSayisi = X.length;
        int ozellikSayisi = X[0].length;
        agirliklar = new double[ozellikSayisi];
        double oncekiMse = Double.MAX_VALUE;
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
                gradyanLineerW[j] = (2.0 / satirSayisi) * gradyanLineerW[j];
                agirliklar[j] -= ogrenmeOrani * gradyanLineerW[j];
            }
            gradyanLineerB = (2.0 / satirSayisi) * gradyanLineerB;
            b -= ogrenmeOrani * gradyanLineerB;

            double mse = toplamKareHata / satirSayisi;
            hataGecmisi.add(mse);

            if (Math.abs(oncekiMse - mse) < tolerans) {
                System.out.println("Lineer Regresyon Erken Durdurma -> Epoch: " + epoch + " | Son MSE: " + mse);
                break;
            }
            oncekiMse = mse;
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
    double[] gradyanW;
    double gradyanB = 0.0;
    double ogrenmeOrani = 0.05;
    int iterasyon = 10000;
    double tolerans = 1e-6;
    List<Double> hataGecmisi = new ArrayList<>();

    double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    void egit(double[][] X, double[] y) {
        int satirSayisi = X.length;
        int ozellikSayisi = X[0].length;
        agirliklar = new double[ozellikSayisi];
        double oncekiLoss = Double.MAX_VALUE;
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

            double ortalamaLoss = toplamLoss / satirSayisi;
            hataGecmisi.add(ortalamaLoss);

            if (Math.abs(oncekiLoss - ortalamaLoss) < tolerans) {
                System.out.println("Lojistik Regresyon Erken Durdurma -> Epoch: " + epoch + " | Son Loss: " + ortalamaLoss);
                break;
            }
            oncekiLoss = ortalamaLoss;
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

class GrafikPenceresi extends JPanel {
    List<Double> lineerHata;
    List<Double> lojistikHata;
    double lineerDogruluk;
    double lojistikDogruluk;

    GrafikPenceresi(List<Double> lineerHata, List<Double> lojistikHata, double lineerDogruluk, double lojistikDogruluk) {
        this.lineerHata = lineerHata;
        this.lojistikHata = lojistikHata;
        this.lineerDogruluk = lineerDogruluk;
        this.lojistikDogruluk = lojistikDogruluk;
        setBackground(Color.WHITE);
    }

    static void goster(List<Double> lineerHata, List<Double> lojistikHata, double lineerDogruluk, double lojistikDogruluk) {
        JFrame pencere = new JFrame("Lineer vs Lojistik Regresyon Performans Grafikleri");
        pencere.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        pencere.setSize(950, 500);
        pencere.add(new GrafikPenceresi(lineerHata, lojistikHata, lineerDogruluk, lojistikDogruluk));
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

        int solX = 60, solY = 50, solW = (genislik / 2) - 90, solH = yukseklik - 110;
        g2.setColor(Color.BLACK);
        g2.setFont(new Font("SansSerif", Font.BOLD, 14));
        g2.drawString("Epoch - Hata (MSE / Loss) Eğrisi", solX + 40, solY - 15);

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
        g2.drawString("Epoch", solX + solW / 2 - 15, solY + solH + 35);

        g2.setStroke(new BasicStroke(2.2f));
        cizgiCiz(g2, lineerHata, maxEpoch, maxHata, solX, solY, solW, solH, new Color(30, 100, 220));
        cizgiCiz(g2, lojistikHata, maxEpoch, maxHata, solX, solY, solW, solH, new Color(220, 50, 50));

        g2.setColor(new Color(30, 100, 220));
        g2.fillRect(solX + solW - 150, solY + 10, 12, 12);
        g2.setColor(Color.BLACK);
        g2.drawString("Lineer (MSE)", solX + solW - 132, solY + 21);

        g2.setColor(new Color(220, 50, 50));
        g2.fillRect(solX + solW - 150, solY + 30, 12, 12);
        g2.setColor(Color.BLACK);
        g2.drawString("Lojistik (Loss)", solX + solW - 132, solY + 41);

        int sagX = (genislik / 2) + 50, sagY = 50, sagW = (genislik / 2) - 100, sagH = yukseklik - 110;
        g2.setFont(new Font("SansSerif", Font.BOLD, 14));
        g2.drawString("Test Doğruluk Oranı (Accuracy %)", sagX + 40, sagY - 15);

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

        int barGenislik = 70;
        int bar1X = sagX + sagW / 4 - barGenislik / 2;
        int bar1H = (int) (lineerDogruluk * sagH);
        g2.setColor(new Color(30, 100, 220));
        g2.fillRect(bar1X, sagY + sagH - bar1H, barGenislik, bar1H);
        g2.setColor(Color.BLACK);
        g2.drawString(String.format("%%%.2f", lineerDogruluk * 100), bar1X + 12, sagY + sagH - bar1H - 8);
        g2.drawString("Lineer Reg.", bar1X + 4, sagY + sagH + 20);

        int bar2X = sagX + (3 * sagW / 4) - barGenislik / 2;
        int bar2H = (int) (lojistikDogruluk * sagH);
        g2.setColor(new Color(220, 50, 50));
        g2.fillRect(bar2X, sagY + sagH - bar2H, barGenislik, bar2H);
        g2.setColor(Color.BLACK);
        g2.drawString(String.format("%%%.2f", lojistikDogruluk * 100), bar2X + 12, sagY + sagH - bar2H - 8);
        g2.drawString("Lojistik Reg.", bar2X + 2, sagY + sagH + 20);
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
