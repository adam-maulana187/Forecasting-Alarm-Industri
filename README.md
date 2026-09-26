# LSTM Downtime Mesin Pengemasan — Aplikasi Streamlit

Aplikasi interaktif untuk melatih dan mencoba model LSTM yang memprediksi
`%downtime` mesin pengemasan pada interval berikutnya, berdasarkan riwayat
status mesin (idle, production, downtime, performance loss, dst).

Mendukung dua mode:
- **Regresi** — prediksi nilai `%downtime` (0–1)
- **Klasifikasi** — prediksi apakah interval berikutnya "downtime tinggi"
  (di atas threshold yang bisa kamu atur)

## Cara menjalankan

1. Install dependensi (disarankan di virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

2. Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```

3. Browser akan terbuka otomatis (biasanya di `http://localhost:8501`).

4. Di sidebar:
   - Upload file CSV data mesin pengemasan (format sama dengan
     `sequences_1h_data.csv`: harus punya kolom `interval_start`,
     `equipment_ID`, dan `%downtime`, `%idle`, `%production`,
     `%performance_loss`, `%scheduled_downtime`, `#changes`, `count_sum`)
   - Pilih mode (Regresi/Klasifikasi)
   - Atur window, split data, dan hyperparameter model
   - Klik **Latih Model**

5. Buka tab **Training** untuk lihat kurva belajar, dan tab
   **Evaluasi & Coba Prediksi** untuk melihat metrik, grafik, dan mencoba
   prediksi titik-per-titik.

## Catatan

- Training berjalan langsung di aplikasi (bukan model pre-trained), jadi
  butuh waktu beberapa puluh detik hingga beberapa menit tergantung ukuran
  data dan jumlah epoch.
- Sequence dibangun secara *posisional* per mesin (N interval tercatat
  terakhir), bukan berdasarkan jam kalender berurutan, karena data
  sensor semacam ini biasanya punya banyak jam yang tidak tercatat.
  Jarak waktu antar baris (`gap_hours`) ikut dijadikan fitur.
- Split train/test dilakukan kronologis (melatih dari data lama, menguji
  di data yang lebih baru) supaya evaluasi mencerminkan kondisi nyata.
