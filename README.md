# Facial Expression Recognition

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
<!-- Tambahkan badge lain jika relevan, misal License, Build Status -->

A Python application for real-time facial expression recognition using a webcam. This project detects faces and identifies their expressions (e.g., happy, sad, neutral, angry).

<!-- Optional: Tambahkan screenshot atau GIF demo di sini -->
<!-- ![Demo Screenshot](link_ke_gambar_demo.png) -->

## Requirements

Untuk menjalankan aplikasi ini, Anda memerlukan:

*   **Python:** Versi 3.11 atau yang lebih baru.
*   **Webcam:** Terhubung dan dapat diakses oleh komputer Anda.
*   **Git:** Untuk mengkloning repositori.

## Installation

Ikuti langkah-langkah berikut untuk menyiapkan proyek di komputer lokal Anda:

1.  **Clone Repository:**
    Buka terminal atau command prompt Anda dan jalankan perintah berikut:
    ```bash
    git clone https://github.com/Hdytalhayat/facial_expression_recognition.git
    cd facial_expression_recognition
    ```
    *(Alternatif: Unduh file ZIP dari halaman GitHub dan ekstrak isinya.)*

2.  **Buat dan Aktifkan Virtual Environment:** (Sangat Direkomendasikan)
    Ini membantu mengisolasi dependensi proyek.
    ```bash
    # Buat virtual environment
    python -m venv .venv

    # Aktifkan virtual environment
    # Windows (Command Prompt/PowerShell):
    .\.venv\Scripts\activate
   
    ```
    Anda akan melihat `(.venv)` di awal prompt terminal jika aktivasi berhasil.

3.  **Install Dependencies:**
    Dengan virtual environment aktif, instal semua pustaka Python yang diperlukan:
    ```bash
    pip install -r requirements.txt
    ```

## How to Use

Setelah instalasi selesai dan virtual environment aktif, jalankan aplikasi dengan perintah berikut:
*pilih salah satu antara nomor 1 atau nomor 2

1.  **Jalankan Server:**
    ```bash
    python FERServer.py
    ```

2.  **Jalankan dengan Visualisasi:**
    Untuk menjalankan server *dan* menampilkan jendela pratinjau webcam secara langsung dengan deteksi ekspresi yang ditandai, gunakan flag `--vis`:
    ```bash
    python FERServer.py --vis
    ```

3.  **Hentikan Aplikasi:**
    Tekan `Ctrl + C` di terminal tempat Anda menjalankan skrip untuk menghentikan server.

## Dependencies Utama

Proyek ini bergantung pada pustaka yang tercantum dalam file `requirements.txt`. Beberapa pustaka kunci mungkin termasuk:

*   numpy
*   opencv-python
*   cvzone
*   mediapipe
