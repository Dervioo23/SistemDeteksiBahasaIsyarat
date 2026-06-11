import cv2
import numpy as np
import os

def slice_reference_image(image_path, output_dir):
    """
    Memotong gambar referensi menjadi gambar huruf individual.
    Mengasumsikan tata letak grid.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Konversi ke skala abu-abu
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Ambang batas untuk menemukan latar belakang putih
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Temukan kontur
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter kontur berdasarkan ukuran dan rasio aspek
    min_area = 1000
    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = w / float(h)
            
            # Filter kontur yang sangat lebar (kemungkinan header/teks)
            if aspect_ratio < 3.0: 
                valid_contours.append(c)
            else:
                print(f"Ignored wide contour at ({x},{y}) with AR={aspect_ratio:.2f}")

    bounding_boxes = [cv2.boundingRect(c) for c in valid_contours]
    
    # Urutkan berdasarkan Y terlebih dahulu dengan toleransi (tinggi baris)
    if not bounding_boxes:
        print("No contours found.")
        return

    # Hitung tinggi rata-rata untuk menentukan toleransi baris
    avg_h = np.mean([h for _, _, _, h in bounding_boxes])
    row_tolerance = avg_h * 0.6

    # Urutkan berdasarkan koordinat Y terlebih dahulu
    bounding_boxes.sort(key=lambda b: b[1])

    # Kelompokkan ke dalam baris
    rows = []
    current_row = []
    if bounding_boxes:
        current_y = bounding_boxes[0][1]
        
        for box in bounding_boxes:
            if box[1] > current_y + row_tolerance:
                # Baris baru dimulai
                # Urutkan baris sebelumnya berdasarkan X
                current_row.sort(key=lambda b: b[0])
                rows.append(current_row)
                current_row = [box]
                current_y = box[1]
            else:
                current_row.append(box)
                # Perbarui current_y menjadi rata-rata Y baris untuk menangani sedikit ketidaksejajaran
                # current_y = (current_y * len(current_row) + box[1]) / (len(current_row) + 1)
        
        # Tambahkan baris terakhir
        if current_row:
            current_row.sort(key=lambda b: b[0])
            rows.append(current_row)

    # Ratakan daftar
    sorted_boxes = [box for row in rows for box in row]

    # Simpan gambar
    count = len(sorted_boxes)
    print(f"Found {count} potential letter regions.")
    
    # Bersihkan direktori output terlebih dahulu
    for f in os.listdir(output_dir):
        if f.endswith(".png"):
            os.remove(os.path.join(output_dir, f))

    labels = []
    # Alfabet BISINDO/ASL standar biasanya memiliki 26 huruf
    # Jika kita menemukan lebih banyak, itu mungkin teks tambahan atau noise.
    # Jika kita menemukan lebih sedikit, kita mungkin melewatkan beberapa.
    
    # Coba petakan ke 0..25
    for i, box in enumerate(sorted_boxes):
        if i >= 26: 
            break # Batasi hingga 26 untuk menghindari overflow jika noise ada di akhir
            
        x, y, w, h = box
        # Tambahkan sedikit padding
        pad = 10
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(img.shape[1] - x, w + 2*pad)
        h = min(img.shape[0] - y, h + 2*pad)
        
        roi = img[y:y+h, x:x+w]
        
        # Simpan sebagai 0.png, 1.png, dll. sesuai dengan A, B...
        output_path = os.path.join(output_dir, f"{i}.png")
        cv2.imwrite(output_path, roi)
        print(f"Saved {output_path}")

if __name__ == "__main__":
    # Gunakan jalur gambar yang diunggah
    # Catatan: Saya perlu mengetahui jalur sebenarnya. Saya akan menggunakan placeholder dan meminta agen untuk mengisinya.
    # Sebenarnya, saya dapat menemukan jalur dari daftar artefak.
    # Path: C:/Users/DELL/.gemini/antigravity/brain/1113a6c1-c3ac-4e01-ad3b-c1d3e40ed10e/uploaded_image_1764240454590.png
    
    image_path = r"C:/Users/DELL/.gemini/antigravity/brain/1113a6c1-c3ac-4e01-ad3b-c1d3e40ed10e/uploaded_image_1764240454590.png"
    output_dir = r"c:\Deteksi Bahasa Isyarat2\reference_images"
    slice_reference_image(image_path, output_dir)
