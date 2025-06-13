import gradio as gr
import numpy as np
import tensorflow as tf
from collections import Counter

# Model custom MatrixFactorization
class MatrixFactorization(tf.keras.Model):
    def __init__(self, n_users, n_places, embedding_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.n_users = n_users
        self.n_places = n_places
        self.embedding_dim = embedding_dim

        self.user_embedding = tf.keras.layers.Embedding(
            input_dim=n_users, 
            output_dim=embedding_dim
        )
        self.place_embedding = tf.keras.layers.Embedding(
            input_dim=n_places, 
            output_dim=embedding_dim
        )

    def call(self, inputs):
        user_vec = self.user_embedding(inputs[:, 0])
        place_vec = self.place_embedding(inputs[:, 1])
        return tf.reduce_sum(user_vec * place_vec, axis=1)

    def get_config(self):
        return {
            "n_users": self.n_users,
            "n_places": self.n_places,
            "embedding_dim": self.embedding_dim
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Load model dan ambil nilai n_users dan n_places dari model
model = tf.keras.models.load_model(
    "model_rekomendasi.keras", 
    custom_objects={"MatrixFactorization": MatrixFactorization}
)

# Ambil nilai n_users dan n_places dari model yang sudah dilatih
n_users = model.n_users
n_places = model.n_places

print(f"✅ Model berhasil dimuat. Jumlah user: {n_users}, tempat: {n_places}")

# Mapping ID tempat ke nama (pastikan sesuai dengan dataset asli)
place_id_to_name = {
    0: "Karaoke",
    1: "Aula",
    2: "Lahan Barat",
    3: "Lahan Timur",
    4: "Pondok Teduh",
    5: "Kano"
}

# Fungsi untuk rekomendasi global
def rekomendasi_terbanyak_global():
    rekomendasi_semua_user = []

    for user_id in range(n_users):
        candidates = list(range(n_places))
        input_data = np.array([[user_id, place_id] for place_id in candidates])
        predictions = model.predict(input_data, verbose=0).reshape(-1)

        top_indices = predictions.argsort()[-3:][::-1]  # Ambil top 3
        top_places = [candidates[i] for i in top_indices]
        rekomendasi_semua_user.extend(top_places)

    # Hitung frekuensi rekomendasi
    counter = Counter(rekomendasi_semua_user)
    top_final = counter.most_common(3)

    output = "3 Tempat yang Paling Sering Direkomendasikan:\n"
    for i, (place_id, count) in enumerate(top_final, 1):
        nama = place_id_to_name.get(place_id, f"Tempat ID {place_id}")
        output += f"{i}. {nama} - {count} kali direkomendasikan\n"

    return output.strip()

# Tampilkan di Gradio
demo = gr.Interface(
    fn=rekomendasi_terbanyak_global,
    inputs=[],
    outputs="text",
    title="Top 3 Tempat Terbaik - Cacalan Go",
    description="Data hasil rekomendasi otomatis dari model untuk seluruh user."
)

demo.launch()