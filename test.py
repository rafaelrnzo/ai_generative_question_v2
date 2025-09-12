from openai import OpenAI

# vLLM lokal server
client = OpenAI(
    base_url="http://192.168.100.136:8006/v1",
    api_key="EMPTY"  # vLLM butuh argumen ini, tapi nilainya bebas
)

# Ganti dengan nama model embed yang memang Anda load di vLLM
# contoh: "nomic-embed-text", "bge-base-en-v1.5"
model = "nomic-embed-text"

responses = client.embeddings.create(
    input=[
        "Hello my name is",
        "The best thing about vLLM is that it supports many different models"
    ],
    model=model,
)

for data in responses.data:
    print(len(data.embedding), data.embedding[:10])  # tampilkan panjang + 10 angka awal
