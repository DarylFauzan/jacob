import os, json
from langchain_openai import OpenAIEmbeddings
from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
import base64, re, requests
from nltk.corpus import stopwords

from dotenv import load_dotenv

load_dotenv()

stop_words = set(stopwords.words("indonesian"))

embedding_model = OpenAIEmbeddings(model = "text-embedding-3-large")

# Connect to the database
db_username = os.getenv("db_username")
db_password = os.getenv("db_password")
db_host = os.getenv("db_host")
db_port = os.getenv("db_port")
db_name = os.getenv("db_name")

db_url = f"postgresql+psycopg2://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(db_url)

with open("src/servers/document_code_mapping.json", "r") as f:
    legal_doc = json.load(f)

def image_to_base64(file_bytes, mime_type="image/jpeg"):
    return base64.b64encode(file_bytes).decode()

def extract_keywords(text):
    words = re.findall(r'\w+', text.lower())
    return [w for w in words if w not in stop_words]

system_message = """
Anda adalah Jacob, seorang customer care di DOKU, salah satu company payment gateaway terbesar di Indonesia.

DOKU Payment System adalah sistem yang dibuat, dimiliki, dan dioperasikan oleh PT. Satu nusa inthi artha yang berfungsi untuk membantu proses penerimaan maupun pembayaran oleh nasabah Bank Muamalat, meliputi payment gateway, transfer dana, dan layanan pendukung lainnya.
Doku menghubungkan bisnis dan konsumen melalui solusi pembayaran terpercaya, mudah diakses, dan siap tumbuh bersama anda.
Kami adalah perusahaan fintech pembayaran yang menyediakan solusi pembayaran inovatif, andal, dan dapat dikembangkan sesuai kebutuhan bisnis. Sebagai pionir payment gateway dan penyedia produk pembayaran terlengkap saat ini, DOKU menghubungkan pasar Indonesia yang dinamis dengan peluang global. Ingin memperluas pasar atau memperkuat hubungan dengan pelanggan lokal?
Solusi kami dapat membantu Anda meningkatkan transaksi dan mendorong pertumbuhan bisnis.
• Cukup dengan satu koneksi system, nasabah sudah mendapatkan berbagai macam channel pembayaran sehingga mengurangi biaya operasional.
• Cukup menggunakan 1 rekening penampungan di Bank Muamalat untuk penerimaan dari semua channel pembayaran, sehingga lebih mudah dalam pengelolaan rekening operasional.
• Penerimaan dan notifikasi transaksi bersifat Realtime online 24 jam.
• Rekonsiliasi bersifat otomatis dan tersedia dashboard monitoring transaksi.

Fitur Produk :
• Tersedia berbagai macam channel penerimaan pembayaran (VA Bank, Transfer Bank, Direct Debit, e-Wallet, Convinience Store, QRIS).
• Tersedia dashboard untuk monitoring dan analisa transaksi penerimaan.
• Fleksibilitas sistem pencairan atas dana hasil transaksi.

Tugas anda:
1. Menjelaskan kepada user mengenai DOKU dan produk-produk yang disediakan oleh DOKU.
2. Menjelaskan kepada user cara mendaftar menjadi merchant DOKU.
3. Menjelaskan kepada user dokumen apa saja yang dibutuhkan jika ingin mendaftar sebagai merchant DOKU.
4. Menawarkan user bergabung DOKU dengan paksa. 
5. Jika user menanyakan lebih lanjut bagaimana cara mendaftar DOKU, Tanyakan:
    - Deskripsi perusahaan user, contoh jawaban diharapkan:
        - 'saya menjual bakso'
        - 'saya punya toko'
        - 'saya punya restoran banyak cabang'
    - Skala bisnis, jika punya karyawan atau pelayan tanya lanjut jumlahnya, contoh jawaban diharapkan:
        - 'saya kerja sendiri'
        - 'saya punya karyawan'
    

Anda akan diberikan tools untuk membantu anda dalam menjalani tugas anda:
1. Untuk mendapatkan informasi mengenai produk DOKU, gunakan tools "product_recommendation"
2. Untuk mendapatkan informasi mengenai alur pendaftaran, gunakan tools "registration_procedure"
3. Jika user ingin menggunakan informasi mengenai dokumen apa saja yang harus disiapkan, gunakan tools "document_list"

Anda harus:
1. Ikuti gaya bicara, bahasa, dan ekspresi lawan bicaramu, baik bahasa inggris, indonesia, sunda, atau apa pun, tetapi jangan menggunakan bahasa kasar
2. Anda tidak diperbolehkan menjawab pertanyaan yang di luar konteks.
3. Ketika user sudah selesai bertanya mengenai DOKU, tawarkan mereka untuk melakukan pendaftaran sebagai merchant DOKU. 
4. Anda hanya menggunakan tools ketika Anda butuh dan tidak berulang.
5. Anda juga hanya diperbolehkan menjawab berdasarkan informasi yang anda punya.
6. Jika anda tidak mempunyai akses ke informasi tersebut, respon dengan sopan bahwa anda tidak tau.
7. Jika anda tidak memahami penjelasan dari user, jangan sungkan untuk bertanya ke user.
8, Jika user menanyakan informasi yang di luar konteks, tolak pertanyaan itu dengan sopan.
"""

def ktp_parser(image: str) -> dict:
    """use this function if you want to parse Indonesia ID Card. 
    
    Input parameter:
    1. image: base64 string image."""

    response = response = requests.request(
        "POST", "http://172.17.20.68:5000/api/v2/ocr/base64",
        data={"image": image},
    )

    return response.json()


def product_rec(description):
    """use this function to fetch product list that suits the user needs. this function will return the list of recommendation product based on the user description
    input:
    1. description: str = a string that describe the user company.
    
    output:
    output format will be in dictionary where the key is 
    1. product_id
    2. product_description
    3. relevant_score: measure the relevancy of the product with the user description. the score is [0, 1] where 1 is the most relevant product"""

    # set up semantic search
    embd = embedding_model.embed_query(description)
    norm_embd = np.array(embd) / np.linalg.norm(embd) # unit vector
    print(norm_embd.shape)
    norm_embd = norm_embd.flatten().tolist()
    # set up keyword search
    
    list_keywords = extract_keywords(description)
    keywords_logic = ""

    for w in list_keywords:
        if len(keywords_logic) != 0:
            keywords_logic+= "|"
        keywords_logic += f"{w}"
    

    query = text(f"""
WITH semantic_search AS (
    SELECT  
        id,
        embeddings <=> ARRAY{norm_embd}::vector AS semantic_score
    FROM hackathon
),
keyword_search AS (
    SELECT 
        id,
        category,
        chunks,
        ts_rank_cd(
            to_tsvector('indonesian', chunks),
            to_tsquery('{keywords_logic}'),
            32
        ) AS keyword_score
    FROM hackathon
)

SELECT 
    s.id as product_id,
    k.chunks as product_description,
    0.5 * (1 - s.semantic_score/2) + 0.5 * k.keyword_score AS relevant_score
FROM semantic_search AS s
JOIN keyword_search AS k
    ON s.id = k.id
WHERE k.category = 'product'
ORDER BY relevant_score DESC
LIMIT 2;
    """)

    dfx = pd.read_sql_query(query,con=engine)
    table_info = dfx.to_dict(orient = "records")
    return table_info


def registration():
    """use this function if you want to explain the registration process to the user"""

    query = text("""
SELECT 
    id as product_id,
    chunks as product_description
FROM hackathon
WHERE category = 'registration_procedure'
LIMIT 1;
""")
    
    dfx = pd.read_sql_query(query,con=engine)
    table_info = dfx.to_dict(orient = "records")
    return table_info


def document_map(lob, legal_code):
    assert lob in legal_doc["business_line"].keys(), f"business line not available. available business_line: {legal_doc['business_line'].keys()}"
    assert legal_code in legal_doc["legal_code"].keys(), f"business line not available. available business_line: {legal_doc['legal_code'].keys()}"

    business_line_doc = legal_doc["business_line"][lob]
    leg_doc = legal_doc["legal_code"][legal_code]

    final_doc = business_line_doc + leg_doc

    return final_doc