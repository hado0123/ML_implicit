# cd ML_implicit\API
# uvicorn app:app --reload
# http://localhost:8000/recommend?user_id=1
# http://127.0.0.1:8000/docs 


import os
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix, csr_matrix
from implicit.als import AlternatingLeastSquares
from sqlalchemy import create_engine


# .env 파일 로드
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# SQLAlchemy로 MySQL 연결
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# 쿼리 실행해서 pandas로 불러오기
query = """
SELECT 
    o.userId AS user_id,
    oi.itemId AS item_id,
    CAST(SUM(oi.count) AS UNSIGNED) AS purchase_count
FROM orders o, orderitems oi
where o.id = oi.orderId
group by oi.itemId, o.userId
order by o.userId, oi.itemId;
"""
data = pd.read_sql(query, engine)
print(data)

app = FastAPI()

# 허용할 origin 설정
origins = [
    os.getenv("FRONTEND_APP_URL"),
    # 필요하다면 도메인 추가
    # "https://내도메인.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # 허용할 origin
    allow_credentials=True,         # 쿠키 인증 허용 여부
    allow_methods=["*"],            # 허용할 HTTP 메서드 (GET, POST 등)
    allow_headers=["*"],            # 허용할 HTTP 헤더
)


# =======================
# 데이터 및 모델 준비
# =======================

# data = pd.DataFrame({
#     'user_id': [0, 1, 1, 2, 3, 3],
#     'item_id': [101, 101, 102, 103, 102, 104],
#     'purchase_count': [1, 1, 2, 1, 1, 1]
# })


len_user = len(data['user_id'])
len_item = len(data['item_id'])
len_count = len(data['purchase_count'])


# 인코딩
user_enc = LabelEncoder()
item_enc = LabelEncoder()
data['user_idx'] = user_enc.fit_transform(data['user_id'])
data['item_idx'] = item_enc.fit_transform(data['item_id'])

# 희소행렬
matrix = coo_matrix(
    (data['purchase_count'], (data['user_idx'], data['item_idx']))
)

user_item_matrix = matrix.tocsr() # 사용자 X 아이템
print("user_item_matrix shape:", user_item_matrix.shape) 
print(user_item_matrix.toarray())

# ALS 모델 학습
model = AlternatingLeastSquares(factors=10, iterations=15)
model.fit(user_item_matrix) 

# print("item_factors shape:", model.item_factors.shape)
# print("user_factors shape:", model.user_factors.shape)


# =======================
# API 엔드포인트
# =======================

# Query(..., ) ...은 필수 입력값을 의미 or 디폴트값을 가져올 수 있음. 뒤에는 해당 파라미터에 대한 설명을 붙임
# limit: int = Query(10, description="가져올 데이터 개수") 기본값 지정 방법
@app.get("/recommend")
def recommend(user_id: int = Query(..., description="원본 user_id 입력 (예: 3)"), top_n: int = 3):
    # 유저가 존재하지 않는 경우
    if user_id not in data['user_id'].values:
        raise HTTPException(status_code=404, detail="해당 user_id는 데이터에 없습니다.")

    # user_idx 변환
    user_idx = user_enc.transform([user_id])[0]
    # print('user_idx: ', user_idx)

    # 유저 벡터 추출
    user_vector = csr_matrix(user_item_matrix[user_idx])
    # print('user_vector:', user_vector.toarray())

    # indices만 출력
    # print("indices:", user_vector.indices)

    # 추천
    item_indices, scores = model.recommend(
        userid=user_idx, 
        user_items=user_vector,
        N=top_n)
    item_ids = item_enc.inverse_transform(item_indices)

    # 해당 인덱스에 대응하는 실제 item_id 보기
    # print("item_ids:", item_enc.inverse_transform(user_vector.indices))
    print("item_ids: ", item_ids)

    # 결과 반환
    result = [
        {"id": int(item_id), "score": float(score)}
        for item_id, score in zip(item_ids, scores)
    ]
    return result