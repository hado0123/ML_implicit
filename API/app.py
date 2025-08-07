# main.py
from fastapi import FastAPI, Query, HTTPException
from typing import List
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix, csr_matrix
from implicit.als import AlternatingLeastSquares
from sqlalchemy import create_engine

# DB 연결 정보
DB_USER = 'root'
DB_PASSWORD = '1234'
DB_HOST = '127.0.0.1'  # 또는 EC2 퍼블릭 IP
DB_PORT = '3306'
DB_NAME = 'shopmax'

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

# =======================
# 데이터 및 모델 준비
# =======================
# data = pd.DataFrame({
#     'user_id': [0, 1, 1, 2, 3, 3],
#     'item_id': [101, 101, 102, 103, 102, 104],
#     'purchase_count': [1, 1, 2, 1, 1, 1]
# })

# 인코딩
user_enc = LabelEncoder()
item_enc = LabelEncoder()
data['user_idx'] = user_enc.fit_transform(data['user_id'])
data['item_idx'] = item_enc.fit_transform(data['item_id'])

# 희소행렬
matrix = coo_matrix(
    (data['purchase_count'], (data['user_idx'], data['item_idx']))
)

print(matrix.toarray()) 

user_item_matrix = matrix.tocsr()

# ALS 모델 학습
model = AlternatingLeastSquares(factors=10, iterations=15)
model.fit(matrix.T.tocsr())

# =======================
# API 엔드포인트
# =======================

@app.get("/recommend")
def recommend(user_id: int = Query(..., description="원본 user_id 입력 (예: 3)"), top_n: int = 3):
    # 유저가 존재하지 않는 경우
    if user_id not in data['user_id'].values:
        raise HTTPException(status_code=404, detail="해당 user_id는 데이터에 없습니다.")

    # user_idx 변환
    user_idx = user_enc.transform([user_id])[0]

    # 유저 벡터 추출
    user_vector = csr_matrix(user_item_matrix[user_idx])

    # 추천
    item_indices, scores = model.recommend(userid=user_idx, user_items=user_vector, N=top_n)
    item_ids = item_enc.inverse_transform(item_indices)

    # 결과 반환
    result = [
        {"item_id": int(item_id), "score": float(score)}
        for item_id, score in zip(item_ids, scores)
    ]
    return result
