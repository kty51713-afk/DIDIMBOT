import os
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DB_PATH = "faiss_index"
CSV_PATHS = [
    "./data/국민연금공단_장애심사 상병별 심사결과 현황_20241231.csv",
    "./data/한국산업인력공단_국가직무능력표준 정보_20251231.csv",
    "./data/장애인_일자리_시설_RAG최적화_CP949.csv",
]


embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

def read_csv_with_fallback(path: str) -> pd.DataFrame:
    for enc in ["cp949", "euc-kr", "utf-8", "utf-8-sig"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"CSV 인코딩을 읽을 수 없습니다: {path}")

def get_vectorstore():
    if os.path.exists(DB_PATH):
        print("기존 FAISS 인덱스 로드")
        vectorstore = FAISS.load_local(
            DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("로드 완료! 문서 수:", vectorstore.index.ntotal)
        return vectorstore

    print("새 FAISS 인덱스 생성")
    docs = []

    for path in CSV_PATHS:
        df = read_csv_with_fallback(path)
        df = df.fillna("")
        print(f"로드 완료: {path} / 행수: {len(df)}")

        for _, row in df.iterrows():
            content = " ".join(
                [str(v).strip() for v in row.values if str(v).strip()]
            )
            if content:
                docs.append(
                    Document(
                        page_content=content,
                        metadata={"source": os.path.basename(path)}
                    )
                )

    print("문서 수:", len(docs))

    if not docs:
        raise ValueError("문서가 비어 있습니다. CSV 경로와 내용을 확인하세요.")

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(DB_PATH)
    return vectorstore

vectorstore = get_vectorstore()