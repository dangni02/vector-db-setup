"""
FEVER Multi-Agent System with Real Vector DB
filtered_train.jsonl에서 claim을 읽고 Vector DB에서 근거를 검색하여 답변 생성
"""
import os
import json
import random
from dotenv import load_dotenv
from openai import OpenAI
from crewai import Agent, Task, Crew, Process
import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print(f"API Key loaded: {os.getenv('OPENAI_API_KEY')[:10]}...")


print("Connecting to DB...")
# ---------------------------
# Vector DB Connection
# ---------------------------
def connect_db():
    """Vector DB 연결"""
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="postgres",
        user="postgres",
        password="mypassword"
    )
    register_vector(conn)
    print("✅ DB Connected!")
    return conn

'''
def get_embedding(text, model="text-embedding-3-small"):
    """OpenAI로 텍스트 임베딩 생성"""
    response = client.embeddings.create(
        input=text,
        model=model,
        dimensions=768
    )
    return response.data[0].embedding
'''
    
    # sentence-transformers 추가
from sentence_transformers import SentenceTransformer

# 전역 변수로 모델 로드 (한 번만)
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def get_embedding(text):
    """sentence-transformers로 임베딩 생성 (768차원)"""
    return embedding_model.encode(text).tolist()

def retrieve_from_vectordb(query, k=5):
    """Vector DB에서 유사 문서 검색"""
    conn = connect_db()
    cur = conn.cursor()
    
    # 쿼리 임베딩 생성
    query_embedding = get_embedding(query)
    
    # Vector 검색 (L2 distance)
    cur.execute("""
        SELECT 
            id,
            content,
            embedding <-> %s::vector AS distance
        FROM fever_documents
        ORDER BY distance
        LIMIT %s;
    """, (query_embedding, k))
    
    results = cur.fetchall()
    conn.close()
    
    # 문서 내용만 반환
    docs = [row[1] for row in results if row[1]]
    return docs

# ---------------------------
# Agents 정의
# ---------------------------
retriever_agent = Agent(
    role="Retriever",
    goal="질문에 가장 관련 있는 문서를 Vector DB에서 검색해 제공한다.",
    backstory="pgvector를 통해 semantic search를 수행하는 검색 전문가",
    verbose=True,
    allow_delegation=False
)

answerer_agent = Agent(
    role="Answerer",
    goal="근거를 바탕으로 claim의 진위를 판단하고 초안을 생성한다.",
    backstory="FEVER 데이터셋의 fact verification 전문가. SUPPORTS/REFUTES/NOT ENOUGH INFO를 판단",
    verbose=True,
    allow_delegation=False
)

judge_agent = Agent(
    role="Judge",
    goal="여러 초안을 비교해 가장 사실성 있는 답변을 선택한다.",
    backstory="사실 검증과 일관성 평가 전문가. 근거와의 정합성을 중시",
    verbose=True,
    allow_delegation=False
)

editor_agent = Agent(
    role="Editor",
    goal="최종 답변을 명확하고 논리적인 형식으로 다듬는다.",
    backstory="학술 논문 수준의 편집 기준을 맞추는 전문가",
    verbose=True,
    allow_delegation=False
)

# ---------------------------
# Task 함수들
# ---------------------------
def retriever_task_fn(claim, k=5):
    """Vector DB에서 관련 문서 검색"""
    print(f"\n🔍 Retrieving documents for: {claim}")
    docs = retrieve_from_vectordb(claim, k=k)
    context = "\n\n".join([f"[Doc {i+1}] {doc}" for i, doc in enumerate(docs)])
    print(f"✅ Retrieved {len(docs)} documents")
    return {"context": context, "docs": docs}

def answerer_task_fn(claim, context, style="balanced", temp=0.3):
    """근거를 바탕으로 답변 생성"""
    prompt = f"""
You are a fact-checking expert for the FEVER dataset.

Claim: {claim}

Evidence from database:
{context}

Task: Determine if the claim is:
- SUPPORTS (evidence supports the claim)
- REFUTES (evidence contradicts the claim)  
- NOT ENOUGH INFO (insufficient evidence)

Provide:
1. Label: [SUPPORTS/REFUTES/NOT ENOUGH INFO]
2. Reasoning: Brief explanation
3. Key evidence: Which documents are most relevant

Style: {style}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=temp,
        messages=[
            {"role": "system", "content": "You are a FEVER fact-checking expert."},
            {"role": "user", "content": prompt}
        ]
    )
    return resp.choices[0].message.content

def judge_task_fn(answers, docs, mode="llm"):
    """여러 답변 중 최선 선택"""
    context = "\n\n".join([f"[Doc {i+1}] {doc}" for i, doc in enumerate(docs)])
    
    if mode == "voting":
        # 간단한 투표 방식
        labels = []
        for ans in answers:
            if "SUPPORTS" in ans.upper():
                labels.append("SUPPORTS")
            elif "REFUTES" in ans.upper():
                labels.append("REFUTES")
            else:
                labels.append("NOT ENOUGH INFO")
        return max(set(labels), key=labels.count)
    else:
        # LLM 판단
        prompt = f"""
We have multiple draft fact-check results:

{json.dumps(answers, ensure_ascii=False, indent=2)}

Original evidence:
{context}

Task:
1. Choose the most accurate verdict (SUPPORTS/REFUTES/NOT ENOUGH INFO)
2. Provide confidence score (0-1)
3. Explain which draft is most reliable and why
"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are an impartial fact-checking judge."},
                {"role": "user", "content": prompt}
            ]
        )
        return resp.choices[0].message.content

def editor_task_fn(text, strength="light"):
    """최종 편집"""
    if strength == "light":
        prompt = f"Polish grammar and clarity, keep original reasoning:\n\n{text}"
    else:
        prompt = f"Rewrite in clear academic style with structured reasoning:\n\n{text}"
    
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        messages=[
            {"role": "system", "content": "You are an academic editor."},
            {"role": "user", "content": prompt}
        ]
    )
    return resp.choices[0].message.content

# ---------------------------
# Pipeline
# ---------------------------
def run_fever_pipeline(claim, k=5, judge_mode="llm", editor_strength="strong"):
    """FEVER claim에 대한 전체 파이프라인 실행"""
    print(f"\n{'='*60}")
    print(f"Processing Claim: {claim}")
    print(f"{'='*60}")
    
    # (1) Retriever: Vector DB에서 검색
    retr = retriever_task_fn(claim, k)
    context, docs = retr["context"], retr["docs"]

    # (2) Answerer: 여러 스타일로 답변 생성
    print("\n📝 Generating answers with different styles...")
    answers = []
    styles = ["precise", "balanced", "creative"]
    for style in styles:
        print(f"  - Style: {style}")
        ans = answerer_task_fn(claim, context, style=style, temp=0.7)
        answers.append(ans)

    # (3) Judge: 최선의 답변 선택
    print("\n⚖️ Judging answers...")
    judged = judge_task_fn(answers, docs, mode=judge_mode)

    # (4) Editor: 최종 편집
    print("\n✍️ Editing final answer...")
    final = editor_task_fn(judged, strength=editor_strength)

    return {
        "claim": claim,
        "docs": docs,
        "answers": answers,
        "judged": judged,
        "final": final
    }

# ---------------------------
# JSONL 파일 처리
# ---------------------------
def load_claims_from_jsonl(filepath, limit=5):
    """filtered_train.jsonl에서 claim 로드"""
    claims = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            data = json.loads(line)
            claims.append({
                "id": data.get("id"),
                "claim": data.get("claim"),
                "label": data.get("label")  # 정답 레이블
            })
    return claims

def evaluate_predictions(results):
    """예측 결과 평가"""
    correct = 0
    total = len(results)
    
    for res in results:
        predicted = res["final"].upper()
        ground_truth = res["label"].upper()
        
        # 간단한 매칭 (실제로는 더 정교한 파싱 필요)
        if ground_truth in predicted:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"Evaluation Results:")
    print(f"  Total: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"{'='*60}")
    
    return accuracy

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # JSONL 파일 경로
    JSONL_PATH = r"your/path/to/filtered_train.jsonl"
    
    # 테스트할 claim 수
    NUM_CLAIMS = 3
    
    print("🚀 FEVER Multi-Agent System Starting...")
    print(f"Loading {NUM_CLAIMS} claims from {JSONL_PATH}\n")
    
    # Claim 로드
    claims = load_claims_from_jsonl(JSONL_PATH, limit=NUM_CLAIMS)
    
    # 각 claim에 대해 파이프라인 실행
    results = []
    for i, claim_data in enumerate(claims):
        print(f"\n\n{'#'*60}")
        print(f"CLAIM {i+1}/{NUM_CLAIMS}")
        print(f"Ground Truth: {claim_data['label']}")
        print(f"{'#'*60}")
        
        result = run_fever_pipeline(
            claim=claim_data["claim"],
            k=5,
            judge_mode="llm",
            editor_strength="strong"
        )
        
        result["label"] = claim_data["label"]  # 정답 추가
        results.append(result)
        
        # 결과 출력
        print(f"\n📊 RESULT for Claim {i+1}:")
        print(f"Claim: {result['claim']}")
        print(f"\nFinal Answer:\n{result['final']}")
        print(f"\nGround Truth: {claim_data['label']}")
    
    # 전체 평가
    evaluate_predictions(results)
    
    # 결과 저장
    output_file = "fever_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Results saved to {output_file}")