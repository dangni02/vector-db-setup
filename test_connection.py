"""
Vector DB 연결 및 검색 테스트
"""
import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np

def connect_db():
    """데이터베이스 연결"""
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="postgres",
            user="postgres",
            password="mypassword"
        )
        register_vector(conn)
        return conn
    except Exception as e:
        print(f"❌ 연결 실패: {e}")
        return None

def test_connection():
    """기본 연결 테스트"""
    print("🔌 데이터베이스 연결 테스트...\n")
    
    conn = connect_db()
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        
        # 테이블 존재 확인
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'fever_documents'
            );
        """)
        exists = cur.fetchone()[0]
        
        if not exists:
            print("❌ fever_documents 테이블이 없습니다!")
            print("   데이터베이스 복원이 제대로 안된 것 같습니다.")
            return False
        
        # 문서 수 확인
        cur.execute("SELECT COUNT(*) FROM fever_documents;")
        count = cur.fetchone()[0]
        print(f"✅ 총 문서 수: {count:,}\n")
        
        # 샘플 데이터 확인
        cur.execute("""
            SELECT pk_id, id, chunk_id, content 
            FROM fever_documents 
            LIMIT 3;
        """)
        
        print("📄 샘플 데이터:")
        for row in cur.fetchall():
            content = row[3][:80] if row[3] else "(empty)"
            print(f"  ID: {row[1]} | {content}...")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ 테스트 중 에러: {e}")
        conn.close()
        return False

def test_vector_search():
    """벡터 검색 테스트"""
    print("\n🔍 벡터 유사도 검색 테스트...\n")
    
    conn = connect_db()
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        
        # 랜덤 쿼리 벡터 (768차원)
        query_vector = np.random.rand(768).tolist()
        
        # L2 distance로 유사 문서 검색
        cur.execute("""
            SELECT 
                id,
                content,
                embedding <-> %s::vector AS distance
            FROM fever_documents
            ORDER BY distance
            LIMIT 5;
        """, (query_vector,))
        
        results = cur.fetchall()
        
        if not results:
            print("❌ 검색 결과가 없습니다.")
            return False
        
        print("상위 5개 유사 문서:")
        for idx, (doc_id, content, distance) in enumerate(results, 1):
            content_preview = content[:100] if content else "(empty)"
            print(f"\n{idx}. ID: {doc_id}")
            print(f"   내용: {content_preview}...")
            print(f"   거리: {distance:.4f}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ 벡터 검색 중 에러: {e}")
        conn.close()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Vector DB 연결 테스트")
    print("=" * 60 + "\n")
    
    if test_connection():
        if test_vector_search():
            print("\n" + "=" * 60)
            print("✅ 모든 테스트 통과!")
            print("=" * 60)
        else:
            print("\n❌ 벡터 검색 테스트 실패")
    else:
        print("\n❌ 연결 테스트 실패")
        print("\n💡 해결 방법:")
        print("   1. 컨테이너 실행 확인: docker ps")
        print("   2. 로그 확인: docker logs vector_db")
        print("   3. 재시작: docker restart vector_db")