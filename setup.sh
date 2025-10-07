#!/bin/bash

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🚀 Vector DB 초기 세팅 시작...${NC}\n"

# ⚠️ 여기만 수정하세요! Google Drive FILE_ID
GDRIVE_FILE_ID="1Ccxpye8vr1upD4-GrWfS0szyFLUTA_uX"
BACKUP_FILE="backup.dump"

# .env 파일 로드
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo -e "${GREEN}✅ .env 파일 로드 완료${NC}\n"
fi

# gdown 설치 확인
if ! command -v gdown &> /dev/null; then
    echo -e "${BLUE}📦 gdown 설치중...${NC}"
    pip3 install gdown
fi

# 백업 파일 다운로드
if [ ! -f "$BACKUP_FILE" ]; then
    echo -e "${BLUE}📥 Google Drive에서 다운로드중... (3GB+, 시간이 걸립니다)${NC}"
    gdown "https://drive.google.com/uc?id=${GDRIVE_FILE_ID}" -O $BACKUP_FILE
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ 다운로드 실패!${NC}"
        echo -e "${BLUE}수동 다운로드: https://drive.google.com/file/d/${GDRIVE_FILE_ID}/view${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ 다운로드 완료!${NC}\n"
else
    echo -e "${GREEN}✅ 백업 파일 이미 존재 (다운로드 건너뜀)${NC}\n"
fi

FILE_SIZE=$(ls -lh $BACKUP_FILE | awk '{print $5}')
echo -e "${GREEN}   파일 크기: ${FILE_SIZE}${NC}\n"

# Docker 실행
echo -e "${BLUE}🐳 PostgreSQL + pgvector 시작...${NC}"
docker-compose down -v 2>/dev/null || true
docker-compose up -d

echo -e "${BLUE}⏳ PostgreSQL 준비중 (최대 30초)...${NC}"
RETRY_COUNT=0
MAX_RETRIES=30

until docker exec vector_db pg_isready -U postgres > /dev/null 2>&1; do
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo -e "${RED}❌ PostgreSQL 시작 실패!${NC}"
        echo -e "${BLUE}로그 확인: docker logs vector_db${NC}"
        exit 1
    fi
    echo "   대기중... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 1
done

echo -e "${GREEN}✅ PostgreSQL 준비 완료!${NC}\n"

# 백업 파일을 볼륨으로 직접 복원 (docker cp 사용 안함)
echo -e "${BLUE}📦 데이터베이스 복원중... (5-10분 소요)${NC}\n"

# 파일을 stdin으로 직접 전달
cat $BACKUP_FILE | docker exec -i vector_db pg_restore \
    -U postgres \
    -d postgres \
    --no-owner \
    --no-acl

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}🎉🎉🎉 세팅 완료! 🎉🎉🎉${NC}"
    echo -e "\n${BLUE}=== 접속 정보 ===${NC}"
    echo "Host: localhost"
    echo "Port: ${POSTGRES_PORT:-5432}"
    echo "Database: ${POSTGRES_DB:-postgres}"
    echo "User: ${POSTGRES_USER:-postgres}"
    echo "Password: ${POSTGRES_PASSWORD:-12345}"
    echo -e "\n${BLUE}테스트:${NC}"
    echo "python3 test_connection.py"
else
    echo -e "\n${RED}❌ 복원 실패!${NC}"
    echo -e "${BLUE}로그 확인: docker logs vector_db${NC}"
    exit 1
fi