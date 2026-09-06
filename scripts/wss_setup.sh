#!/usr/bin/env bash
# wss(TLS) 준비 자동화: self-signed 인증서를 만들고, 제출 페이지에 등록할 값을 출력합니다.
#
#   ./scripts/wss_setup.sh <공인 IP 또는 도메인> [출력 디렉토리]
#   예) ./scripts/wss_setup.sh 203.0.113.7
#
# 출력되는 "인증서 지문"을 제출 페이지의 지문 칸에 붙여넣으면, 평가 서버는 그 지문과
# 일치하는 인증서만 신뢰합니다(pinning) — 중간자가 끼어들면 지문 불일치로 거부됩니다.
# 도메인 + 정식 인증서(Let's Encrypt 등)를 쓰는 팀은 이 스크립트가 필요 없습니다.
# 공유 망(연구실 공용 LAN, 기숙사, 카페 Wi-Fi)에서 서버를 돌린다면 wss를 꼭 쓰세요 —
# ws://는 같은 망의 타인이 제출 토큰을 볼 수 있습니다.
set -euo pipefail

HOST="${1:?사용법: ./scripts/wss_setup.sh <공인 IP 또는 도메인> [출력 디렉토리(기본 .)]}"
DIR="${2:-.}"
mkdir -p "$DIR"
CERT="$DIR/cert.pem"
KEY="$DIR/key.pem"

if [ -e "$CERT" ] || [ -e "$KEY" ]; then
    echo "중단: $CERT 또는 $KEY 가 이미 있습니다. 재발급하려면 지우고 다시 실행하세요." >&2
    echo "(주의: 인증서를 새로 만들면 지문이 바뀌므로 제출 페이지의 지문도 갱신해야 합니다.)" >&2
    exit 1
fi

# IP면 IP SAN, 도메인이면 DNS SAN — 지문 pinning에는 영향 없지만 표준 도구 호환용.
if [[ "$HOST" =~ ^[0-9.]+$ ]]; then SAN="IP:$HOST"; else SAN="DNS:$HOST"; fi
openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
    -subj "/CN=$HOST" -addext "subjectAltName=$SAN" \
    -keyout "$KEY" -out "$CERT" 2>/dev/null
chmod 600 "$KEY"
FP=$(openssl x509 -in "$CERT" -noout -fingerprint -sha256 | cut -d= -f2)

cat <<EOF
생성 완료
  인증서:  $CERT
  개인키:  $KEY   (절대 공유·커밋 금지 — .gitignore에 *.pem 포함됨)

제출 페이지에 등록할 값
  서버 주소:     wss://$HOST:<포트>
  인증서 지문:   $FP

서버 실행 (지문 등록 후)
  evalhost-demo --port 8000 --token "\$(cat token.txt)" \\
      --certfile $CERT --keyfile $KEY
EOF
