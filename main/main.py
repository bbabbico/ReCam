# -*- coding: utf-8 -*-
"""
카메라 재배치 시각화 지도 - FastAPI 서버
카카오 맵 API를 활용한 우선순위별 카메라 설치 지점 + 과잉 설치 지점 시각화
"""
import os
import sys
import math
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="카메라 재배치 시각화 지도")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 데이터 로드 및 전처리 ──────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
OUTPUT_DIR = os.path.join(PARENT_DIR, "output_v2")

def load_data():
    """결과_전체군집화데이터.csv를 로드하고 우선순위를 계산"""
    csv_path = os.path.join(OUTPUT_DIR, "결과_전체군집화데이터.csv")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # 위험점수 계산 (EPDO 기반)
    df["위험점수"] = df["총사고건수"] + df["총사망자수"] * 10 + df["총중상자수"] * 5

    # NaN 좌표 제거
    df = df.dropna(subset=["위도", "경도"])
    
    # JSON 직렬화를 위한 NaN 처리
    df = df.fillna({
        "시도명": "",
        "사고위험지역명": "",
        "사고분석유형명": "",
        "군집라벨": "",
        "총사고건수": 0,
        "총사망자수": 0,
        "총중상자수": 0,
        "반경내카메라수": 0,
        "위험점수": 0
    })

    # 카테고리 분류
    results = []

    # 1) 과잉 구역 → 과잉세분화 컬럼 기반 3분류
    surplus = df[df["군집라벨"].str.contains("카메라 설치지점", na=False)].copy()
    if "과잉세분화" in surplus.columns:
        # 과잉 (재배치 1순위 공급원)
        true_ex = surplus[surplus["과잉세분화"].str.contains("과잉", na=False)].copy()
        true_ex["category"] = "과잉"
        true_ex["priority"] = "과잉 (재배치 1순위 공급원)"
        results.append(true_ex)

        # 재배치 가능 (재배치 2순위 공급원)
        realloc = surplus[surplus["과잉세분화"].str.contains("재배치가능", na=False)].copy()
        realloc["category"] = "재배치가능"
        realloc["priority"] = "재배치 가능 (재배치 2순위 공급원)"
        results.append(realloc)

        # 적정 (현행 유지)
        proper = surplus[surplus["과잉세분화"].str.contains("적정", na=False)].copy()
        proper["category"] = "적정"
        proper["priority"] = "적정 (현행 유지 권고)"
        results.append(proper)
    else:
        # 과잉세분화 컬럼이 없으면 기존 방식
        surplus["category"] = "과잉"
        surplus["priority"] = "과잉 (카메라 재배치 대상)"
        results.append(surplus)

    # 2) 카메라 미설치(0대) 지점만 재배치 대상
    no_cam = df[df["반경내카메라수"] == 0].copy()

    # 1순위: 사망위험 + 사고다발
    p1 = no_cam[no_cam["군집라벨"].str.contains("사망위험|사고다발", na=False)].copy()
    p1["category"] = "1순위"
    p1["priority"] = "1순위 (사망위험+사고다발)"
    results.append(p1)

    # 부족 군집 내 세분화
    deficit = no_cam[no_cam["군집라벨"].str.contains("부족", na=False)].sort_values(
        "위험점수", ascending=False
    )
    top_10_idx = int(len(deficit) * 0.1)

    # 2순위: 부족 상위 10%
    p2 = deficit.iloc[:top_10_idx].copy()
    p2["category"] = "2순위"
    p2["priority"] = "2순위 (부족 상위 10%)"
    results.append(p2)

    # 3순위: 부족 나머지
    p3 = deficit.iloc[top_10_idx:].copy()
    p3["category"] = "3순위"
    p3["priority"] = "3순위 (부족 나머지)"
    results.append(p3)

    return pd.concat(results, ignore_index=True)


# 서버 시작 시 데이터 로드
print("데이터 로딩 중...")
DF = load_data()
print(f"총 {len(DF):,}건 로드 완료")
CATEGORIES = ["과잉", "재배치가능", "적정", "1순위", "2순위", "3순위"]
for cat in CATEGORIES:
    count = len(DF[DF["category"] == cat])
    if count > 0:
        print(f"  {cat}: {count:,}건")


# ── API 엔드포인트 ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """index.html 제공"""
    html_path = os.path.join(BASE_DIR, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/markers")
async def get_markers(
    category: str = None,
    sido: str = None,
    min_lat: float = None,
    max_lat: float = None,
    min_lng: float = None,
    max_lng: float = None,
    limit: int = 3000,
):
    """
    지도 마커 데이터를 JSON으로 반환.
    - category: '과잉', '1순위', '2순위', '3순위' 필터
    - sido: 시도명 필터
    - min_lat/max_lat/min_lng/max_lng: 바운드 필터 (뷰포트 기반)
    - limit: 최대 반환 수
    """
    df = DF.copy()

    # 카테고리 필터
    if category:
        cats = category.split(",")
        df = df[df["category"].isin(cats)]

    # 시도 필터
    if sido and sido != "전체":
        df = df[df["시도명"] == sido]

    # 바운드 필터
    if all(v is not None for v in [min_lat, max_lat, min_lng, max_lng]):
        df = df[
            (df["위도"] >= min_lat)
            & (df["위도"] <= max_lat)
            & (df["경도"] >= min_lng)
            & (df["경도"] <= max_lng)
        ]

    # 위험점수 순 정렬 후 limit
    df = df.sort_values("위험점수", ascending=False).head(limit)

    markers = []
    for _, row in df.iterrows():
        lat = row["위도"]
        lng = row["경도"]
        if math.isnan(lat) or math.isnan(lng):
            continue
        markers.append(
            {
                "lat": round(lat, 6),
                "lng": round(lng, 6),
                "category": row["category"],
                "priority": row["priority"],
                "name": row.get("사고위험지역명", ""),
                "sido": row.get("시도명", ""),
                "accidents": int(row.get("총사고건수", 0)),
                "deaths": int(row.get("총사망자수", 0)),
                "serious": int(row.get("총중상자수", 0)),
                "cameras": int(row.get("반경내카메라수", 0)),
                "riskScore": int(row.get("위험점수", 0)),
                "cluster": row.get("군집라벨", ""),
                "accidentType": row.get("사고분석유형명", ""),
            }
        )

    return JSONResponse(content={"total": len(markers), "markers": markers})


@app.get("/api/stats")
async def get_stats():
    """전체 통계 요약 반환"""
    stats = {}
    for cat in CATEGORIES:
        subset = DF[DF["category"] == cat]
        if len(subset) == 0:
            continue
        stats[cat] = {
            "count": int(len(subset)),
            "avgAccidents": round(float(subset["총사고건수"].mean()), 1),
            "avgDeaths": round(float(subset["총사망자수"].mean()), 2),
            "avgSerious": round(float(subset["총중상자수"].mean()), 1),
            "avgCameras": round(float(subset["반경내카메라수"].mean()), 1),
            "avgRisk": round(float(subset["위험점수"].mean()), 1),
        }

    # 시도 목록
    sido_list = sorted(DF["시도명"].dropna().unique().tolist())

    return JSONResponse(
        content={"stats": stats, "sidoList": sido_list, "totalPoints": int(len(DF))}
    )


# ── 실행 ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000, reload=False)
