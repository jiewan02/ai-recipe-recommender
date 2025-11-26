# graph_server.py (새 파일로 만들거나, 기존 app.py에 합쳐도 됨)
import math
from typing import Dict, Any, List

from neo4j import GraphDatabase
from extractor_model_old import extract_keywords  # 기존 키워드 추출 모델
import time

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "password"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))


def build_cypher_from_keywords_relaxed(kw: dict, limit: int = 50):
    where_clauses = []
    params = {}

    # 1) must_ingredients: "이 문자열이 이름에 포함되는 재료" 있으면 OK
    ing_filters = []
    if kw.get("must_ingredients"):
        for idx, ing in enumerate(kw["must_ingredients"]):
            key = f"must_ing_{idx}"
            clause = f"""
            EXISTS {{
                MATCH (r)-[:HAS_INGREDIENT]->(mi:Ingredient)
                WHERE mi.name CONTAINS ${key}
            }}
            """
            ing_filters.append(clause)
            params[key] = ing
    if ing_filters:
        where_clauses.append("(" + " AND ".join(ing_filters) + ")")

    # 2) exclude_ingredients: 이름에 포함되면 안 됨
    exc_filters = []
    if kw.get("exclude_ingredients"):
        for idx, ing in enumerate(kw["exclude_ingredients"]):
            key = f"exc_ing_{idx}"
            clause = f"""
            NOT EXISTS {{
                MATCH (r)-[:HAS_INGREDIENT]->(ei:Ingredient)
                WHERE ei.name CONTAINS ${key}
            }}
            """
            exc_filters.append(clause)
            params[key] = ing
    if exc_filters:
        where_clauses.append("(" + " AND ".join(exc_filters) + ")")

    # 3) 시간 조건
    if kw.get("max_cook_time_min"):
        where_clauses.append("r.time_min IS NOT NULL AND r.time_min <= $tmax")
        params["tmax"] = kw["max_cook_time_min"]

    where_str = ""
    if where_clauses:
        where_str = "WHERE " + " AND ".join([w.strip() for w in where_clauses])

    cypher = f"""
    MATCH (r:Recipe)
    {where_str}
    RETURN
        r.recipe_id AS recipe_id,
        r.title AS title,
        r.name AS name,
        r.views AS views,
        r.time_min AS time_min,
        r.difficulty AS difficulty,
        r.servings AS servings,
        r.image_url AS image_url
    ORDER BY r.views DESC
    LIMIT {limit}
    """

    return cypher, params


def compute_score(rec: Dict[str, Any], kw: Dict[str, Any]) -> float:
    """
    간단한 scoring 함수:
      - 조회수(log) 기반 가중치
      - 유저가 max_cook_time_min 요구했을 때 이를 만족하면 보너스
    필요하면 나중에 더 고도화하면 됨.
    """
    score = 0.0

    views = rec.get("views")
    if isinstance(views, (int, float)) and views is not None:
        score += math.log1p(max(views, 0)) / 10.0  # 조회수 많은 레시피 가볍게 우대

    # 시간 조건을 만족하면 보너스
    tmax = kw.get("max_cook_time_min")
    time_min = rec.get("time_min")
    if isinstance(tmax, (int, float)) and isinstance(time_min, (int, float)):
        if time_min <= tmax:
            score += 1.0
        else:
            score -= 0.5

    # 난이도, 기타 태그도 나중에 추가 가능
    return float(score)

def merge_value(v1, v2):
    # Case 1: 둘 다 list → union
    if isinstance(v1, list) and isinstance(v2, list):
        return list({*v1, *v2})

    # Case 2: 둘 다 dict → 재귀 병합
    if isinstance(v1, dict) and isinstance(v2, dict):
        merged = {}
        for k in v1.keys() | v2.keys():
            merged[k] = merge_value(v1.get(k), v2.get(k))
        return merged

    # Case 3: 둘 중 하나가 None이면, 다른 것을 반환
    if v1 is None:
        return v2
    if v2 is None:
        return v1

    # Case 4: 둘 다 scalar/str/bool인데 값이 다르면?
    # 일반적으로 kw → keywords 순으로 우선순위 결정
    # 필요하면 concatenate로 바꿀 수 있음.
    return v2 if v2 not in (None, "", []) else v1


def merge_dicts(kw: dict, keywords: dict) -> dict:
    merged = {}
    all_keys = set(kw.keys()) | set(keywords.keys())

    for key in all_keys:
        merged[key] = merge_value(kw.get(key), keywords.get(key))

    return merged
def graph_rag_search(user_prompt: str,keywords:dict, top_k: int = 5) -> Dict[str, Any]:
    # 1) 키워드 추출
    t0 = time.time()
    kw = extract_keywords(user_prompt)
    t1 = time.time()

    ##### kw + keyword
    kw = merge_dicts(kw, keywords)
    t2 = time.time()

    # 2) Cypher 쿼리 생성 (relaxed 버전 사용)
    cypher, params = build_cypher_from_keywords_relaxed(kw, limit=50)
    t3 = time.time()

    # 3) Neo4j에서 후보 레시피 검색
    with driver.session() as session:
        result = session.run(cypher, **params)
        rows = list(result)
    t4 = time.time()

    # 4) row들을 dict로 변환 + score 계산
    candidates: List[Dict[str, Any]] = []
    for rec in rows:
        item = {
            "recipe_id": rec["recipe_id"],
            "title": rec["title"],
            "name": rec["name"],
            "views": rec["views"],
            "time_min": rec["time_min"],
            "difficulty": rec["difficulty"],
            "servings": rec["servings"],
            "image_url": rec["image_url"],
        }
        item["score"] = compute_score(item, kw)
        candidates.append(item)
    t5 = time.time()

    # score 기준으로 정렬해서 상위 top_k만
    candidates.sort(key=lambda x: x["score"], reverse=True)
    t6 = time.time()
    selected = candidates[:top_k]

    print(f"⏱️ 키워드 추출 소요 시간 : {t1 - t0:.4f}초")
    print(f"⏱️ 이전 키워드 반영     : {t2 - t1:.4f}초")
    print(f"⏱️ DB검색 쿼리 생성    : {t3 - t2:.4f}초")
    print(f"⏱️ DB검색            : {t4 - t3:.4f}초")
    print(f"⏱️ 검색결과 정렬       : {t5 - t4:.4f}초")
    print("=====================================")
    print(f"⏱️ 총 경과 시간       : {t5 - t0:.4f}초")

    return {
        "keywords": kw,
        "recipes": selected,
    }