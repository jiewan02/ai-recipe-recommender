# rag_flask/app.py
from flask import Flask, request, jsonify, Response
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
import re
import torch
import requests
from bs4 import BeautifulSoup
import time
import json
from graph_similarity_v2 import RecipeGraphSimilarity
from jiewan_model_v2 import graph_rag_search_with_scoring_explanation
# from jiewan_model import graph_rag_search_with_scoring_explanation
# from graph_server import graph_rag_search 


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 환경변수(.env) 로드
load_dotenv()
client = OpenAI()  # OPENAI_API_KEY 자동 사용

app = Flask(__name__)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

similarity_service = RecipeGraphSimilarity(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

def json_line(obj):
    return json.dumps(obj, ensure_ascii=False) + "\n"

# ===== 데이터 & 임베딩 로드 =====
# 미리 만들어둔 임베딩
embeddings = np.load("recipe_embeddings.npy").astype("float32")
emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

emb_norm_t = torch.from_numpy(emb_norm).to(DEVICE)  # shape: (N, D)

# 레시피 메타데이터
# Parquet이 싫으면 df = pd.read_csv("recipes_with_embedding_text.csv") 도 가능
# df = pd.read_parquet("recipes_with_embedding_text.parquet")
df = pd.read_csv("dataset_preprocessed.csv")

EMBED_MODEL = "text-embedding-3-small"


def embed_query(text: str, keywords: dict) -> np.ndarray:
    """OpenAI 임베딩으로 쿼리 벡터 생성"""
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text],
    )
    v = np.array(resp.data[0].embedding, dtype="float32")
    v = v / np.linalg.norm(v)
    v_t = torch.from_numpy(v).to(DEVICE)  # (D,)
    return v_t


def get_recipe(id):
    res = requests.get(f"https://www.10000recipe.com/recipe/{id}")
    soup = BeautifulSoup(res.content, features="html.parser")
    main_title = soup.select_one("div.view2_summary h3").text

    infos = soup.select_one("div.view2_summary_info")
    info1 = infos.select_one("span.view2_summary_info1")
    info1 = info1.text if info1 else ""
    info2 = infos.select_one("span.view2_summary_info2")
    info2 = info2.text if info2 else ""
    info3 = infos.select_one("span.view2_summary_info3")
    info3 = info3.text if info3 else ""
    infos = [info1,info2,info3]


    # Fallback: desktop layout style that sometimes appears
    image_url = ""
    img_tag = soup.select_one("div.centeredcrop img")
    if not img_tag:
        img_tag = soup.select_one("div.view3_pic img")

    if img_tag and img_tag.get("src"):
        image_url = img_tag["src"].strip()

    grid = soup.select_one("div.cont_ingre2")
    result = {
        "재료": [],
        "조리도구": []
    }

    # 모든 big titles 찾기 (ex: 재료 / 조리도구)
    big_sections = grid.select("div.best_tit")

    for section in big_sections:
        title = section.get_text(strip=True)

        # 제목 다음에 오는 ready_ingre3 블록 찾기
        next_div = section.find_next_sibling("div", class_="ready_ingre3")
        if not next_div:
            continue

        # 재료 처리
        if "재료" in title:
            for li in next_div.select("li"):
                # 재료명
                name_tag = li.select_one("div.ingre_list_name a")
                if not name_tag:
                    continue
                name = name_tag.get_text(strip=True)

                # 용량
                qty_tag = li.select_one("span.ingre_list_ea")
                qty = qty_tag.get_text(strip=True) if qty_tag else ""

                result["재료"].append((name, qty))

        # 조리도구 처리
        elif "조리도구" in title:
            for li in next_div.select("li"):
                name_tag = li.select_one("div.ingre_list_name")
                if name_tag:
                    tool_name = name_tag.get_text(strip=True)
                    result["조리도구"].append(tool_name)



    steps = []
    for cont in soup.select("div.view_step_cont"):
        step = {"text": "","tools": "", "img_url": ""}
        body = cont.select_one("div.media-body")
        if body:
            main_text_node = body.find(string=True, recursive=False)
            if main_text_node:
                # steps.append(main_text_node.strip())
                step["text"] = main_text_node.strip()

                tools = body.select("p")
                if tools:
                    step["tools"] = tools[0].text.strip()

        img = cont.find("div", id=re.compile(r"stepimg\d+")).select_one("img")
        if img:
            step["img_url"] = img["src"]
        steps.append(step)
        

    return {"title": main_title,"infos":infos, "image_url": image_url, "steps": steps, "grid_info": result }


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json() or {}
    query = (data.get("query") or "").strip()
    keywords = (data.get("matchedKeywords") or {})
    top_k = int(data.get("top_k", 5))

    if not query:
        return jsonify({"error": "query is required"}), 400

    # 1) 쿼리 임베딩
    q = embed_query(query, keywords)

    # 2) 코사인 유사도 (정규화된 벡터이므로 dot = cosine)
    sims_t = emb_norm_t @ q  # (N,)

    # 3) 상위 K개 인덱스
    k = min(top_k, sims_t.shape[0])
    scores_t, idxs_t = torch.topk(sims_t, k)

    idxs = idxs_t.cpu().numpy()
    scores = scores_t.cpu().numpy().tolist()

    # 4) df에서 메타데이터 꺼내서 JSON으로 묶기
    results = []
    for idx, score in zip(idxs, scores):
        row = df.iloc[idx]

        results.append(
            {
                "index": int(idx),
                "score": float(score),
                "name": row.get("요리명", ""),
                "types": row.get("요리종류별명", []),
                "intro": row.get("요리소개_cleaned", ""),
                "servings": row.get("요리인분명", ""),
                "difficulty": row.get("요리난이도명", ""),
                "time": row.get("요리시간명", ""),
                "ingredients": row.get("재료", []),
            }
        )

    return jsonify({"results": results})

# @app.route("/graph-search", methods=["POST"])
# def graph_search_endpoint():
#     data = request.get_json() or {}
#     query = (data.get("query") or "").strip()
#     keywords = (data.get("matchedKeywords") or {})
#     top_k = int(data.get("top_k", 5))

#     if not query:
#         return jsonify({"error": "query is required"}), 400

#     try:
#         start = time.time()
#         res = graph_rag_search(query, keywords, top_k=top_k)
#         end = time.time()
#         print(f"⏱️ 작업 소요 시간: {end - start:.4f}초")
#     except Exception as e:
#         print("[ERROR] graph_rag_search failed:", e)
#         return jsonify({"error": "graph search failed", "detail": str(e)}), 500

#     # Express 쪽에서 쓰기 편하도록 기존 /search와 비슷한 형태로 맞추기
#     # res["recipes"] = [{recipe_id, title, name, views, time_min, difficulty, servings, score}, ...]
#     return jsonify({
#         "results": res["recipes"],   # 메인 추천 리스트
#         "keywords": res["keywords"], # 디버깅/로그용 (원하면 프론트에서 안 써도 됨)
#     })

# @app.route("/jiewan-search", methods=["POST"])
# def graph_search_endpoint():
#     data = request.get_json() or {}
#     query = (data.get("query") or "").strip()
#     keywords = (data.get("matchedKeywords") or {})
#     top_k = int(data.get("top_k", 5))

#     if not query:
#         return jsonify({"error": "query is required"}), 400

#     try:
#         start = time.time()
#         res = graph_rag_search_with_scoring_explanation(query, top_k=top_k)
#         end = time.time()
#         print(f"⏱️ 작업 소요 시간: {end - start:.4f}초")
#     except Exception as e:
#         print("[ERROR] graph_rag_search failed:", e)
#         return jsonify({"error": "graph search failed", "detail": str(e)}), 500

#     # Express 쪽에서 쓰기 편하도록 기존 /search와 비슷한 형태로 맞추기
#     # res["recipes"] = [{recipe_id, title, name, views, time_min, difficulty, servings, score}, ...]
#     return jsonify({
#         "results": res["recipes"],   # 메인 추천 리스트
#         "keywords": res["keywords"], # 디버깅/로그용 (원하면 프론트에서 안 써도 됨)
#     })

@app.route("/jiewan-search-v2", methods=["POST"])
def graph_search_endpoint():
    data = request.get_json() or {}
    query = (data.get("query") or "").strip()
    filterKeywords = (data.get("filterKeywords") or {})
    top_k = int(data.get("top_k", 5))

    if not query:
        return jsonify({"error": "query is required"}), 400

    try:
        start = time.time()
        res = graph_rag_search_with_scoring_explanation(query,filterKeywords = filterKeywords, top_k=top_k)
        end = time.time()
        print(f"⏱️ 작업 소요 시간: {end - start:.4f}초")
    except Exception as e:
        print("[ERROR] graph_rag_search failed:", e)
        return jsonify({"error": "graph search failed", "detail": str(e)}), 500

    # Express 쪽에서 쓰기 편하도록 기존 /search와 비슷한 형태로 맞추기
    # res["recipes"] = [{recipe_id, title, name, views, time_min, difficulty, servings, score}, ...]
    return jsonify({
        "results": res["recipes"],   # 메인 추천 리스트
        "keywords": res["keywords"], # 디버깅/로그용 (원하면 프론트에서 안 써도 됨)
    })

@app.route("/crawl-recipe/<int:recipe_id>", methods=["GET"]) #아래 엔드포인트랑 합치기
def crawl_recipe_endpoint(recipe_id):

    top_n = 3
    min_shared_ings = int(2)  # 기본값: 최소 2개 재료 공유
    try:
        start = time.time()
        result = similarity_service.get_similar_recipes(
            recipe_id=recipe_id,
            top_n=top_n,
            min_shared_ings=min_shared_ings,
        )
        end = time.time()
        print(f"⏱️ similar-recipes 작업 소요 시간: {end - start:.4f}초")
        data = get_recipe(recipe_id)
        return jsonify({
            "id": recipe_id,
            "data": data,
            "overall": result["overall"],         # 전체 그래프 기반 유사 레시피
        "ingredients": result["ingredients"], # 재료 기반 유사 레시피
        })
    
    except Exception as e:
        print("[ERROR] get_recipe failed:", e)
        return jsonify({"error": "crawl_failed", "detail": str(e)}), 500

@app.route("/similar-recipes", methods=["POST"])
def similar_recipes_endpoint():
    data = request.get_json() or {}
    recipe_id = data.get("recipe_id")
    top_n = 3
    min_shared_ings = int(2)  # 기본값: 최소 2개 재료 공유

    try:
        start = time.time()
        result = similarity_service.get_similar_recipes(
            recipe_id=recipe_id,
            top_n=top_n,
            min_shared_ings=min_shared_ings,
        )
        end = time.time()
        print(f"⏱️ similar-recipes 작업 소요 시간: {end - start:.4f}초")
    except Exception as e:
        print("[ERROR] similar_recipes failed:", e)
        return jsonify({"error": "similar_recipes failed", "detail": str(e)}), 500

    # 프론트에서 바로 쓰기 편하도록 구조 유지
    return jsonify({
        "recipe_id": recipe_id,
        "overall": result["overall"],         # 전체 그래프 기반 유사 레시피
        "ingredients": result["ingredients"], # 재료 기반 유사 레시피
    })


if __name__ == "__main__":
    # Node 서버랑 포트 안 겹치게 5001으로 예시
    app.run(host="0.0.0.0", port=8001, debug=True)