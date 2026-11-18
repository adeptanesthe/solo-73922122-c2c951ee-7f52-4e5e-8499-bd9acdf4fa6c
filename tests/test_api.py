"""API 端到端测试：会话创建、校验、判断/参数版本、异步求解与迟到结果作废。"""

import io
import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app

client = TestClient(app)


def png_bytes(color=(200, 30, 30)):
    img = Image.new("RGB", (64, 48), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_session(n_imgs=3, dets=None):
    if dets is None:
        dets = [
            {"id": "a", "frame": 0, "rect": [1, 1, 8, 8], "vector": [1, 0]},
            {"id": "b", "frame": 1, "rect": [11, 1, 8, 8], "vector": [1, 0]},
        ]
    files = [("images", (f"f{i}.png", png_bytes(), "image/png")) for i in range(n_imgs)]
    files.append(("detections", ("det.json", json.dumps({"detections": dets}), "application/json")))
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 200, r.text
    return r.json()


def wait_job(job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.05)
    raise TimeoutError("求解任务超时未完成")


# ---------------------------------------------------------------- 会话与校验

def test_create_session_ok():
    p = make_session()
    assert p["n_frames"] == 3 and len(p["detections"]) == 2
    assert p["version"] == 1
    img = client.get(f"/api/sessions/{p['session_id']}/frames/0")
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"


def test_too_many_images_rejected():
    files = [("images", (f"f{i}.png", png_bytes(), "image/png")) for i in range(11)]
    files.append(("detections", ("d.json", json.dumps([]), "application/json")))
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 400 and "10" in r.json()["detail"]


def test_too_many_detections_rejected():
    dets = [{"id": f"b{i}", "frame": 0, "rect": [0, 0, 4, 4], "vector": [1]} for i in range(31)]
    files = [("images", ("f0.png", png_bytes(), "image/png")),
             ("detections", ("d.json", json.dumps(dets), "application/json"))]
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 400 and "30" in r.json()["detail"]


@pytest.mark.parametrize("det,needle", [
    ({"id": "a", "frame": 0, "rect": [0, 0, 4, 4], "vector": [0, 0]}, "非零"),
    ({"id": "a", "frame": 5, "rect": [0, 0, 4, 4], "vector": [1]}, "帧号"),
    ({"id": "a", "frame": 0, "rect": [0, 0, -4, 4], "vector": [1]}, "宽"),
    ({"id": "a", "frame": 0, "rect": [0, 0, 4, 4], "vector": [1, float("nan")]}, "非法"),
])
def test_bad_detection_rejected(det, needle):
    files = [("images", ("f0.png", png_bytes(), "image/png")),
             ("detections", ("d.json", json.dumps([det]), "application/json"))]
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 400 and needle in r.json()["detail"]


def test_vector_dim_mismatch_rejected():
    dets = [
        {"id": "a", "frame": 0, "rect": [0, 0, 4, 4], "vector": [1, 0]},
        {"id": "b", "frame": 0, "rect": [8, 0, 4, 4], "vector": [1, 0, 0]},
    ]
    files = [("images", ("f0.png", png_bytes(), "image/png")),
             ("detections", ("d.json", json.dumps(dets), "application/json"))]
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 400 and "维度" in r.json()["detail"]


# ---------------------------------------------------------------- 示例与求解

def test_example_solves_to_two_animals():
    p = client.post("/api/example").json()
    assert p["n_frames"] == 6 and len(p["detections"]) == 11
    job = client.post(f"/api/sessions/{p['session_id']}/solve").json()
    done = wait_job(job["job_id"])
    assert done["status"] == "done"
    res = done["result"]
    assert res["feasible"] and res["animal_count"] == 2 and res["n_trajectories"] == 2
    assert res["total_cost"] == pytest.approx(res["assoc_cost"] + res["penalty_cost"])


def test_conflicting_judgments_report_infeasible_without_count():
    p = client.post("/api/example").json()
    sid = p["session_id"]
    # A0 与 B0 同处第 0 帧，标记同只 → 无解
    r = client.put(f"/api/sessions/{sid}/judgments",
                   json={"judgments": [{"a": "A0", "b": "B0", "type": "same"}]})
    assert r.status_code == 200
    job = client.post(f"/api/sessions/{sid}/solve").json()
    done = wait_job(job["job_id"])
    res = done["result"]
    assert res["feasible"] is False
    assert "animal_count" not in res
    assert "无解" in res["message"]


def test_same_judgment_changes_result():
    p = client.post("/api/example").json()
    sid = p["session_id"]
    # 把 B0 与 A1 判为同只（它们帧差 1），全局解必须让二者同轨
    client.put(f"/api/sessions/{sid}/judgments",
               json={"judgments": [{"a": "B0", "b": "A1", "type": "same"}]})
    done = wait_job(client.post(f"/api/sessions/{sid}/solve").json()["job_id"])
    res = done["result"]
    assert res["feasible"]
    for t in res["trajectories"]:
        ids = {b["id"] for b in t["boxes"]}
        assert not ({"B0", "A1"} <= ids) or True  # 占位，下面统一检查
    same_traj = [t for t in res["trajectories"]
                 if {"B0", "A1"} <= {b["id"] for b in t["boxes"]}]
    assert len(same_traj) == 1


# ---------------------------------------------------------------- 版本与迟到结果

def test_version_bumps_and_stale_job_detectable():
    p = make_session()
    sid = p["session_id"]
    assert p["version"] == 1

    r = client.put(f"/api/sessions/{sid}/params",
                   json={"w_dist": 2.0, "w_app": 50.0, "penalty": 100.0})
    assert r.status_code == 200 and r.json()["version"] == 2

    job = client.post(f"/api/sessions/{sid}/solve").json()
    assert job["version"] == 2  # 任务记录启动时版本

    # 求解期间继续修改 → 版本再 +1，旧任务结果即为“迟到结果”
    r = client.put(f"/api/sessions/{sid}/judgments",
                   json={"judgments": [{"a": "a", "b": "b", "type": "different"}]})
    assert r.json()["version"] == 3

    done = wait_job(job["job_id"])
    assert done["version"] == 2  # 结果仍携带旧版本号
    current = client.get(f"/api/sessions/{sid}").json()
    assert current["version"] == 3
    assert done["version"] != current["version"]  # 前端据此丢弃，不覆盖新输入


def test_params_validation():
    p = make_session()
    r = client.put(f"/api/sessions/{p['session_id']}/params",
                   json={"w_dist": -1, "w_app": 1, "penalty": 1})
    assert r.status_code == 422


def test_unknown_session_404():
    assert client.get("/api/sessions/nope").status_code == 404
    assert client.post("/api/sessions/nope/solve").status_code == 404


def test_healthz():
    assert client.get("/healthz").json() == {"ok": True}


# ---------------------------------------------------------------- 向量核查

VECTOR_DETS = [
    {"id": "a", "frame": 0, "rect": [0, 0, 8, 8], "vector": [1, 0]},
    {"id": "b", "frame": 1, "rect": [10, 0, 8, 8], "vector": [1, 0]},
    {"id": "c", "frame": 2, "rect": [20, 0, 8, 8], "vector": [0.97, 0.243]},
    {"id": "d", "frame": 3, "rect": [30, 0, 8, 8], "vector": [0, 1]},
    {"id": "e", "frame": 2, "rect": [20, 20, 8, 8], "vector": [1, 0]},
    {"id": "f", "frame": 3, "rect": [30, 20, 8, 8], "vector": [0, 1]},
]


def make_vector_session(judgments):
    p = make_session(n_imgs=4, dets=VECTOR_DETS)
    sid = p["session_id"]
    r = client.put(f"/api/sessions/{sid}/judgments", json={"judgments": judgments})
    assert r.status_code == 200, r.text
    return sid, r.json()["version"]


def run_vector_check(sid):
    job = client.post(f"/api/sessions/{sid}/vector-check").json()
    return wait_job(job["job_id"]), job


def test_vector_check_unique_recompute():
    sid, version = make_vector_session([
        {"a": "a", "b": "b", "type": "same"},
        {"a": "a", "b": "c", "type": "different"},
        {"a": "e", "b": "f", "type": "different"},
    ])
    done, job = run_vector_check(sid)
    assert job["version"] == version
    assert done["status"] == "done" and done["result"]["available"]
    res = done["result"]
    assert res["recommendation"] == "recompute"
    assert res["unique"] is True
    assert res["boxes_to_recompute"] == ["c"]
    assert len(res["judgments"]) == 3
    assert all(0.0 <= j["similarity"] <= 1.0 for j in res["judgments"])
    assert res["threshold"] is not None
    assert res["separation"] >= 0.05


def test_vector_check_already_separable():
    sid, _ = make_vector_session([
        {"a": "a", "b": "b", "type": "same"},
        {"a": "e", "b": "f", "type": "different"},
    ])
    done, _ = run_vector_check(sid)
    res = done["result"]
    assert res["available"] and res["recommendation"] == "none"
    assert res["already_separable"] is True
    assert res["boxes_to_recompute"] == []


def test_vector_check_requires_both_types():
    sid, _ = make_vector_session([{"a": "a", "b": "b", "type": "same"}])
    done, _ = run_vector_check(sid)
    res = done["result"]
    assert res["available"] is False and res["reason"] == "need_both_types"
    assert res["boxes_to_recompute"] is None


def test_vector_check_no_suggestion_when_trajectory_infeasible():
    p = client.post("/api/example").json()
    sid = p["session_id"]
    # A0 与 B0 同帧却标同只 → 轨迹判断本身无解
    client.put(f"/api/sessions/{sid}/judgments", json={"judgments": [
        {"a": "A0", "b": "B0", "type": "same"},
        {"a": "B0", "b": "B1", "type": "different"},
    ]})
    done, _ = run_vector_check(sid)
    res = done["result"]
    assert res["available"] is False and res["feasible"] is False
    assert res["reason"] == "trajectory_infeasible"
    assert res["boxes_to_recompute"] is None


def test_vector_check_scope_limit_via_api():
    dets = [{"id": f"b{i}", "frame": i % 3, "rect": [0, 0, 4, 4],
             "vector": [1, 0] if i % 2 == 0 else [0, 1]} for i in range(11)]
    p = make_session(n_imgs=3, dets=dets)
    sid = p["session_id"]
    js = [{"a": f"b{2*k}", "b": f"b{2*k+1}", "type": "same"} for k in range(5)]
    js.append({"a": "b9", "b": "b10", "type": "different"})
    client.put(f"/api/sessions/{sid}/judgments", json={"judgments": js})
    done, _ = run_vector_check(sid)
    res = done["result"]
    assert res["available"] is False and res["reason"] == "too_many_boxes"
    assert res["boxes_to_recompute"] is None


def test_vector_check_stale_version_reported():
    """核查期间改判 → 任务结果仍携带旧版本，前端据此丢弃，不覆盖新输入。"""
    sid, _ = make_vector_session([
        {"a": "a", "b": "b", "type": "same"},
        {"a": "e", "b": "f", "type": "different"},
    ])
    before = client.get(f"/api/sessions/{sid}").json()["version"]
    job = client.post(f"/api/sessions/{sid}/vector-check").json()
    assert job["version"] == before
    # 立即改判（核查可能已完成，但结果版本仍是旧版本）
    r = client.put(f"/api/sessions/{sid}/judgments", json={"judgments": [
        {"a": "a", "b": "b", "type": "same"},
    ]})
    assert r.json()["version"] == before + 1
    done = wait_job(job["job_id"])
    assert done["version"] == before
    assert done["version"] != client.get(f"/api/sessions/{sid}").json()["version"]


def test_vector_check_ambiguous_reported():
    dets = [
        {"id": "a", "frame": 0, "rect": [0, 0, 8, 8], "vector": [1, 0]},
        {"id": "b", "frame": 1, "rect": [10, 0, 8, 8], "vector": [0.92, 0.391918]},
        {"id": "c", "frame": 2, "rect": [20, 0, 8, 8], "vector": [1, 0]},
        {"id": "d", "frame": 3, "rect": [30, 0, 8, 8], "vector": [0.90, 0.435890]},
        {"id": "e", "frame": 0, "rect": [0, 20, 8, 8], "vector": [1, 0]},
        {"id": "f", "frame": 1, "rect": [10, 20, 8, 8], "vector": [0, 1]},
    ]
    p = make_session(n_imgs=4, dets=dets)
    sid = p["session_id"]
    client.put(f"/api/sessions/{sid}/judgments", json={"judgments": [
        {"a": "a", "b": "b", "type": "same"},
        {"a": "c", "b": "d", "type": "different"},
        {"a": "e", "b": "f", "type": "different"},
    ]})
    done, _ = run_vector_check(sid)
    res = done["result"]
    assert res["available"] is True
    assert res["recommendation"] == "ambiguous" and res["unique"] is False
    assert res["boxes_to_recompute"] is None
    assert {tuple(c["boxes_to_recompute"]) for c in res["candidate_sets"]} == {("c",), ("d",)}
    for c in res["candidate_sets"]:
        assert c["retained_same"] >= 1 and c["retained_different"] >= 1
    assert "不唯一" in res["message"]


def test_vector_check_unknown_session_404():
    assert client.post("/api/sessions/nope/vector-check").status_code == 404


def test_vector_check_job_kind_and_local_data():
    """核查任务不改动会话数据，且返回的框号只来自当前会话。"""
    sid, _ = make_vector_session([
        {"a": "a", "b": "b", "type": "same"},
        {"a": "a", "b": "c", "type": "different"},
    ])
    before = client.get(f"/api/sessions/{sid}").json()
    done, _ = run_vector_check(sid)
    assert done["kind"] == "vector_check"
    after = client.get(f"/api/sessions/{sid}").json()
    assert after == before
    ids = {d["id"] for d in before["detections"]}
    for j in done["result"]["judgments"]:
        assert j["a"] in ids and j["b"] in ids
