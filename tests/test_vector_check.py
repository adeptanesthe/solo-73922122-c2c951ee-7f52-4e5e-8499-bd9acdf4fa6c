"""外观向量一致性核查单元测试。"""

import copy
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.vector_check import analyze_vector_consistency


def vec(cos_with, target=(1.0, 0.0)):
    """返回与 target 余弦相似度恰为 cos_with 的二维单位向量。"""
    c = max(-1.0, min(1.0, cos_with))
    s = math.sqrt(max(0.0, 1.0 - c * c))
    tx, ty = target
    ox, oy = -ty, tx  # 与 target 正交
    return [c * tx + s * ox, c * ty + s * oy]


def det(bid, frame, vector):
    return {"id": bid, "frame": frame, "rect": [0, 0, 10, 10], "vector": vector}


def j(a, b, t):
    return {"a": a, "b": b, "type": t}


# ------------------------------------------------------------ 已可区分

def test_already_separable():
    dets = [det("a", 0, [1, 0]), det("b", 1, [1, 0]),
            det("c", 2, [1, 0]), det("d", 3, [0, 1])]
    js = [j("a", "b", "same"), j("c", "d", "different")]
    r = analyze_vector_consistency(dets, js)
    assert r["available"] and r["feasible"]
    assert r["recommendation"] == "none"
    assert r["already_separable"] is True
    assert r["boxes_to_recompute"] == []
    assert r["separation"] >= 0.05
    assert r["threshold_range"][0] <= r["threshold"] <= r["threshold_range"][1]
    assert len(r["judgments"]) == 2
    assert all("similarity" in x for x in r["judgments"])


def test_gap_exactly_at_margin_is_separable():
    # min同只 0.55，max异只 0.50 → 间隔恰为 0.05
    dets = [det("a", 0, [1, 0]), det("b", 1, vec(0.55)),
            det("c", 2, [1, 0]), det("d", 3, vec(0.50))]
    js = [j("a", "b", "same"), j("c", "d", "different")]
    r = analyze_vector_consistency(dets, js)
    assert r["recommendation"] == "none"
    assert r["separation"] == pytest.approx(0.05, abs=1e-6)


# ------------------------------------------------------------ 唯一重算建议

def test_gap_below_margin_needs_recompute():
    # 同只 0.54 / 异只 0.50，间隔 0.04 < 0.05
    dets = [det("a", 0, [1, 0]), det("b", 1, vec(0.54)),
            det("c", 2, vec(0.97)), det("d", 3, vec(0.96))]
    js = [j("a", "b", "same"), j("c", "d", "different")]
    r = analyze_vector_consistency(dets, js)
    # 移除任一框都会清空某一类判断 → 无合格框集
    assert r["available"] is False and r["reason"] == "no_qualifying_set"
    assert r["boxes_to_recompute"] is None


def test_unique_offending_box_with_clean_other_diff():
    dets = [
        det("a", 0, [1, 0]),
        det("b", 1, [1, 0]),
        det("c", 2, vec(0.97)),   # 与 a 的异只判断相似度偏高
        det("d", 3, [0, 1]),
    ]
    js = [
        j("a", "b", "same"),       # 1.00
        j("a", "c", "different"),  # 0.97
        j("b", "d", "different"),  # 0.00
    ]
    r = analyze_vector_consistency(dets, js)
    assert r["unique"] is True
    assert r["boxes_to_recompute"] == ["c"]
    cand = r["candidate_sets"][0]
    # 移除 d 只能把间隔拉到 0.03，仍不达标，因此不会成为候选
    assert cand["max_different_similarity"] == 0.0
    assert cand["min_same_similarity"] == pytest.approx(1.0, abs=1e-6)


def test_multiple_boxes_removed_together():
    # c、d 各自带一条与同只判断高相似的异只判断，必须同时移除
    dets = [
        det("a", 0, [1, 0]),
        det("b", 1, [1, 0]),
        det("c", 2, vec(0.96)),
        det("d", 3, vec(0.96)),
        det("e", 4, [1, 0]),
        det("f", 5, [0, 1]),
    ]
    js = [
        j("a", "b", "same"),        # 1.00
        j("a", "c", "different"),   # 0.96
        j("b", "d", "different"),   # 0.96
        j("e", "f", "different"),   # 0.00
    ]
    r = analyze_vector_consistency(dets, js)
    assert r["recommendation"] == "recompute" and r["unique"] is True
    assert r["boxes_to_recompute"] == ["c", "d"]
    assert r["threshold"] is not None
    assert r["separation"] >= 0.05
    cand = r["candidate_sets"][0]
    assert cand["retained_same"] >= 1 and cand["retained_different"] >= 1


# ------------------------------------------------------------ 不唯一

def test_ambiguous_when_multiple_minimal_sets():
    # 同只 0.92 / 异只 c-d 0.90（间隔 0.02）；另有一条低相似异只 e-f
    # 移除 c 或移除 d 都能恢复区分 → 两组同样最少
    dets = [
        det("a", 0, [1, 0]),
        det("b", 1, vec(0.92)),
        det("c", 2, [1, 0]),
        det("d", 3, vec(0.90)),
        det("e", 4, [1, 0]),
        det("f", 5, [0, 1]),
    ]
    js = [
        j("a", "b", "same"),
        j("c", "d", "different"),
        j("e", "f", "different"),
    ]
    r = analyze_vector_consistency(dets, js)
    assert r["recommendation"] == "ambiguous"
    assert r["unique"] is False
    assert r["available"] is True
    assert r["boxes_to_recompute"] is None
    assert r["threshold"] is None
    assert len(r["candidate_sets"]) == 2
    sets = {tuple(c["boxes_to_recompute"]) for c in r["candidate_sets"]}
    assert sets == {("c",), ("d",)}
    for c in r["candidate_sets"]:
        assert c["retained_same"] >= 1 and c["retained_different"] >= 1
        assert c["threshold"] is not None
    assert "不唯一" in r["message"]


# ------------------------------------------------------------ 不给建议的情形

def test_no_qualifying_set():
    dets = [
        det("a", 0, [1, 0]),
        det("b", 1, [1, 0]),
        det("c", 2, vec(0.98)),
    ]
    # 两类判断共享框 a：移除任何框都会导致某一类判断被清空
    js = [j("a", "b", "same"), j("a", "c", "different")]
    r = analyze_vector_consistency(dets, js)
    assert r["available"] is False
    assert r["reason"] == "no_qualifying_set"
    assert r["recommendation"] == "none"
    assert r["boxes_to_recompute"] is None


def test_requires_both_types():
    dets = [det("a", 0, [1, 0]), det("b", 1, [1, 0])]
    r = analyze_vector_consistency(dets, [j("a", "b", "same")])
    assert r["available"] is False and r["reason"] == "need_both_types"
    assert r["boxes_to_recompute"] is None

    r2 = analyze_vector_consistency(dets, [])
    assert r2["available"] is False and r2["reason"] == "need_both_types"


def test_trajectory_infeasible_no_suggestion():
    dets = [det("a", 0, [1, 0]), det("b", 1, [1, 0])]
    js = [j("a", "b", "same")]
    r = analyze_vector_consistency(
        dets, js, trajectory_feasible=False, trajectory_message="无解：示例"
    )
    assert r["available"] is False and r["feasible"] is False
    assert r["reason"] == "trajectory_infeasible"
    assert r["boxes_to_recompute"] is None


def test_too_many_boxes():
    # 引用框数只有 4，但会话检测框 11 > 10：按“涉及”口径仍应放行
    vectors = [[1, 0], [1, 0], [1, 0], [0, 1]]
    vectors += [[1, 0] for _ in range(7)]
    dets = [det(f"b{i}", i, vectors[i]) for i in range(11)]
    js = [j("b0", "b1", "same"), j("b2", "b3", "different")]
    r = analyze_vector_consistency(dets, js)
    assert r["available"] is True
    assert r["scope"]["boxes"] == 4

    # 引用 11 个框 → 超限
    js2 = [j(f"b{2 * k}", f"b{2 * k + 1}", "same") for k in range(5)]
    js2.append(j("b0", "b10", "different"))
    r2 = analyze_vector_consistency(dets, js2)
    assert r2["available"] is False and r2["reason"] == "too_many_boxes"
    assert r2["boxes_to_recompute"] is None


def test_too_many_judgments():
    dets = [det("a", 0, [1, 0]), det("b", 1, [1, 0]),
            det("c", 2, [1, 0]), det("d", 3, [0, 1])]
    js = [j("a", "b", "same"), j("c", "d", "different")]
    js += [j("a", "b", "same")] * 19  # 共 21 条
    r = analyze_vector_consistency(dets, js)
    assert r["available"] is False and r["reason"] == "too_many_judgments"
    assert r["boxes_to_recompute"] is None


def test_scope_boundary_10_boxes_20_judgments_allowed():
    """恰好 10 个框、20 条判断仍应核查。"""
    # 10 个框全部被判断引用；判断共 20 条
    vecs = [[1, 0], [1, 0], [1, 0], [0, 1], [1, 0],
            [1, 0], [1, 0], [0, 1], [1, 0], [1, 0]]
    dets = [det(f"b{i}", i, vecs[i]) for i in range(10)]
    js = [j("b0", "b1", "same"), j("b4", "b5", "same"), j("b8", "b9", "same"),
          j("b2", "b3", "different"), j("b6", "b7", "different")]
    js += js * 3  # 共 20 条
    r = analyze_vector_consistency(dets, js)
    assert r["available"] is True
    assert r["scope"]["boxes"] == 10 and r["scope"]["judgments"] == 20


# ------------------------------------------------------------ 确定性 / 纯净性

def test_does_not_mutate_inputs():
    dets = [det("a", 0, [1, 0]), det("b", 1, [1, 0]),
            det("c", 2, vec(0.9)), det("d", 3, [0, 1])]
    js = [j("a", "b", "same"), j("a", "c", "different"), j("b", "d", "different")]
    d0, j0 = copy.deepcopy(dets), copy.deepcopy(js)
    analyze_vector_consistency(dets, js)
    assert dets == d0 and js == j0


def test_similarity_values_reported():
    dets = [det("a", 0, [1, 0]), det("b", 1, [1, 0]),
            det("c", 2, [1, 0]), det("d", 3, [0, 1])]
    js = [j("a", "b", "same"), j("c", "d", "different")]
    r = analyze_vector_consistency(dets, js)
    sims = {(x["a"], x["b"]): x["similarity"] for x in r["judgments"]}
    assert sims[("a", "b")] == pytest.approx(1.0)
    assert sims[("c", "d")] == pytest.approx(0.0)
