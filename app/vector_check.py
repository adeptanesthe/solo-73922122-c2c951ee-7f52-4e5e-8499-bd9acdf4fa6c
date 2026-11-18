"""外观向量一致性核查。

用途
----
视觉工程师复核交叉遮挡片段时，需要判断“哪些检测框的外观向量值得重新
提取”。本模块只使用**人工同只/异只判断**与检测框外观向量的余弦相似度：

- 同只判断的相似度应整体高于异只判断；
- 若存在阈值 t，使所有同只相似度 ≥ t、所有异只相似度 < t，且两类之间
  （最小同只相似度 − 最大异只相似度）至少相隔 ``MIN_MARGIN``（0.05），
  则认为当前向量足以区分两类；
- 若不能区分，则枚举“移除哪些框”（这些框的向量被怀疑，重新提取后即可
  恢复区分）的最小子集。为保证结论仍有依据，移除后必须至少保留一条
  同只判断和一条异只判断；
- 若存在多组同样大小的最小框集，明确标记结果不唯一，不擅自指定其中一组，
  也不断定“向量出错”；
- 若不存在合格框集，或当前轨迹判断本身无解，则不给出任何重算建议。

范围限制
--------
只核查涉及不超过 10 个框、20 条人工判断的片段（枚举 2^10 个子集）。
“涉及”指被人工判断引用的框，而非会话内全部检测框。

本模块不修改求解器的轨迹与计数逻辑。
"""

from __future__ import annotations

import itertools
import math

MIN_MARGIN = 0.05
MAX_CHECK_BOXES = 10
MAX_CHECK_JUDGMENTS = 20
_EPS = 1e-12


def _cosine(u, v) -> float:
    """余弦相似度（向量在输入时已保证非零）。"""
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(x * x for x in v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    val = sum(a * b for a, b in zip(u, v)) / (nu * nv)
    return max(-1.0, min(1.0, val))


def _r(x: float) -> float:
    return round(float(x), 6)


def _unavailable(reason: str, message: str, *, scope=None, judgments=None,
                 feasible: bool = True, base: dict | None = None) -> dict:
    out = dict(base or {})
    out.update(
        {
            "available": False,
            "feasible": feasible,
            "reason": reason,
            "message": message,
            "scope": scope,
            "judgments": judgments or [],
            "recommendation": "none",
            "boxes_to_recompute": None,
            "threshold": None,
            "threshold_range": None,
            "separation": None,
            "unique": None,
            "already_separable": False,
            "candidate_sets": [],
        }
    )
    return out


def _scope(detections, judgments) -> tuple[dict, list[str], dict[str, list[float]]]:
    """返回 (scope 信息, 被判断引用的框号列表, 框号→向量)。"""
    id_to_vec = {str(d["id"]): list(d["vector"]) for d in detections}
    referenced = []
    seen = set()
    for j in judgments:
        for bid in (str(j["a"]), str(j["b"])):
            if bid not in seen:
                seen.add(bid)
                referenced.append(bid)
    scope = {
        "boxes": len(referenced),
        "judgments": len(judgments),
        "max_boxes": MAX_CHECK_BOXES,
        "max_judgments": MAX_CHECK_JUDGMENTS,
        "within_limits": (
            len(referenced) <= MAX_CHECK_BOXES and len(judgments) <= MAX_CHECK_JUDGMENTS
        ),
    }
    return scope, referenced, id_to_vec


def _judgment_rows(judgments, id_to_vec) -> list[dict]:
    rows = []
    for j in judgments:
        a, b = str(j["a"]), str(j["b"])
        sim = _cosine(id_to_vec[a], id_to_vec[b])
        rows.append(
            {
                "a": a,
                "b": b,
                "type": j["type"],
                "similarity": _r(sim),
                "_sim": sim,
            }
        )
    return rows


def _separates(rows, margin: float = MIN_MARGIN) -> tuple[bool, float | None, dict | None]:
    """检查 rows 中同只/异只是否被阈值分开至少 margin。

    返回 (是否可分, 最小同只相似度, 阈值信息)。
    """
    same = [r["_sim"] for r in rows if r["type"] == "same"]
    diff = [r["_sim"] for r in rows if r["type"] == "different"]
    if not same or not diff:
        return False, None, None
    lo = max(diff)
    hi = min(same)
    gap = hi - lo
    if gap + _EPS < margin:
        return False, hi, None
    return True, hi, {"low": lo, "high": hi, "threshold": (lo + hi) / 2.0, "gap": gap}


def _public_rows(rows) -> list[dict]:
    return [{k: v for k, v in r.items() if k != "_sim"} for r in rows]


def _candidate(boxes, retained_rows, dropped_rows, sep) -> dict:
    same_count = sum(1 for r in retained_rows if r["type"] == "same")
    diff_count = sum(1 for r in retained_rows if r["type"] == "different")
    return {
        "boxes_to_recompute": list(boxes),
        "threshold": _r(sep["threshold"]),
        "threshold_range": [_r(sep["low"]), _r(sep["high"])],
        "separation": _r(sep["gap"]),
        "min_same_similarity": _r(sep["high"]),
        "max_different_similarity": _r(sep["low"]),
        "retained_same": same_count,
        "retained_different": diff_count,
        "retained_judgments": _public_rows(retained_rows),
        "dropped_judgments": _public_rows(dropped_rows),
    }


def analyze_vector_consistency(detections, judgments, *,
                               trajectory_feasible: bool = True,
                               trajectory_message: str | None = None,
                               margin: float = MIN_MARGIN,
                               max_boxes: int = MAX_CHECK_BOXES,
                               max_judgments: int = MAX_CHECK_JUDGMENTS) -> dict:
    """核查外观向量一致性并给出重算建议（纯函数）。

    detections: 规范化检测框 [{"id","frame","rect","vector"}]
    judgments:  [{"a":框号,"b":框号,"type":"same"|"different"}]
    trajectory_feasible: 当前人工判断下轨迹问题是否有解（由求解器给出）。
    """
    # 1) 轨迹判断本身无解 → 不给重算建议
    if not trajectory_feasible:
        return _unavailable(
            "trajectory_infeasible",
            "当前人工判断下轨迹问题本身无解，无法据此核查外观向量，不提供重算建议。"
            + (f"（{trajectory_message}）" if trajectory_message else ""),
            feasible=False,
        )

    scope, referenced, id_to_vec = _scope(detections, judgments)
    if scope["judgments"] > max_judgments:
        return _unavailable(
            "too_many_judgments",
            f"涉及 {scope['judgments']} 条人工判断，超过核查上限 {max_judgments}，"
            "不提供重算建议。",
            scope=scope,
        )
    if scope["boxes"] > max_boxes:
        return _unavailable(
            "too_many_boxes",
            f"涉及 {scope['boxes']} 个检测框，超过核查上限 {max_boxes}，"
            "不提供重算建议。",
            scope=scope,
        )

    same_n = sum(1 for j in judgments if j["type"] == "same")
    diff_n = sum(1 for j in judgments if j["type"] == "different")
    if same_n == 0 or diff_n == 0:
        missing = "同只与异只" if same_n == 0 and diff_n == 0 else (
            "同只" if same_n == 0 else "异只"
        )
        return _unavailable(
            "need_both_types",
            f"核查需要同时包含同只与异只判断；当前缺少{missing}判断，不提供重算建议。",
            scope=scope,
            judgments=_public_rows(_judgment_rows(judgments, id_to_vec)),
        )

    all_rows = _judgment_rows(judgments, id_to_vec)
    public_all = _public_rows(all_rows)
    # 去除内部字段前先保留一份带 _sim 的副本用于计算
    base = {
        "scope": scope,
        "margin_required": _r(margin),
        "judgments": public_all,
    }
    # 2) 当前全部判断是否已可区分
    ok, _, sep = _separates(all_rows, margin)
    if ok:
        cand = _candidate([], all_rows, [], sep)
        return {
            **base,
            "available": True,
            "feasible": True,
            "reason": None,
            "recommendation": "none",
            "already_separable": True,
            "unique": True,
            "boxes_to_recompute": [],
            "threshold": cand["threshold"],
            "threshold_range": cand["threshold_range"],
            "separation": cand["separation"],
            "min_same_similarity": cand["min_same_similarity"],
            "max_different_similarity": cand["max_different_similarity"],
            "candidate_sets": [cand],
            "message": (
                f"当前同只/异只判断的余弦相似度已可区分（最小同只 "
                f"{cand['min_same_similarity']:.4f}，最大异只 "
                f"{cand['max_different_similarity']:.4f}，间隔 "
                f"{cand['separation']:.4f} ≥ {margin:g}），无需重算检测框。"
            ),
        }

    # 3) 枚举最小“移除框”集合（移除后其相关判断一并排除）
    qualifying: list[tuple[tuple[str, ...], list[dict], list[dict], dict]] = []
    min_size = None
    for k in range(0, len(referenced) + 1):
        if min_size is not None and k > min_size:
            break
        for combo in itertools.combinations(referenced, k):
            removed = set(combo)
            retained = [r for r in all_rows if r["a"] not in removed and r["b"] not in removed]
            ok, _, sep2 = _separates(retained, margin)
            if not ok:
                continue
            retained_same = sum(1 for r in retained if r["type"] == "same")
            retained_diff = sum(1 for r in retained if r["type"] == "different")
            if retained_same < 1 or retained_diff < 1:
                continue
            dropped = [r for r in all_rows if r["a"] in removed or r["b"] in removed]
            if min_size is None:
                min_size = k
            if k == min_size:
                qualifying.append((combo, retained, dropped, sep2))

    if not qualifying:
        return _unavailable(
            "no_qualifying_set",
            "找不到能让同只与异只（各至少保留一条判断）按余弦相似度区分开的"
            "框集合，不提供重算建议。",
            scope=scope,
            judgments=public_all,
            base=base,
        )

    candidates = [
        _candidate(list(combo), retained, dropped, sep2)
        for combo, retained, dropped, sep2 in qualifying
    ]
    unique = len(candidates) == 1
    if not unique:
        boxes_txt = "；".join(
            "{" + "、".join(c["boxes_to_recompute"]) + "}" for c in candidates
        )
        return {
            **base,
            "available": True,
            "feasible": True,
            "reason": None,
            "recommendation": "ambiguous",
            "already_separable": False,
            "unique": False,
            "boxes_to_recompute": None,
            "threshold": None,
            "threshold_range": None,
            "separation": None,
            "candidate_sets": candidates,
            "message": (
                f"存在 {len(candidates)} 组同样最少（均需重算 {min_size} 个框）的框集合："
                f"{boxes_txt}。结果不唯一，不能据此断定哪个框的外观向量出错；"
                "请结合画面人工复核，或补充同只/异只判断后重试。"
            ),
        }

    cand = candidates[0]
    boxes = cand["boxes_to_recompute"]
    return {
        **base,
        "available": True,
        "feasible": True,
        "reason": None,
        "recommendation": "recompute",
        "already_separable": False,
        "unique": True,
        "boxes_to_recompute": boxes,
        "threshold": cand["threshold"],
        "threshold_range": cand["threshold_range"],
        "separation": cand["separation"],
        "min_same_similarity": cand["min_same_similarity"],
        "max_different_similarity": cand["max_different_similarity"],
        "candidate_sets": candidates,
        "message": (
            f"建议重新提取检测框 {'、'.join(boxes)} 的外观向量；"
            f"排除这些框相关的判断后，剩余同只/异只判断可用阈值 "
            f"{cand['threshold']:.4f}（区间 {cand['threshold_range'][0]:.4f}~"
            f"{cand['threshold_range'][1]:.4f}）区分，间隔 "
            f"{cand['separation']:.4f} ≥ {margin:g}。"
            "此为向量一致性提示，不代表已确认这些框的向量有误。"
        ),
    }
