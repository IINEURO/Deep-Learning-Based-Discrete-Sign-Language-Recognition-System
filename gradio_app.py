#!/usr/bin/env python3
"""
软件名称：基于深度学习的离散手语识别系统 V1.0
Software Name: Deep Learning Based Discrete Sign Language Recognition System V1.0
版本号：V1.0
"""
from __future__ import annotations

import argparse
import base64
import sys
import threading
import time
from pathlib import Path
from typing import Any

import gradio as gr

try:
    import cv2
    import numpy as np
    import torch
except ImportError as exc:
    raise SystemExit("Missing dependencies for gradio_app.py. Run: pip install -r requirements.txt") from exc

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from infer import ensure_ckpt_config, infer_sequence, load_vocab, pick_device, resolve_mediapipe_apis
from slr_baseline.keypoints import extract_frame_keypoints, load_frames_from_source
from slr_baseline.model import SignBiLSTMBaseline
from slr_baseline.utils import sample_frame_indices

_LOGO_CANDIDATES = [
    PROJECT_ROOT / "assets" / "logo_cutout.png",
    PROJECT_ROOT / "assets" / "logo.png",
    PROJECT_ROOT / "assets" / "LOGO.png",
]
LOGO_PATH = next((p for p in _LOGO_CANDIDATES if p.exists()), _LOGO_CANDIDATES[0])
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best.pt"
DEFAULT_VOCAB = PROJECT_ROOT / "meta" / "vocab_gloss.json"


class InferenceEngine:
    def __init__(self, checkpoint_path: str, vocab_path: str, device_arg: str = "auto") -> None:
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self.vocab_path = str(Path(vocab_path).resolve())
        self.device = pick_device(device_arg)

        ckpt_path = Path(self.checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        vocab_path_obj = Path(self.vocab_path)
        if not vocab_path_obj.exists():
            raise FileNotFoundError(f"Vocab not found: {vocab_path_obj}")

        ckpt = torch.load(ckpt_path, map_location=self.device)
        cfg = ensure_ckpt_config(ckpt)

        self.model_t = int(cfg["num_frames"])
        self.input_dim = int(cfg["input_dim"])
        self.use_normalize = bool(cfg.get("normalize", False))
        self.use_velocity = bool(cfg.get("use_velocity", False))
        self.vocab = load_vocab(vocab_path_obj)

        self.model = SignBiLSTMBaseline(
            num_classes=int(cfg["num_classes"]),
            input_dim=self.input_dim,
            proj_dim=int(cfg["proj_dim"]),
            hidden_size=int(cfg["hidden_size"]),
            num_layers=int(cfg["num_layers"]),
            dropout=float(cfg["dropout"]),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        mp_hands, mp_pose = resolve_mediapipe_apis()
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def close(self) -> None:
        try:
            self.hands.close()
        except Exception:
            pass
        try:
            self.pose.close()
        except Exception:
            pass

    def extract_frame_keypoints_from_rgb(self, frame_rgb: np.ndarray) -> np.ndarray:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return extract_frame_keypoints(frame_bgr, self.hands, self.pose)

    def predict_from_keypoints_clip(self, keypoints_t554: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        return infer_sequence(
            model=self.model,
            keypoints_t554=keypoints_t554,
            input_dim=self.input_dim,
            use_normalize=self.use_normalize,
            use_velocity=self.use_velocity,
            device=self.device,
            topk=int(top_k),
            vocab=self.vocab,
        )


_ENGINE_LOCK = threading.Lock()
_ENGINE_CACHE: dict[str, Any] = {
    "key": None,
    "engine": None,
}


def get_engine(checkpoint_path: str, vocab_path: str, device_arg: str) -> InferenceEngine:
    key = (str(Path(checkpoint_path).resolve()), str(Path(vocab_path).resolve()), device_arg)
    with _ENGINE_LOCK:
        if _ENGINE_CACHE["engine"] is not None and _ENGINE_CACHE["key"] == key:
            return _ENGINE_CACHE["engine"]

        old_engine = _ENGINE_CACHE["engine"]
        if old_engine is not None:
            old_engine.close()

        engine = InferenceEngine(*key)
        _ENGINE_CACHE["key"] = key
        _ENGINE_CACHE["engine"] = engine
        return engine


def _safe_video_path(video_input) -> str | None:
    if video_input is None:
        return None
    if isinstance(video_input, str):
        return video_input
    if isinstance(video_input, dict):
        return video_input.get("path") or video_input.get("name")
    return None


def _video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 25.0
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 1e-6:
        return 25.0
    return fps


def _timeline_rows(segments: list[dict[str, Any]]) -> list[list[Any]]:
    return [
        [
            idx + 1,
            f"{seg['start']:.2f}s - {seg['end']:.2f}s",
            seg["label"],
            f"{seg['score']:.4f}",
        ]
        for idx, seg in enumerate(segments)
    ]


def _aggregate_topk_from_segments(segments: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if not segments:
        return []
    bucket: dict[str, float] = {}
    for seg in segments:
        bucket[seg["label"]] = bucket.get(seg["label"], 0.0) + float(seg["score"])
    ranked = sorted(bucket.items(), key=lambda x: x[1], reverse=True)[: max(1, int(top_k))]
    total = sum(score for _, score in ranked) or 1.0
    return [{"rank": i + 1, "label": label, "score": score / total} for i, (label, score) in enumerate(ranked)]


def _logo_html(logo_path: Path, width_px: int = 80) -> str:
    if not logo_path.exists():
        return "<div style='font-size:48px;text-align:center;'>🤟</div>"
    data = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    return (
        "<div style='display:flex;align-items:center;justify-content:center;'>"
        f"<img src='data:image/png;base64,{data}' "
        f"style='width:{width_px}px;height:auto;display:block;border-radius:10px;' "
        "alt='logo' />"
        "</div>"
    )


def predict_image(image, top_k: int, checkpoint_path: str, vocab_path: str, device_arg: str):
    if image is None:
        return "未检测到输入", 0.0, [], "0.000 秒"

    t0 = time.perf_counter()
    try:
        engine = get_engine(checkpoint_path, vocab_path, device_arg)
        kpt = engine.extract_frame_keypoints_from_rgb(image)
        clip = np.repeat(kpt[None, ...], repeats=engine.model_t, axis=0).astype(np.float32)
        topk = engine.predict_from_keypoints_clip(clip, top_k=top_k)
    except Exception as exc:
        return f"推理失败: {exc}", 0.0, [], "0.000 秒"

    table = [[item["rank"], item["gloss"], f"{item['prob']:.4f}"] for item in topk]
    top1 = topk[0]
    elapsed = time.perf_counter() - t0
    return str(top1["gloss"]), float(top1["prob"]), table, f"{elapsed:.3f} 秒"


def predict_video(video, interval_sec: float, top_k: int, checkpoint_path: str, vocab_path: str, device_arg: str):
    video_path = _safe_video_path(video)
    if not video_path:
        return "未检测到输入", 0.0, [], "0.000 秒", "", []

    t0 = time.perf_counter()
    try:
        engine = get_engine(checkpoint_path, vocab_path, device_arg)
        frames = load_frames_from_source(video_path)  # BGR frames
        if not frames:
            raise RuntimeError("未能从视频读取帧")

        fps = _video_fps(video_path)
        interval = max(0.2, float(interval_sec))
        seg_len = max(1, int(round(interval * fps)))

        all_kpts = []
        for frame_bgr in frames:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            all_kpts.append(engine.extract_frame_keypoints_from_rgb(frame_rgb))
        kpt_arr = np.asarray(all_kpts, dtype=np.float32)

        segments: list[dict[str, Any]] = []
        for start_f in range(0, len(frames), seg_len):
            end_f = min(len(frames), start_f + seg_len)
            seg = kpt_arr[start_f:end_f]
            if seg.shape[0] <= 0:
                continue
            idx = sample_frame_indices(seg.shape[0], engine.model_t)
            clip = seg[np.asarray(idx, dtype=np.int32)]
            preds = engine.predict_from_keypoints_clip(clip, top_k=top_k)
            top1 = preds[0]
            segments.append(
                {
                    "start": start_f / fps,
                    "end": end_f / fps,
                    "label": str(top1["gloss"]),
                    "score": float(top1["prob"]),
                }
            )

        if not segments:
            return "未识别", 0.0, [], "0.000 秒", "", []

        agg_topk = _aggregate_topk_from_segments(segments, top_k=max(1, int(top_k)))
        top1 = agg_topk[0]
        topk_table = [[item["rank"], item["label"], f"{item['score']:.4f}"] for item in agg_topk]
        transcript = " ".join(seg["label"] for seg in segments)
        elapsed = time.perf_counter() - t0
        return top1["label"], float(top1["score"]), topk_table, f"{elapsed:.3f} 秒", transcript, _timeline_rows(segments)
    except Exception as exc:
        return f"推理失败: {exc}", 0.0, [], "0.000 秒", "", []


def _new_webcam_state(interval_sec: float, running: bool = True) -> dict[str, Any]:
    now = time.perf_counter()
    interval = max(0.2, float(interval_sec))
    return {
        "running": running,
        "start_ts": now,
        "next_emit_ts": now + interval,
        "interval_sec": interval,
        "kpts": [],
        "segments": [],
    }


def start_webcam_session(interval_sec: float):
    state = _new_webcam_state(interval_sec, running=True)
    return state, "会话状态：运行中", "", []


def stop_webcam_session(state: dict[str, Any] | None):
    if state is None:
        state = _new_webcam_state(1.0, running=False)
    state["running"] = False
    transcript = " ".join(seg["label"] for seg in state.get("segments", []))
    return state, "会话状态：已停止", transcript, _timeline_rows(state.get("segments", []))


def clear_webcam_session(interval_sec: float):
    state = _new_webcam_state(interval_sec, running=False)
    return state, "会话状态：已清空（未开始）", "等待输入", 0.0, [], "0.000 秒", "", []


def predict_webcam(
    frame,
    top_k: int,
    interval_sec: float,
    state: dict[str, Any] | None,
    checkpoint_path: str,
    vocab_path: str,
    device_arg: str,
):
    t0 = time.perf_counter()
    if state is None:
        state = _new_webcam_state(interval_sec, running=True)

    interval = max(0.2, float(interval_sec))
    state["interval_sec"] = interval

    if frame is None:
        status = "会话状态：运行中（等待摄像头画面）" if state.get("running") else "会话状态：已停止"
        transcript = " ".join(seg["label"] for seg in state.get("segments", []))
        return "等待输入", 0.0, [], f"{(time.perf_counter()-t0):.3f} 秒", transcript, _timeline_rows(state.get("segments", [])), state, status

    try:
        engine = get_engine(checkpoint_path, vocab_path, device_arg)
    except Exception as exc:
        return f"推理失败: {exc}", 0.0, [], "0.000 秒", "", [], state, "会话状态：错误"

    now = time.perf_counter()
    kpt = engine.extract_frame_keypoints_from_rgb(frame)
    state.setdefault("kpts", [])
    state["kpts"].append((now, kpt))
    keep_after = now - max(10.0, interval * 4.0)
    state["kpts"] = [(ts, x) for ts, x in state["kpts"] if ts >= keep_after]

    if state.get("running", False):
        while now >= float(state["next_emit_ts"]):
            seg_end = float(state["next_emit_ts"])
            seg_start = seg_end - interval
            seg_kpts = [x for ts, x in state["kpts"] if seg_start <= ts < seg_end]
            if not seg_kpts:
                seg_kpts = [state["kpts"][-1][1]]

            seg_arr = np.asarray(seg_kpts, dtype=np.float32)
            idx = sample_frame_indices(seg_arr.shape[0], engine.model_t)
            clip = seg_arr[np.asarray(idx, dtype=np.int32)]
            preds = engine.predict_from_keypoints_clip(clip, top_k=top_k)
            top1 = preds[0]
            state.setdefault("segments", [])
            rel_start = seg_start - float(state["start_ts"])
            rel_end = seg_end - float(state["start_ts"])
            state["segments"].append(
                {
                    "start": rel_start,
                    "end": rel_end,
                    "label": str(top1["gloss"]),
                    "score": float(top1["prob"]),
                }
            )
            state["next_emit_ts"] = seg_end + interval

    # current-frame preview prediction
    clip_now = np.repeat(kpt[None, ...], repeats=engine.model_t, axis=0).astype(np.float32)
    preds_now = engine.predict_from_keypoints_clip(clip_now, top_k=top_k)
    top1_now = preds_now[0]
    topk_table = [[item["rank"], item["gloss"], f"{item['prob']:.4f}"] for item in preds_now]

    for seg in state.get("segments", []):
        seg["start"] = max(0.0, float(seg["start"]))
        seg["end"] = max(0.0, float(seg["end"]))

    transcript = " ".join(seg["label"] for seg in state.get("segments", []))
    elapsed = time.perf_counter() - t0
    status = f"会话状态：{'运行中' if state.get('running') else '已停止'}，已输出片段 {len(state.get('segments', []))} 个"
    return (
        str(top1_now["gloss"]),
        float(top1_now["prob"]),
        topk_table,
        f"{elapsed:.3f} 秒",
        transcript,
        _timeline_rows(state.get("segments", [])),
        state,
        status,
    )


CSS = """
.header-wrap {
  border: 1px solid #6b3ef2;
  border-radius: 14px;
  padding: 14px 16px;
  background: linear-gradient(135deg, #5f2cff 0%, #8a4dff 45%, #b67bff 100%);
  margin-bottom: 10px;
}
.card {
  border: 1px solid #e6eaf2;
  border-radius: 14px;
  padding: 12px;
  background: #ffffff;
}
#logo {
  border-radius: 10px;
  overflow: hidden;
}
.header-title {
  color: #ffffff;
}
.header-title h1 {
  margin: 0;
  line-height: 1.2;
  font-size: 2rem;
}
.header-title .sub {
  margin-top: 8px;
  font-size: 1rem;
  opacity: 0.95;
}
.header-title .en {
  margin-top: 6px;
  font-size: 0.95rem;
  font-style: italic;
  opacity: 0.92;
}
"""


with gr.Blocks(title="基于深度学习的离散手语识别系统 V1.0") as demo:
    with gr.Row(elem_classes=["header-wrap"], equal_height=True):
        with gr.Column(scale=1, min_width=90):
            if LOGO_PATH.exists():
                gr.HTML(_logo_html(LOGO_PATH, width_px=80), elem_id="logo")
            else:
                gr.HTML("<div style='font-size:48px;text-align:center;'>🤟</div>")
        with gr.Column(scale=8):
            gr.HTML(
                """
                <div class="header-title">
                  <h1>基于深度学习的离散手语识别系统 V1.0</h1>
                  <div class="sub">支持图片识别、视频识别和实时摄像头识别</div>
                  <div class="en">Deep Learning Based Discrete Sign Language Recognition System V1.0</div>
                </div>
                """
            )

    with gr.Accordion("模型配置", open=False):
        ckpt_in = gr.Textbox(label="Checkpoint 路径", value=str(DEFAULT_CHECKPOINT))
        vocab_in = gr.Textbox(label="Vocab 路径", value=str(DEFAULT_VOCAB))
        device_in = gr.Dropdown(choices=["auto", "cpu", "cuda"], value="auto", label="推理设备")

    with gr.Tabs():
        with gr.Tab("图片识别"):
            with gr.Row():
                with gr.Column(scale=5):
                    with gr.Group(elem_classes=["card"]):
                        image_input = gr.Image(label="上传图片", type="numpy")
                        image_topk = gr.Slider(1, 10, value=5, step=1, label="Top-K")
                        image_btn = gr.Button("开始识别", variant="primary")
                with gr.Column(scale=5):
                    with gr.Group(elem_classes=["card"]):
                        image_top1 = gr.Textbox(label="Top1 类别", interactive=False)
                        image_conf = gr.Number(label="置信度", precision=4, interactive=False)
                        image_topk_list = gr.Dataframe(
                            headers=["排名", "类别", "置信度"],
                            datatype=["number", "str", "str"],
                            row_count=5,
                            column_count=(3, "fixed"),
                            interactive=False,
                            label="Top-K 列表",
                        )
                        image_time = gr.Textbox(label="推理耗时", interactive=False)

            image_btn.click(
                fn=predict_image,
                inputs=[image_input, image_topk, ckpt_in, vocab_in, device_in],
                outputs=[image_top1, image_conf, image_topk_list, image_time],
            )

        with gr.Tab("视频识别"):
            with gr.Row():
                with gr.Column(scale=5):
                    with gr.Group(elem_classes=["card"]):
                        video_input = gr.Video(label="上传视频")
                        video_interval = gr.Slider(0.2, 3.0, value=1.0, step=0.1, label="时间间隔（秒）")
                        video_topk = gr.Slider(1, 10, value=5, step=1, label="Top-K")
                        video_btn = gr.Button("开始识别", variant="primary")
                with gr.Column(scale=5):
                    with gr.Group(elem_classes=["card"]):
                        video_top1 = gr.Textbox(label="Top1 类别", interactive=False)
                        video_conf = gr.Number(label="置信度", precision=4, interactive=False)
                        video_topk_list = gr.Dataframe(
                            headers=["排名", "类别", "置信度"],
                            datatype=["number", "str", "str"],
                            row_count=5,
                            column_count=(3, "fixed"),
                            interactive=False,
                            label="Top-K 列表",
                        )
                        video_time = gr.Textbox(label="推理耗时", interactive=False)
                        video_transcript = gr.Textbox(label="时序识别文本（按时间片段排列）", interactive=False)
                        video_timeline = gr.Dataframe(
                            headers=["序号", "时间段", "字符", "置信度"],
                            datatype=["number", "str", "str", "str"],
                            row_count=8,
                            column_count=(4, "fixed"),
                            interactive=False,
                            label="视频时间片段结果",
                        )

            video_btn.click(
                fn=predict_video,
                inputs=[video_input, video_interval, video_topk, ckpt_in, vocab_in, device_in],
                outputs=[video_top1, video_conf, video_topk_list, video_time, video_transcript, video_timeline],
            )

        with gr.Tab("实时摄像头"):
            with gr.Row():
                with gr.Column(scale=5):
                    with gr.Group(elem_classes=["card"]):
                        webcam_input = gr.Image(
                            label="摄像头输入（请先允许浏览器调用摄像头）",
                            sources=["webcam"],
                            type="numpy",
                        )
                        webcam_topk = gr.Slider(1, 10, value=5, step=1, label="Top-K")
                        webcam_interval = gr.Slider(0.2, 3.0, value=1.0, step=0.1, label="输出时间间隔（秒）")
                        with gr.Row():
                            webcam_start_btn = gr.Button("开始会话", variant="primary")
                            webcam_stop_btn = gr.Button("停止会话")
                            webcam_clear_btn = gr.Button("清空会话")
                        webcam_btn = gr.Button("手动识别当前帧")
                        gr.Markdown("会按设定时间间隔持续输出字符并累计时间片段；如果自动更新不稳定，可点击手动识别。")
                with gr.Column(scale=5):
                    with gr.Group(elem_classes=["card"]):
                        webcam_status = gr.Textbox(label="会话状态", interactive=False, value="会话状态：未开始")
                        webcam_top1 = gr.Textbox(label="Top1 类别", interactive=False)
                        webcam_conf = gr.Number(label="置信度", precision=4, interactive=False)
                        webcam_topk_list = gr.Dataframe(
                            headers=["排名", "类别", "置信度"],
                            datatype=["number", "str", "str"],
                            row_count=5,
                            column_count=(3, "fixed"),
                            interactive=False,
                            label="Top-K 列表",
                        )
                        webcam_time = gr.Textbox(label="推理耗时", interactive=False)
                        webcam_transcript = gr.Textbox(label="时序识别文本（累计）", interactive=False)
                        webcam_timeline = gr.Dataframe(
                            headers=["序号", "时间段", "字符", "置信度"],
                            datatype=["number", "str", "str", "str"],
                            row_count=10,
                            column_count=(4, "fixed"),
                            interactive=False,
                            label="摄像头时间片段结果",
                        )

            webcam_state = gr.State(_new_webcam_state(1.0, running=False))

            webcam_start_btn.click(
                fn=start_webcam_session,
                inputs=[webcam_interval],
                outputs=[webcam_state, webcam_status, webcam_transcript, webcam_timeline],
            )
            webcam_stop_btn.click(
                fn=stop_webcam_session,
                inputs=[webcam_state],
                outputs=[webcam_state, webcam_status, webcam_transcript, webcam_timeline],
            )
            webcam_clear_btn.click(
                fn=clear_webcam_session,
                inputs=[webcam_interval],
                outputs=[
                    webcam_state,
                    webcam_status,
                    webcam_top1,
                    webcam_conf,
                    webcam_topk_list,
                    webcam_time,
                    webcam_transcript,
                    webcam_timeline,
                ],
            )

            webcam_event = dict(
                fn=predict_webcam,
                inputs=[webcam_input, webcam_topk, webcam_interval, webcam_state, ckpt_in, vocab_in, device_in],
                outputs=[
                    webcam_top1,
                    webcam_conf,
                    webcam_topk_list,
                    webcam_time,
                    webcam_transcript,
                    webcam_timeline,
                    webcam_state,
                    webcam_status,
                ],
            )
            if hasattr(webcam_input, "stream"):
                webcam_input.stream(**webcam_event)
            webcam_input.change(**webcam_event)
            webcam_btn.click(**webcam_event)

    gr.Markdown("注：当前为“基于深度学习的离散手语识别系统”V1.0 实际推理界面，调用 checkpoint + vocab 进行预测。")

demo.queue(default_concurrency_limit=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于深度学习的离散手语识别系统 V1.0 - Gradio UI")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="启用公网 HTTPS 分享链接（远程访问摄像头推荐）")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=not args.no_browser,
        share=args.share,
        theme=gr.themes.Soft(),
        css=CSS,
    )
