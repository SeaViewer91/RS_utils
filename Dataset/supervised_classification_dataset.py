#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FG/BG 인터랙티브 샘플링 → 감독분류(SVM/RF/MLP) → 모폴로지 후처리 → 정교한 외곽 폴리곤 → YOLO-seg Export
+ (2025-10-31) 업데이트
  1) Dataset Split 버튼 추가
  2) Model Evaluation 버튼 추가
  3) Image List 버튼 추가
+ (2025-11-26) 업데이트
  4) BBox 샘플링 모드 추가 (드래그로 박스 그려 내부 픽셀 일괄 추가)
  5) Option에 BBox Subsampling(0~1) / All Pixels 선택 추가 (기본 0.3)
  6) 그린 BBox를 프리뷰에 유지 (FG=시안, BG=마젠타)
  7) "Clear BBoxes (this image)" 버튼으로 현재 이미지의 BBox 프리뷰 초기화
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib

# ----- PyTorch (MLP) -----
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
except Exception:
    TORCH_OK = False  # 설치 안됨 시 MLP 비활성 안내

# ---------------- 기본 설정 ----------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CANVAS_W, CANVAS_H = 1280, 800
CLASS_ID = 0  # YOLO-seg 클래스 ID (단일 클래스 가정)

# 모폴로지 기본값 (Option에서 변경 가능)
DEFAULT_KERNEL_SIZE = 3   # 3x3
DEFAULT_OPEN_ITER   = 1
DEFAULT_CLOSE_ITER  = 1

# 컨투어(정교화) 기본값 (Option에서 변경 가능)
DEFAULT_CHAIN_APPROX = "High precision (CHAIN_APPROX_NONE)"
DEFAULT_EPSILON_PX   = 0.5
DEFAULT_CNT_BLUR_ENABLE = True
DEFAULT_CNT_BLUR_METHOD = "Gaussian"   # 또는 "Median"
DEFAULT_CNT_BLUR_KSIZE  = 3            # 홀수

# MLP 기본 (Option에서 변경 가능)
DEFAULT_MLP_LAYERS = 3
DEFAULT_MLP_NEURONS_TEXT = "10,10,10"
DEFAULT_MLP_USE_DROPOUT = True
DEFAULT_MLP_DROPOUT_RATE = 0.05

# BBox 기본 (Option에서 변경 가능)
DEFAULT_BBOX_SUBSAMPLE_ENABLE = True
DEFAULT_BBOX_SUBSAMPLE_RATIO = 0.3  # 0~1

# --------------- 유틸 ---------------
def imread_unicode(path: Path, flags=cv2.IMREAD_COLOR):
    if not path.exists():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, flags)

def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    ok, buf = cv2.imencode(ext if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"] else ".png", image)
    if not ok:
        return False
    buf.tofile(str(path))
    return True

def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def extract_features_image(image_bgr: np.ndarray) -> np.ndarray:
    """전체 픽셀 특징: BGR + HSV + Lab + 정규화 좌표(x,y) => (H*W, 11)"""
    H, W = image_bgr.shape[:2]
    bgr = image_bgr.reshape(-1, 3).astype(np.float32)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    xn = (xx.astype(np.float32) / max(1, W-1)).reshape(-1, 1)
    yn = (yy.astype(np.float32) / max(1, H-1)).reshape(-1, 1)
    return np.concatenate([bgr, hsv, lab, xn, yn], axis=1)

def feature_at_xy(image_bgr: np.ndarray, x: int, y: int) -> np.ndarray:
    """한 픽셀 특징 (1, 11)"""
    H, W = image_bgr.shape[:2]
    if x < 0 or x >= W or y < 0 or y >= H:
        return np.zeros((1, 11), np.float32)
    b, g, r = image_bgr[y, x].astype(np.float32)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)
    hh, ss, vv = hsv[y, x].astype(np.float32)
    ll, aa, bb = lab[y, x].astype(np.float32)
    xn = float(x) / max(1, W-1)
    yn = float(y) / max(1, H-1)
    return np.array([[b, g, r, hh, ss, vv, ll, aa, bb, xn, yn]], dtype=np.float32)

def hsv_lab_from_rgb(r: int, g: int, b: int):
    """CSV 로드 시 RGB만 있을 때 HSV/Lab를 복원"""
    px = np.array([[[b, g, r]]], dtype=np.uint8)  # OpenCV는 BGR
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV).astype(np.float32)[0, 0]
    lab = cv2.cvtColor(px, cv2.COLOR_BGR2Lab).astype(np.float32)[0, 0]
    return hsv, lab

def postprocess_mask(mask01: np.ndarray, ksize: int, open_iter: int, close_iter: int) -> np.ndarray:
    """이진(0/1) 마스크에 모폴로지(열림→닫힘) 적용"""
    u8 = (mask01 > 0).astype(np.uint8) * 255
    ksize = max(3, int(ksize))
    if ksize % 2 == 0: ksize += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    if open_iter > 0:
        u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, kernel, iterations=int(open_iter))
    if close_iter > 0:
        u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kernel, iterations=int(close_iter))
    return (u8 > 0).astype(np.uint8)

def find_outer_contours(mask01: np.ndarray,
                        epsilon_px: float = DEFAULT_EPSILON_PX,
                        chain_approx_mode: str = DEFAULT_CHAIN_APPROX,
                        blur_enable: bool = DEFAULT_CNT_BLUR_ENABLE,
                        blur_method: str = DEFAULT_CNT_BLUR_METHOD,
                        blur_ksize: int = DEFAULT_CNT_BLUR_KSIZE) -> List[np.ndarray]:
    """
    정교한 외곽 컨투어 추출
    """
    u8 = (mask01 > 0).astype(np.uint8) * 255
    if blur_enable:
        k = max(3, int(blur_ksize))
        if k % 2 == 0: k += 1
        if blur_method.lower().startswith("median"):
            u8 = cv2.medianBlur(u8, k)
        else:
            u8 = cv2.GaussianBlur(u8, (k, k), 0)
        _, u8 = cv2.threshold(u8, 127, 255, cv2.THRESH_BINARY)

    chain_flag = cv2.CHAIN_APPROX_NONE if chain_approx_mode.startswith("High") else cv2.CHAIN_APPROX_SIMPLE
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, chain_flag)

    outs = []
    for c in contours:
        if len(c) < 3:
            continue
        if epsilon_px and epsilon_px > 0:
            approx = cv2.approxPolyDP(c, float(epsilon_px), True)
        else:
            approx = c
        if len(approx) >= 3:
            outs.append(approx)
    return outs

def contours_to_yoloseg(contours: List[np.ndarray], W: int, H: int, class_id: int) -> List[str]:
    lines = []
    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        if pts.shape[0] < 3:
            continue
        xs = pts[:, 0] / float(W)
        ys = pts[:, 1] / float(H)
        coords = []
        for x, y in zip(xs, ys):
            coords.append(f"{x:.6f}"); coords.append(f"{y:.6f}")
        lines.append(f"{class_id} " + " ".join(coords))
    return lines

# ----- PyTorch MLP 정의 -----
class MLPNet(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: List[int], num_classes: int = 2,
                 use_dropout: bool = True, dropout_rate: float = 0.05):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.ReLU(inplace=True))
            if use_dropout and dropout_rate > 0:
                layers.append(nn.Dropout(p=float(dropout_rate)))
            prev = int(h)
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# --------------- 앱 ---------------
class App:
    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("FG/BG Interactive Seg → YOLO-seg")

        # 경로/이미지 상태
        self.image_dir: Optional[Path] = None
        self.output_dir: Optional[Path] = None
        self.image_paths: List[Path] = []
        self.idx = 0
        self.img_bgr: Optional[np.ndarray] = None

        # 분류/표시 상태
        self.last_mask: Optional[np.ndarray] = None
        self.last_contours: List[np.ndarray] = []
        self.overlay_transp = tk.DoubleVar(value=10.0)  # 10~100 (%)

        # 샘플/학습 데이터
        self.curr_fg: List[Tuple[int, int]] = []
        self.curr_bg: List[Tuple[int, int]] = []
        self.current_label = tk.StringVar(value="FG")

        self.X_list: List[np.ndarray] = []
        self.y_list: List[int] = []
        self.RGB_list: List[Tuple[int, int, int]] = []
        self.Cls_list: List[str] = []

        # Train/Valid/Test 역할 관리
        self.data_roles: List[str] = []
        self.random_split: bool = False

        # 모델 (sklearn/torch)
        self.model: Optional[Pipeline] = None
        self.torch_model: Optional[nn.Module] = None
        self.torch_scaler: Optional[StandardScaler] = None
        self.is_trained: bool = False
        self.last_train_date: Optional[str] = None
        self.model_type_var = tk.StringVar(value="SVM - rbf")
        self.epochs_var = tk.IntVar(value=100)

        # 모폴로지 옵션
        self.morph_kernel_size = DEFAULT_KERNEL_SIZE
        self.morph_open_iter = DEFAULT_OPEN_ITER
        self.morph_close_iter = DEFAULT_CLOSE_ITER

        # 컨투어 옵션
        self.chain_approx_mode = tk.StringVar(value=DEFAULT_CHAIN_APPROX)
        self.poly_epsilon_px = tk.DoubleVar(value=DEFAULT_EPSILON_PX)
        self.cnt_blur_enable = tk.BooleanVar(value=DEFAULT_CNT_BLUR_ENABLE)
        self.cnt_blur_method = tk.StringVar(value=DEFAULT_CNT_BLUR_METHOD)
        self.cnt_blur_ksize = tk.IntVar(value=DEFAULT_CNT_BLUR_KSIZE)

        # MLP 옵션
        self.mlp_layers_var = tk.IntVar(value=DEFAULT_MLP_LAYERS)
        self.mlp_neurons_text = tk.StringVar(value=DEFAULT_MLP_NEURONS_TEXT)
        self.mlp_use_dropout = tk.BooleanVar(value=DEFAULT_MLP_USE_DROPOUT)
        self.mlp_dropout_rate = tk.DoubleVar(value=DEFAULT_MLP_DROPOUT_RATE)

        # 캔버스/렌더링(줌/팬)
        self.canvas = None
        self.tk_img = None
        self.tk_overlay = None
        self.base_scale = 1.0
        self.zoom = 1.0
        self.center_xy = (0.5, 0.5)

        # ------ BBox 상태 ------
        self.bbox_mode = tk.BooleanVar(value=False)     # BBox 모드 토글
        self.bbox_dragging = False
        self.bbox_start = (0, 0)                        # 이미지 좌표
        self.bbox_end = (0, 0)                          # 이미지 좌표
        self.bbox_history: List[Tuple[int, int, int, int, str]] = []  # (x1,y1,x2,y2, "FG"/"BG")
        self.bbox_subsample_enable = tk.BooleanVar(value=DEFAULT_BBOX_SUBSAMPLE_ENABLE)
        self.bbox_subsample_ratio = tk.DoubleVar(value=DEFAULT_BBOX_SUBSAMPLE_RATIO)

        self._build_ui()
        self._ask_and_load_dir(initial=True)

    # ---------- UI ----------
    def _build_ui(self):
        top = ttk.Frame(self.master); top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        ttk.Button(top, text="Load Images", command=self._ask_and_load_dir).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Prev", command=self.on_prev).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Next", command=self.on_next).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Image List", command=self.on_image_list).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Initialization", command=self.on_init_dataset).pack(side=tk.LEFT, padx=8)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Label(top, text="Model:").pack(side=tk.LEFT)
        models = ["SVM - linear", "SVM - rbf", "SVM - poly", "RF", "MLP (PyTorch)"]
        ttk.Combobox(top, textvariable=self.model_type_var, state="readonly",
                     values=models, width=18).pack(side=tk.LEFT, padx=4)
        ttk.Label(top, text="Epochs:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Spinbox(top, from_=1, to=5000, textvariable=self.epochs_var, width=6).pack(side=tk.LEFT, padx=4)

        ttk.Button(top, text="Train", command=self.on_train).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Classify", command=self.on_classify).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Export YOLO-seg", command=self.on_export_yolo).pack(side=tk.LEFT, padx=6)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(top, text="Save Weights", command=self.on_save_weights).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Load Weights", command=self.on_load_weights).pack(side=tk.LEFT, padx=2)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(top, text="Export Dataset", command=self.on_export_dataset).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Dataset Load", command=self.on_dataset_load).pack(side=tk.LEFT, padx=2)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(top, text="Dataset Split", command=self.on_dataset_split).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Model Evaluation", command=self.on_model_evaluation).pack(side=tk.LEFT, padx=2)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(top, text="Option", command=self.on_option).pack(side=tk.LEFT, padx=2)

        mid = ttk.Frame(self.master); mid.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 4))
        self.lbl_info = ttk.Label(mid, text="Path: -"); self.lbl_info.pack(side=tk.LEFT, padx=4)
        self.lbl_counts = ttk.Label(mid, text="FG_curr:0 BG_curr:0 | X_total:0"); self.lbl_counts.pack(side=tk.RIGHT, padx=4)

        lab = ttk.Frame(self.master); lab.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 6))
        ttk.Label(lab, text="Current Label:").pack(side=tk.LEFT)
        ttk.Radiobutton(lab, text="FG(대상)", variable=self.current_label, value="FG").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(lab, text="BG(배경)", variable=self.current_label, value="BG").pack(side=tk.LEFT, padx=2)

        # ---- BBox 모드 토글 + 현재 이미지용 초기화 버튼 ----
        ttk.Checkbutton(lab, text="BBox Mode (drag)", variable=self.bbox_mode).pack(side=tk.LEFT, padx=10)
        ttk.Button(lab, text="Clear BBoxes (this image)", command=self.on_clear_bboxes_this_image).pack(side=tk.LEFT, padx=4)

        ttk.Button(lab, text="Undo last", command=self.on_undo).pack(side=tk.LEFT, padx=10)
        ttk.Button(lab, text="Clear points (this image)", command=self.on_clear_current_points).pack(side=tk.LEFT, padx=2)

        ol = ttk.Frame(self.master); ol.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 6))
        ttk.Label(ol, text="Overlay Transparency (%)").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Scale(ol, from_=10.0, to=100.0, variable=self.overlay_transp, command=lambda e: self._redraw()
                 ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(ol, text="Clear Cls. Results", command=self.on_clear_cls).pack(side=tk.LEFT, padx=8)

        self.canvas = tk.Canvas(self.master, bg="#222222", width=CANVAS_W, height=CANVAS_H)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 마우스/키 바인딩
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<ButtonRelease-1>", self.on_b1_up)
        self.canvas.bind("<B1-Motion>", self.on_b1_motion)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.master.bind("f", lambda e: self.current_label.set("FG"))
        self.master.bind("b", lambda e: self.current_label.set("BG"))
        # 휠(Windows/Mac)
        self.master.bind("<MouseWheel>", self.on_wheel_zoom)
        # 리눅스
        self.master.bind("<Button-4>", self.on_wheel_zoom_linux)
        self.master.bind("<Button-5>", self.on_wheel_zoom_linux)

    # ---------- 디렉토리/이미지 ----------
    def _ask_and_load_dir(self, initial=False):
        d = filedialog.askdirectory(title="이미지 폴더를 선택하세요")
        if not d:
            if initial: return
            return
        self.image_dir = Path(d)
        self.output_dir = self.image_dir / "yolo-labels"
        self.image_paths = self._gather_images(self.image_dir)
        self.idx = 0
        if not self.image_paths:
            messagebox.showwarning("알림", "선택 폴더에 이미지가 없습니다.")
        self._load_image(self.idx)

    def _gather_images(self, directory: Path) -> List[Path]:
        return [p for p in sorted(directory.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS]

    def _load_image(self, idx: int):
        if not self.image_paths:
            self.img_bgr = None
            self._redraw()
            return
        self.idx = max(0, min(idx, len(self.image_paths)-1))
        p = self.image_paths[self.idx]
        self.lbl_info.config(text=f"[{self.idx+1}/{len(self.image_paths)}] {p.name}")

        img = imread_unicode(p, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("오류", f"이미지 로딩 실패: {p}")
            return
        self.img_bgr = img
        self.last_mask = None; self.last_contours = []
        self.curr_fg.clear(); self.curr_bg.clear()
        self.bbox_history.clear()  # 이미지 변경 시 BBox 프리뷰 초기화
        self.bbox_dragging = False
        self._update_counts()

        # 보기 초기화
        H, W = img.shape[:2]
        self.base_scale = min(CANVAS_W / W, CANVAS_H / H)
        self.zoom = 1.0
        self.center_xy = (0.5, 0.5)
        self._redraw()

    # (신규) 이미지 리스트 팝업
    def on_image_list(self):
        if not self.image_paths:
            messagebox.showwarning("알림", "먼저 Load Images로 폴더를 열어주세요.")
            return

        win = tk.Toplevel(self.master)
        win.title("Image List")
        win.geometry("400x400")
        win.transient(self.master)
        win.grab_set()

        frm = ttk.Frame(win)
        frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        lb = tk.Listbox(frm, selectmode=tk.SINGLE)
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sb = ttk.Scrollbar(frm, orient=tk.VERTICAL, command=lb.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        lb.config(yscrollcommand=sb.set)

        for i, p in enumerate(self.image_paths, start=1):
            mark = "  "
            if (i-1) == self.idx:
                mark = "▶ "
            lb.insert(tk.END, f"{mark}{i:03d}  {p.name}")

        lb.selection_set(self.idx)
        lb.see(self.idx)

        def on_select(event=None):
            sel = lb.curselection()
            if not sel:
                return
            new_idx = sel[0]
            self._load_image(new_idx)
            win.destroy()

        lb.bind("<Double-Button-1>", on_select)

        btns = ttk.Frame(win)
        btns.pack(fill=tk.X, pady=(6,0))
        ttk.Button(btns, text="Open", command=on_select).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Close", command=win.destroy).pack(side=tk.RIGHT, padx=4)

    # ---------- 좌표 변환/그리기 ----------
    def _image_to_canvas(self, x_img: float, y_img: float):
        if self.img_bgr is None: return 0, 0
        H, W = self.img_bgr.shape[:2]
        disp_scale = self.base_scale * self.zoom
        cx = int(self.center_xy[0] * W * disp_scale)
        cy = int(self.center_xy[1] * H * disp_scale)
        top_left_x = int(-cx + CANVAS_W // 2)
        top_left_y = int(-cy + CANVAS_H // 2)
        return int(x_img * disp_scale + top_left_x), int(y_img * disp_scale + top_left_y)

    def _canvas_to_image(self, x_can: int, y_can: int):
        if self.img_bgr is None: return None, None
        H, W = self.img_bgr.shape[:2]
        disp_scale = self.base_scale * self.zoom
        cx = int(self.center_xy[0] * W * disp_scale)
        cy = int(self.center_xy[1] * H * disp_scale)
        top_left_x = int(-cx + CANVAS_W // 2)
        top_left_y = int(-cy + CANVAS_H // 2)
        xi = int(round((x_can - top_left_x) / disp_scale))
        yi = int(round((y_can - top_left_y) / disp_scale))
        if xi < 0 or xi >= W or yi < 0 or yi >= H: return None, None
        return xi, yi

    def _redraw(self):
        self.canvas.delete("all")
        if self.img_bgr is None: return

        H, W = self.img_bgr.shape[:2]
        disp_scale = self.base_scale * self.zoom
        resized = cv2.resize(self.img_bgr, (max(1, int(W*disp_scale)), max(1, int(H*disp_scale))), interpolation=cv2.INTER_AREA)
        self.tk_img = ImageTk.PhotoImage(bgr_to_pil(resized))

        cx = int(self.center_xy[0] * W * disp_scale)
        cy = int(self.center_xy[1] * H * disp_scale)
        top_left_x = int(-cx + CANVAS_W // 2)
        top_left_y = int(-cy + CANVAS_H // 2)
        self.canvas.create_image(top_left_x, top_left_y, image=self.tk_img, anchor=tk.NW)

        # ----- 과거 BBox 프리뷰(테두리) -----
        for (x1, y1, x2, y2, cls_str) in self.bbox_history:
            x1c, y1c = self._image_to_canvas(x1, y1)
            x2c, y2c = self._image_to_canvas(x2, y2)
            color = "#00FFFF" if cls_str == "FG" else "#FF00FF"
            self.canvas.create_rectangle(x1c, y1c, x2c, y2c, outline=color, width=2)

        # ----- 드래그 중인 임시 BBox -----
        if self.bbox_dragging:
            x1, y1 = self.bbox_start
            x2, y2 = self.bbox_end
            x1c, y1c = self._image_to_canvas(x1, y1)
            x2c, y2c = self._image_to_canvas(x2, y2)
            color = "#00FFFF" if self.current_label.get() == "FG" else "#FF00FF"
            self.canvas.create_rectangle(x1c, y1c, x2c, y2c, outline=color, width=2, dash=(4,2))

        # ----- 분류 오버레이 -----
        if self.last_mask is not None:
            transp = float(self.overlay_transp.get())
            alpha = 1.0 - max(10.0, min(100.0, transp)) / 100.0
            overlay = self._make_overlay_with_contours(self.img_bgr, self.last_mask, alpha, self.last_contours)
            overlay = cv2.resize(overlay, (resized.shape[1], resized.shape[0]), interpolation=cv2.INTER_NEAREST)
            self.tk_overlay = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(top_left_x, top_left_y, image=self.tk_overlay, anchor=tk.NW)

        # 포인트 시각화
        for (x, y) in self.curr_fg:
            self._draw_dot(x, y, "#00FF00")
        for (x, y) in self.curr_bg:
            self._draw_dot(x, y, "#FF5555")

    def _draw_dot(self, x: int, y: int, color: str):
        cx, cy = self._image_to_canvas(x, y)
        r = 3
        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill=color, outline=color)

    def _make_overlay_with_contours(self, bgr: np.ndarray, mask01: np.ndarray, alpha: float,
                                    contours: List[np.ndarray]) -> np.ndarray:
        base = bgr.copy()
        green = np.zeros_like(base); green[:] = (0, 255, 0)
        m3 = (mask01 > 0).astype(np.float32)[..., None]
        out = (base * (1 - alpha * m3) + green * (alpha * m3)).astype(np.uint8)
        if contours:
            cv2.drawContours(out, contours, -1, (0, 0, 255), 1)
        return out

    # ---------- 이벤트 ----------
    def on_left_click(self, e):
        # Ctrl+좌클릭 → 팬
        if (e.state & 0x0004) != 0:
            xi, yi = self._canvas_to_image(e.x, e.y)
            if xi is None: return
            H, W = self.img_bgr.shape[:2]
            self.center_xy = (xi / float(W), yi / float(H))
            self._redraw()
            return

        if self.img_bgr is None: return
        xi, yi = self._canvas_to_image(e.x, e.y)
        if xi is None: return

        # BBox 모드: 드래그 시작
        if self.bbox_mode.get():
            self.bbox_dragging = True
            self.bbox_start = (xi, yi)
            self.bbox_end = (xi, yi)
            self._redraw()
            return

        # 포인트 샘플 모드
        if self.current_label.get() == "FG":
            self.curr_fg.append((xi, yi)); ylab = 1; cls_str = "FG"
        else:
            self.curr_bg.append((xi, yi)); ylab = 0; cls_str = "BG"

        feat = feature_at_xy(self.img_bgr, xi, yi)  # (1,11)
        self.X_list.append(feat[0].copy()); self.y_list.append(ylab)
        b, g, r = self.img_bgr[yi, xi].tolist()
        self.RGB_list.append((int(r), int(g), int(b)))
        self.Cls_list.append(cls_str)
        self.data_roles.append(self._assign_role_for_index(len(self.X_list)-1))
        self._update_counts(); self._redraw()

    def on_b1_motion(self, e):
        if not self.bbox_mode.get() or not self.bbox_dragging:
            return
        xi, yi = self._canvas_to_image(e.x, e.y)
        if xi is None: return
        self.bbox_end = (xi, yi)
        self._redraw()

    def on_b1_up(self, e):
        if not self.bbox_mode.get() or not self.bbox_dragging:
            return
        self.bbox_dragging = False
        x2, y2 = self._canvas_to_image(e.x, e.y)
        if x2 is None:
            self._redraw()
            return
        x1, y1 = self.bbox_start
        x1, x2 = sorted([x1, x2]); y1, y2 = sorted([y1, y2])
        if x2 - x1 < 1 or y2 - y1 < 1:
            self._redraw()
            return

        # 박스 내부 픽셀 좌표
        xs = np.arange(x1, x2+1)
        ys = np.arange(y1, y2+1)
        xx, yy = np.meshgrid(xs, ys)
        coords = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (N,2)

        # Subsample 옵션
        if self.bbox_subsample_enable.get():
            ratio = max(0.0, min(1.0, float(self.bbox_subsample_ratio.get())))
            if ratio < 1.0:
                n = coords.shape[0]
                k = int(round(n * ratio))
                if k < 1:
                    k = 1
                idxs = np.random.choice(n, size=k, replace=False)
                coords = coords[idxs]

        # 특징 생성
        feats = []
        rgbs = []
        for (x, y) in coords:
            feats.append(feature_at_xy(self.img_bgr, int(x), int(y))[0])
            b, g, r = self.img_bgr[int(y), int(x)].tolist()
            rgbs.append([int(r), int(g), int(b)])
        feats = np.asarray(feats, dtype=np.float32)
        rgbs = np.asarray(rgbs, dtype=np.int32)

        if feats.size == 0:
            self._redraw()
            return

        cls_str = self.current_label.get()
        ylab = 1 if cls_str == "FG" else 0

        n_add = feats.shape[0]
        for i in range(n_add):
            self.X_list.append(feats[i])
            self.y_list.append(ylab)
            self.RGB_list.append(tuple(rgbs[i]))
            self.Cls_list.append(cls_str)
            self.data_roles.append(self._assign_role_for_index(len(self.X_list)-1))

        # 히스토리 기록(프리뷰 유지)
        self.bbox_history.append((x1, y1, x2, y2, cls_str))

        self._update_counts()
        self._redraw()
        messagebox.showinfo("BBox", f"{cls_str} 샘플 {n_add}개 추가됨.")

    def on_right_click(self, e):
        if self.img_bgr is None: return
        xi, yi = self._canvas_to_image(e.x, e.y)
        if xi is None: return
        self.curr_bg.append((xi, yi))
        feat = feature_at_xy(self.img_bgr, xi, yi)
        self.X_list.append(feat[0].copy()); self.y_list.append(0)
        b, g, r = self.img_bgr[yi, xi].tolist()
        self.RGB_list.append((int(r), int(g), int(b))); self.Cls_list.append("BG")
        self.data_roles.append(self._assign_role_for_index(len(self.X_list)-1))
        self._update_counts(); self._redraw()

    def on_wheel_zoom(self, e):
        if (e.state & 0x0004) == 0:
            return
        factor = 1.1 if e.delta > 0 else (1 / 1.1)
        self._apply_zoom(factor, e.x, e.y)

    def on_wheel_zoom_linux(self, e):
        if (e.state & 0x0004) == 0: return
        if e.num == 4: self._apply_zoom(1.1, e.x, e.y)
        elif e.num == 5: self._apply_zoom(1/1.1, e.x, e.y)

    def _apply_zoom(self, factor: float, x_can: int, y_can: int):
        if self.img_bgr is None: return
        self.zoom = max(0.2, min(10.0, self.zoom * factor))
        xi, yi = self._canvas_to_image(x_can, y_can)
        if xi is not None:
            H, W = self.img_bgr.shape[:2]
            self.center_xy = (xi / float(W), yi / float(H))
        self._redraw()

    def on_prev(self):
        if not self.image_paths: return
        self._load_image(self.idx - 1)

    def on_next(self):
        if not self.image_paths: return
        self._load_image(self.idx + 1)

    def on_undo(self):
        if self.curr_fg or self.curr_bg:
            if self.current_label.get() == "FG" and self.curr_fg:
                self.curr_fg.pop(); self._remove_last_from_dataset("FG")
            elif self.curr_bg:
                self.curr_bg.pop(); self._remove_last_from_dataset("BG")
            self._update_counts(); self._redraw()

    def _remove_last_from_dataset(self, cls_str: str):
        for i in range(len(self.Cls_list) - 1, -1, -1):
            if self.Cls_list[i] == cls_str:
                self.Cls_list.pop(i); self.RGB_list.pop(i)
                self.y_list.pop(i); self.X_list.pop(i)
                self.data_roles.pop(i)
                break

    def on_clear_current_points(self):
        if not (self.curr_fg or self.curr_bg): return
        ans = messagebox.askyesnocancel("확인", "현재 이미지 포인트 삭제\nYes: 누적에서도 삭제\nNo: 누적 유지\nCancel: 취소")
        if ans is None: return
        if ans:
            for _ in range(len(self.curr_fg)): self._remove_last_from_dataset("FG")
            for _ in range(len(self.curr_bg)): self._remove_last_from_dataset("BG")
        self.curr_fg.clear(); self.curr_bg.clear()
        self._update_counts(); self._redraw()

    def on_clear_bboxes_this_image(self):
        if not self.bbox_history:
            return
        if messagebox.askyesno("확인", "현재 이미지에서 그린 모든 BBox 프리뷰를 지울까요? (데이터셋에는 이미 반영됨)"):
            self.bbox_history.clear()
            self._redraw()

    def on_clear_cls(self):
        self.last_mask = None; self.last_contours = []
        self._redraw()

    def _update_counts(self):
        self.lbl_counts.config(text=f"FG_curr:{len(self.curr_fg)} BG_curr:{len(self.curr_bg)} | X_total:{len(self.X_list)}")

    # ---------- Dataset Split ----------
    def _assign_role_for_index(self, idx: int) -> str:
        if self.random_split:
            return "TRAIN"
        n = idx + 1
        ratio = (idx) / max(1, n)
        if ratio < 0.8:
            return "TRAIN"
        elif ratio < 0.9:
            return "VALID"
        else:
            return "TEST"

    def on_dataset_split(self):
        n = len(self.X_list)
        if n == 0:
            messagebox.showwarning("알림", "분할할 데이터가 없습니다.")
            return

        idxs = np.arange(n)
        np.random.shuffle(idxs)
        n_train = int(n * 0.8)
        n_valid = int(n * 0.1)
        train_idx = idxs[:n_train]
        valid_idx = idxs[n_train:n_train+n_valid]
        test_idx = idxs[n_train+n_valid:]

        roles = [""] * n
        for i in train_idx: roles[i] = "TRAIN"
        for i in valid_idx: roles[i] = "VALID"
        for i in test_idx: roles[i] = "TEST"
        self.data_roles = roles
        self.random_split = True
        messagebox.showinfo("완료", f"Dataset 무작위 분할\nTrain: {len(train_idx)} / Valid: {len(valid_idx)} / Test: {len(test_idx)}")

    # ---------- 학습 ----------
    def on_train(self):
        if len(self.X_list) < 10 or len(set(self.y_list)) < 2:
            messagebox.showwarning("경고", "학습을 위해 FG/BG 포인트를 더 추가하세요(최소 10개, 두 클래스 모두).")
            return

        X_train, y_train = self._get_subset("TRAIN")
        _X_valid, _y_valid = self._get_subset("VALID")
        if len(X_train) < 5 or len(set(y_train)) < 2:
            messagebox.showwarning("경고", f"Train 세트가 부족합니다. (현재 Train={len(X_train)})")
            return

        X = np.array(X_train, dtype=np.float32)
        y = np.array(y_train, dtype=np.int32)
        epochs = max(1, int(self.epochs_var.get()))
        mname = self.model_type_var.get()

        pop = tk.Toplevel(self.master); pop.title("Training...")
        lbl = ttk.Label(pop, text=f"{mname} | Epoch 0 / {epochs}"); lbl.pack(padx=12, pady=(12, 4))
        pb = ttk.Progressbar(pop, orient="horizontal", length=360, mode="determinate", maximum=epochs)
        pb.pack(padx=12, pady=(0, 12))
        pop.transient(self.master); pop.grab_set(); pop.update()

        if mname.startswith("SVM"):
            kernel = "rbf"
            if "linear" in mname: kernel = "linear"
            elif "poly" in mname: kernel = "poly"
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel=kernel, C=3.0, gamma="scale"))])
            n = len(y)
            for ep in range(1, epochs + 1):
                idxs = np.random.randint(0, n, size=n)
                pipe.fit(X[idxs], y[idxs])
                pb["value"] = ep; lbl.config(text=f"{mname} | Epoch {ep} / {epochs}"); pop.update()
            self.model = pipe
            self.torch_model = None; self.torch_scaler = None

        elif mname == "RF":
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            rf = RandomForestClassifier(n_estimators=0, warm_start=True, random_state=42, n_jobs=-1)
            for ep in range(1, epochs + 1):
                rf.n_estimators = ep
                rf.fit(Xs, y)
                pb["value"] = ep; lbl.config(text=f"RF | Trees {ep} / {epochs}"); pop.update()
            self.model = Pipeline([("scaler", scaler), ("clf", rf)])
            self.torch_model = None; self.torch_scaler = None

        else:
            if not TORCH_OK:
                pop.destroy()
                messagebox.showerror("오류", "PyTorch가 설치되어 있지 않습니다. pip install torch 로 설치하세요.")
                return
            scaler = StandardScaler().fit(X)
            Xs = scaler.transform(X).astype(np.float32)
            X_tensor = torch.from_numpy(Xs)
            y_tensor = torch.from_numpy(y.astype(np.int64))

            layers = max(1, int(self.mlp_layers_var.get()))
            neurons = self._parse_neurons(self.mlp_neurons_text.get(), layers)
            use_do = bool(self.mlp_use_dropout.get())
            do_rate = max(0.0, min(0.9, float(self.mlp_dropout_rate.get())))
            model = MLPNet(input_dim=11, hidden_sizes=neurons, num_classes=2,
                           use_dropout=use_do, dropout_rate=do_rate)

            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()

            model.train()
            B = min(256, max(16, len(y)//4))
            for ep in range(1, epochs+1):
                idxs = np.random.permutation(len(y))
                Xb = X_tensor[idxs]; yb = y_tensor[idxs]
                for i in range(0, len(y), B):
                    xb = Xb[i:i+B]; ybt = yb[i:i+B]
                    optimizer.zero_grad()
                    logits = model(xb)
                    loss = criterion(logits, ybt)
                    loss.backward()
                    optimizer.step()
                pb["value"] = ep; lbl.config(text=f"MLP | Epoch {ep} / {epochs}"); pop.update()

            self.torch_model = model.eval()
            self.torch_scaler = scaler
            self.model = None

        self.is_trained = True
        self.last_train_date = datetime.now().strftime("%Y%m%d")
        pop.destroy()
        messagebox.showinfo("완료", f"학습 완료: {mname}, epochs={epochs}, Train 샘플={len(y)}")

    def _parse_neurons(self, text: str, layers: int) -> List[int]:
        try:
            parts = [int(max(1, int(p.strip()))) for p in text.split(",") if p.strip() != ""]
        except Exception:
            parts = []
        if not parts:
            parts = [10]*layers
        if len(parts) < layers:
            parts = parts + [parts[-1]]*(layers - len(parts))
        elif len(parts) > layers:
            parts = parts[:layers]
        return parts

    def _get_subset(self, role: str):
        Xs = []
        ys = []
        for x, y, r in zip(self.X_list, self.y_list, self.data_roles):
            if r == role:
                Xs.append(x)
                ys.append(y)
        return Xs, ys

    # ---------- 분류 ----------
    def on_classify(self):
        if self.img_bgr is None:
            return
        mname = self.model_type_var.get()
        if not self.is_trained or (self.model is None and self.torch_model is None):
            messagebox.showwarning("경고", "먼저 Train 또는 Load Weights를 수행하세요.")
            return

        feats = extract_features_image(self.img_bgr)
        try:
            if mname == "MLP (PyTorch)" or (self.torch_model is not None and self.model is None):
                Xs = self.torch_scaler.transform(feats).astype(np.float32)
                with torch.no_grad():
                    logits = self.torch_model(torch.from_numpy(Xs))
                    pred = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
            else:
                pred = self.model.predict(feats).astype(np.uint8)
        except Exception as e:
            messagebox.showerror("오류", f"분류 실패: {e}")
            return

        H, W = self.img_bgr.shape[:2]
        mask = pred.reshape(H, W)

        mask_post = postprocess_mask(mask, self.morph_kernel_size, self.morph_open_iter, self.morph_close_iter)
        self.last_mask = mask_post

        self.last_contours = find_outer_contours(
            mask_post,
            epsilon_px=float(self.poly_epsilon_px.get()),
            chain_approx_mode=self.chain_approx_mode.get(),
            blur_enable=bool(self.cnt_blur_enable.get()),
            blur_method=self.cnt_blur_method.get(),
            blur_ksize=int(self.cnt_blur_ksize.get()),
        )
        self._redraw()

    # ---------- YOLO-seg Export ----------
    def on_export_yolo(self):
        if self.img_bgr is None or self.last_mask is None:
            messagebox.showwarning("경고", "Classify 후 Export 할 수 있습니다.")
            return
        H, W = self.img_bgr.shape[:2]
        contours = find_outer_contours(
            self.last_mask,
            epsilon_px=float(self.poly_epsilon_px.get()),
            chain_approx_mode=self.chain_approx_mode.get(),
            blur_enable=bool(self.cnt_blur_enable.get()),
            blur_method=self.cnt_blur_method.get(),
            blur_ksize=int(self.cnt_blur_ksize.get()),
        )
        if not contours:
            messagebox.showwarning("경고", "외곽 폴리곤이 없습니다.")
            return
        lines = contours_to_yoloseg(contours, W, H, CLASS_ID)
        img_path = self.image_paths[self.idx]
        out_txt = (self.output_dir / img_path.stem).with_suffix(".txt")
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(out_txt, "w", encoding="utf-8") as f:
                for ln in lines: f.write(ln + "\n")
        except Exception as e:
            messagebox.showerror("오류", f"YOLO 라벨 저장 실패: {e}")
            return

        preview = self.img_bgr.copy()
        cv2.drawContours(preview, contours, -1, (0, 0, 255), 1)
        imwrite_unicode((self.output_dir / f"{img_path.stem}_preview.jpg"), preview)
        messagebox.showinfo("완료", f"저장: {out_txt.name}\n(yolo-labels 폴더)")

    # ---------- 가중치 저장/로드 ----------
    def on_save_weights(self):
        if not self.is_trained:
            messagebox.showwarning("경고", "학습된 모델이 없습니다.")
            return
        mname = self.model_type_var.get()
        epochs = int(self.epochs_var.get())
        ymd = self.last_train_date or datetime.now().strftime("%Y%m%d")

        if mname == "MLP (PyTorch)" or (self.torch_model is not None and self.model is None):
            default_name = f"MLP_{epochs}_{ymd}.pt"
            path = filedialog.asksaveasfilename(
                title="MLP 모델 저장", defaultextension=".pt", initialfile=default_name,
                filetypes=[("PyTorch", "*.pt"), ("All files", "*.*")]
            )
            if not path: return
            saveobj = {
                "state_dict": self.torch_model.state_dict(),
                "arch": {
                    "input_dim": 11,
                    "hidden_sizes": self._parse_neurons(self.mlp_neurons_text.get(), int(self.mlp_layers_var.get())),
                    "num_classes": 2,
                    "use_dropout": bool(self.mlp_use_dropout.get()),
                    "dropout_rate": float(self.mlp_dropout_rate.get()),
                },
                "scaler": self.torch_scaler,
                "meta": {"model_type": "MLP (PyTorch)", "epochs": epochs, "date": ymd}
            }
            try:
                torch.save(saveobj, path)
            except Exception as e:
                messagebox.showerror("오류", f"저장 실패: {e}"); return
            messagebox.showinfo("완료", f"저장됨: {Path(path).name}")
        else:
            model_name = mname.replace(" ", "")
            default_name = f"{model_name}_{epochs}_{ymd}.joblib"
            path = filedialog.asksaveasfilename(
                title="모델 저장", defaultextension=".joblib", initialfile=default_name,
                filetypes=[("Joblib", "*.joblib"), ("All files", "*.*")]
            )
            if not path: return
            payload = {"model": self.model, "meta": {"model_type": mname, "epochs": epochs, "date": ymd}}
            try: joblib.dump(payload, path)
            except Exception as e:
                messagebox.showerror("오류", f"저장 실패: {e}"); return
            messagebox.showinfo("완료", f"저장됨: {Path(path).name}")

    def on_load_weights(self):
        path = filedialog.askopenfilename(
            title="모델 불러오기",
            filetypes=[("Model files", "*.pt *.joblib"), ("PyTorch", "*.pt"), ("Joblib", "*.joblib"), ("All files", "*.*")]
        )
        if not path: return
        path = Path(path)
        if path.suffix.lower() == ".pt":
            if not TORCH_OK:
                messagebox.showerror("오류", "PyTorch가 설치되어 있지 않습니다. pip install torch 로 설치하세요.")
                return
            try:
                payload = torch.load(path, map_location="cpu")
            except Exception as e:
                messagebox.showerror("오류", f"불러오기 실패: {e}"); return
            arch = payload.get("arch", {})
            model = MLPNet(
                input_dim=int(arch.get("input_dim", 11)),
                hidden_sizes=list(arch.get("hidden_sizes", [10,10,10])),
                num_classes=int(arch.get("num_classes", 2)),
                use_dropout=bool(arch.get("use_dropout", True)),
                dropout_rate=float(arch.get("dropout_rate", 0.05)),
            )
            try:
                model.load_state_dict(payload["state_dict"])
            except Exception as e:
                messagebox.showerror("오류", f"가중치 로드 실패: {e}"); return
            self.torch_model = model.eval()
            self.torch_scaler = payload.get("scaler", None)
            self.model = None
            self.is_trained = True
            meta = payload.get("meta", {})
            self.model_type_var.set(meta.get("model_type", "MLP (PyTorch)"))
            self.epochs_var.set(int(meta.get("epochs", self.epochs_var.get())))
            self.last_train_date = meta.get("date", datetime.now().strftime("%Y%m%d"))
            self.mlp_layers_var.set(len(arch.get("hidden_sizes", [10,10,10])))
            self.mlp_neurons_text.set(",".join(str(int(n)) for n in arch.get("hidden_sizes", [10,10,10])))
            self.mlp_use_dropout.set(bool(arch.get("use_dropout", True)))
            self.mlp_dropout_rate.set(float(arch.get("dropout_rate", 0.05)))
            messagebox.showinfo("완료", f"MLP 모델 로드: {path.name}")
        else:
            try:
                payload = joblib.load(path)
            except Exception as e:
                messagebox.showerror("오류", f"불러오기 실패: {e}"); return
            if not isinstance(payload, dict) or "model" not in payload:
                messagebox.showerror("오류", "잘못된 모델 파일입니다."); return
            self.model = payload["model"]
            self.torch_model = None; self.torch_scaler = None
            self.is_trained = True
            meta = payload.get("meta", {})
            self.model_type_var.set(meta.get("model_type", self.model_type_var.get()))
            self.epochs_var.set(int(meta.get("epochs", self.epochs_var.get())))
            self.last_train_date = meta.get("date", datetime.now().strftime("%Y%m%d"))
            messagebox.showinfo("완료", f"모델 로드: {path.name}")

    # ---------- 데이터셋 Export/Load ----------
    def on_export_dataset(self):
        if not self.RGB_list:
            messagebox.showwarning("경고", "내보낼 데이터가 없습니다.")
            return
        path = filedialog.asksaveasfilename(
            title="Export Dataset (CSV)", defaultextension=".csv", initialfile="dataset.csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")]
        )
        if not path: return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["IDX", "R", "G", "B", "Cls", "Set"])
                for i, ((r, g, b), cls, role) in enumerate(zip(self.RGB_list, self.Cls_list, self.data_roles), start=1):
                    w.writerow([i, r, g, b, cls, role])
        except Exception as e:
            messagebox.showerror("오류", f"CSV 저장 실패: {e}"); return
        messagebox.showinfo("완료", f"데이터셋 저장: {Path(path).name}")

    def on_dataset_load(self):
        path = filedialog.askopenfilename(
            title="Dataset Load (CSV)",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")]
        )
        if not path: return
        loaded = 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    r = int(float(row["R"])); g = int(float(row["G"])); b = int(float(row["B"]))
                    cls = row["Cls"].strip().upper()
                    ylab = 1 if cls == "FG" else 0
                    hsv, lab = hsv_lab_from_rgb(r, g, b)
                    feat = np.array([[b, g, r, hsv[0], hsv[1], hsv[2], lab[0], lab[1], lab[2], 0.5, 0.5]], dtype=np.float32)
                    self.X_list.append(feat[0]); self.y_list.append(ylab)
                    self.RGB_list.append((r, g, b)); self.Cls_list.append("FG" if ylab==1 else "BG")
                    role = row.get("Set", "").strip().upper()
                    if role not in ("TRAIN", "VALID", "TEST"):
                        role = self._assign_role_for_index(len(self.X_list)-1)
                    self.data_roles.append(role)
                    loaded += 1
        except Exception as e:
            messagebox.showerror("오류", f"CSV 로드 실패: {e}"); return
        self._update_counts()
        messagebox.showinfo("완료", f"로딩된 항목: {loaded}")

    # ---------- 옵션(모폴로지 + 컨투어 + MLP + BBox) ----------
    def on_option(self):
        win = tk.Toplevel(self.master); win.title("Options")
        frm = ttk.Frame(win); frm.pack(padx=12, pady=12)

        # ---- Morphology
        ttk.Label(frm, text="Morphology").grid(row=0, column=0, sticky="w", padx=4, pady=(0,4), columnspan=2)
        ttk.Label(frm, text="Kernel size (odd ≥ 3):").grid(row=1, column=0, sticky="e", padx=4, pady=4)
        ks_var = tk.IntVar(value=self.morph_kernel_size)
        ttk.Spinbox(frm, from_=3, to=99, increment=2, textvariable=ks_var, width=6).grid(row=1, column=1, padx=4, pady=4)

        ttk.Label(frm, text="Open iterations:").grid(row=2, column=0, sticky="e", padx=4, pady=4)
        op_var = tk.IntVar(value=self.morph_open_iter)
        ttk.Spinbox(frm, from_=0, to=10, textvariable=op_var, width=6).grid(row=2, column=1, padx=4, pady=4)

        ttk.Label(frm, text="Close iterations:").grid(row=3, column=0, sticky="e", padx=4, pady=4)
        cl_var = tk.IntVar(value=self.morph_close_iter)
        ttk.Spinbox(frm, from_=0, to=10, textvariable=cl_var, width=6).grid(row=3, column=1, padx=4, pady=4)

        # ---- Contour Refinement
        ttk.Separator(frm, orient=tk.HORIZONTAL).grid(row=4, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Label(frm, text="Contour Refinement").grid(row=5, column=0, sticky="w", padx=4, pady=(0,4), columnspan=2)

        ttk.Label(frm, text="Chain Approx:").grid(row=6, column=0, sticky="e", padx=4, pady=4)
        ca_var = tk.StringVar(value=self.chain_approx_mode.get())
        ttk.Combobox(frm, textvariable=ca_var, state="readonly",
                     values=["High precision (CHAIN_APPROX_NONE)", "Simple (CHAIN_APPROX_SIMPLE)"],
                     width=28).grid(row=6, column=1, padx=4, pady=4)

        ttk.Label(frm, text="Polygon ε (px, 0=off):").grid(row=7, column=0, sticky="e", padx=4, pady=4)
        eps_var = tk.DoubleVar(value=float(self.poly_epsilon_px.get()))
        ttk.Spinbox(frm, from_=0.0, to=10.0, increment=0.1, textvariable=eps_var, width=6).grid(row=7, column=1, padx=4, pady=4)

        ttk.Label(frm, text="Pre-contour blur:").grid(row=8, column=0, sticky="e", padx=4, pady=4)
        blur_enable_var = tk.BooleanVar(value=bool(self.cnt_blur_enable.get()))
        ttk.Checkbutton(frm, variable=blur_enable_var).grid(row=8, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(frm, text="Blur method:").grid(row=9, column=0, sticky="e", padx=4, pady=4)
        blur_method_var = tk.StringVar(value=self.cnt_blur_method.get())
        ttk.Combobox(frm, textvariable=blur_method_var, state="readonly",
                     values=["Gaussian", "Median"], width=10).grid(row=9, column=1, padx=4, pady=4)

        ttk.Label(frm, text="Blur ksize (odd ≥ 3):").grid(row=10, column=0, sticky="e", padx=4, pady=4)
        blur_ksize_var = tk.IntVar(value=int(self.cnt_blur_ksize.get()))
        ttk.Spinbox(frm, from_=3, to=31, increment=2, textvariable=blur_ksize_var, width=6).grid(row=10, column=1, padx=4, pady=4)

        # ---- MLP (PyTorch)
        ttk.Separator(frm, orient=tk.HORIZONTAL).grid(row=11, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Label(frm, text="MLP (PyTorch)").grid(row=12, column=0, sticky="w", padx=4, pady=(0,4), columnspan=2)

        ttk.Label(frm, text="# Hidden Layers:").grid(row=13, column=0, sticky="e", padx=4, pady=4)
        layers_var = tk.IntVar(value=int(self.mlp_layers_var.get()))
        ttk.Spinbox(frm, from_=1, to=10, textvariable=layers_var, width=6).grid(row=13, column=1, padx=4, pady=4)

        ttk.Label(frm, text="Neurons per layer (comma):").grid(row=14, column=0, sticky="e", padx=4, pady=4)
        neurons_var = tk.StringVar(value=self.mlp_neurons_text.get())
        ttk.Entry(frm, textvariable=neurons_var, width=24).grid(row=14, column=1, padx=4, pady=4)

        ttk.Label(frm, text="Use Dropout:").grid(row=15, column=0, sticky="e", padx=4, pady=4)
        use_do_var = tk.BooleanVar(value=bool(self.mlp_use_dropout.get()))
        ttk.Checkbutton(frm, variable=use_do_var).grid(row=15, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(frm, text="Dropout rate (0~0.9):").grid(row=16, column=0, sticky="e", padx=4, pady=4)
        do_rate_var = tk.DoubleVar(value=float(self.mlp_dropout_rate.get()))
        ttk.Spinbox(frm, from_=0.0, to=0.9, increment=0.01, textvariable=do_rate_var, width=6).grid(row=16, column=1, padx=4, pady=4)

        # ---- BBox 옵션
        ttk.Separator(frm, orient=tk.HORIZONTAL).grid(row=17, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Label(frm, text="BBox Sampling").grid(row=18, column=0, sticky="w", padx=4, pady=(0,4), columnspan=2)

        ttk.Label(frm, text="Use Subsample (ratio):").grid(row=19, column=0, sticky="e", padx=4, pady=4)
        subs_en_var = tk.BooleanVar(value=bool(self.bbox_subsample_enable.get()))
        subs_ratio_var = tk.DoubleVar(value=float(self.bbox_subsample_ratio.get()))
        ttk.Checkbutton(frm, variable=subs_en_var).grid(row=19, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(frm, text="Ratio (0~1):").grid(row=20, column=0, sticky="e", padx=4, pady=4)
        ttk.Spinbox(frm, from_=0.0, to=1.0, increment=0.05, textvariable=subs_ratio_var, width=6).grid(row=20, column=1, padx=4, pady=4)

        def apply_and_close():
            # Morph
            ks = int(ks_var.get());  ks = ks + (ks % 2 == 0)
            self.morph_kernel_size = max(3, ks)
            self.morph_open_iter = max(0, int(op_var.get()))
            self.morph_close_iter = max(0, int(cl_var.get()))
            # Contour
            self.chain_approx_mode.set(ca_var.get())
            eps = float(eps_var.get())
            self.poly_epsilon_px.set(max(0.0, min(10.0, eps)))
            self.cnt_blur_enable.set(bool(blur_enable_var.get()))
            self.cnt_blur_method.set(blur_method_var.get())
            k2 = int(blur_ksize_var.get()); k2 = k2 + (k2 % 2 == 0)
            self.cnt_blur_ksize.set(max(3, k2))
            # MLP
            self.mlp_layers_var.set(max(1, int(layers_var.get())))
            self.mlp_neurons_text.set(neurons_var.get())
            self.mlp_use_dropout.set(bool(use_do_var.get()))
            self.mlp_dropout_rate.set(max(0.0, min(0.9, float(do_rate_var.get()))))
            # BBox
            self.bbox_subsample_enable.set(bool(subs_en_var.get()))
            self.bbox_subsample_ratio.set(max(0.0, min(1.0, float(subs_ratio_var.get()))))
            win.destroy()

        btns = ttk.Frame(frm); btns.grid(row=21, column=0, columnspan=2, pady=(8,0))
        ttk.Button(btns, text="OK", command=apply_and_close).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side=tk.LEFT, padx=4)

    # ---------- 데이터셋 초기화 ----------
    def on_init_dataset(self):
        if messagebox.askyesno("확인", "누적 학습 데이터셋(X,y)을 모두 초기화할까요?"):
            self.X_list.clear(); self.y_list.clear()
            self.RGB_list.clear(); self.Cls_list.clear()
            self.data_roles.clear()
            self.random_split = False
            self.is_trained = False
            self.model = None
            self.torch_model = None; self.torch_scaler = None
            self.last_mask = None; self.last_contours = []
            self.curr_fg.clear(); self.curr_bg.clear()
            self.bbox_history.clear(); self.bbox_dragging = False
            self._update_counts(); self._redraw()
            messagebox.showinfo("완료", "누적 데이터셋이 초기화되었습니다. 새로 샘플링하세요.")

    # ---------- 모델 평가 ----------
    def on_model_evaluation(self):
        if not self.is_trained:
            messagebox.showwarning("경고", "먼저 모델을 학습시키거나 불러오세요.")
            return

        X_test, y_test = self._get_subset("TEST")
        if len(X_test) == 0:
            messagebox.showwarning("경고", "Test 세트가 없습니다. Dataset Split을 하거나 일부 샘플을 Test로 지정하세요.")
            return

        mname = self.model_type_var.get()
        try:
            if mname == "MLP (PyTorch)" or (self.torch_model is not None and self.model is None):
                Xs = self.torch_scaler.transform(np.array(X_test, dtype=np.float32)).astype(np.float32)
                with torch.no_grad():
                    logits = self.torch_model(torch.from_numpy(Xs))
                    y_pred = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int32)
            else:
                y_pred = self.model.predict(np.array(X_test, dtype=np.float32)).astype(np.int32)
        except Exception as e:
            messagebox.showerror("오류", f"평가 실패: {e}")
            return

        y_true = np.array(y_test, dtype=np.int32)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        total = len(y_true)
        acc = (tp + tn) / total if total else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        win = tk.Toplevel(self.master); win.title("Model Evaluation (Test set)")
        frm = ttk.Frame(win); frm.pack(padx=12, pady=12)

        ttk.Label(frm, text="Confusion Matrix (rows=true, cols=pred)").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0,6))
        ttk.Label(frm, text="          Pred 0    Pred 1").grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Label(frm, text=f"True 0 :   {tn:4d}      {fp:4d}").grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Label(frm, text=f"True 1 :   {fn:4d}      {tp:4d}").grid(row=3, column=0, columnspan=2, sticky="w")

        ttk.Separator(frm, orient=tk.HORIZONTAL).grid(row=4, column=0, columnspan=2, sticky="ew", pady=8)

        ttk.Label(frm, text=f"Total samples (Test): {total}").grid(row=5, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Label(frm, text=f"Accuracy : {acc:.4f}").grid(row=6, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Label(frm, text=f"Precision: {prec:.4f}").grid(row=7, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Label(frm, text=f"Recall   : {rec:.4f}").grid(row=8, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Label(frm, text=f"F1-score : {f1:.4f}").grid(row=9, column=0, columnspan=2, sticky="w", pady=2)

        ttk.Button(frm, text="Close", command=win.destroy).grid(row=10, column=0, columnspan=2, pady=(10,0))

# --------------- main ---------------
def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
