#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FG/BG 인터랙티브 샘플링 → 감독분류(SVM/RF) → 모폴로지 후처리 → 외곽 폴리곤(정교) → YOLO-seg Export
- 시작 시 폴더 선택 팝업 + Load Images 버튼
- 모델: SVM(linear/rbf/poly), RF / Epochs 기본 100 (조절 가능), 진행 팝업 표시
- Classify 결과 오버레이 투명도(10~100%), Clear 버튼
- Export Dataset(CSV: IDX,R,G,B,Cls) / Dataset Load(CSV 불러와 이어서 학습)
- 줌/팬: Ctrl+휠(줌), Ctrl+클릭(해당 지점을 화면 중앙으로 팬)
- 모폴로지 기본: 커널 3x3, 열림1, 닫힘1 (Option에서 변경)
- 컨투어 정교화: CHAIN_APPROX_NONE 기본, epsilon(px)로 단순화 강도 제어(0=끄기),
  컨투어 추출 전 선택적 블러(가우시안/미디언) + 재이진화 옵션 추가 (Option에서 변경)
- YOLO-seg 저장 + 화면에는 붉은색(1px) 경계선 표시
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

# ---------------- 기본 설정 ----------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CANVAS_W, CANVAS_H = 1280, 800
CLASS_ID = 0  # YOLO-seg 클래스 ID (단일 클래스 가정)

# 모폴로지 기본값 (Option에서 변경 가능)
DEFAULT_KERNEL_SIZE = 3   # 3x3
DEFAULT_OPEN_ITER   = 1
DEFAULT_CLOSE_ITER  = 1

# 컨투어(정교화) 기본값 (Option에서 변경 가능)
DEFAULT_CHAIN_APPROX = "High precision (CHAIN_APPROX_NONE)"   # 또는 "Simple (CHAIN_APPROX_SIMPLE)"
DEFAULT_EPSILON_PX   = 0.5   # 0이면 단순화 끔
DEFAULT_CNT_BLUR_ENABLE = True
DEFAULT_CNT_BLUR_METHOD = "Gaussian"   # 또는 "Median"
DEFAULT_CNT_BLUR_KSIZE  = 3           # 홀수

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
    정교한 외곽 컨투어 추출:
      - 체인 압축: CHAIN_APPROX_NONE(정밀) / CHAIN_APPROX_SIMPLE
      - epsilon_px(픽셀)로 다각형 단순화 강도 제어(0이면 단순화 해제)
      - 컨투어 전처리: 선택적 블러 + 재이진화(경계 매끄럽게)
    """
    u8 = (mask01 > 0).astype(np.uint8) * 255

    # 선택적 블러 + 재이진화
    if blur_enable:
        k = max(3, int(blur_ksize))
        if k % 2 == 0: k += 1
        if blur_method.lower().startswith("median"):
            u8 = cv2.medianBlur(u8, k)
        else:  # Gaussian
            u8 = cv2.GaussianBlur(u8, (k, k), 0)
        _, u8 = cv2.threshold(u8, 127, 255, cv2.THRESH_BINARY)

    # 체인 모드
    if chain_approx_mode.startswith("High"):
        chain_flag = cv2.CHAIN_APPROX_NONE
    else:
        chain_flag = cv2.CHAIN_APPROX_SIMPLE

    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, chain_flag)

    outs = []
    for c in contours:
        if len(c) < 3:
            continue
        if epsilon_px and epsilon_px > 0:
            approx = cv2.approxPolyDP(c, float(epsilon_px), True)
        else:
            approx = c  # 단순화 끔(가장 정교)
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
        self.last_mask: Optional[np.ndarray] = None   # 모폴로지 후 최종 마스크(0/1)
        self.last_contours: List[np.ndarray] = []
        self.overlay_transp = tk.DoubleVar(value=10.0)  # 10~100 (%)

        # 샘플/학습 데이터
        self.curr_fg: List[Tuple[int, int]] = []
        self.curr_bg: List[Tuple[int, int]] = []
        self.current_label = tk.StringVar(value="FG")

        self.X_list: List[np.ndarray] = []           # (11,) 특징
        self.y_list: List[int] = []                  # 1(FG), 0(BG)
        self.RGB_list: List[Tuple[int, int, int]] = []  # (R,G,B)
        self.Cls_list: List[str] = []                # "FG"/"BG"

        # 모델
        self.model: Optional[Pipeline] = None
        self.is_trained: bool = False
        self.last_train_date: Optional[str] = None
        self.model_type_var = tk.StringVar(value="SVM - rbf")
        self.epochs_var = tk.IntVar(value=100)

        # 모폴로지 옵션
        self.morph_kernel_size = DEFAULT_KERNEL_SIZE
        self.morph_open_iter = DEFAULT_OPEN_ITER
        self.morph_close_iter = DEFAULT_CLOSE_ITER

        # 컨투어 옵션(정교화)
        self.chain_approx_mode = tk.StringVar(value=DEFAULT_CHAIN_APPROX)
        self.poly_epsilon_px = tk.DoubleVar(value=DEFAULT_EPSILON_PX)
        self.cnt_blur_enable = tk.BooleanVar(value=DEFAULT_CNT_BLUR_ENABLE)
        self.cnt_blur_method = tk.StringVar(value=DEFAULT_CNT_BLUR_METHOD)
        self.cnt_blur_ksize = tk.IntVar(value=DEFAULT_CNT_BLUR_KSIZE)

        # 캔버스/렌더링(줌/팬)
        self.canvas = None
        self.tk_img = None
        self.tk_overlay = None
        self.base_scale = 1.0
        self.zoom = 1.0
        self.center_xy = (0.5, 0.5)  # 이미지 내 정규화 중심

        self._build_ui()
        self._ask_and_load_dir(initial=True)

    # ---------- UI ----------
    def _build_ui(self):
        top = ttk.Frame(self.master); top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        ttk.Button(top, text="Load Images", command=self._ask_and_load_dir).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Prev", command=self.on_prev).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Next", command=self.on_next).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Initialization", command=self.on_init_dataset).pack(side=tk.LEFT, padx=8)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Label(top, text="Model:").pack(side=tk.LEFT)
        ttk.Combobox(top, textvariable=self.model_type_var, state="readonly",
                     values=["SVM - linear", "SVM - rbf", "SVM - poly", "RF"], width=14).pack(side=tk.LEFT, padx=4)
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
        ttk.Button(top, text="Option", command=self.on_option).pack(side=tk.LEFT, padx=2)

        mid = ttk.Frame(self.master); mid.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 4))
        self.lbl_info = ttk.Label(mid, text="Path: -"); self.lbl_info.pack(side=tk.LEFT, padx=4)
        self.lbl_counts = ttk.Label(mid, text="FG_curr:0 BG_curr:0 | X_total:0"); self.lbl_counts.pack(side=tk.RIGHT, padx=4)

        lab = ttk.Frame(self.master); lab.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 6))
        ttk.Label(lab, text="Current Label:").pack(side=tk.LEFT)
        ttk.Radiobutton(lab, text="FG(대상)", variable=self.current_label, value="FG").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(lab, text="BG(배경)", variable=self.current_label, value="BG").pack(side=tk.LEFT, padx=2)
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
        self.canvas.bind("<Button-1>", self.on_left_click)   # 좌클릭: 샘플 or Ctrl+클릭: 팬
        self.canvas.bind("<Button-3>", self.on_right_click)  # 우클릭: BG 샘플
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
        self._update_counts()

        # 보기 초기화
        H, W = img.shape[:2]
        self.base_scale = min(CANVAS_W / W, CANVAS_H / H)
        self.zoom = 1.0
        self.center_xy = (0.5, 0.5)
        self._redraw()

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

        # 오버레이(분류 결과 + 붉은 윤곽)
        if self.last_mask is not None:
            transp = float(self.overlay_transp.get())
            alpha = 1.0 - max(10.0, min(100.0, transp)) / 100.0  # 10%→0.9, 100%→0.0
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
        # 붉은색 윤곽(선 두께 1px)
        if contours:
            cv2.drawContours(out, contours, -1, (0, 0, 255), 1)
        return out

    # ---------- 이벤트 ----------
    def on_left_click(self, e):
        # Ctrl+클릭 → 팬(해당 지점을 중앙에 두기)
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
        if self.current_label.get() == "FG":
            self.curr_fg.append((xi, yi)); ylab = 1; cls_str = "FG"
        else:
            self.curr_bg.append((xi, yi)); ylab = 0; cls_str = "BG"

        feat = feature_at_xy(self.img_bgr, xi, yi)  # (1,11)
        self.X_list.append(feat[0].copy()); self.y_list.append(ylab)
        b, g, r = self.img_bgr[yi, xi].tolist()
        self.RGB_list.append((int(r), int(g), int(b)))
        self.Cls_list.append(cls_str)
        self._update_counts(); self._redraw()

    def on_right_click(self, e):
        if self.img_bgr is None: return
        xi, yi = self._canvas_to_image(e.x, e.y)
        if xi is None: return
        self.curr_bg.append((xi, yi))
        feat = feature_at_xy(self.img_bgr, xi, yi)
        self.X_list.append(feat[0].copy()); self.y_list.append(0)
        b, g, r = self.img_bgr[yi, xi].tolist()
        self.RGB_list.append((int(r), int(g), int(b))); self.Cls_list.append("BG")
        self._update_counts(); self._redraw()

    def on_wheel_zoom(self, e):
        if (e.state & 0x0004) == 0:  # Ctrl 없는 휠은 무시
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

    def on_clear_cls(self):
        self.last_mask = None; self.last_contours = []
        self._redraw()

    def _update_counts(self):
        self.lbl_counts.config(text=f"FG_curr:{len(self.curr_fg)} BG_curr:{len(self.curr_bg)} | X_total:{len(self.X_list)}")

    # ---------- 학습 ----------
    def on_train(self):
        if len(self.X_list) < 10 or len(set(self.y_list)) < 2:
            messagebox.showwarning("경고", "학습을 위해 FG/BG 포인트를 더 추가하세요(최소 10개, 두 클래스 모두).")
            return
        X = np.array(self.X_list, dtype=np.float32)
        y = np.array(self.y_list, dtype=np.int32)
        epochs = max(1, int(self.epochs_var.get()))
        mname = self.model_type_var.get()

        # 진행 팝업
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
                idxs = np.random.randint(0, n, size=n)  # 부트스트랩
                pipe.fit(X[idxs], y[idxs])
                pb["value"] = ep; lbl.config(text=f"{mname} | Epoch {ep} / {epochs}"); pop.update()
            self.model = pipe
        else:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            rf = RandomForestClassifier(n_estimators=0, warm_start=True, random_state=42, n_jobs=-1)
            for ep in range(1, epochs + 1):
                rf.n_estimators = ep
                rf.fit(Xs, y)
                pb["value"] = ep; lbl.config(text=f"RF | Trees {ep} / {epochs}"); pop.update()
            self.model = Pipeline([("scaler", scaler), ("clf", rf)])

        self.is_trained = True
        self.last_train_date = datetime.now().strftime("%Y%m%d")
        pop.destroy()
        messagebox.showinfo("완료", f"학습 완료: {mname}, epochs={epochs}, 샘플={len(y)}")

    # ---------- 분류 ----------
    def on_classify(self):
        if self.img_bgr is None:
            return
        if not self.is_trained or self.model is None:
            messagebox.showwarning("경고", "먼저 Train 또는 Load Weights를 수행하세요.")
            return
        feats = extract_features_image(self.img_bgr)
        try:
            pred = self.model.predict(feats).astype(np.uint8)
        except Exception as e:
            messagebox.showerror("오류", f"분류 실패: {e}")
            return
        H, W = self.img_bgr.shape[:2]
        mask = pred.reshape(H, W)  # 1=FG, 0=BG

        # 모폴로지 후처리(옵션에서 설정한 값 사용)
        mask_post = postprocess_mask(mask, self.morph_kernel_size, self.morph_open_iter, self.morph_close_iter)
        self.last_mask = mask_post

        # 정교한 외곽 컨투어 보관(옵션 반영)
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
        # 현재 옵션으로 다시 추출(일관성)
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

        # 프리뷰 저장(컨투어 붉은색)
        preview = self.img_bgr.copy()
        cv2.drawContours(preview, contours, -1, (0, 0, 255), 1)
        imwrite_unicode((self.output_dir / f"{img_path.stem}_preview.jpg"), preview)
        messagebox.showinfo("완료", f"저장: {out_txt.name}\n(yolo-labels 폴더)")

    # ---------- 가중치 저장/로드 ----------
    def on_save_weights(self):
        if not self.is_trained or self.model is None:
            messagebox.showwarning("경고", "학습된 모델이 없습니다.")
            return
        model_name = self.model_type_var.get().replace(" ", "")  # "SVM-rbf" 등
        epochs = int(self.epochs_var.get())
        ymd = self.last_train_date or datetime.now().strftime("%Y%m%d")
        default_name = f"{model_name}_{epochs}_{ymd}.joblib"
        path = filedialog.asksaveasfilename(
            title="모델 저장", defaultextension=".joblib", initialfile=default_name,
            filetypes=[("Joblib", "*.joblib"), ("All files", "*.*")]
        )
        if not path: return
        payload = {"model": self.model, "meta": {"model_type": self.model_type_var.get(),
                                                 "epochs": epochs, "date": ymd}}
        try: joblib.dump(payload, path)
        except Exception as e:
            messagebox.showerror("오류", f"저장 실패: {e}"); return
        messagebox.showinfo("완료", f"저장됨: {Path(path).name}")

    def on_load_weights(self):
        path = filedialog.askopenfilename(
            title="모델 불러오기",
            filetypes=[("Joblib", "*.joblib"), ("All files", "*.*")]
        )
        if not path: return
        try: payload = joblib.load(path)
        except Exception as e:
            messagebox.showerror("오류", f"불러오기 실패: {e}"); return
        if not isinstance(payload, dict) or "model" not in payload:
            messagebox.showerror("오류", "잘못된 모델 파일입니다."); return
        self.model = payload["model"]; self.is_trained = True
        meta = payload.get("meta", {})
        self.model_type_var.set(meta.get("model_type", self.model_type_var.get()))
        self.epochs_var.set(int(meta.get("epochs", self.epochs_var.get())))
        self.last_train_date = meta.get("date", datetime.now().strftime("%Y%m%d"))
        messagebox.showinfo("완료", f"모델 로드: {Path(path).name}")

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
                w.writerow(["IDX", "R", "G", "B", "Cls"])
                for i, ((r, g, b), cls) in enumerate(zip(self.RGB_list, self.Cls_list), start=1):
                    w.writerow([i, r, g, b, cls])
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
                    feat = np.array([[b, g, r, hsv[0], hsv[1], hsv[2], lab[0], lab[1], lab[2], 0.5, 0.5]],
                                    dtype=np.float32)
                    self.X_list.append(feat[0]); self.y_list.append(ylab)
                    self.RGB_list.append((r, g, b)); self.Cls_list.append("FG" if ylab==1 else "BG")
                    loaded += 1
        except Exception as e:
            messagebox.showerror("오류", f"CSV 로드 실패: {e}"); return
        self._update_counts()
        messagebox.showinfo("완료", f"로딩된 항목: {loaded}")

    # ---------- 옵션(모폴로지 + 컨투어 정교화) ----------
    def on_option(self):
        win = tk.Toplevel(self.master); win.title("Options (Morphology & Contour)")
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
        sep = ttk.Separator(frm, orient=tk.HORIZONTAL); sep.grid(row=4, column=0, columnspan=2, sticky="ew", pady=8)
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

        def apply_and_close():
            # Morph
            ks = int(ks_var.get())
            if ks % 2 == 0: ks += 1
            self.morph_kernel_size = max(3, ks)
            self.morph_open_iter = max(0, int(op_var.get()))
            self.morph_close_iter = max(0, int(cl_var.get()))

            # Contour
            self.chain_approx_mode.set(ca_var.get())
            self.poly_epsilon_px.set(float(eps_var.get()))
            self.cnt_blur_enable.set(bool(blur_enable_var.get()))
            self.cnt_blur_method.set(blur_method_var.get())
            k2 = int(blur_ksize_var.get())
            if k2 % 2 == 0: k2 += 1
            self.cnt_blur_ksize.set(max(3, k2))

            win.destroy()

        btns = ttk.Frame(frm); btns.grid(row=11, column=0, columnspan=2, pady=(8,0))
        ttk.Button(btns, text="OK", command=apply_and_close).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side=tk.LEFT, padx=4)

    # ---------- 데이터셋 초기화 ----------
    def on_init_dataset(self):
        if messagebox.askyesno("확인", "누적 학습 데이터셋(X,y)을 모두 초기화할까요?"):
            self.X_list.clear(); self.y_list.clear()
            self.RGB_list.clear(); self.Cls_list.clear()
            self.is_trained = False; self.model = None
            self.last_mask = None; self.last_contours = []
            self.curr_fg.clear(); self.curr_bg.clear()
            self._update_counts(); self._redraw()
            messagebox.showinfo("완료", "누적 데이터셋이 초기화되었습니다. 새로 샘플링하세요.")

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
