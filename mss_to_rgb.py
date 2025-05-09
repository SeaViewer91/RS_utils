"""
멀티스펙트럴 이미지(MSS)에서 R, G, B 이미지를 선별하여 RGB 이미지로 합성
"""

import os
import re
import rasterio
import numpy as np
import cv2

# === 사용자 설정 ===
# RedEdge 멀티스펙트럼 이미지가 저장된 최상위 디렉토리 경로
input_dir = "E:/MSS to RGB/geo001"

def normalize_band(band_array):
    """밴드 배열을 uint8 [0,255] 범위로 정규화"""
    arr = band_array.astype(np.float32)  # 배열을 float32로 변환
    min_val, max_val = np.nanmin(arr), np.nanmax(arr)  # 최소·최대값 계산
    if max_val > min_val:
        arr = (arr - min_val) / (max_val - min_val) * 255.0  # 선형 스케일링
    else:
        arr = np.zeros_like(arr)  # 단일값일 경우 0 배열 생성
    return arr.astype(np.uint8)  # uint8로 변환

def composite_rgb_to_jpg(input_dir):
    """
    TIFF 파일명에서 접두부, 밴드 문자(B/G/R), 숫자 추출
    예시: 220718KW034R0125.tif, 220718KW034B0125.tif, 220718KW034G0125.tif
    B, G, R 밴드 그룹핑 후 JPEG RGB 합성
    """
    pattern = re.compile(r"^(.+?)([BGR])(\d+)\.tif$", re.IGNORECASE)  # 파일명 패턴
    CHANNEL_ORDER = ['B', 'G', 'R']  # 합성 채널 순서

    for root, dirs, files in os.walk(input_dir):
        groups = {}
        for fname in files:
            match = pattern.match(fname)
            if not match:
                continue
            base, band, code = match.groups()
            key = f"{base}{code}"
            groups.setdefault(key, {})[band.upper()] = os.path.join(root, fname)

        if not groups:
            continue

        out_dir = os.path.join(root, 'RGB')
        os.makedirs(out_dir, exist_ok=True)  # out_dir 생성

        for key, ch_files in groups.items():
            if any(b not in ch_files for b in CHANNEL_ORDER):
                continue  # 채널 누락 시 건너뛰기

            norm_bands = []
            for band in CHANNEL_ORDER:
                with rasterio.open(ch_files[band]) as src:
                    band_arr = src.read(1)  # 밴드 읽기
                norm_bands.append(normalize_band(band_arr))  # 밴드 정규화

            bgr = cv2.merge(norm_bands)  # BGR 병합
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # RGB 변환

            out_path = os.path.join(out_dir, f"{key}_RGB.jpg")
            cv2.imwrite(out_path, rgb)  # JPEG 저장
            print(f"JPEG RGB 합성 이미지 저장: {out_path}")  # 저장 로그

if __name__ == "__main__":
    composite_rgb_to_jpg(input_dir)  # 함수 호출
# === 스크립트 종료 ===
