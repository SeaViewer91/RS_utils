"""
EXIF 정보를 읽어서 디지털 줌을 적용했는지 여부를 구분해주는 함수.
"""

from PIL import Image, ExifTags
from typing import Optional, Tuple

def get_focal_and_zoom(image_path: str) -> Tuple[Optional[float], float]:
    """
    이미지 파일의 EXIF에서
    1) 실제 초점거리(FocalLength, mm 단위)
    2) 디지털 줌 배율(DigitalZoomRatio)
    를 읽어 반환한다.
    """
    img = Image.open(image_path)
    exif_raw = img._getexif()
    if not exif_raw:
        raise ValueError("EXIF 정보를 찾을 수 없습니다.")

    exif = {
        ExifTags.TAGS.get(code, code): val
        for code, val in exif_raw.items()
    }

    focal = exif.get('FocalLength')
    if focal is None:
        focal_length: Optional[float] = None
    elif isinstance(focal, tuple):
        num, den = focal
        focal_length = (num / den) if den else None
    else:
        focal_length = float(focal)

    zoom = exif.get('DigitalZoomRatio')
    try:
        zoom_ratio = float(zoom) if zoom is not None else 1.0
    except (TypeError, ValueError):
        zoom_ratio = 1.0

    return focal_length, zoom_ratio

def is_zoom(image_path: str) -> int:
    """
    디지털 줌 배율이 1.1 초과이면 1, 아니면 0을 반환
    """
    _, zoom_ratio = get_focal_and_zoom(image_path)
    return 1 if zoom_ratio > 1.1 else 0

if __name__ == '__main__':
    # 테스트할 이미지 파일 경로
    path = r'src\DJI_20250714120517_0001_V_웨이포인트0.JPG'
    try:
        # 디지털 줌 배율과 줌 여부 계산
        _, zoom_ratio = get_focal_and_zoom(path)
        zoom_flag = is_zoom(path)
        # 함께 출력
        print(f"디지털 줌 배율: {zoom_ratio} 배, 줌 여부: {'줌' if zoom_flag else '줌 아님'}")
    except Exception as e:
        print(f"오류 발생: {e}")
