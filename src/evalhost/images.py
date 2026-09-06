"""관측 이미지 디코드. JPEG는 꺼낸 카메라만, depth는 16-bit mm PNG를 미터로 변환한다."""

from io import BytesIO

import numpy as np

try:  # simplejpeg이 있으면 자동으로 쓴다 (없으면 PIL 폴백 — 순수 선택 의존성)
    import simplejpeg

    def decode_jpeg(buf) -> np.ndarray:
        """JPEG 바이트를 (H, W, 3) uint8 RGB 배열로 디코드한다."""
        return simplejpeg.decode_jpeg(bytes(buf), colorspace="RGB")
except ImportError:
    def decode_jpeg(buf) -> np.ndarray:
        """JPEG 바이트를 (H, W, 3) uint8 RGB 배열로 디코드한다."""
        from PIL import Image

        return np.asarray(Image.open(BytesIO(buf)).convert("RGB"))


class LazyImages(dict):
    """관측 이미지 dict. **꺼낸 카메라만** JPEG 디코드한다 (디코드 결과는 캐시된다).

    `obs["images"]["head_l"]` -> numpy (H, W, 3) uint8. 정책이 손목 캠을 안 쓰면 그 캠은
    디코드 비용이 아예 들지 않는다. bytes가 아니면(무압축 전송) 그대로 통과시킨다.
    """

    def __getitem__(self, cam):
        img = super().__getitem__(cam)
        if isinstance(img, (bytes, bytearray)):
            img = decode_jpeg(img)
            super().__setitem__(cam, img)
        return img

    def get(self, cam, default=None):
        return self[cam] if cam in self else default

    def values(self):
        return (self[cam] for cam in self)

    def items(self):
        return ((cam, self[cam]) for cam in self)


def decode_depth(data) -> np.ndarray:
    """head depth: 16-bit(mm) PNG -> (H, W) float32 **미터**. 0은 무효값 그대로 0으로 남는다.

    이 depth는 시뮬레이터의 이상화(idealized) depth다. 실기 ZED Mini가 주는 스테레오
    depth와 달리 노이즈·구멍(무텍스처/반사면)·시차 한계가 전혀 없는 완벽한 값이다. 이 값에
    그대로 의존하는 정책은 실기에서 성능이 떨어진다 — 학습 시 노이즈/드롭아웃 증강을 권장한다.
    손목(wrist)에는 depth가 제공되지 않는다(head_l만).

    Args:
        data: 16-bit PNG 바이트. 이미 배열이면 그대로 통과한다.

    Returns:
        (H, W) float32 미터 배열.
    """
    if not isinstance(data, (bytes, bytearray)):
        return data  # 이미 배열이면 그대로 (테스트·무압축 경로)
    from PIL import Image

    return np.asarray(Image.open(BytesIO(data)), np.float32) / 1000.0
