import numpy as np
import pytest
from apt_colorize import APTColorizer2D


@pytest.fixture
def dummy_gray_image():
    ch_a = np.full((32, 128), 100, dtype=np.uint8)
    ch_b = np.full((32, 128), 50, dtype=np.uint8)
    interleaved = np.empty((64, 128), dtype=np.uint8)
    interleaved[::2] = ch_a
    interleaved[1::2] = ch_b
    return interleaved


def test_colorize_array_shape(dummy_gray_image):
    colorizer = APTColorizer2D()
    out = colorizer.colorize(dummy_gray_image)
    assert out.shape == (64, 128, 3)
    assert out.dtype == np.uint8


def test_colorize_from_file(tmp_path):
    colorizer = APTColorizer2D()
    img = np.random.randint(0, 255, size=(64, 128), dtype=np.uint8)
    path = tmp_path / "test.png"
    from PIL import Image
    Image.fromarray(img).save(path)
    out = colorizer.colorize(str(path))
    assert out.shape[0] == 64
    assert out.shape[1] == 128
    assert out.shape[2] == 3


def test_colorize_invalid_input():
    colorizer = APTColorizer2D()
    with pytest.raises(TypeError):
        colorizer.colorize(12345)

def test_colorize_odd_height(dummy_gray_image):
    colorizer = APTColorizer2D()
    cropped = dummy_gray_image[:-1]  
    result = colorizer.colorize(cropped)
 
    assert result.shape[0] % 2 == 0

