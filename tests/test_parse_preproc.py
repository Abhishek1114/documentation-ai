import numpy as np
from PIL import Image
from src.parsing.parse import _remove_ruling


def test_remove_ruling():
    # White 400x300 image
    img_arr = np.full((300, 400), 255, dtype=np.uint8)

    # 3px-thick horizontal line of length 300
    img_arr[30:33, 50:350] = 0
    # 3px-thick vertical line of length 200
    img_arr[50:250, 20:23] = 0
    # 20x20 black square
    img_arr[100:120, 100:120] = 0
    # 40x8 bar
    img_arr[150:158, 100:140] = 0
    # 12x50 vertical stem
    img_arr[100:150, 200:212] = 0

    img = Image.fromarray(img_arr)
    out_img = _remove_ruling(img, k=60)
    out = np.array(out_img)

    # Assert both lines are fully white after _remove_ruling(k=60)
    assert np.all(out[30:33, 50:350] == 255), "Horizontal line not fully whitened"
    assert np.all(out[50:250, 20:23] == 255), "Vertical line not fully whitened"

    # Assert square, bar and stem remain mostly dark
    assert (out[100:120, 100:120] < 128).mean() > 0.8, "Square did not remain mostly dark"
    assert (out[150:158, 100:140] < 128).mean() > 0.8, "Bar did not remain mostly dark"
    assert (out[100:150, 200:212] < 128).mean() > 0.8, "Stem did not remain mostly dark"
