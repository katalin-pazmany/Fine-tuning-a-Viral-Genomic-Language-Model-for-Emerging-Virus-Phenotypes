"""Add a photographic bloom to a Gephi black-background render.

Multi-scale Gaussian blur of the bright filaments, Screen-composited back over
the sharp original -> the light bleeds outward into a halo. Also trims the empty
black page margins so the network fills the frame.
"""
import numpy as np
from PIL import Image, ImageFilter

SRC = "Untitled.pdf.png"
OUT_FULL = "viral_network_glow.png"

im = Image.open(SRC).convert("RGB")

# qlmanage stamps a bright 1px frame around the thumbnail -> strip a small border
b = 24
im = im.crop((b, b, im.width - b, im.height - b))

# --- trim black margins (keep a small pad) ---
a = np.asarray(im).astype(np.float32) / 255.0
lum = a.max(2)
mask = lum > 0.10          # ignore the faint thumbnail vignette at the page edges
ys, xs = np.where(mask)
pad = 90
y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, a.shape[0])
x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, a.shape[1])
im = im.crop((x0, y0, x1, y1))
base = np.asarray(im).astype(np.float32) / 255.0
print("cropped to", im.size)


def screen(x, y):
    return 1.0 - (1.0 - x) * (1.0 - y)


# highlight pass: only reasonably bright pixels contribute to the glow
hi = np.clip((base - 0.06) / (1.0 - 0.06), 0, 1)
hi_im = Image.fromarray((hi * 255).astype(np.uint8))

result = base.copy()
# a few blur scales, each screened on -> soft wide halo + tight core glow
for radius, gain in [(4, 0.9), (12, 0.7), (30, 0.45)]:
    g = np.asarray(hi_im.filter(ImageFilter.GaussianBlur(radius))).astype(np.float32) / 255.0
    result = screen(result, np.clip(g * gain, 0, 1))

out = Image.fromarray((result * 255).astype(np.uint8))
out.save(OUT_FULL)
print("saved", OUT_FULL, out.size)
