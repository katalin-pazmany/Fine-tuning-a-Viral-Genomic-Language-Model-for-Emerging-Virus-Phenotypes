"""Bloom + title + family legend onto the Gephi family-coloured PNG (EID2 slide style)."""
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont

SRC = "/Users/puszedli/Untitled2.png"
OUT = "/Users/puszedli/viral-phenotype-finetuning/outputs/figures_umap/viral_network_family_titled.png"

TITLE = "The Vir2vec viral\nembedding space"
SUBTITLE = "1,086 human-associated viruses, linked to\ntheir nearest neighbours in the model's\nrepresentation and coloured by family"

# family -> (hex, share%)  read from the Gephi Appearance panel
FAMILIES = [
    ("Picornaviridae",   "#C86FD8", "24.2%"),
    ("Papillomaviridae", "#7CB342", "18.6%"),
    ("Anelloviridae",    "#5BB8F0", "5.9%"),
    ("Adenoviridae",     "#E8E6F0", "5.6%"),
    ("Peribunyaviridae", "#F0952A", "3.9%"),
    ("Flaviviridae",     "#E85D70", "3.4%"),
    ("Caliciviridae",    "#4CBFA0", "2.4%"),
    ("other families",   "#6E6E6E", "35.6%"),
]


def screen(x, y):
    return 1.0 - (1.0 - x) * (1.0 - y)


def bloom(rgb):
    hi = np.clip((rgb - 0.06) / 0.94, 0, 1)
    hi_im = Image.fromarray((hi * 255).astype(np.uint8))
    out = rgb.copy()
    for radius, gain in [(4, 0.9), (12, 0.7), (30, 0.45)]:
        g = np.asarray(hi_im.filter(ImageFilter.GaussianBlur(radius))).astype(np.float32) / 255.0
        out = screen(out, np.clip(g * gain, 0, 1))
    return np.clip(out, 0, 1)


def font(sz, bold=False):
    idx = 2 if bold else 7          # Avenir Next: Demi Bold / Regular
    try:
        return ImageFont.truetype("/System/Library/Fonts/Avenir Next.ttc", sz, index=idx)
    except Exception:
        return ImageFont.load_default()


net = Image.open(SRC).convert("RGB")
# trim black margins
a = np.asarray(net).astype(np.float32) / 255.0
m = a.max(2) > 0.06
ys, xs = np.where(m)
pad = 40
net = net.crop((max(xs.min()-pad,0), max(ys.min()-pad,0),
                min(xs.max()+pad,net.width), min(ys.max()+pad,net.height)))
# glow
net = Image.fromarray((bloom(np.asarray(net).astype(np.float32)/255.0)*255).astype(np.uint8))
S = net.height

# --- slide canvas: network left, text panel right ---
panel = int(net.width * 1.05)
W, H = net.width + panel, S
img = Image.new("RGB", (W, H), (0, 0, 0))
img.paste(net, (0, 0))
d = ImageDraw.Draw(img)

px = net.width + int(S * 0.05)          # left edge of panel text
TITLE_C, SUB_C, TXT_C = (174, 190, 235), (150, 150, 160), (235, 235, 235)

d.multiline_text((px, int(S*0.11)), TITLE, fill=TITLE_C,
                 font=font(int(S*0.048), bold=True), spacing=int(S*0.010))
d.multiline_text((px, int(S*0.28)), SUBTITLE, fill=SUB_C,
                 font=font(int(S*0.026)), spacing=int(S*0.013))

# legend
ly = int(S*0.47)
sw = int(S*0.032)
dy = int(S*0.058)
d.text((px, ly - int(S*0.052)), "Virus family", fill=TXT_C, font=font(int(S*0.036), bold=True))
for name, hexc, share in FAMILIES:
    c = tuple(int(hexc.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    d.rounded_rectangle([px, ly, px+sw, ly+sw], radius=int(sw*0.22), fill=c)
    d.text((px+sw+int(S*0.018), ly-int(S*0.002)), name, fill=TXT_C, font=font(int(S*0.032)))
    d.text((px+int(S*0.52), ly-int(S*0.002)), share, fill=SUB_C, font=font(int(S*0.030)))
    ly += dy

img.save(OUT)
print("saved", OUT, img.size)
