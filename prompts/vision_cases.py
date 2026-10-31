"""Fine-detail images for the image-fidelity probe (experiments/image_fidelity.py).

Each image is a 1536 x 1536 white page of random 5-character codes printed in bands of shrinking font size (36 px
down to 7 px). Reading a band needs enough resolution: an endpoint that silently downscales the image (to save
compute) loses the small bands first, and when the model bills image tokens by pixel count (Qwen-VL) its prompt
token count drops too. Codes avoid look-alike characters (0/O, 1/I/L, 5/S, 8/B, 2/Z).
"""
import base64
import io
import random

from PIL import Image, ImageDraw, ImageFont

ALPHABET = "ACDEFHJKMNPRTUVWXY34679"
SIZES = [36, 24, 16, 12, 9, 7]
PER_BAND = 4
SIDE = 1536
FONT = "/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Regular.ttf"
PROMPT = ("This image contains short codes made of 5 letters or digits, printed in rows from large to tiny. "
          "Read every code you can, in order from top to bottom and left to right, one per line. Write only the "
          "codes, exactly as printed; skip any you cannot read.")


def make(seed: int):
    """-> (png bytes, {font size: [codes]})"""
    rng = random.Random(f"llmverify-vision-v1:{seed}")
    img = Image.new("L", (SIDE, SIDE), 255)
    d = ImageDraw.Draw(img)
    bands, y = {}, 60
    for size in SIZES:
        font = ImageFont.truetype(FONT, size)
        codes = ["".join(rng.choice(ALPHABET) for _ in range(5)) for _ in range(PER_BAND)]
        bands[size] = codes
        x = 80
        for c in codes:
            d.text((x, y), c, fill=0, font=font)
            x += (SIDE - 160) // PER_BAND
        y += int(size * 1.6) + 140
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue(), bands


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def grade(bands: dict, reply: str) -> dict:
    """Share of each band's codes that appear in the reply (case-insensitive, spaces ignored)."""
    text = (reply or "").upper().replace(" ", "")
    return {size: sum(c in text for c in codes) / len(codes) for size, codes in bands.items()}
