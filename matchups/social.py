"""Generate a deterministic 1200x630 matchup share card."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int, bold: bool = False):
    candidates = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        str(Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf")),
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _spread_label(home: str, away: str, spread: float) -> str:
    if spread > 0:
        return f"{home} -{abs(spread):.1f}"
    if spread < 0:
        return f"{away} -{abs(spread):.1f}"
    return "PICK'EM"


def render_social_card(detail: dict) -> bytes:
    game, prediction, status, result = (
        detail["game"], detail["prediction"], detail["status"], detail["result"]
    )
    image = Image.new("RGB", (1200, 630), "#0B0F14")
    draw = ImageDraw.Draw(image)
    accent = "#35D08A" if status["label"] == "HIGH" else "#3D95CE"
    text, dim, surface, border = "#E7ECF3", "#93A0B1", "#121821", "#263244"

    draw.rounded_rectangle((54, 46, 1146, 584), radius=30, fill=surface, outline=border, width=2)
    draw.rounded_rectangle((54, 46, 1146, 58), radius=6, fill=accent)
    draw.text((92, 87), "JOSCHO ANALYTICS", font=_font(27, True), fill=accent)
    demo = "HISTORICAL DEMO" if game.get("historical_demo") else "WEEKLY PREDICTION"
    badge_font = _font(20, True)
    badge_box = draw.textbbox((0, 0), demo, font=badge_font)
    badge_w = badge_box[2] - badge_box[0] + 36
    draw.rounded_rectangle((1054 - badge_w, 82, 1054, 122), radius=12, fill="#1A2230")
    draw.text((1072 - badge_w, 91), demo, font=badge_font, fill=dim)

    away, home = game["away_team"], game["home_team"]
    draw.text((92, 161), f"{away}  @  {home}", font=_font(76, True), fill=text)
    draw.text(
        (94, 253),
        f"{game['season']} WEEK {game['week']}  ·  {game.get('gameday') or 'DATE TBD'}",
        font=_font(25, True),
        fill=dim,
    )

    draw.rounded_rectangle((92, 315, 410, 490), radius=18, fill="#0E141C", outline=border, width=2)
    draw.text((118, 342), "MODEL MARGIN", font=_font(21, True), fill=dim)
    margin = float(prediction["projected_margin"])
    winner = home if margin >= 0 else away
    draw.text((118, 386), f"{winner} {abs(margin):.1f}", font=_font(42, True), fill=text)

    draw.rounded_rectangle((441, 315, 759, 490), radius=18, fill="#0E141C", outline=border, width=2)
    draw.text((467, 342), "MARKET", font=_font(21, True), fill=dim)
    draw.text(
        (467, 391),
        _spread_label(home, away, float(prediction["market_spread"])),
        font=_font(36, True),
        fill=text,
    )

    draw.rounded_rectangle((790, 315, 1108, 490), radius=18, fill="#0E141C", outline=accent, width=3)
    draw.text((816, 342), "MODEL EDGE", font=_font(21, True), fill=dim)
    draw.text((816, 386), f"{abs(float(prediction['model_edge'])):.1f} PTS", font=_font(42, True), fill=accent)

    status_text = status["label"]
    if result.get("status") == "final" and result.get("ats_result"):
        status_text += f"  ·  {result['ats_result']}"
    draw.text((94, 530), status_text, font=_font(25, True), fill=accent)
    draw.text((1108, 530), "joschoanalytics.streamlit.app", font=_font(20), fill=dim, anchor="ra")

    out = BytesIO()
    image.save(out, format="PNG", optimize=True)
    return out.getvalue()
