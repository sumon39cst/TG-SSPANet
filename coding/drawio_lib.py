"""Tiny draw.io (mxGraph XML) writer with a matching matplotlib preview.

Coordinates are given in millimetres with the origin at the bottom-left (as in 07_paper_figures.py) and converted to
draw.io pixels (origin top-left).  Every primitive is written as an editable draw.io cell and also drawn in a preview
PNG, stacked in insertion order exactly like draw.io, so the layout can be checked without the draw.io application.
The preview uses DejaVu Sans Bold, which is wider than draw.io's Helvetica, so it is a conservative overflow check.
"""
import base64
import html
import io
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

PREVIEW_DPI = int(os.environ.get("DRAWIO_PREVIEW_DPI", 96))   # 288 gives print-resolution (3x) figure PNGs
S = 8.0                       # px per mm
PT = 0.3528 * S               # px per typographic point at this scale
FONT = "Helvetica"


class Diagram:
    def __init__(self, W, H, name, ymin=0.0):
        """W, H: canvas in mm; content below y = ymin (mm) is cropped from the page."""
        self.W, self.H, self.name, self.ymin = W, H, name, ymin
        self.cells = []; self.n = 2; self.zc = 1.0
        Hc = H - ymin
        self.fig = plt.figure(figsize=(W * S / 96, Hc * S / 96), dpi=96)
        self.ax = self.fig.add_axes([0, 0, 1, 1]); self.ax.set_xlim(0, W * S); self.ax.set_ylim(Hc * S, 0)
        self.ax.set_aspect("equal"); self.ax.axis("off")

    # ---------------------------------------------------------------- helpers
    def px(self, x, y):
        return x * S, (self.H - y) * S

    def _z(self):
        self.zc += 1; return self.zc

    def _id(self):
        self.n += 1; return f"c{self.n}"

    def _vertex(self, value, style, X, Y, w, h):
        self.cells.append(f'<mxCell id="{self._id()}" value="{value}" style="{style}" vertex="1" parent="1">'
                          f'<mxGeometry x="{X:.1f}" y="{Y:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry"/></mxCell>')

    @staticmethod
    def _html(text):
        """Plain text with \\n -> <br>, _{..} -> subscript, ^{..} -> superscript; escaped for the XML attribute."""
        t = html.escape(text, quote=False)
        t = re.sub(r"_\{(.*?)\}", r"<sub>\1</sub>", t); t = re.sub(r"\^\{(.*?)\}", r"<sup>\1</sup>", t)
        return html.escape(t.replace("\n", "<br>"))

    @staticmethod
    def _plain(text):
        t = re.sub(r"_\{(.*?)\}", r"$_{\1}$", text); t = re.sub(r"\^\{(.*?)\}", r"$^{\1}$", t)
        return t.replace("‖", "|")

    @staticmethod
    def _shade(c, f):
        c = np.array(matplotlib.colors.to_rgb(c))
        return tuple(np.clip(c * f if f < 1 else c + (1 - c) * (f - 1), 0, 1))

    # ---------------------------------------------------------------- primitives
    def box(self, x, y, w, h, text="", fill="#FFFFFF", stroke="#555555", fs=6.2, bold=True, rounded=True,
            dashed=False, color="#000000", lw=1.2, italic=False, align="center", valign="middle", r=1.0):
        X, Y = self.px(x, y + h); Wp, Hp = w * S, h * S
        fstyle = (1 if bold else 0) + (2 if italic else 0)
        arc = max(4, int(r * S / max(min(Wp, Hp), 1) * 100))
        style = (f"rounded={1 if rounded else 0};arcSize={arc};whiteSpace=wrap;html=1;fillColor={fill};"
                 f"strokeColor={stroke};strokeWidth={lw};fontFamily={FONT};fontSize={round(fs * PT)};fontStyle={fstyle};"
                 f"fontColor={color};align={align};verticalAlign={valign};spacing=2;"
                 f"{'dashed=1;dashPattern=6 4;' if dashed else ''}")
        self._vertex(self._html(text), style, X, Y, Wp, Hp)
        z = self._z()
        self.ax.add_patch(FancyBboxPatch((X, Y), Wp, Hp, boxstyle=f"round,pad=0,rounding_size={r * S if rounded else 0}",
                                         fc="none" if fill == "none" else fill, ec="none" if stroke == "none" else stroke,
                                         lw=lw * 0.75, ls=(0, (6, 4)) if dashed else "-", zorder=z))
        if text:
            tx = X + Wp / 2 if align == "center" else (X + 3 if align == "left" else X + Wp - 3)
            ty = Y + Hp / 2 if valign == "middle" else (Y + 3 if valign == "top" else Y + Hp - 3)
            self.ax.text(tx, ty, self._plain(text), ha=align, va={"middle": "center", "top": "top", "bottom": "bottom"}[valign],
                         fontsize=fs * PT * 0.75, color=color, fontweight="bold" if bold else "normal",
                         style="italic" if italic else "normal", zorder=z + 0.5, linespacing=1.15, family="DejaVu Sans")

    def text(self, x, y, s, fs=6.2, ha="center", va="center", bold=True, color="#000000", italic=False):
        lines = s.split("\n"); fpx = fs * PT
        est_w = max(len(l) for l in lines) * fpx * 0.62 + 10
        est_h = len(lines) * fpx * 1.3 + 6
        X, Y = self.px(x, y)
        X0 = X - est_w / 2 if ha == "center" else (X if ha == "left" else X - est_w)
        Y0 = Y - est_h / 2 if va == "center" else (Y if va == "top" else Y - est_h)
        valign = {"center": "middle", "top": "top", "bottom": "bottom"}[va]
        style = (f"text;html=1;whiteSpace=nowrap;align={ha};verticalAlign={valign};fontFamily={FONT};"
                 f"fontSize={round(fpx)};fontStyle={(1 if bold else 0) + (2 if italic else 0)};fontColor={color};spacing=0;")
        self._vertex(self._html(s), style, X0, Y0, est_w, est_h)
        self.ax.text(X, Y, self._plain(s), ha=ha, va=va, fontsize=fpx * 0.75, color=color,
                     fontweight="bold" if bold else "normal", style="italic" if italic else "normal", zorder=self._z(),
                     linespacing=1.15, family="DejaVu Sans")

    def cube(self, x, y, w, h, d, fill, stroke, label=None, above=False, fs=5.8):
        """Feature-map block: front face (x, y, w, h) in mm, depth d drawn up-right at 45 degrees."""
        size = d * 0.5 * S
        X, Y = self.px(x, y + h); Wp, Hp = w * S, h * S
        style = (f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.08;darkOpacity2=0.2;"
                 f"size={size:.1f};flipH=1;fillColor={fill};strokeColor={stroke};strokeWidth=1.2;")
        self._vertex("", style, X, Y - size, Wp + size, Hp + size)
        z = self._z()
        self.ax.add_patch(Polygon([(X, Y), (X + size, Y - size), (X + Wp + size, Y - size), (X + Wp, Y)],
                                  fc=self._shade(fill, 1.25), ec=stroke, lw=0.9, zorder=z))
        self.ax.add_patch(Polygon([(X + Wp, Y), (X + Wp + size, Y - size), (X + Wp + size, Y + Hp - size), (X + Wp, Y + Hp)],
                                  fc=self._shade(fill, 0.8), ec=stroke, lw=0.9, zorder=z))
        self.ax.add_patch(Rectangle((X, Y), Wp, Hp, fc=fill, ec=stroke, lw=0.9, zorder=z))
        if label:
            cx = x + w / 2 + d * 0.25
            if above:
                self.text(cx, y + h + d * 0.5 + 1.0, label, fs=fs, va="bottom")
            else:
                self.text(cx, y - 1.0, label, fs=fs, va="top")

    def node(self, x, y, s, r=2.0, fs=8):
        X, Y = self.px(x, y); R = r * S
        style = (f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#FFFFFF;strokeColor=#333333;strokeWidth=1.3;"
                 f"fontFamily={FONT};fontSize={round(fs * PT)};fontStyle=1;")
        self._vertex(html.escape(s), style, X - R, Y - R, 2 * R, 2 * R)
        z = self._z()
        self.ax.add_patch(Circle((X, Y), R, fc="white", ec="#333333", lw=1, zorder=z))
        self.ax.text(X, Y, s, ha="center", va="center", fontsize=fs * PT * 0.75, fontweight="bold", zorder=z + 0.5,
                     family="DejaVu Sans")

    def edge(self, pts, color="#333333", lw=1.4, dashed=False, arrow=True):
        P = [self.px(*p) for p in pts]
        style = (f"endArrow={'block' if arrow else 'none'};endFill=1;endSize=7;html=1;rounded=0;strokeColor={color};"
                 f"strokeWidth={lw};{'dashed=1;dashPattern=5 4;' if dashed else ''}")
        way = "".join(f'<mxPoint x="{a:.1f}" y="{b:.1f}"/>' for a, b in P[1:-1])
        self.cells.append(
            f'<mxCell id="{self._id()}" value="" style="{style}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
            f'<mxPoint x="{P[0][0]:.1f}" y="{P[0][1]:.1f}" as="sourcePoint"/><mxPoint x="{P[-1][0]:.1f}" y="{P[-1][1]:.1f}" as="targetPoint"/>'
            + (f'<Array as="points">{way}</Array>' if way else "") + "</mxGeometry></mxCell>")
        z = self._z(); xs, ys = zip(*P)
        self.ax.plot(xs, ys, color=color, lw=lw * 0.75, ls=(0, (5, 4)) if dashed else "-", zorder=z)
        if arrow:
            self.ax.annotate("", xy=P[-1], xytext=P[-2], zorder=z,
                             arrowprops=dict(arrowstyle="-|>", color=color, lw=lw * 0.75, mutation_scale=10))

    def image(self, x, y, w, h, png_bytes, border="#333333"):
        X, Y = self.px(x, y + h); Wp, Hp = w * S, h * S
        b64 = base64.b64encode(png_bytes).decode()
        style = (f"shape=image;verticalLabelPosition=bottom;verticalAlign=top;imageAspect=0;"
                 f"image=data:image/png,{b64};imageBorder={border};strokeWidth=1;")
        self._vertex("", style, X, Y, Wp, Hp)
        z = self._z(); arr = plt.imread(io.BytesIO(png_bytes))
        self.ax.imshow(arr, extent=(X, X + Wp, Y + Hp, Y), zorder=z)
        self.ax.add_patch(Rectangle((X, Y), Wp, Hp, fc="none", ec=border, lw=0.8, zorder=z + 0.1))

    # ---------------------------------------------------------------- output
    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        W, H = int(self.W * S), int((self.H - self.ymin) * S)
        xml = (f'<mxfile host="drawio" type="device"><diagram name="{self.name}" id="{self.name}">'
               f'<mxGraphModel dx="{W}" dy="{H}" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" '
               f'fold="1" page="1" pageScale="1" pageWidth="{W}" pageHeight="{H}" math="0" shadow="0" background="#FFFFFF">'
               f'<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(self.cells) +
               "</root></mxGraphModel></diagram></mxfile>")
        p = os.path.join(out_dir, self.name + ".drawio")
        io.open(p, "w", encoding="utf8").write(xml)
        self.fig.savefig(os.path.join(out_dir, self.name + "_preview.png"), dpi=PREVIEW_DPI)
        plt.close(self.fig)
        return p


def png_from_array(arr, cmap="gray", vmin=0, vmax=1, over=None, ocmap="jet", oalpha=0.45, gt=None, pred=None,
                   size=200, interp="bilinear", rgb=False):
    """Render an image (+ overlay / contours) to PNG bytes of `size` px (scaled with the preview resolution)."""
    size = int(size * PREVIEW_DPI / 96)
    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(arr, cmap=None if rgb else cmap, vmin=vmin, vmax=vmax, interpolation=interp)
    if over is not None:
        ax.imshow(over, cmap=ocmap, vmin=0, vmax=1, alpha=oalpha, interpolation=interp,
                  extent=(-0.5, arr.shape[1] - 0.5, arr.shape[0] - 0.5, -0.5))
    if gt is not None:
        ax.contour(gt.astype(float), levels=[0.5], colors="#00E676", linewidths=1.6)
    if pred is not None:
        ax.contour(pred.astype(float), levels=[0.5], colors="#FF1744", linewidths=1.6)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100); plt.close(fig)
    return buf.getvalue()
