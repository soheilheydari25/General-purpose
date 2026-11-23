#!/usr/bin/env python3
"""
Culvert 2D (single barrel, 15.9 m slide-rail shoring BOX, 4 rails / 3 bays, 2 strut levels) - staged excavation, 3D time-lapse
======================================================================
Derived from the 2B model (v3) and revised for a single-barrel culvert using
"Typical cross section J" and the supplier slide-rail plan / Section A / Section B.

Levels in metres, 0.0 = pavement level.  x runs along the culvert axis
(x = 0 is the shoring line next to the traffic = ROAD SIDE, culvert extends to -x),
y runs along the road (chainage).

Box (supplier drawing): trench 16.294 x 2.790, post c/c 15.864 x 2.360, clear 15.724 x 2.060,
4 sliding rails (post length 8.665) at 5.307 / 5.250 / 5.307 c/c, 3 bays, 4 panel tiers
(outer upper 1.35, outer lower 2.385, inner upper 1.95, inner lower 2.385), excavation depth 7.1 m,
2 DSR strut levels at the two middle rails (upper at the outer upper/lower joint,
lower with 2.42 m clearance beneath it for the pipes).

Sequence (9 stages, as 2B, but the digger works FROM THE ROAD SIDE: panels are set road end -> far end,
and everything is pulled out far end -> road end, so the road-side end is the LAST to come out)
   1  Digger pad built from the road side
   2  Rails + outer panels set one by one from the road side (panel, rail, panel, rail ...) - half in / half out
   3  Panels and rails pushed down one by one, excavating inside as needed
   4  Rails to full depth, upper struts fitted, excavation undermines the outer panels
   5  Inner panels installed, lower struts fitted, excavation to foundation, inner lower panels
   6  Foundation test, then pipes placed and surveyed
   7  Backfill in layers; each strut removed when the fill reaches just under it; inner panels lifted as fill rises
   8  Outer panels + rails out far end -> road side (fill >= 1.5 m below the pad), backfill to existing level
   9  Far end excavated by 1:1 stepped batter / benching for end structure + rock protection

Usage
  pip install numpy plotly
  python culvert_2D_staging_3d.py            # -> interactive HTML
  python culvert_2D_staging_3d.py --video    # -> MP4 (needs: pip install kaleido, ffmpeg)
  python culvert_2D_staging_3d.py --png 5    # -> stage_5.png
"""
import argparse
import copy
import math
import os
import shutil
import subprocess
import tempfile

import numpy as np
import plotly.graph_objects as go

# =============================== PARAMETERS ================================
# existing ground (dotted line): (x along culvert, z).  x=0 = shoring line (road side).  Falls to the outlet level beyond the far end.
GROUND_PTS = [(-30, -7.40), (-24.5, -7.40), (-17.0, -1.90), (-14.5, -1.55), (-10.0, -1.30),
              (-6.0, -1.05), (-2.0, -0.88), (0.0, -0.80), (2.4, -1.82),
              (5.15, -3.40), (6.2, -4.35), (6.4, -4.45), (6.6, -5.99), (7.1, -6.27),
              (9.3, -5.90), (13.9, -5.83), (18.9, -5.55), (24.0, -5.68)]

LAYER_BOT = [-2.3, -3.8, -5.3, -6.8, -7.9]   # 1.5 m lifts (last lift 1.1 m): 7.1 m from the road-end ground (-0.8) to foundation
NL = len(LAYER_BOT)
Z_FOUND = LAYER_BOT[-1]                      # foundation level

PIPE_OD, PIPE_L, N_PIPES = 1.2, 2.45, 6      # single barrel, 6 pipes (pipe 1 against the existing culvert)
N_BARRELS = 1
BARREL_Y = [0.0]
BEDDING = 0.4                                # crown stays 0.1 m under the lower strut (2.42 m strut clearance)
JOINT = -0.8                                 # joint with the existing culvert
PIPE_ORDER = [1, 0, 2, 3, 4, 5]              # pipe 2 goes in before pipe 1 (it is slid under the strut)
POX = {1: 0.45, 4: -0.40}                    # pipes that straddle a strut: lowered beside it, then slid under it

# digger pad (orange line): top level, plan half-width.  Built from the road side outwards.
PADX0, PAD_W = -19.5, 6.0
PAD_PROFILE = ([PADX0, -15.0, 0.0], [-2.40, -0.80, -0.80])

# slide-rail box (supplier plan + Section A/B): 4 rails (8.665 m), 3 bays, closed ends, 2.06 m clear width
RAIL_X, RAIL_W = [-16.079, -10.772, -5.522, -0.215], 0.30      # far end -> road side (post c/c 5.307 / 5.250 / 5.307)
RAIL_LEN = 8.665                                               # post length - longer than the excavation is deep
CLEAR_HW = 1.03                                                # clear width 2.06 m
RAIL_Z = (-8.67, -8.67 + RAIL_LEN)                             # seated: toe just under the panel stack, top stands ~0.8 m above the pad
RAIL_Y = (CLEAR_HW - 0.02, CLEAR_HW + 0.36)
OUT_Y, IN_Y = (CLEAR_HW + 0.14, CLEAR_HW + 0.28), (CLEAR_HW, CLEAR_HW + 0.14)                # outer / inner panel planes
TIER = dict(out=(-4.285, -0.55), iu=(-6.235, -4.285), il=(-8.62, -6.235))   # outer 1.35+2.385 / inner upper 1.95 / inner lower 2.385 (0.25 upstand over the pad)
OUT_SPLIT = -1.90                                              # joint between outer upper (1.35) and outer lower (2.385) panels
BAYS = [(RAIL_X[i] + RAIL_W / 2 + 0.01, RAIL_X[i + 1] - RAIL_W / 2 - 0.01) for i in range(3)]   # far bay, middle bay, road bay
END_OUT = [(-16.294, -16.154), (-0.14, 0.0)]
END_IN = [(-16.154, -16.014), (-0.28, -0.14)]
VX0, VX1, BOX_IN = -16.01, -0.28, CLEAR_HW + 0.08               # excavated void inside the box (15.7 m x 2.06 m)
HID = 7.0                                                       # lift height at which an element is "out"
P_OUT, P_RAIL1, P_RAIL0 = 1.4, 1.85, 4.15                       # half-in penetration offsets (stage 2/3)

# DSR struts at the two middle rails: z range of each level (upper at the outer upper/lower joint,
# lower with 2.42 m clearance beneath it), strut index = level * 2 + j, j: 0 = rail 3 (far side), 1 = rail 2
STRUT_Z = [(-2.05, -1.75), (-6.20, -5.90)]
STRUT_RAILS = [1, 2]

# stage 9 end excavation (far end): stepped 1:1, 1.5 m benches, per lift (top -> bottom)
XCUT = [-9.0, -10.5, -12.0, -13.5, -15.0]
YCUT = [8.0, 6.5, 5.0, 3.5, 2.5]

TOTAL_DAYS = 12                                         # programme length (days)
XMIN, XMAX, YMAX, ZMIN, ZMAX = -26.0, 15.0, 9.0, -12.0, 9.0
DROP = HID

# derived
PIPE_Z0 = Z_FOUND + BEDDING
PIPE_ZC = PIPE_Z0 + PIPE_OD / 2
PIPES = [(JOINT - (i + 1) * PIPE_L, JOINT - i * PIPE_L) for i in range(N_PIPES)]

# colours
C_OUT, C_OUT2, C_IU, C_IL, C_RAIL = "#ff7a2b", "#f06a1c", "#ff9a45", "#e8481f", "#6f1d3b"
C_STRUT, C_STRUT2 = "#2b6cb0", "#1e4e8c"
C_PIPE, C_BED, C_EXIST = "#2fa75d", "#b3b0a6", "#9aa0a6"
C_LAYER = ["#d2b17a", "#c49a64", "#b48a58", "#a47b4f", "#93673f"]
C_BASE = "#8c8a83"
C_BF = ["#eadba8", "#dccb92", "#eadba8", "#dccb92", "#eadba8"]
C_GRASS, C_ASPH, C_GRAVEL, C_PAD = "#86a95a", "#3b4048", "#bdb7a6", "#a89f8e"
C_DIG, C_TRACK = "#f2b705", "#3a3f45"
# ===========================================================================

_GX = [p[0] for p in GROUND_PTS]
_GZ = [p[1] for p in GROUND_PTS]
ground = lambda x: float(np.interp(x, _GX, _GZ))
clip = lambda v, lo, hi: min(max(v, lo), hi)


class Mesh:
    def __init__(self):
        self.V, self.F, self.C = [], [], []

    def _v(self, p):
        self.V.append(tuple(p))
        return len(self.V) - 1

    def quad(self, a, b, c, d, col):
        self.F += [(a, b, c), (a, c, d)]
        self.C += [col, col]

    def tri(self, a, b, c, col):
        self.F.append((a, b, c))
        self.C.append(col)

    def box(self, x0, x1, y0, y1, z0, z1, col, top=None):
        if x1 - x0 < 1e-4 or y1 - y0 < 1e-4 or z1 - z0 < 1e-4:
            return
        p = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
             (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        v = [self._v(q) for q in p]
        self.quad(v[0], v[3], v[2], v[1], col)
        self.quad(v[4], v[5], v[6], v[7], top or col)
        self.quad(v[0], v[1], v[5], v[4], col)
        self.quad(v[2], v[3], v[7], v[6], col)
        self.quad(v[3], v[0], v[4], v[7], col)
        self.quad(v[1], v[2], v[6], v[5], col)

    def obeam(self, p0, p1, wy, wz, col):
        """oriented prism from p0 to p1 with a wy x wz cross-section (used for the digger)."""
        p0, p1 = np.array(p0, float), np.array(p1, float)
        u = p1 - p0
        u /= np.linalg.norm(u)
        v = np.cross((0, 0, 1.0), u)
        v = v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-6 else np.array([0, 1.0, 0])
        w = np.cross(u, v)
        rings = []
        for p in (p0, p1):
            rings.append([self._v(p + sv * wy / 2 * v + sw * wz / 2 * w)
                          for sv, sw in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        for i in range(4):
            j = (i + 1) % 4
            self.quad(rings[0][i], rings[0][j], rings[1][j], rings[1][i], col)
        self.quad(*rings[0], col)
        self.quad(*rings[1][::-1], col)

    def cylinder_x(self, x0, x1, zc, r, col, n=18, yc=0.0):
        r0, r1 = [], []
        for k in range(n):
            a = 2 * math.pi * k / n
            y, z = yc + r * math.cos(a), zc + r * math.sin(a)
            r0.append(self._v((x0, y, z)))
            r1.append(self._v((x1, y, z)))
        for k in range(n):
            k2 = (k + 1) % n
            self.quad(r0[k], r0[k2], r1[k2], r1[k], col)
        c0, c1 = self._v((x0, yc, zc)), self._v((x1, yc, zc))
        for k in range(n):
            k2 = (k + 1) % n
            self.tri(c0, r0[k2], r0[k], col)
            self.tri(c1, r1[k], r1[k2], col)

    def trace(self, name, opacity=1.0):
        op = opacity
        if self.V:
            V = np.round(np.array(self.V, float), 3)
            F = np.array(self.F, int)
            x, y, z = V[:, 0].tolist(), V[:, 1].tolist(), V[:, 2].tolist()
            i, j, k = F[:, 0].tolist(), F[:, 1].tolist(), F[:, 2].tolist()
            fc = list(self.C)
        else:
            x = y = z = [0, 0, 0]
            i, j, k, fc, op = [0], [1], [2], ["#ffffff"], 0.0
        return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, facecolor=fc, opacity=op,
                         flatshading=True, name=name, hoverinfo="skip", showscale=False,
                         lighting=dict(ambient=0.68, diffuse=0.7, specular=0.0,
                                       roughness=1.0, fresnel=0.0),
                         lightposition=dict(x=-3000, y=-2000, z=5000))


def pad_profile(x):
    return float(np.interp(x, *PAD_PROFILE))


def pad_reach(pad):
    """x of the far edge of the pad: it is built from the road side (x = 0) outwards"""
    return PADX0 * pad


def surf_z(x, pad):
    """existing ground, or the digger pad where it has been built"""
    g = ground(x)
    if PADX0 <= x <= 0 and x >= pad_reach(pad) - 1e-6:
        return max(g, pad_profile(x))
    return g


# --------------------------------- soil -------------------------------------
def x_slabs():
    left = set(np.arange(XMIN, 0.01, 1.0)) | {BOX_X0, VX0, VX1, PADX0, 0.0} | set(XCUT)
    left = sorted(v for v in left if XMIN <= v <= 0)
    right = set(np.arange(0, XMAX + 0.01, 1.0)) | {2.6, 11.7, XMAX}
    right = sorted(v for v in right if 0 <= v <= XMAX)
    return list(zip(left[:-1], left[1:])), list(zip(right[:-1], right[1:]))


BOX_X0 = RAIL_X[0] - RAIL_W / 2
SLABS_L, SLABS_R = x_slabs()


class SoilBoxes:
    """Collects soil boxes and merges neighbours along x (fewer faces, no internal faces)."""
    def __init__(self):
        self.items = {}

    def add(self, xa, xb, ya, yb, z0, z1, col, top=None):
        if z1 - z0 < 1e-3:
            return
        key = (round(ya, 3), round(yb, 3), round(z0, 3), round(z1, 3), col, top)
        self.items.setdefault(key, []).append((xa, xb))

    def flush(self, m):
        for (ya, yb, z0, z1, col, top), segs in self.items.items():
            segs.sort()
            cur = list(segs[0])
            for a, b in segs[1:]:
                if abs(a - cur[1]) < 1e-6:
                    cur[1] = b
                else:
                    m.box(cur[0], cur[1], ya, yb, z0, z1, col, top=top)
                    cur = [a, b]
            m.box(cur[0], cur[1], ya, yb, z0, z1, col, top=top)


def layer_h(k, top):
    zt = top if k == 0 else min(LAYER_BOT[k - 1], top)
    return max(0.0, min(zt, top) - LAYER_BOT[k])


def build_soil(m, st):
    sb = SoilBoxes()
    # ---------- right of the shoring line (traffic embankment) ----------
    for xa, xb in SLABS_R:
        xm = (xa + xb) / 2
        T = 0.0 if xm <= 11.7 else max(ground(xm), -(xm - 11.7) / 1.5)
        sb.add(xa, xb, -YMAX, YMAX, ZMIN, Z_FOUND, C_BASE)
        top = C_GRAVEL if xm < 2.6 else (C_ASPH if xm < 11.7 else C_GRASS)
        for k in range(NL):
            h = layer_h(k, T)
            if h > 1e-3:
                sb.add(xa, xb, -YMAX, YMAX, LAYER_BOT[k], LAYER_BOT[k] + h, C_LAYER[k],
                       top=top if k == 0 else None)
    # ---------- culvert side (x < 0): dotted existing ground, pad, box void, end cut ----------
    for xa, xb in SLABS_L:
        xm = (xa + xb) / 2
        G = ground(xm)
        built = PADX0 <= xm <= 0 and xa >= pad_reach(st["pad"]) - 1e-6
        PT = max(G, pad_profile(xm)) if built else G
        sb.add(xa, xb, -YMAX, YMAX, ZMIN, Z_FOUND, C_BASE)
        for k in range(NL):
            zb, col = LAYER_BOT[k], C_LAYER[k]
            edges = sorted({-YMAX, -PAD_W, -YCUT[k], -BOX_IN, BOX_IN, YCUT[k], PAD_W, YMAX})
            for ya, yb in zip(edges[:-1], edges[1:]):
                ym = (ya + yb) / 2
                on_pad = built and abs(ym) < PAD_W
                T = PT if on_pad else G
                H = layer_h(k, T)
                if H <= 1e-3:
                    continue
                tcol = (C_PAD if (on_pad and T > G + 1e-6) else C_GRASS) if k == 0 else None
                interior = VX0 <= xm <= VX1 and abs(ym) < BOX_IN
                cutreg = xm < XCUT[k] and abs(ym) < YCUT[k]
                segs = []
                if interior:
                    nat, fil = H * (1 - st["inb"][k]), H * st["fa"][k]
                    if nat > 1e-3:
                        segs.append((zb, zb + nat, col, tcol if nat >= H - 1e-6 else None))
                    if fil > 1e-3:
                        segs.append((zb, zb + fil, C_BF[k],
                                     C_PAD if (k == 0 and st["fa"][0] > 0.999) else None))
                else:
                    segs.append((zb, zb + H, col, tcol))
                if cutreg:
                    zt = zb + H * (1 - st["c9"][k])
                    segs = [(a, min(b, zt), c, tp if b <= zt + 1e-9 else None) for a, b, c, tp in segs]
                for a, b, c, tp in segs:
                    sb.add(xa, xb, ya, yb, a, b, c, top=tp)
    sb.flush(m)


# ------------------------------ shoring box ---------------------------------
def build_box(m, st):
    live = lambda dz: dz < HID - 1e-6
    # sliding rails (posts)
    for k, x in enumerate(RAIL_X):
        dz = st["rail"][k]
        if live(dz):
            for s in (-1, 1):
                ya, yb = sorted((s * RAIL_Y[0], s * RAIL_Y[1]))
                m.box(x - RAIL_W / 2, x + RAIL_W / 2, ya, yb, RAIL_Z[0] + dz, RAIL_Z[1] + dz, C_RAIL)
    # panels: columns 0 = far end, 1..3 = bays (far -> road), 4 = road end
    for grp, col in (("out", C_OUT), ("iu", C_IU), ("il", C_IL)):
        za, zb = TIER[grp]
        wall_y = OUT_Y if grp == "out" else IN_Y
        end_x = END_OUT if grp == "out" else END_IN
        parts = [(za, OUT_SPLIT - 0.02, C_OUT2), (OUT_SPLIT, zb, C_OUT)] if grp == "out" else [(za, zb, col)]
        for c in range(5):
            dz = st[grp][c]
            if not live(dz):
                continue
            for pa, pb, pc in parts:
                if c in (1, 2, 3):
                    x0, x1 = BAYS[c - 1]
                    for s in (-1, 1):
                        ya, yb = sorted((s * wall_y[0], s * wall_y[1]))
                        m.box(x0, x1, ya, yb, pa + dz, pb + dz, pc)
                else:
                    x0, x1 = end_x[0 if c == 0 else 1]
                    m.box(x0, x1, -(CLEAR_HW - 0.02), CLEAR_HW - 0.02, pa + dz, pb + dz, pc)
            if grp == "out" and c in (1, 2, 3):                       # lifting lugs
                x0, x1 = BAYS[c - 1]
                for lx in (x0 + 1.2, x1 - 1.2):
                    for s in (-1, 1):
                        ya, yb = sorted((s * OUT_Y[0], s * OUT_Y[1]))
                        m.box(lx - 0.15, lx + 0.15, ya, yb, zb + dz, zb + dz + 0.22, C_RAIL)


def build_struts(m, st):
    """DSR struts: a tube across the 2.06 m clear width with a spreader plate on each wall (0.8 m spacers)."""
    for s_i, v in enumerate(st["strut"]):
        if v <= 1e-3:
            continue
        lvl, j = divmod(s_i, 2)
        za, zb = STRUT_Z[lvl]
        dz = (1 - v) * 3.0                                              # lowered in / lifted out from above
        x = RAIL_X[STRUT_RAILS[j]]
        m.box(x - 0.14, x + 0.14, -CLEAR_HW, CLEAR_HW, za + dz, zb + dz, C_STRUT)
        for sg in (-1, 1):
            ya, yb = sorted((sg * CLEAR_HW, sg * (CLEAR_HW - 0.10)))
            m.box(x - 0.40, x + 0.40, ya, yb, za + dz - 0.15, zb + dz + 0.15, C_STRUT2)


# --------------------------------- pipes ------------------------------------
def build_culvert(m, st):
    for yc in BARREL_Y:                                                  # existing culvert
        m.cylinder_x(JOINT, 9.0, PIPE_ZC, PIPE_OD / 2, C_EXIST, yc=yc)
    if st["bed"] > 0:                                                    # bedding under the barrel
        hw = max(abs(y) for y in BARREL_Y) + PIPE_OD / 2 + 0.3
        xe = JOINT + st["bed"] * (PIPES[-1][0] - JOINT)
        m.box(xe, JOINT, -hw, hw, Z_FOUND, PIPE_Z0, C_BED)
    for i, (xa, xb) in enumerate(PIPES):
        for b, yc in enumerate(BARREL_Y):
            loc = clip(st["pipes"][N_BARRELS * i + b], 0, 1)
            if loc <= 0:
                continue
            ox = st["pox"][i]
            m.cylinder_x(xa + ox + 0.04, xb + ox, PIPE_ZC + (1 - loc) * DROP, PIPE_OD / 2, C_PIPE, yc=yc)


# --------------------------------- digger -----------------------------------
def build_digger(m, st):
    if st["dig_on"] < 0.5:
        return
    x0, y0 = st["dig_x"], st["dig_y"]
    z0 = surf_z(x0, st["pad"])
    ang = math.radians(st["dig_yaw"])
    c, s = math.cos(ang), math.sin(ang)
    P = lambda lx, ly, lz: (x0 + lx * c - ly * s, y0 + lx * s + ly * c, z0 + lz)
    for ly in (-1.0, 1.0):                                                    # tracks
        m.obeam(P(-1.7, ly, 0.4), P(1.7, ly, 0.4), 0.8, 0.8, C_TRACK)
    m.obeam(P(-1.2, 0, 1.2), P(1.2, 0, 1.2), 2.4, 0.9, C_DIG)                 # body
    m.obeam(P(-1.0, 0.5, 2.0), P(0.0, 0.5, 2.0), 1.0, 0.9, C_DIG)             # cab
    m.obeam(P(1.0, 0, 1.6), P(3.5, 0, 3.5), 0.45, 0.45, C_DIG)                # boom
    m.obeam(P(3.5, 0, 3.5), P(5.2, 0, 2.0), 0.35, 0.35, C_DIG)                # stick
    m.obeam(P(5.2, 0, 2.0), P(5.6, 0, 1.3), 0.9, 0.5, C_TRACK)                # bucket


# -------------------------------- traffic -----------------------------------
def build_traffic(m, idx):
    m.box(0.3, 1.0, -YMAX, YMAX, 0.0, 0.8, "#e8e8e8")                    # temporary barrier
    for y in np.arange(-YMAX + 1, YMAX, 3.0):
        m.box(7.1, 7.25, y, y + 1.5, 0.0, 0.02, "#f5f5f5")
    cars = [(4.9, 6.8 - (idx * 0.5) % 13.6, "#c0392b"), (9.4, -6.8 + (idx * 0.65 + 5) % 13.6, "#2c6fbb")]   # kept inside the block (|y| < 9)
    for cx, cy, col in cars:
        m.box(cx - 0.9, cx + 0.9, cy - 2.1, cy + 2.1, 0.0, 0.7, col)
        m.box(cx - 0.8, cx + 0.8, cy - 1.0, cy + 1.0, 0.7, 1.3, "#d6e4ee")


def build_scene(st, idx):
    soil, box, culv, traf, dig = Mesh(), Mesh(), Mesh(), Mesh(), Mesh()
    build_soil(soil, st)
    build_box(box, st)
    build_culvert(culv, st)
    build_struts(culv, st)
    build_traffic(traf, idx)
    build_digger(dig, st)
    return [soil.trace("ground"), box.trace("shoring box", opacity=0.5), culv.trace("pipes"),
            traf.trace("traffic"), dig.trace("digger")]


# ------------------------------- timeline -----------------------------------
STAGES = [
    "Digger pad built from the road side",
    "Panels + rails set from the road side",
    "Panels and rails pushed down",
    "Rails to depth, upper struts, dig under outer panels",
    "Inner panels, lower struts, excavate to foundation",
    "Foundation test, pipes placed",
    "Backfill in layers, struts + inner panels out",
    "Outer panels + rails out (road side last)",
    "Far end excavated (1:1 / bench)",
]

BAY_C = [(a + b) / 2 for a, b in BAYS]                         # far bay, middle bay, road bay
COL_X = {0: -16.2, 1: BAY_C[0], 2: BAY_C[1], 3: BAY_C[2], 4: -0.2}   # panel column -> x
# installation order, ROAD SIDE FIRST (panel, rail, panel, rail ...): (group, column / rail index, x, label)
COLS = [("out", 4, -0.2, "road end panel"), ("rail", 3, RAIL_X[3], "rail 1 (road side)"),
        ("out", 3, BAY_C[2], "outer panel, bay 1"), ("rail", 2, RAIL_X[2], "rail 2"),
        ("out", 2, BAY_C[1], "outer panel, bay 2"), ("rail", 1, RAIL_X[1], "rail 3"),
        ("out", 1, BAY_C[0], "outer panel, bay 3"), ("rail", 0, RAIL_X[0], "rail 4 (far end)"),
        ("out", 0, -16.2, "far end panel")]
DIGX = lambda x: clip(x, -19.0, -2.0)
ROAD_FIRST = [4, 3, 2, 1, 0]                                   # panel columns, road end -> far end
FAR_FIRST = [0, 1, 2, 3, 4]


def build_timeline():
    cur = dict(pad=0.0, dig_on=0.0, dig_x=-2.0, dig_y=3.8, dig_yaw=270.0,
               rail=[HID] * 4, out=[HID] * 5, iu=[HID] * 5, il=[HID] * 5,
               inb=[0.0] * NL, fa=[0.0] * NL, c9=[0.0] * NL, bed=0.0,
               pipes=[0.0] * (N_BARRELS * N_PIPES), pox=[0.0] * N_PIPES, strut=[0.0] * 4, day=0.0)
    out = []

    def go_(stage, sub, frames, days, **tgt):
        start = copy.deepcopy(cur)
        for f in range(1, frames + 1):
            t = f / frames
            fr = copy.deepcopy(cur)
            for k, v in tgt.items():
                if isinstance(v, list):
                    fr[k] = [a + (b - a) * t for a, b in zip(start[k], v)]
                else:
                    fr[k] = start[k] + (v - start[k]) * t
            fr["day"] = start["day"] + days * t
            fr["stage"], fr["sub"] = stage, sub
            out.append(fr)
        for k, v in tgt.items():
            cur[k] = copy.deepcopy(v)
        cur["day"] = start["day"] + days

    def upd(key, idx, val):
        v = list(cur[key])
        v[idx] = val
        return v

    def col_step(stage, text, i, frames, days, outv, railv):
        grp, c, x, name = COLS[i]
        key, val = ("out", outv) if grp == "out" else ("rail", railv)
        go_(stage, f"{text}: {name} ({i + 1} of {len(COLS)})", frames, days,
            **{key: upd(key, c, val)}, dig_x=DIGX(x))

    def strut_step(stage, text, lvl, j, val, label):
        go_(stage, f"{text} {label} strut at rail {4 - STRUT_RAILS[j]}",
            2, 0.5, strut=upd("strut", lvl * 2 + j, val), dig_x=DIGX(RAIL_X[STRUT_RAILS[j]]))

    # ---- 1  digger pad, from the road side
    go_(1, "Existing ground follows the dotted line on the drawing", 2, 1)
    go_(1, "Digger walks in from the road side", 1, 0.5, dig_on=1.0)
    go_(1, "Building the digger pad (orange line) outwards from the road side", 4, 2, pad=1.0, dig_x=-9.0)
    go_(1, "Digger back at the road end, ready to set the first panel", 1, 0.5, dig_x=-2.0)
    # ---- 2  rails + outer panels from the road side, half in / half out
    for i in range(len(COLS)):
        col_step(2, "Setting", i, 1, 0.5, P_OUT, P_RAIL0)
    # ---- 3  push panels and rails down one by one (road side first), dig inside as needed
    for i in range(len(COLS)):
        col_step(3, "Pushing down", i, 1, 0.5, 0.0, P_RAIL1)
    go_(3, "Excavating inside the box to ease penetration - lift 1 of 5", 2, 1,
        inb=upd("inb", 0, 1.0), dig_x=-8.0)
    # ---- 4  rails to depth, upper struts, dig under the outer panels
    for c in (3, 2, 1, 0):
        go_(4, f"Pushing rail {4 - c} of 4 to full depth", 1, 0.5,
            rail=upd("rail", c, 0.0), dig_x=DIGX(RAIL_X[c]))
    strut_step(4, "Fitting the upper", 0, 1, 1.0, "DSR")
    strut_step(4, "Fitting the upper", 0, 0, 1.0, "DSR")
    go_(4, "Excavating - lift 2 of 5", 2, 1, inb=upd("inb", 1, 1.0), dig_x=-8.0)
    go_(4, "Excavating under the outer panels - lift 3 of 5", 2, 1, inb=upd("inb", 2, 1.0))
    # ---- 5  inner panels, lower struts, excavate to foundation
    for c in ROAD_FIRST:
        go_(5, f"Installing inner upper panel (column {5 - c} of 5, road side first)", 1, 0.5,
            iu=upd("iu", c, 0.0), dig_x=DIGX(COL_X[c]))
    go_(5, "Excavating - lift 4 of 5", 2, 1, inb=upd("inb", 3, 1.0), dig_x=-8.0)
    strut_step(5, "Fitting the lower", 1, 1, 1.0, "DSR")
    strut_step(5, "Fitting the lower", 1, 0, 1.0, "DSR")
    go_(5, "Excavating to foundation level - lift 5 of 5", 2, 1, inb=upd("inb", 4, 1.0), dig_x=-8.0)
    for c in ROAD_FIRST:
        go_(5, f"Installing inner lower panel (column {5 - c} of 5, road side first)", 1, 0.5,
            il=upd("il", c, 0.0), dig_x=DIGX(COL_X[c]))
    # ---- 6  foundation test, bedding, pipes (2 clear of the strut first, 5 likewise)
    go_(6, "Foundation tested and signed off", 2, 1)
    go_(6, "Laying bedding", 2, 0.5, bed=1.0)
    for i in PIPE_ORDER:
        cur["pox"][i] = POX.get(i, 0.0)
        xm = (PIPES[i][0] + PIPES[i][1]) / 2
        tag = " - lowered beside the strut" if i in POX else ""
        go_(6, f"Lowering pipe {i + 1} of {N_PIPES}{tag} - surveyed for conformance", 2, 0.5,
            pipes=upd("pipes", i, 1.0), dig_x=DIGX(xm))
        if i in POX:
            go_(6, f"Sliding pipe {i + 1} under the lower strut to its joint", 1, 0.25, pox=upd("pox", i, 0.0),
                dig_x=DIGX(xm))
    # ---- 7  backfill in layers; each strut out when the fill reaches just under it
    go_(7, "Backfill layer 1 of 5 (foundation to -6.8), compacted full footprint", 2, 1, fa=upd("fa", 4, 1.0), dig_x=-8.0)
    go_(7, "Lifting inner lower panels 1.5 m", 2, 0.5, il=[1.5] * 5)
    go_(7, "Backfill layer 2 up to just under the lower struts", 2, 0.5, fa=upd("fa", 3, 0.35))
    strut_step(7, "Removing the lower", 1, 0, 0.0, "DSR")
    strut_step(7, "Removing the lower", 1, 1, 0.0, "DSR")
    go_(7, "Backfill layer 2 of 5 completed (to -5.3)", 2, 0.5, fa=upd("fa", 3, 1.0), dig_x=-8.0)
    go_(7, "Lifting inner lower panels clear of the fill", 2, 0.5, il=[3.3] * 5)
    go_(7, "Backfill layer 3 of 5 (to -3.8)", 2, 1, fa=upd("fa", 2, 1.0))
    go_(7, "Lifting inner panels as the fill rises", 2, 0.5, il=[HID] * 5, iu=[2.4] * 5)
    go_(7, "Inner panels fully out", 2, 0.5, iu=[HID] * 5)
    go_(7, "Rails lifted partly clear", 2, 0.5, rail=[3.3] * 4)
    # ---- 8  upper struts, outer panels + rails out far end -> road side, finish backfill
    go_(8, "Backfill layer 4 of 5 (to -2.3, 1.5 m below the pad, just under the upper struts)", 2, 1,
        fa=upd("fa", 1, 1.0), dig_x=-8.0)
    strut_step(8, "Removing the upper", 0, 0, 0.0, "DSR")
    strut_step(8, "Removing the upper", 0, 1, 0.0, "DSR")
    for c in FAR_FIRST:
        go_(8, f"Extracting outer panel (column {c + 1} of 5, far end first - road side last)", 1, 0.5,
            out=upd("out", c, HID), dig_x=DIGX(COL_X[c]))
    for c in (0, 1, 2, 3):
        go_(8, f"Extracting rail {c + 1} of 4 (far end first - road side last)", 1, 0.5,
            rail=upd("rail", c, HID), dig_x=DIGX(RAIL_X[c]))
    go_(8, "Backfill layer 5 of 5 - back to the existing level", 2, 1, fa=upd("fa", 0, 1.0), dig_x=-8.0)
    # ---- 9  end excavation (1:1 stepped batter / benching) at the far end
    go_(9, "Digger repositions on the pad to open the far end", 2, 0.5, dig_x=-4.0, dig_y=0.0, dig_yaw=180.0)
    for k in range(NL):
        go_(9, f"Excavating the end - bench {k + 1} of {NL} (1:1 stepped, 1.5 m vertical)", 2, 1,
            c9=upd("c9", k, 1.0))
    go_(9, "Complete - end ready for headwall and rock protection", 2, 1)
    tot = out[-1]["day"]                                   # rescale the programme to a fixed length
    for fr in out:
        fr["day"] = fr["day"] * TOTAL_DAYS / tot
    return out


# ------------------------------- plotting -----------------------------------
FONT = "Inter, Segoe UI, Helvetica, Arial, sans-serif"


def annotations(st):
    lines = []
    for i, name in enumerate(STAGES, 1):
        if i == st["stage"]:
            lines.append(f'<b style="color:#c2410c">&#9654; {i}. {name}</b>')
        elif i < st["stage"]:
            lines.append(f'<span style="color:#334155">&#10003; {i}. {name}</span>')
        else:
            lines.append(f'<span style="color:#94a3b8">&#9675; {i}. {name}</span>')
    key = ("<br><br><span style='color:#64748b'>Key:</span> "
           f"<span style='color:{C_OUT}'><b>panels</b></span> "
           f"<span style='color:{C_RAIL}'><b>rails</b></span> "
           f"<span style='color:{C_STRUT}'><b>struts</b></span> "
           f"<span style='color:{C_PIPE}'><b>new pipes</b></span> "
           f"<span style='color:{C_EXIST}'><b>existing culvert</b></span> "
           "<span style='color:#b59a4a'><b>backfill</b></span> <span style='color:#8a8272'><b>pad</b></span>")
    return [
        dict(x=0.01, y=0.985, xref="paper", yref="paper", xanchor="left", yanchor="top",
             showarrow=False, align="left",
             text=(f'<b style="font-size:20px">Stage {st["stage"]} of {len(STAGES)}</b><br>'
                   f'<span style="font-size:14px;color:#475569">{st["sub"]}</span>')),
        dict(x=0.99, y=0.985, xref="paper", yref="paper", xanchor="right", yanchor="top",
             showarrow=False, align="right",
             text=f'<span style="font-size:12px;color:#64748b">PROGRAMME</span><br> <br>'
                  f'<b style="font-size:22px">Day {st["day"]:.0f}</b>'),
        dict(x=0.01, y=0.80, xref="paper", yref="paper", xanchor="left", yanchor="top",
             showarrow=False, align="left", font=dict(size=12), text="<br>".join(lines) + key),
    ]


def base_layout(camera=None):
    cam = camera or dict(eye=dict(x=-0.20, y=-0.44, z=1.18), center=dict(x=-0.10, y=0.0, z=-0.32))
    return dict(
        font=dict(family=FONT), paper_bgcolor="#f4f7fb", plot_bgcolor="#f4f7fb",
        margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
        scene=dict(xaxis=dict(visible=False, range=[XMIN, XMAX]),
                   yaxis=dict(visible=False, range=[-YMAX, YMAX]),
                   zaxis=dict(visible=False, range=[ZMIN, ZMAX]),
                   aspectmode="data", camera=cam, bgcolor="#f4f7fb"))


def make_figure(timeline):
    frames, steps, last = [], [], 0
    for i, st in enumerate(timeline):
        frames.append(go.Frame(data=build_scene(st, i), name=str(i), traces=[0, 1, 2, 3, 4],
                               layout=dict(annotations=annotations(st))))
        steps.append(dict(method="animate", label=str(st["stage"]) if st["stage"] != last else "",
                          args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                               transition=dict(duration=0))]))
        last = st["stage"]
    fig = go.Figure(data=frames[0].data, frames=frames)
    lay = base_layout()
    lay["annotations"] = annotations(timeline[0])

    def play(label, ms):
        return dict(label=label, method="animate",
                    args=[None, dict(frame=dict(duration=ms, redraw=True), fromcurrent=True,
                                     transition=dict(duration=0), mode="immediate")])
    pause = dict(label="Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate",
                                    transition=dict(duration=0))])
    last_idx = {st["stage"]: i for i, st in enumerate(timeline)}
    jump = [dict(label=str(s), method="animate",
                 args=[[str(last_idx[s])], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                transition=dict(duration=0))])
            for s in sorted(last_idx)]
    lay["updatemenus"] = [
        dict(type="buttons", direction="left", x=0.01, y=0.02, xanchor="left", yanchor="bottom",
             pad=dict(r=4, t=4), bgcolor="#ffffff", bordercolor="#cbd5e1",
             buttons=[play("&#9654; Play", 140), play("Slow", 320), pause]),
        dict(type="buttons", direction="left", x=0.99, y=0.02, xanchor="right", yanchor="bottom",
             pad=dict(r=2, t=4), bgcolor="#ffffff", bordercolor="#cbd5e1", buttons=jump),
    ]
    lay["sliders"] = [dict(active=0, steps=steps, x=0.2, y=0.015, len=0.5, xanchor="left",
                           yanchor="bottom", pad=dict(t=0, b=0), currentvalue=dict(visible=False),
                           ticklen=0, minorticklen=0, bgcolor="#e2e8f0", activebgcolor="#c2410c",
                           font=dict(size=10, color="#64748b"))]
    fig.update_layout(**lay)
    return fig


# --------------------------------- export -----------------------------------
def frame_figure(fig, i, angle=None):
    fr = fig.frames[i]
    cam = None
    if angle is not None:
        r = 1.3
        cam = dict(eye=dict(x=r * math.cos(angle), y=r * math.sin(angle), z=1.15),
                   center=dict(x=-0.04, y=0.0, z=-0.30))
    lay = base_layout(cam)
    lay["annotations"] = fr.layout.annotations
    return go.Figure(data=fr.data, layout=lay)


def export_video(fig, timeline, path, size=(1280, 720), fps=12, hold=14):
    tmp = tempfile.mkdtemp()
    n = len(timeline)
    seq = []
    for i in range(n):
        seq.append(i)
        if i == n - 1 or timeline[i + 1]["stage"] != timeline[i]["stage"]:
            seq += [i] * hold
    files = {}
    for i in range(n):
        ang = math.radians(-140 + 35 * i / (n - 1))
        f = os.path.join(tmp, f"f{i:04d}.png")
        frame_figure(fig, i, ang).write_image(f, width=size[0], height=size[1], scale=1)
        files[i] = f
    lst = os.path.join(tmp, "seq")
    os.makedirs(lst)
    for k, i in enumerate(seq):
        shutil.copy(files[i], os.path.join(lst, f"s{k:05d}.png"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", os.path.join(lst, "s%05d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", path], check=True)
    shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="Culvert 2D - single barrel, slide-rail shoring box with struts, staging 3D")
    ap.add_argument("--out", default="culvert_2D_staging_3d_12days.html")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--png", type=int, metavar="STAGE")
    a = ap.parse_args()
    tl = build_timeline()
    fig = make_figure(tl)
    fig.write_html(a.out, include_plotlyjs=True, auto_play=False,
                   config=dict(displaylogo=False, scrollZoom=True))
    print(f"interactive model -> {a.out} ({len(tl)} frames)")
    if a.png:
        idx = max(i for i, s in enumerate(tl) if s["stage"] == a.png)
        frame_figure(fig, idx).write_image(f"stage_{a.png}.png", width=1280, height=720)
    if a.video:
        export_video(fig, tl, os.path.splitext(a.out)[0] + ".mp4")


if __name__ == "__main__":
    main()
