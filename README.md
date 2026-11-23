# General-purpose

3D staged-excavation and slide-rail shoring models (Plotly time-lapse) for Bruce Highway Stage 2 culverts, 12-day programme.

| Script | Culvert | Notes |
|---|---|---|
| `culvert_2D_staging_3d_12days.py` | WP-02D | single barrel (6 x 1.2 m pipes), 16.3 x 2.06 m box, 4 rails, 2 strut levels, digger works from the road side |
| `culvert_2B_staging_3d_v3_12days.py` | WP-02B | 5.5 m wide excavation, twin barrel |
| `culvert_2B_staging_3d_v2_12days.py` | WP-02B | 8.57 m rails, single barrel |

## Run

```
pip install numpy plotly
python culvert_2D_staging_3d_12days.py          # -> interactive HTML (play / pause / stage buttons)
python culvert_2D_staging_3d_12days.py --png 5  # -> stage_5.png (needs: pip install kaleido)
python culvert_2D_staging_3d_12days.py --video  # -> MP4 (needs kaleido + ffmpeg)
```

The programme length is `TOTAL_DAYS = 12` near the top of each script.
