"""Generate PanelConvention_IY.xlsx from scratch (the old convention file is not read).

Panel layout comes from the right-hand table (cols F:H) of the PanelOverview sheet in
the panel log (F = cycle+round, G = round-1 marker, H = round-2 marker). Slide list
is the folder names in the Batch2 Slides dir; sample and panel are defined below. Every cell is a plain string:
<Sample>_<Sample_ID>_<Panel>_<header>.
"""
from pathlib import Path

import openpyxl
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

STUDY_DIR = Path(r"Z:\Multiplex_IHC_studies\UCSD_AndyLowey_Lustgarten")
SLIDES_DIR = STUDY_DIR / "OHSU_mIHC_UCSD_Lowey" / "Batch2" / "Slides"  # one folder per slide
PANEL_LOG = STUDY_DIR / "StudyInfo" / "UCSD_Syki_PDAC_PanelLog2a.xlsx"
OUT = STUDY_DIR / "PanelConvention_IY.xlsx"

SAMPLE = "KB_UCSD"
PANEL = "D30"
HE_HEADER = "C001R1_HE"


def read_slide_ids():
    return sorted(p.name for p in SLIDES_DIR.iterdir() if p.is_dir())


def read_panel_headers():
    ws = openpyxl.load_workbook(PANEL_LOG, data_only=True)["PanelOverview"]
    headers = []
    for r in range(2, ws.max_row + 1):
        cr = ws.cell(r, 6).value
        marker = ws.cell(r, 7).value or ws.cell(r, 8).value
        if cr and marker:
            headers.append(f"{cr.strip()}_{str(marker).strip()}")
    return headers


def main():
    slide_ids = read_slide_ids()
    headers = [HE_HEADER] + read_panel_headers()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"

    ws.append(["Sample", "Sample_ID", "Panel"] + headers)
    for sid in slide_ids:
        ws.append([SAMPLE, sid, PANEL] + [f"{SAMPLE}_{sid}_{PANEL}_{h}" for h in headers])

    for c in ws[1]:
        c.font = Font(bold=True)
    for j in range(1, 4):
        ws.column_dimensions[get_column_letter(j)].width = 12
    for j in range(4, 4 + len(headers)):
        ws.column_dimensions[get_column_letter(j)].width = 42
    ws.freeze_panes = "D2"

    wb.save(OUT)
    print(f"Wrote {OUT}: {len(slide_ids)} slides x {len(headers)} columns")


if __name__ == "__main__":
    main()
