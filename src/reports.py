import argparse
import json
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "data" / ".matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .config import build_config


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_events(db_path: Path, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Banco nao encontrado: {db_path}")

    con = sqlite3.connect(str(db_path))
    base_sql = (
        "SELECT id, ts_start, ts_end, duration, classes, conf_max, "
        "video_path, snapshot_path, meta_path FROM events"
    )
    params = []
    if start and end:
        base_sql += " WHERE ts_start >= ? AND ts_start < ?"
        params = [start, end]
    elif start:
        base_sql += " WHERE ts_start >= ?"
        params = [start]
    elif end:
        base_sql += " WHERE ts_start < ?"
        params = [end]

    df = pd.read_sql_query(base_sql, con, params=params)
    con.close()
    if df.empty:
        return df

    df["ts_start_dt"] = pd.to_datetime(df["ts_start"], errors="coerce")
    df["ts_end_dt"] = pd.to_datetime(df["ts_end"], errors="coerce")
    df["date"] = df["ts_start_dt"].dt.date
    df["hour"] = df["ts_start_dt"].dt.hour
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)
    df["classes"] = df["classes"].fillna("").astype(str)

    zones = []
    for _, row in df.iterrows():
        zone_value = None
        try:
            meta_path = row.get("meta_path")
            if isinstance(meta_path, str) and meta_path and os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as handle:
                    meta = json.load(handle)
                zone_value = meta.get("zones") if isinstance(meta.get("zones"), list) else None
        except Exception:
            zone_value = None
        zones.append(";".join(sorted(set(zone_value))) if zone_value else None)
    df["zones"] = zones
    return df


def save_csvs(df: pd.DataFrame, outdir: Path) -> None:
    ensure_dir(outdir)
    df.to_csv(outdir / "events_full.csv", index=False)
    df.groupby("hour").size().reset_index(name="events").to_csv(outdir / "summary_by_hour.csv", index=False)

    cls = df.assign(cls=df["classes"].str.split(",")).explode("cls")
    cls["cls"] = cls["cls"].str.strip()
    cls = cls[cls["cls"] != ""]
    cls.groupby("cls").size().reset_index(name="events").to_csv(outdir / "summary_by_class.csv", index=False)

    if "zones" in df.columns and df["zones"].notna().any():
        zt = df.assign(zone=df["zones"].fillna("").str.split(";")).explode("zone")
        zt = zt[zt["zone"] != ""]
        zt.groupby("zone").size().reset_index(name="events").to_csv(outdir / "summary_by_zone.csv", index=False)


def make_charts(df: pd.DataFrame, outdir: Path) -> None:
    ensure_dir(outdir)

    by_hour = df.groupby("hour").size()
    plt.figure()
    by_hour.plot(kind="bar")
    plt.title("Eventos por hora")
    plt.xlabel("Hora")
    plt.ylabel("Eventos")
    plt.tight_layout()
    plt.savefig(outdir / "chart_events_by_hour.png")
    plt.close()

    cls = df.assign(cls=df["classes"].str.split(",")).explode("cls")
    cls["cls"] = cls["cls"].str.strip()
    cls = cls[cls["cls"] != ""]
    plt.figure()
    cls.groupby("cls").size().plot(kind="bar")
    plt.title("Eventos por classe")
    plt.xlabel("Classe")
    plt.ylabel("Eventos")
    plt.tight_layout()
    plt.savefig(outdir / "chart_events_by_class.png")
    plt.close()

    by_day = df.groupby("date").size()
    plt.figure()
    by_day.plot(kind="bar")
    plt.title("Eventos por dia")
    plt.xlabel("Dia")
    plt.ylabel("Eventos")
    plt.tight_layout()
    plt.savefig(outdir / "chart_events_by_day.png")
    plt.close()


def kpi(df: pd.DataFrame):
    total_events = int(len(df))
    total_duration = float(df["duration"].sum())
    avg_duration = float(df["duration"].mean()) if total_events else 0.0
    start = str(df["ts_start"].min()) if total_events else "-"
    end = str(df["ts_end"].max()) if total_events else "-"
    return total_events, total_duration, avg_duration, start, end


def build_pdf(df: pd.DataFrame, charts_dir: Path, out_pdf: Path) -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="T", fontSize=18, leading=22, spaceAfter=12, alignment=1))
    styles.add(ParagraphStyle(name="H", fontSize=14, leading=18, spaceAfter=8, spaceBefore=12))
    styles.add(ParagraphStyle(name="B", fontSize=11, leading=15))

    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4)
    story = [Paragraph("Relatorio - Monitoramento Amazonia (MVP)", styles["T"])]
    total_events, total_duration, avg_duration, start, end = kpi(df)
    story.append(
        Paragraph(
            f"<b>Periodo:</b> {start} -> {end}<br/>"
            f"<b>Eventos:</b> {total_events} &nbsp;&nbsp; "
            f"<b>Duracao total (s):</b> {total_duration:.1f} &nbsp;&nbsp; "
            f"<b>Duracao media (s):</b> {avg_duration:.1f}",
            styles["B"],
        )
    )
    story.append(Spacer(1, 10))

    for chart in ["chart_events_by_hour.png", "chart_events_by_class.png", "chart_events_by_day.png"]:
        path = charts_dir / chart
        if path.exists():
            story.append(Paragraph(chart.replace("_", " ").replace(".png", "").title(), styles["H"]))
            story.append(RLImage(str(path), width=480, height=280))
            story.append(Spacer(1, 10))

    sample = df.sort_values("ts_start_dt").head(10).copy()
    rows = [["Inicio", "Duracao(s)", "Classes", "Conf. Max."]]
    for _, row in sample.iterrows():
        rows.append([str(row["ts_start"]), f"{row['duration']:.1f}", row["classes"], f"{row['conf_max']:.2f}"])
    table = Table(rows, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), (0.9, 0.9, 0.9)),
                ("GRID", (0, 0), (-1, -1), 0.25, (0.6, 0.6, 0.6)),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ]
        )
    )
    story.append(Paragraph("Amostras recentes", styles["H"]))
    story.append(table)
    doc.build(story)


def calcular_periodo(tipo: str):
    today = datetime.today()
    if tipo == "diario":
        start = today - timedelta(days=1)
    elif tipo == "semanal":
        start = today - timedelta(days=7)
    elif tipo == "mensal":
        start = today - timedelta(days=30)
    else:
        raise ValueError("Tipo de relatorio invalido")
    return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")


def main() -> None:
    config = build_config()
    parser = argparse.ArgumentParser(description="Exporta events.db para CSV e PDF")
    parser.add_argument("--db", default=str(config.db_path), help="Caminho do banco SQLite")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--tipo", choices=["diario", "semanal", "mensal"], help="Tipo automatico de relatorio")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.tipo:
        args.start, args.end = calcular_periodo(args.tipo)
    if args.end and len(args.end) == 10:
        end_dt = datetime.strptime(args.end, "%Y-%m-%d") + timedelta(days=1)
        args.end = end_dt.strftime("%Y-%m-%d")

    label = args.tipo or "custom"
    outroot = Path(args.out) if args.out else config.reports_root / f"{label}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    charts_dir = ensure_dir(outroot / "charts")

    df = load_events(Path(args.db), args.start, args.end)
    if df.empty:
        print("[INFO] Nenhum evento encontrado no periodo informado.")
        return

    save_csvs(df, outroot)
    make_charts(df, charts_dir)
    build_pdf(df, charts_dir, outroot / "relatorio_mvp.pdf")
    print(f"[OK] Relatorio '{label}' gerado em: {outroot}")


if __name__ == "__main__":
    main()
