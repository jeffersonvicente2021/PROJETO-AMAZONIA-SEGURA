import os
import sqlite3
import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle

# ---------- Utils ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_events(db_path: Path, start: str = None, end: str = None) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Banco não encontrado: {db_path}")

    con = sqlite3.connect(str(db_path))
    base_sql = "SELECT id, ts_start, ts_end, duration, classes, conf_max, video_path, snapshot_path, meta_path FROM events"
    params = []
    if start and end:
        # ts_start está em ISO (YYYY-MM-DD_HH:MM:SS) → comparação lexicográfica funciona
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

    # Normalizações
    df["ts_start_dt"] = pd.to_datetime(df["ts_start"], errors="coerce")
    df["ts_end_dt"]   = pd.to_datetime(df["ts_end"], errors="coerce")
    df["date"]        = df["ts_start_dt"].dt.date
    df["hour"]        = df["ts_start_dt"].dt.hour
    df["duration"]    = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)
    df["classes"]     = df["classes"].fillna("").astype(str)

    # Opcional: tenta ler "zones" do meta.json se existir (se não existir, fica vazio)
    zones = []
    for _, row in df.iterrows():
        z = None
        try:
            meta_path = row.get("meta_path")
            if isinstance(meta_path, str) and meta_path and os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                # se um dia o script passar a salvar "zones" no meta, pegamos aqui
                z = meta.get("zones") if isinstance(meta.get("zones"), list) else None
        except Exception:
            z = None
        # padroniza: "HORIZON;RIVER" / None
        zones.append(";".join(sorted(set(z))) if z else None)
    df["zones"] = zones

    return df

def save_csvs(df: pd.DataFrame, outdir: Path):
    ensure_dir(outdir)
    df.to_csv(outdir / "events_full.csv", index=False)

    # por hora
    by_hour = df.groupby("hour").size().reset_index(name="events")
    by_hour.to_csv(outdir / "summary_by_hour.csv", index=False)

    # por classe (explode classes "canoa,lancha")
    cls = df.assign(cls=df["classes"].str.split(",")).explode("cls")
    cls["cls"] = cls["cls"].str.strip()
    cls = cls[cls["cls"] != ""]
    by_class = cls.groupby("cls").size().reset_index(name="events")
    by_class.to_csv(outdir / "summary_by_class.csv", index=False)

    # por zona, se houver
    if "zones" in df.columns and df["zones"].notna().any():
        zt = df.assign(zone=df["zones"].fillna("").str.split(";")).explode("zone")
        zt = zt[zt["zone"] != ""]
        by_zone = zt.groupby("zone").size().reset_index(name="events")
        by_zone.to_csv(outdir / "summary_by_zone.csv", index=False)

def make_charts(df: pd.DataFrame, outdir: Path):
    ensure_dir(outdir)

    # 1) eventos por hora
    by_hour = df.groupby("hour").size()
    plt.figure()
    by_hour.plot(kind="bar")
    plt.title("Eventos por hora")
    plt.xlabel("Hora")
    plt.ylabel("Eventos")
    plt.tight_layout()
    plt.savefig(outdir / "chart_events_by_hour.png")
    plt.close()

    # 2) eventos por classe
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

    # 3) eventos por dia
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
    total_dur = float(df["duration"].sum())
    avg_dur = float(df["duration"].mean()) if total_events else 0.0
    start = str(df["ts_start"].min()) if total_events else "-"
    end   = str(df["ts_end"].max()) if total_events else "-"
    return total_events, total_dur, avg_dur, start, end

def build_pdf(df: pd.DataFrame, charts_dir: Path, out_pdf: Path):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="T", fontSize=18, leading=22, spaceAfter=12, alignment=1))
    styles.add(ParagraphStyle(name="H", fontSize=14, leading=18, spaceAfter=8, spaceBefore=12))
    styles.add(ParagraphStyle(name="B", fontSize=11, leading=15))

    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4)
    story = []

    story.append(Paragraph("Relatório – Monitoramento Amazônia (MVP)", styles["T"]))

    te, td, av, st, en = kpi(df)
    kpi_txt = f"""
    <b>Período:</b> {st} → {en}<br/>
    <b>Eventos:</b> {te} &nbsp;&nbsp; <b>Duração total (s):</b> {td:.1f} &nbsp;&nbsp; <b>Duração média (s):</b> {av:.1f}
    """
    story.append(Paragraph(kpi_txt, styles["B"]))
    story.append(Spacer(1, 10))

    # Gráficos
    charts = ["chart_events_by_hour.png", "chart_events_by_class.png", "chart_events_by_day.png"]
    for c in charts:
        p = charts_dir / c
        if p.exists():
            story.append(Paragraph(c.replace("_", " ").replace(".png", "").title(), styles["H"]))
            story.append(RLImage(str(p), width=480, height=280))
            story.append(Spacer(1, 10))

    # Tabela de amostras (até 10)
    samp = df.sort_values("ts_start_dt").head(10).copy()
    rows = [["Início", "Duração(s)", "Classes", "Conf. Máx."]]
    for _, r in samp.iterrows():
        rows.append([str(r["ts_start"]), f"{r['duration']:.1f}", r["classes"], f"{r['conf_max']:.2f}"])
    t = Table(rows, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), (0.9,0.9,0.9)),
        ("GRID", (0,0), (-1,-1), 0.25, (0.6,0.6,0.6)),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("ALIGN", (1,1), (-1,-1), "CENTER")
    ]))
    story.append(Paragraph("Amostras recentes", styles["H"]))
    story.append(t)

    doc.build(story)

from datetime import datetime, timedelta

def calcular_periodo(tipo: str):
    hoje = datetime.today()
    if tipo == "diario":
        inicio = hoje - timedelta(days=1)
    elif tipo == "semanal":
        inicio = hoje - timedelta(days=7)
    elif tipo == "mensal":
        inicio = hoje - timedelta(days=30)
    else:
        raise ValueError("Tipo de relatório inválido")
    return inicio.strftime("%Y-%m-%d"), hoje.strftime("%Y-%m-%d")

def main():
    parser = argparse.ArgumentParser(description="Exporta events.db para CSV e PDF")
    parser.add_argument("--db", default="data_events/events.db", help="Caminho do banco SQLite")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--tipo", choices=["diario", "semanal", "mensal"], help="Tipo automático de relatório")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    # Definindo início e fim com base no tipo
    if args.tipo:
        args.start, args.end = calcular_periodo(args.tipo)

    # Ajuste do end (inclui o dia todo)
    if args.end and len(args.end) == 10:
        end_dt = datetime.strptime(args.end, "%Y-%m-%d") + timedelta(days=1)
        args.end = end_dt.strftime("%Y-%m-%d")

    outroot = Path(args.out) if args.out else Path("reports") / f"{args.tipo}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    charts_dir = ensure_dir(outroot / "charts")

    df = load_events(Path(args.db), args.start, args.end)
    if df.empty:
        print("[INFO] Nenhum evento encontrado no período informado.")
        return

    save_csvs(df, outroot)
    make_charts(df, charts_dir)
    build_pdf(df, charts_dir, outroot / "relatorio_mvp.pdf")

    print(f"[OK] Relatório '{args.tipo}' gerado em: {outroot}")


if __name__ == "__main__":
    main()
