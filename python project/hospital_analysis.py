"""
Hospital Appointment Operations Analysis  -  single-file edition
ABES Engineering College, Ghaziabad | Nakul (Roll No. 2026B01010862)
Implements SRS (IEEE 830-1998) requirements FR-1.1 to FR-4.2.

Dashboard :  streamlit run hospital_analysis.py     (http://127.0.0.1:8501)
Console   :  python hospital_analysis.py [path_to_csv]
Needs     :  pandas>=2.0, numpy>=1.24, streamlit>=1.30 (streamlit only for dashboard)
Operational scheduling data only - no clinical inference.
"""
import sys
import time

import numpy as np
import pandas as pd

CSV_PATH = "09_hospital_appointments.csv"

# =====================================================================
# SUBSYSTEM A - Dataset Ingestion & Data Hygiene (FR-1.1 .. FR-1.3)
# =====================================================================
EXPECTED_COLUMNS = [
    "Appointment_ID", "Appointment_Date", "Department", "Visit_Type",
    "SMS_Reminder", "Attendance_Status", "Waiting_Time", "Consultation_Fee",
]


def load_data(source="09_hospital_appointments.csv"):
    """FR-1.1: Load the CSV (path or uploaded file object) and validate schema."""
    df = pd.read_csv(source, encoding="utf-8")
    df.columns = df.columns.str.strip()
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Light standardisation of text columns and types
    for col in ["Appointment_ID", "Department", "Visit_Type",
                "SMS_Reminder", "Attendance_Status"]:
        df[col] = df[col].astype("string").str.strip()
    df["Appointment_Date"] = pd.to_datetime(df["Appointment_Date"], errors="coerce")
    df["Waiting_Time"] = pd.to_numeric(df["Waiting_Time"], errors="coerce")
    df["Consultation_Fee"] = pd.to_numeric(df["Consultation_Fee"], errors="coerce")
    return df[EXPECTED_COLUMNS]


def schema_report(df, preview_rows=5):
    """FR-1.1: record count, column data types and sample rows."""
    dtypes = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(t) for t in df.dtypes],
        "Null Count": df.isna().sum().values,
    })
    return {"total_records": len(df), "dtypes": dtypes, "preview": df.head(preview_rows)}


def remove_duplicates(df):
    """FR-1.3: report duplicate Appointment_ID frequency, keep first occurrence."""
    dup_mask = df["Appointment_ID"].duplicated(keep=False)
    dup_rows = df[dup_mask].sort_values("Appointment_ID")
    dup_freq = (df["Appointment_ID"].value_counts()
                .loc[lambda s: s > 1].rename("Occurrences").reset_index())
    n_removed = int(df["Appointment_ID"].duplicated(keep="first").sum())
    clean = df.drop_duplicates(subset="Appointment_ID", keep="first").reset_index(drop=True)
    return clean, {"duplicates_removed": n_removed,
                   "duplicate_frequency": dup_freq,
                   "duplicate_rows": dup_rows}


def impute_waiting_time(df):
    """FR-1.2: log NaN waiting times, impute with department median
    (global median as fallback if a whole department is empty)."""
    df = df.copy()
    missing_before = int(df["Waiting_Time"].isna().sum())
    missing_by_dept = (df[df["Waiting_Time"].isna()]
                       .groupby("Department").size().rename("Missing Count").reset_index())
    dept_median = df.groupby("Department")["Waiting_Time"].transform("median")
    df["Waiting_Time_Imputed"] = df["Waiting_Time"].isna()
    df["Waiting_Time"] = df["Waiting_Time"].fillna(dept_median)
    df["Waiting_Time"] = df["Waiting_Time"].fillna(df["Waiting_Time"].median())
    return df, {"missing_before": missing_before,
                "missing_after": int(df["Waiting_Time"].isna().sum()),
                "missing_by_department": missing_by_dept}


def clean_pipeline(source="09_hospital_appointments.csv"):
    """Run the full hygiene layer. Returns (clean_df, log_dict)."""
    raw = load_data(source)
    schema = schema_report(raw)
    deduped, dup_log = remove_duplicates(raw)
    cleaned, null_log = impute_waiting_time(deduped)

    # Statistical Accuracy NFR: record-count reconciliation
    assert len(raw) - dup_log["duplicates_removed"] == len(cleaned), "Row-count reconciliation failed"
    assert cleaned["Appointment_ID"].is_unique, "Appointment_ID not unique after dedup"

    log = {"schema": schema, "raw_rows": len(raw), "clean_rows": len(cleaned),
           **dup_log, **null_log}
    return cleaned, log


# =====================================================================
# SUBSYSTEMS B, C and D (recommendations)  (FR-2.x, FR-3.x, FR-4.2)
# All floats rounded to 2 decimal places.
# =====================================================================
BAND_ORDER = ["Immediate", "Standard", "Delayed", "Critical"]


# ---------------- Subsystem B ----------------
def attendance_breakdown(df):
    """FR-2.1: counts and % share for every attendance category."""
    out = df["Attendance_Status"].value_counts().rename("Appointments").to_frame()
    out["Share (%)"] = (out["Appointments"] / len(df) * 100).round(2)
    return out.reset_index(names="Attendance_Status")


def department_noshow_ranking(df):
    """FR-2.2: department no-show % via groupby().agg(), ranked high -> low."""
    g = df.groupby("Department").agg(
        Total_Appointments=("Appointment_ID", "count"),
        No_Shows=("Attendance_Status", lambda s: (s == "No-Show").sum()),
    )
    g["No_Show_Rate (%)"] = (g["No_Shows"] / g["Total_Appointments"] * 100).round(2)
    g = g.sort_values("No_Show_Rate (%)", ascending=False).reset_index()
    g.insert(0, "Rank", range(1, len(g) + 1))
    return g


def sms_crosstab(df):
    """FR-2.3: SMS_Reminder (Yes/No) vs Attendance_Status — counts and row %."""
    counts = pd.crosstab(df["SMS_Reminder"], df["Attendance_Status"], margins=True, margins_name="Total")
    pct = (pd.crosstab(df["SMS_Reminder"], df["Attendance_Status"], normalize="index") * 100).round(2)
    return counts, pct


def fee_summary(df):
    """FR-2.4: mean fee per department + total fees from attended appointments."""
    mean_fee = (df.groupby("Department")["Consultation_Fee"].mean().round(2)
                .sort_values(ascending=False).rename("Mean_Fee").reset_index())
    attended = df[df["Attendance_Status"] == "Attended"]
    attended_fee = (attended.groupby("Department")["Consultation_Fee"].sum().round(2)
                    .rename("Attended_Fee_Total").reset_index())
    mean_fee = mean_fee.merge(attended_fee, on="Department", how="left").fillna(0)
    total = round(float(attended["Consultation_Fee"].sum()), 2)
    return mean_fee, total


# ---------------- Subsystem C ----------------
def department_wait(df):
    """FR-3.1: mean waiting time (minutes) per department."""
    return (df.groupby("Department")["Waiting_Time"].agg(["mean", "median", "max"]).round(2)
            .rename(columns={"mean": "Mean_Wait (min)", "median": "Median_Wait (min)",
                             "max": "Max_Wait (min)"})
            .sort_values("Mean_Wait (min)", ascending=False).reset_index())


def volume_trends(df):
    """FR-3.2: appointment counts by date x department + top peak days."""
    pivot = (df.groupby(["Appointment_Date", "Department"]).size()
             .unstack(fill_value=0).sort_index())
    daily = df.groupby("Appointment_Date").size().rename("Appointments")
    peak_days = daily.sort_values(ascending=False).head(10).reset_index()
    return pivot, daily.reset_index(), peak_days


def visit_type_wait(df):
    """FR-3.3: mean and median waiting time — New vs Follow-up."""
    return (df.groupby("Visit_Type")["Waiting_Time"].agg(["count", "mean", "median"]).round(2)
            .rename(columns={"count": "Appointments", "mean": "Mean_Wait (min)",
                             "median": "Median_Wait (min)"}).reset_index())


def dispersion_metrics(df):
    """FR-3.4: NumPy median, Q1, Q3, standard deviation of waiting time."""
    w = df["Waiting_Time"].to_numpy(dtype=float)
    q1, q3 = np.percentile(w, [25, 75])
    return {
        "Mean (min)": round(float(np.mean(w)), 2),
        "Median (min)": round(float(np.median(w)), 2),
        "Q1 (min)": round(float(q1), 2),
        "Q3 (min)": round(float(q3), 2),
        "IQR (min)": round(float(q3 - q1), 2),
        "Std Dev (min)": round(float(np.std(w, ddof=1)), 2),
        "Upper Fence (min)": round(float(q3 + 1.5 * (q3 - q1)), 2),
    }


def flag_outliers_and_bands(df, upper_fence=None):
    """FR-3.5: flag Wait > Q3 + 1.5*IQR and classify into waiting-time bands.
    Bands: Immediate (<=15), Standard (15-45], Delayed (45-fence], Critical (> fence).
    `upper_fence` may be passed so filtered views use the full-dataset threshold."""
    df = df.copy()
    if upper_fence is None:
        upper_fence = dispersion_metrics(df)["Upper Fence (min)"]
    df["Is_Outlier"] = df["Waiting_Time"] > upper_fence
    bins = [-np.inf, 15, 45, max(upper_fence, 45.01), np.inf]
    df["Waiting_Band"] = pd.cut(df["Waiting_Time"], bins=bins, labels=BAND_ORDER)
    return df, upper_fence


def band_summary(df):
    s = df["Waiting_Band"].value_counts().reindex(BAND_ORDER, fill_value=0).rename("Appointments")
    out = s.to_frame()
    out["Share (%)"] = (out["Appointments"] / max(len(df), 1) * 100).round(2)
    return out.reset_index(names="Waiting_Band")


# ---------------- Subsystem D: recommendations ----------------
def generate_recommendations(df):
    """FR-4.2: five evidence-backed recommendations computed from the data
    (scheduling buffers, SMS outreach, peak-day staffing). Operational only."""
    ns = department_noshow_ranking(df)
    top_ns = ns.iloc[0]
    wait = department_wait(df)
    top_wait = wait.iloc[0]
    counts, pct = sms_crosstab(df)
    ns_yes = pct.loc["Yes", "No-Show"] if "No-Show" in pct.columns and "Yes" in pct.index else float("nan")
    ns_no = pct.loc["No", "No-Show"] if "No-Show" in pct.columns and "No" in pct.index else float("nan")
    vt = visit_type_wait(df).set_index("Visit_Type")
    _, _, peak = volume_trends(df)
    peak_day = peak.iloc[0]
    fence = dispersion_metrics(df)["Upper Fence (min)"]
    n_out = int((df["Waiting_Time"] > fence).sum())

    recs = [
        ("Add scheduling buffers / overbooking in the highest no-show department",
         f"{top_ns['Department']} has the highest no-show rate ({top_ns['No_Show_Rate (%)']}%). "
         f"Controlled overbooking or buffer slots can recover lost capacity."),
        ("Expand SMS reminder coverage",
         f"No-show rate is {ns_yes}% with an SMS reminder vs {ns_no}% without. "
         f"Send reminders for every booking, especially in high no-show departments."),
        ("Re-balance staffing in the longest-queue department",
         f"{top_wait['Department']} averages {top_wait['Mean_Wait (min)']} min waiting "
         f"(overall median {dispersion_metrics(df)['Median (min)']} min). "
         f"Add support staff or stagger arrival slots there."),
        ("Separate New and Follow-up consultation slots",
         " | ".join(f"{k}: mean {r['Mean_Wait (min)']} min, median {r['Median_Wait (min)']} min"
                    for k, r in vt.iterrows())
         + ". Use dedicated slot lengths for the longer-waiting visit type."),
        ("Plan peak-day staffing and review outlier delays",
         f"Busiest day: {pd.Timestamp(peak_day['Appointment_Date']).date()} with "
         f"{peak_day['Appointments']} appointments. {n_out} appointments exceeded the IQR "
         f"outlier threshold of {fence} min — investigate these peak days first."),
    ]
    return recs


# =====================================================================
# SUBSYSTEM D - Interactive Streamlit dashboard (FR-4.1, FR-4.2)
# =====================================================================
def run_dashboard():
    import streamlit as st

    st.set_page_config(page_title="Hospital Appointment Operations Analysis", layout="wide")
    st.title("🏥 Hospital Appointment Operations Analysis")
    st.caption("Operational scheduling metadata only — no clinical inference is made. "
               "| ABES Engineering College • Nakul (2026B01010862)")
    # ---------- Data loading ----------
    uploaded = st.sidebar.file_uploader("Upload updated CSV (optional)", type="csv")


    @st.cache_data(show_spinner=False)
    def run_pipeline(source):
        t0 = time.perf_counter()
        df, log = clean_pipeline(source)
        return df, log, time.perf_counter() - t0


    try:
        data, log, load_time = run_pipeline(uploaded if uploaded is not None
                                            else "09_hospital_appointments.csv")
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        st.stop()

    # ---------- Global outlier threshold (computed on full clean data) ----------
    disp_all = dispersion_metrics(data)
    data, fence = flag_outliers_and_bands(data, disp_all["Upper Fence (min)"])

    # ---------- FR-4.1 Interactive filtering ----------
    departments = sorted(data["Department"].unique())
    chosen = st.sidebar.multiselect("Department filter", departments, default=departments)
    df = data[data["Department"].isin(chosen)]
    if df.empty:
        st.warning("Select at least one department.")
        st.stop()

    t0 = time.perf_counter()
    att = attendance_breakdown(df)
    noshow = department_noshow_ranking(df)
    ct_counts, ct_pct = sms_crosstab(df)
    fee_tbl, fee_total = fee_summary(df)
    dept_wait = department_wait(df)
    pivot, daily, peak = volume_trends(df)
    vt_wait = visit_type_wait(df)
    disp = dispersion_metrics(df)
    bands = band_summary(df)
    recs = generate_recommendations(df)
    calc_time = time.perf_counter() - t0

    # ---------- KPI cards ----------
    k = st.columns(5)
    k[0].metric("Valid Appointments", f"{len(df):,}")
    k[1].metric("Attendance Rate", f"{(df['Attendance_Status'] == 'Attended').mean() * 100:.2f} %")
    k[2].metric("No-Show Rate", f"{(df['Attendance_Status'] == 'No-Show').mean() * 100:.2f} %")
    k[3].metric("Median Wait", f"{disp['Median (min)']} min")
    k[4].metric("Attended Fees", f"₹ {fee_total:,.2f}")
    st.caption(f"Load+clean: {load_time:.3f}s • Compute: {calc_time:.3f}s "
               f"(target < 2.5s for 100k rows)")

    tab_data, tab_att, tab_queue, tab_band, tab_rec = st.tabs(
        ["📂 Data Hygiene", "✅ Attendance & Revenue", "⏱ Queue Analytics",
         "📊 Waiting Bands & Outliers", "💡 Recommendations"])

    with tab_data:
        st.subheader("FR-1.1 Schema & preview")
        st.write(f"Raw records: **{log['raw_rows']}** → Clean records: **{log['clean_rows']}**")
        st.dataframe(log["schema"]["dtypes"], use_container_width=True, hide_index=True)
        st.dataframe(log["schema"]["preview"], use_container_width=True, hide_index=True)
        st.subheader("FR-1.2 Missing Waiting_Time (imputed with department median)")
        st.write(f"Missing before: **{log['missing_before']}** • after: **{log['missing_after']}**")
        st.dataframe(log["missing_by_department"], use_container_width=True, hide_index=True)
        st.subheader("FR-1.3 Duplicate Appointment_ID removal")
        st.write(f"Duplicate rows removed: **{log['duplicates_removed']}**")
        st.dataframe(log["duplicate_frequency"], use_container_width=True, hide_index=True)

    with tab_att:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("FR-2.1 Attendance breakdown")
            st.dataframe(att, use_container_width=True, hide_index=True)
        with c2:
            st.subheader("FR-2.2 Department no-show ranking")
            st.dataframe(noshow, use_container_width=True, hide_index=True)
            st.bar_chart(noshow.set_index("Department")["No_Show_Rate (%)"])
        st.subheader("FR-2.3 SMS reminder vs attendance (counts)")
        st.dataframe(ct_counts, use_container_width=True)
        st.subheader("SMS reminder vs attendance (row %)")
        st.dataframe(ct_pct, use_container_width=True)
        st.subheader("FR-2.4 Consultation fees (currency)")
        st.dataframe(fee_tbl, use_container_width=True, hide_index=True)
        st.success(f"Total estimated fees from attended appointments: ₹ {fee_total:,.2f}")

    with tab_queue:
        st.subheader("FR-3.1 Average waiting time by department (minutes)")
        st.dataframe(dept_wait, use_container_width=True, hide_index=True)
        st.bar_chart(dept_wait.set_index("Department")["Mean_Wait (min)"])
        st.subheader("FR-3.2 Daily appointment volume by department")
        st.line_chart(pivot)
        st.write("Top 10 peak demand days")
        st.dataframe(peak, use_container_width=True, hide_index=True)
        st.subheader("FR-3.3 New vs Follow-up waiting time")
        st.dataframe(vt_wait, use_container_width=True, hide_index=True)
        st.subheader("FR-3.4 NumPy dispersion metrics (waiting time)")
        st.dataframe(pd.DataFrame(disp, index=["Value"]).T, use_container_width=True)

    with tab_band:
        st.subheader("FR-3.5 Waiting-time bands")
        st.caption("Immediate ≤ 15 • Standard 15–45 • Delayed 45–upper fence • "
                   f"Critical > {fence} min (Q3 + 1.5×IQR)")
        st.dataframe(bands, use_container_width=True, hide_index=True)
        st.bar_chart(bands.set_index("Waiting_Band")["Appointments"])
        outliers = df[df["Is_Outlier"]].sort_values("Waiting_Time", ascending=False)
        st.warning(f"⚠ {len(outliers)} outlier appointments exceed {fence} min")
        st.dataframe(outliers[["Appointment_ID", "Appointment_Date", "Department",
                               "Visit_Type", "Waiting_Time", "Waiting_Band"]],
                     use_container_width=True, hide_index=True)

    with tab_rec:
        st.subheader("FR-4.2 Five operational recommendations")
        for i, (title, evidence) in enumerate(recs, 1):
            st.markdown(f"**{i}. {title}**")
            st.write(f"Evidence: {evidence}")

    # ---------- Export interface ----------
    st.sidebar.markdown("---")
    st.sidebar.subheader("Export")
    st.sidebar.download_button("Cleaned data (CSV)", df.to_csv(index=False).encode("utf-8"),
                               "cleaned_appointments.csv", "text/csv")
    md = ["# Hospital Appointment Operations Summary\n",
          "## Attendance\n", att.to_markdown(index=False) if hasattr(att, "to_markdown") else att.to_string(index=False),
          "\n\n## Department No-Show Ranking\n", noshow.to_string(index=False),
          "\n\n## Department Waiting Time\n", dept_wait.to_string(index=False),
          "\n\n## Dispersion\n", "\n".join(f"- {k}: {v}" for k, v in disp.items()),
          "\n\n## Recommendations\n", "\n".join(f"{i}. {t} — {e}" for i, (t, e) in enumerate(recs, 1))]
    st.sidebar.download_button("Summary report (Markdown)", "\n".join(md).encode("utf-8"),
                               "summary_report.md", "text/markdown")


# =====================================================================
# Console report (no Streamlit needed)
# =====================================================================
def run_console(path=CSV_PATH):
    t0 = time.perf_counter()
    df, log = clean_pipeline(path)
    disp = dispersion_metrics(df)
    df, fence = flag_outliers_and_bands(df, disp["Upper Fence (min)"])
    print(f"Rows {log['raw_rows']} -> {log['clean_rows']} | duplicates removed "
          f"{log['duplicates_removed']} | missing waits imputed {log['missing_before']}\n")
    print("FR-2.1 Attendance\n", attendance_breakdown(df), "\n")
    print("FR-2.2 No-show ranking\n", department_noshow_ranking(df), "\n")
    counts, pct = sms_crosstab(df)
    print("FR-2.3 SMS crosstab\n", counts, "\n", pct, "\n")
    fee_tbl, fee_total = fee_summary(df)
    print("FR-2.4 Fees\n", fee_tbl, f"\nTotal attended fees: {fee_total}\n")
    print("FR-3.1 Dept wait\n", department_wait(df), "\n")
    _, _, peak = volume_trends(df)
    print("FR-3.2 Peak days\n", peak, "\n")
    print("FR-3.3 Visit type\n", visit_type_wait(df), "\n")
    print("FR-3.4 Dispersion\n", disp, "\n")
    print("FR-3.5 Bands\n", band_summary(df), f"\nOutliers (> {fence} min): {int(df['Is_Outlier'].sum())}\n")
    print("FR-4.2 Recommendations")
    for i, (t, e) in enumerate(generate_recommendations(df), 1):
        print(f" {i}. {t}\n    {e}")
    print(f"\nElapsed: {time.perf_counter() - t0:.3f}s")


def _in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _in_streamlit():
    run_dashboard()
elif __name__ == "__main__":
    run_console(sys.argv[1] if len(sys.argv) > 1 else CSV_PATH)
