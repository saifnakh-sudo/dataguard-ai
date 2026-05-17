import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import json
import requests
import matplotlib.pyplot as plt

from src.phase1_prevention import prevention_filter
from src.phase2_detection import detect_anomalies
from src.phase3_correction import retrain_and_report

st.set_page_config(page_title='DataGuard AI', layout='wide')

st.markdown("""
<div style='text-align:center; padding: 1.5rem 0'>
    <h1 style='font-size:2.5rem'>DataGuard AI</h1>
    <p style='font-size:1.1rem; color:#888'>Training Data Firewall — Prevention | Detection | Correction</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Configuration")
    k_neighbors = st.slider("k-NN neighbors (Phase 1)", 3, 10, 5)
    sigma = st.slider("Sigma multiplier (Phase 1)", 1.0, 3.0, 2.0, 0.1)
    contamination = st.slider("Contamination (Phase 2)", 0.01, 0.20, 0.05, 0.01)
    st.markdown("---")
    st.markdown("**How to use:**\n1. Upload a poisoned CSV\n2. Walk through each tab\n3. Review the final report")

uploaded = st.file_uploader(
    "Upload poisoned dataset (CSV with 'text' and 'label' columns)",
    type=['csv'])

if not uploaded:
    st.info("Upload data/poisoned.csv to get started.")
    st.stop()

df = pd.read_csv(uploaded)
has_ground_truth = 'is_poisoned' in df.columns

st.success(
    f"Loaded {len(df):,} rows | "
    f"Label 0: {(df['label']==0).sum()} | Label 1: {(df['label']==1).sum()}" +
    (f" | True poison injected: {df['is_poisoned'].sum()}" if has_ground_truth else "")
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Phase 1 — Prevention", "Phase 2 — Detection",
     "Phase 3 — Correction", "Final Report"])

# ── TAB 1 ────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Phase 1 — k-NN Prevention Filter")
    st.markdown("Each incoming sample is compared to its **k nearest neighbors of the same class**. If its distance exceeds mean + sigma x sigma_mult, it is **rejected**.")

    if st.button("Run Phase 1", key="run1"):
        with st.spinner("Running k-NN filter..."):
            accepted, rejected = prevention_filter(df, k=k_neighbors, sigma_mult=sigma)
            st.session_state['accepted'] = accepted
            st.session_state['rejected'] = rejected

    if 'accepted' in st.session_state:
        accepted = st.session_state['accepted']
        rejected = st.session_state['rejected']
        c1, c2, c3 = st.columns(3)
        c1.metric("Accepted", len(accepted))
        c2.metric("Rejected", len(rejected))
        if has_ground_truth and 'is_poisoned' in rejected.columns:
            caught = rejected['is_poisoned'].sum()
            total_p = df['is_poisoned'].sum()
            c3.metric("Poison caught", f"{caught}/{total_p}",
                      f"{caught/total_p*100:.1f}%" if total_p > 0 else "")
        st.subheader("Rejected samples (top 50)")
        st.dataframe(rejected.head(50), use_container_width=True)
    else:
        st.info("Click Run Phase 1 to start.")

# ── TAB 2 ────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Phase 2 — Isolation Forest Detector")
    st.markdown("**Isolation Forest** flags anomalous samples for human review — not auto-deleted.")

    if 'accepted' not in st.session_state:
        st.warning("Run Phase 1 first.")
    else:
        if st.button("Run Phase 2", key="run2"):
            with st.spinner("Running Isolation Forest..."):
                flagged_df = detect_anomalies(st.session_state['accepted'], contamination=contamination)
                st.session_state['flagged_df'] = flagged_df

        if 'flagged_df' in st.session_state:
            flagged_df = st.session_state['flagged_df']
            flagged = flagged_df[flagged_df['is_flagged']]
            clean_p2 = flagged_df[~flagged_df['is_flagged']]
            c1, c2, c3 = st.columns(3)
            c1.metric("Flagged", len(flagged))
            c2.metric("Clean", len(clean_p2))
            if has_ground_truth and 'is_poisoned' in flagged.columns:
                caught2 = flagged['is_poisoned'].sum()
                remaining = st.session_state['accepted']['is_poisoned'].sum() if 'is_poisoned' in st.session_state['accepted'].columns else 0
                c3.metric("Poison caught", f"{caught2}/{remaining}",
                          f"{caught2/remaining*100:.1f}%" if remaining > 0 else "")
            st.subheader("Anomaly score distribution (top 50 samples)")
            fig, ax = plt.subplots(figsize=(10, 3))
            top50 = flagged_df.head(50)
            colors = ['#c0392b' if f else '#27ae60' for f in top50['is_flagged']]
            ax.bar(range(len(top50)), top50['anomaly_score'], color=colors)
            ax.set_xlabel("Sample rank")
            ax.set_ylabel("Anomaly score")
            ax.set_title("Anomaly Scores — Higher = More Anomalous  |  Red: Flagged   Green: Clean")
            st.pyplot(fig)
            plt.close()
            st.subheader("Top flagged samples")
            st.dataframe(flagged.head(30), use_container_width=True)
        else:
            st.info("Click Run Phase 2 to start.")

# ── TAB 3 ────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Phase 3 — Quarantine and Retrain")
    st.markdown("Confirmed anomalies are moved to **quarantine** (never deleted — full auditability). The model retrains on the clean remainder.")

    if 'flagged_df' not in st.session_state:
        st.warning("Run Phases 1 and 2 first.")
    else:
        if st.button("Run Phase 3", key="run3"):
            with st.spinner("Retraining model..."):
                flagged_df = st.session_state['flagged_df']
                clean_final = flagged_df[~flagged_df['is_flagged']]
                report = retrain_and_report(clean_final, df)
                st.session_state['report'] = report
                st.session_state['quarantine'] = flagged_df[flagged_df['is_flagged']]
                st.session_state['clean_final'] = clean_final

        if 'report' in st.session_state:
            report = st.session_state['report']
            quarantine = st.session_state['quarantine']
            clean_final = st.session_state['clean_final']
            c1, c2 = st.columns(2)
            c1.metric("Quarantined", len(quarantine))
            c2.metric("Clean dataset", len(clean_final))
            st.subheader("Metrics: Before vs After Defense")
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            before = [report['before_defense'][m] for m in metrics]
            after = [report['after_defense'][m] for m in metrics]
            fig, ax = plt.subplots(figsize=(9, 4))
            x = range(len(metrics))
            w = 0.35
            ax.bar([i - w/2 for i in x], before, w, label='Before Defense', color='#c0392b', alpha=0.85)
            ax.bar([i + w/2 for i in x], after, w, label='After Defense', color='#27ae60', alpha=0.85)
            ax.set_xticks(list(x))
            ax.set_xticklabels([m.capitalize() for m in metrics])
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Score")
            ax.set_title("DataGuard AI — Defense Effectiveness")
            ax.legend()
            for i, (b, a) in enumerate(zip(before, after)):
                ax.text(i - w/2, b + 0.01, f'{b:.3f}', ha='center', fontsize=8)
                ax.text(i + w/2, a + 0.01, f'{a:.3f}', ha='center', fontsize=8)
            st.pyplot(fig)
            plt.close()
            st.json(report)
        else:
            st.info("Click Run Phase 3 to start.")

# ── TAB 4 ────────────────────────────────────────────────────────────────────
with tab4:
    st.header("Audit Report")

    if 'report' not in st.session_state:
        st.warning("Complete all three phases first — go to Phase 1, Phase 2, and Phase 3 tabs and click Run on each one.")
    else:
        report = st.session_state['report']
        b = report['before_defense']
        a = report['after_defense']

        st.markdown(f"""
### Summary

| Metric    | Before Defense | After Defense | Delta |
|-----------|---------------|---------------|-------|
| Accuracy  | {b['accuracy']:.3f} | {a['accuracy']:.3f} | {(a['accuracy']-b['accuracy'])*100:+.1f}pp |
| Precision | {b['precision']:.3f} | {a['precision']:.3f} | {(a['precision']-b['precision'])*100:+.1f}pp |
| Recall    | {b['recall']:.3f} | {a['recall']:.3f} | {(a['recall']-b['recall'])*100:+.1f}pp |
| F1-Score  | {b['f1']:.3f} | {a['f1']:.3f} | {(a['f1']-b['f1'])*100:+.1f}pp |

### Pipeline Summary
- Phase 1 rejected {len(st.session_state['rejected'])} samples via k-NN distance filter
- Phase 2 flagged {len(st.session_state['flagged_df'][st.session_state['flagged_df']['is_flagged']])} samples via Isolation Forest
- Phase 3 quarantined {len(st.session_state['quarantine'])} samples — none deleted, all auditable

### Known Limitations
- Does not protect LLMs at training time (operates at the data preparation stage)
- Assumes the initial reference set is clean
- Does not handle adaptive attackers who are aware of the defense
- Thresholds are tuned for this dataset and may require adjustment for other domains
        """)

        report_md = f"""# DataGuard AI — Audit Report

## Results

| Metric    | Before Defense | After Defense | Delta |
|-----------|---------------|---------------|-------|
| Accuracy  | {b['accuracy']:.4f} | {a['accuracy']:.4f} | {(a['accuracy']-b['accuracy'])*100:+.2f}pp |
| Precision | {b['precision']:.4f} | {a['precision']:.4f} | {(a['precision']-b['precision'])*100:+.2f}pp |
| Recall    | {b['recall']:.4f} | {a['recall']:.4f} | {(a['recall']-b['recall'])*100:+.2f}pp |
| F1-Score  | {b['f1']:.4f} | {a['f1']:.4f} | {(a['f1']-b['f1'])*100:+.2f}pp |
"""
        col1, col2 = st.columns(2)
        col1.download_button("Download audit_report.md",
                             data=report_md,
                             file_name="audit_report.md",
                             mime="text/markdown")
        col2.download_button("Download metrics.json",
                             data=json.dumps(report, indent=2),
                             file_name="metrics.json",
                             mime="application/json")

        st.markdown("---")
        st.subheader("Email Report")

        with st.expander("Send via EmailJS", expanded=True):
            ejs_service    = st.text_input("EmailJS Service ID",  value="service_ex9rkqj")
            ejs_template   = st.text_input("EmailJS Template ID", value="template_fhastq6")
            ejs_pubkey     = st.text_input("EmailJS Public Key",  value="nxtmDO9WPDWongJjX")
            ejs_privatekey = st.text_input("EmailJS Private Key", placeholder="Paste your private key here", type="password")
            to_email       = st.text_input("Recipient email address", placeholder="you@example.com")

            if st.button("Send Report"):
                if not ejs_privatekey:
                    st.error("Please paste your Private Key.")
                elif not to_email:
                    st.error("Please enter a recipient email address.")
                else:
                    payload = {
                        "service_id":  ejs_service,
                        "template_id": ejs_template,
                        "user_id":     ejs_pubkey,
                        "accessToken": ejs_privatekey,
                        "template_params": {
                            "to_email":         to_email,
                            "before_accuracy":  f"{b['accuracy']:.3f}",
                            "after_accuracy":   f"{a['accuracy']:.3f}",
                            "delta_accuracy":   f"{(a['accuracy']-b['accuracy'])*100:+.1f}pp",
                            "before_precision": f"{b['precision']:.3f}",
                            "after_precision":  f"{a['precision']:.3f}",
                            "before_recall":    f"{b['recall']:.3f}",
                            "after_recall":     f"{a['recall']:.3f}",
                            "before_f1":        f"{b['f1']:.3f}",
                            "after_f1":         f"{a['f1']:.3f}",
                        }
                    }
                    try:
                        response = requests.post(
                            "https://api.emailjs.com/api/v1.0/email/send",
                            json=payload,
                            timeout=10
                        )
                        if response.status_code == 200:
                            st.success(f"Report successfully sent to {to_email}.")
                        else:
                            st.error(f"Failed: {response.text}")
                    except Exception as e:
                        st.error(f"Error: {e}")