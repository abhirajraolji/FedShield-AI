
"""FedShield AI — MedMNIST/PathMNIST hospital command center.

Run from the project root:
    python -m streamlit run app.py

This dashboard replays saved experiments; it does not train models.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="FedShield AI | Command Center", page_icon="🛡️", layout="wide")
RESULTS = Path(__file__).resolve().parent / "results"

# Class counts from the original 8,000-image, four-hospital PathMNIST split.
HOSPITAL_COUNTS = [
    [316, 34, 9, 730, 59, 414, 76, 0, 81],
    [84, 237, 202, 104, 187, 277, 118, 6, 349],
    [203, 21, 286, 7, 437, 139, 274, 96, 233],
    [263, 525, 391, 99, 39, 224, 257, 748, 475],
]


@st.cache_data
def load_json(filename):
    path = RESULTS / filename
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def normalize_history(data):
    """Accept accuracy lists, round dictionaries, or common saved dict wrappers."""
    if isinstance(data, dict):
        for key in ("history", "fedprox", "federated_history", "accuracies", "rounds", "results"):
            if key in data:
                return normalize_history(data[key])
        return []
    if not isinstance(data, list):
        return []
    output = []
    for index, item in enumerate(data, 1):
        if isinstance(item, (int, float)):
            output.append({"round": index, "accuracy": float(item)})
        elif isinstance(item, dict):
            accuracy = next((item[k] for k in ("accuracy", "global_accuracy", "test_accuracy", "acc") if k in item), None)
            if isinstance(accuracy, (int, float)):
                output.append({"round": int(item.get("round", index)), "accuracy": float(accuracy)})
    return output


def accuracy_at(history, round_number):
    for item in history:
        if item["round"] == round_number:
            return item["accuracy"]
    return None


def format_accuracy(value):
    return f"{value:.2f}%" if value is not None else "N/A"


comparison = load_json("comparison.json")
security = load_json("security.json")
private = load_json("federated_dp.json")
privacy = load_json("privacy_diagnostics.json")
history_file = load_json("history.json")
fedprox_file = load_json("fedprox.json")

fedavg_history = normalize_history(history_file)
if not fedavg_history and isinstance(comparison, dict):
    fedavg_history = normalize_history(comparison.get("federated_history", []))
fedprox_history = normalize_history(fedprox_file)
private_history = normalize_history(private)

SCENARIOS = {
    "Normal FedAvg": "normal",
    "Hospital 4 Attack": "attacked",
    "Attack + Median Defence": "defended",
}

st.sidebar.title("🛡️ FedShield AI")
st.sidebar.caption("MedMNIST Hospital Collaboration Command Center")
st.sidebar.divider()
scenario = st.sidebar.selectbox("Network scenario", list(SCENARIOS))
security_histories = {
    key: normalize_history(security.get(key, []))
    for key in ("normal", "attacked", "defended")
} if isinstance(security, dict) else {}
security_rounds = max((item["round"] for rows in security_histories.values() for item in rows), default=3)
security_round = st.sidebar.slider("Security experiment round", 1, security_rounds, security_rounds)
st.sidebar.info("Replay mode: these controls select saved experiments. No training runs when you change a setting.")
st.sidebar.caption("Security experiment: Hospital 4, amplified/reversed update scale = −5.")
st.sidebar.divider()
st.sidebar.caption(f"Results directory: {RESULTS}")

st.title("🛡️ FedShield AI")
st.subheader("Hospital Collaboration Command Center")
st.caption("Federated Learning · Differential Privacy · Poisoning Defense · Training Optimization")
st.warning("Research prototype using simulated hospitals and public PathMNIST data. Not for clinical use.")

selected_security = security_histories.get(SCENARIOS[scenario], [])
selected_accuracy = accuracy_at(selected_security, security_round)
metric_columns = st.columns(4)
metric_columns[0].metric("Simulated Hospitals", "4")
metric_columns[1].metric("Training Images", "8,000")
metric_columns[2].metric("Security Round", security_round)
metric_columns[3].metric("Scenario Accuracy", format_accuracy(selected_accuracy))

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏥 Hospital Network",
    "⚔️ Attack Lab",
    "🔐 Privacy Lab",
    "📈 Training Optimization",
    "📋 Audit Report",
])

with tab1:
    st.header("Hospital Network")
    st.write("Four simulated hospitals each train a local CNN on different PathMNIST tissue-image distributions.")
    columns = st.columns(4)
    for i, counts in enumerate(HOSPITAL_COUNTS):
        with columns[i]:
            st.metric(f"Hospital {i + 1}", f"{sum(counts):,} images")
            if i == 3 and scenario != "Normal FedAvg":
                st.error("Attacker in selected security scenario")
            else:
                st.success("Honest participant")
    rows = [
        {"Hospital": f"Hospital {hospital + 1}", "Class": f"Class {class_id}", "Images": count}
        for hospital, counts in enumerate(HOSPITAL_COUNTS)
        for class_id, count in enumerate(counts)
    ]
    distribution = pd.DataFrame(rows)
    fig = px.bar(distribution, x="Hospital", y="Images", color="Class", title="Non-IID Tissue Class Distribution")
    st.plotly_chart(fig, use_container_width=True)
    st.info("Non-IID means the hospitals have different class distributions. Hospital 1, for example, has no Class 7 training images.")
    st.subheader("How Federated Learning Works")
    st.markdown("""
    1. The server sends the global CNN to each hospital.
    2. Hospitals train the model using only their local images.
    3. Hospitals return model parameters, not their raw images.
    4. The server aggregates the parameters to create a new global model.
    5. The process repeats for several communication rounds.
    """)
    st.caption("All four hospitals are simulated sequentially on one computer.")

with tab2:
    st.header("Model Poisoning Attack Lab")
    st.write("Compare saved results for normal FedAvg, poisoned FedAvg, and coordinate-wise median aggregation.")
    if security_histories and all(security_histories.values()):
        normal = accuracy_at(security_histories["normal"], security_round)
        attacked = accuracy_at(security_histories["attacked"], security_round)
        defended = accuracy_at(security_histories["defended"], security_round)
        c1, c2, c3 = st.columns(3)
        c1.metric("Normal FedAvg", format_accuracy(normal))
        c2.metric("Under Attack", format_accuracy(attacked),
                  delta=f"{attacked - normal:+.2f} pp" if normal is not None and attacked is not None else None)
        c3.metric("Median Defence", format_accuracy(defended),
                  delta=f"{defended - attacked:+.2f} pp" if defended is not None and attacked is not None else None)
        if scenario == "Normal FedAvg":
            st.success("All hospitals submit legitimate updates.")
        elif scenario == "Hospital 4 Attack":
            st.error("Hospital 4 submits a reversed, amplified model update. Ordinary FedAvg is vulnerable in this test.")
        else:
            st.info("Hospital 4 still attacks, but coordinate-wise median aggregation reduces the damage in this test.")
        names = {"normal": "Normal FedAvg", "attacked": "Poisoned FedAvg", "defended": "Median Defence"}
        plot_rows = [
            {"Round": item["round"], "Accuracy (%)": item["accuracy"], "Scenario": names[key]}
            for key, history in security_histories.items()
            for item in history
        ]
        fig = px.line(pd.DataFrame(plot_rows), x="Round", y="Accuracy (%)", color="Scenario", markers=True,
                      title="Measured Attack and Defence Performance")
        fig.update_xaxes(dtick=1)
        fig.add_vline(x=security_round, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
        if normal is not None and attacked is not None and defended is not None:
            st.write(f"The attack reduced accuracy by **{normal - attacked:.2f} percentage points** at round {security_round}.")
            st.write(f"Median aggregation recovered **{defended - attacked:.2f} percentage points** relative to the attacked run.")
        st.caption("This experiment uses update scaling −5 and coordinate-wise median. It does not test label flipping or Cosine-Guard.")
    else:
        st.warning("No complete security results found. Run: python -m src.security")

with tab3:
    st.header("Differential Privacy Lab")
    st.write("DP-SGD clips per-example gradients and adds Gaussian noise during local training.")
    st.caption("The recorded private and non-private runs may use different training configurations, so their accuracy gap is not a controlled estimate of DP's effect alone.")
    if isinstance(comparison, dict) and private_history:
        baseline = comparison.get("federated")
        private_final = private_history[-1]["accuracy"]
        c1, c2 = st.columns(2)
        c1.metric("Earlier Non-Private FedAvg", format_accuracy(baseline if isinstance(baseline, (int, float)) else None))
        c2.metric("Private FedAvg", format_accuracy(private_final))
    if private_history:
        fig = px.line(pd.DataFrame(private_history), x="round", y="accuracy", markers=True,
                      title="DP-FedAvg Accuracy Across Rounds", labels={"round": "Round", "accuracy": "Accuracy (%)"})
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, use_container_width=True)
    if isinstance(private, list):
        epsilon_rows = []
        for item in private:
            if not isinstance(item, dict):
                continue
            for hospital, epsilon in enumerate(item.get("cumulative_client_epsilons", []), 1):
                epsilon_rows.append({"Round": item.get("round"), "Hospital": f"Hospital {hospital}", "Epsilon": epsilon})
        if epsilon_rows:
            st.subheader("Cumulative Privacy Budget")
            fig = px.line(pd.DataFrame(epsilon_rows), x="Round", y="Epsilon", color="Hospital", markers=True,
                          title="Recorded Cumulative Epsilon by Hospital")
            fig.update_xaxes(dtick=1)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Opacus RDP accounting, δ = 10⁻⁵. Secure RNG was disabled in the experiment; this is not a production privacy certification.")
    st.subheader("Pooled Privacy Diagnostics")
    if isinstance(privacy, list):
        rows = [{
            "Experiment": item.get("experiment", "Unknown"),
            "Accuracy (%)": item.get("accuracy"),
            "Noise multiplier": item.get("noise_multiplier"),
            "Epsilon": item.get("epsilon"),
        } for item in privacy if isinstance(item, dict)]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption("These diagnostics use pooled training data; they are separate from the federated DP experiment.")
    else:
        st.info("Optional: run python -m src.privacy to create privacy_diagnostics.json.")

with tab4:
    st.header("Training Optimization — FedAvg vs FedProx")
    st.write("Compare the saved ten-round training histories on the same PathMNIST task.")
    if fedavg_history or fedprox_history:
        c1, c2 = st.columns(2)
        c1.metric("FedAvg Final Accuracy", format_accuracy(fedavg_history[-1]["accuracy"] if fedavg_history else None))
        c2.metric("FedProx Final Accuracy", format_accuracy(fedprox_history[-1]["accuracy"] if fedprox_history else None))
        plot_rows = []
        for label, history in (("FedAvg", fedavg_history), ("FedProx", fedprox_history)):
            plot_rows.extend({"Round": item["round"], "Accuracy (%)": item["accuracy"], "Algorithm": label}
                             for item in history)
        fig = px.line(pd.DataFrame(plot_rows), x="Round", y="Accuracy (%)", color="Algorithm", markers=True,
                      title="Recorded Training Performance Across Communication Rounds")
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, use_container_width=True)
        max_round = max(item["round"] for item in fedavg_history + fedprox_history)
        chosen_round = st.slider("Inspect training round", 1, max_round, max_round, key="training_round")
        avg_value = accuracy_at(fedavg_history, chosen_round)
        prox_value = accuracy_at(fedprox_history, chosen_round)
        c1, c2 = st.columns(2)
        c1.metric("FedAvg at selected round", format_accuracy(avg_value))
        c2.metric("FedProx at selected round", format_accuracy(prox_value))
        if avg_value is not None and prox_value is not None:
            winner = "FedAvg" if avg_value > prox_value else "FedProx" if prox_value > avg_value else "Neither (tie)"
            st.info(f"At round {chosen_round}, **{winner}** has the higher recorded accuracy. The difference is **{abs(avg_value - prox_value):.2f} percentage points**.")
        st.markdown("**FedAvg** aggregates locally trained models. **FedProx** adds a proximal penalty during local training to discourage large deviations from the global model.")
        st.caption("The runs must use comparable splits, initialization, training settings, and evaluation data for a controlled algorithm comparison.")
    else:
        st.warning("Missing training histories. Expected results/history.json and results/fedprox.json.")

with tab5:
    st.header("Experiment Audit Report")
    st.write("Export the selected scenario and recorded experiment results as a JSON report.")
    st.write(f"**Scenario:** {scenario}")
    st.write(f"**Security round:** {security_round}")
    st.write(f"**Recorded accuracy:** {format_accuracy(selected_accuracy)}")
    st.write("**Dataset:** PathMNIST (MedMNIST)")
    st.write("**Participants:** 4 simulated hospitals")
    st.write("**Training images:** 8,000")
    limitations = [
        "Four simulated hospitals, not real medical institutions",
        "Public PathMNIST dataset, not private patient records",
        "Dashboard replays recorded experiments rather than training live",
        "No cryptographic secure aggregation implemented",
        "Secure RNG disabled in the differential privacy experiment",
        "Privacy and poisoning experiments were performed separately",
        "No clinical validation",
        "Coordinate-wise median is not robust to every attack",
    ]
    report = {
        "project": "FedShield AI", "created_at": datetime.now().isoformat(),
        "dataset": "PathMNIST", "simulation": True,
        "hospital_count": 4, "training_images": 8000,
        "hospital_class_counts": HOSPITAL_COUNTS,
        "selected_scenario": scenario, "selected_security_round": security_round,
        "selected_accuracy": selected_accuracy,
        "fedavg_history": fedavg_history, "fedprox_history": fedprox_history,
        "security_results": security, "federated_dp_results": private,
        "pooled_privacy_diagnostics": privacy, "comparison": comparison,
        "limitations": limitations,
    }
    st.download_button("Download Audit Report", json.dumps(report, indent=2, default=str),
                       file_name="fedshield_audit_report.json", mime="application/json")
    st.subheader("Limitations")
    for limitation in limitations:
        st.write(f"• {limitation}")

st.divider()
st.caption("FedShield AI — Experimental privacy-preserving federated learning for medical-image classification.")
