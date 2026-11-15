
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st


# ---------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------

st.set_page_config(
    page_title="FedShield AI | Command Center",
    page_icon="🛡️",
    layout="wide"
)

RESULTS = Path(__file__).resolve().parent / "results"

HOSPITAL_COUNTS = [
    [316, 34, 9, 730, 59, 414, 76, 0, 81],
    [84, 237, 202, 104, 187, 277, 118, 6, 349],
    [203, 21, 286, 7, 437, 139, 274, 96, 233],
    [263, 525, 391, 99, 39, 224, 257, 748, 475]
]


# ---------------------------------------
# 2. LOAD SAVED EXPERIMENTS
# ---------------------------------------

@st.cache_data
def load_json(filename):
    path = RESULTS / filename

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


comparison = load_json("comparison.json")
private = load_json("federated_dp.json")
security = load_json("security.json")
privacy = load_json("privacy_diagnostics.json")
fedavg_history = load_json("history.json")
fedprox_history = load_json("fedprox.json")


# ---------------------------------------
# 3. SIDEBAR
# ---------------------------------------

st.sidebar.title("🛡️ FedShield AI")
st.sidebar.caption("Hospital Collaboration Command Center")

st.sidebar.divider()

st.sidebar.subheader("Simulation settings")

scenario = st.sidebar.selectbox(
    "Choose network scenario",
    [
        "Normal FedAvg",
        "Hospital 4 Attack",
        "Attack + Median Defence"
    ]
)

selected_round = st.sidebar.slider(
    "Communication round",
    min_value=1,
    max_value=3,
    value=3
)

st.sidebar.divider()

st.sidebar.info(
    "Replay mode: the controls select actual "
    "recorded experiments. They do not retrain "
    "models or generate new predictions."
)

st.sidebar.caption(
    "Attack: Hospital 4, update scale = -5"
)


# ---------------------------------------
# 4. HEADER
# ---------------------------------------

st.title("🛡️ FedShield AI")
st.subheader("Hospital Collaboration Command Center")

st.caption(
    "Federated Learning | Differential Privacy | "
    "Model Poisoning Defence"
)

st.warning(
    "Research prototype using simulated hospitals "
    "and public PathMNIST data. Not for clinical use."
)


# ---------------------------------------
# 5. SELECT RECORDED SECURITY RESULT
# ---------------------------------------

scenario_keys = {
    "Normal FedAvg": "normal",
    "Hospital 4 Attack": "attacked",
    "Attack + Median Defence": "defended"
}

scenario_key = scenario_keys[scenario]

current_accuracy = None

if security and scenario_key in security:
    history = security[scenario_key]

    matching = [
        item for item in history
        if item["round"] == selected_round
    ]

    if matching:
        current_accuracy = matching[0]["accuracy"]


# ---------------------------------------
# 6. MAIN METRICS
# ---------------------------------------

st.divider()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Connected Hospitals", "4")
c2.metric("Training Images", "8,000")
c3.metric("Selected Round", selected_round)

if current_accuracy is not None:
    c4.metric(
        "Recorded Accuracy",
        f"{current_accuracy:.2f}%"
    )
else:
    c4.metric("Recorded Accuracy", "N/A")


# ---------------------------------------
# 7. COMMAND CENTER TABS
# ---------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏥 Hospital Network",
    "⚔️ Attack Lab",
    "🔐 Privacy Lab",
    "📋 Audit Report",
    "📈 Training Optimization"
])


# =======================================
# TAB 1: HOSPITAL NETWORK
# =======================================

with tab1:
    st.header("Hospital Network")

    st.write(
        "Four simulated hospitals train local CNNs "
        "using different distributions of PathMNIST images."
    )

    cols = st.columns(4)

    for i, counts in enumerate(HOSPITAL_COUNTS):
        with cols[i]:
            st.metric(
                f"Hospital {i + 1}",
                f"{sum(counts):,} images"
            )

            if i == 3 and scenario != "Normal FedAvg":
                st.error("Simulated attacker")
            else:
                st.success("Normal client")

    st.subheader("Non-IID Data Distribution")

    distribution_rows = []

    for i, counts in enumerate(HOSPITAL_COUNTS):
        for class_id, count in enumerate(counts):
            distribution_rows.append({
                "Hospital": f"Hospital {i + 1}",
                "Class": f"Class {class_id}",
                "Images": count
            })

    distribution_df = pd.DataFrame(distribution_rows)

    fig = px.bar(
        distribution_df,
        x="Hospital",
        y="Images",
        color="Class",
        title="Tissue Class Distribution Across Hospitals"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Non-IID means hospitals have different "
        "class distributions. For example, "
        "Hospital 1 has no class 7 training images."
    )

    st.subheader("How Federated Learning Works")

    st.markdown("""
    **1.** The server distributes the global CNN.

    **2.** Hospitals train locally on their own images.

    **3.** Hospitals send trained model parameters.

    **4.** The server aggregates the parameters.

    **5.** The updated global model is evaluated.
    """)

    st.caption(
        "The simulation runs sequentially on one machine. "
        "It does not communicate with real hospitals."
    )


# =======================================
# TAB 2: ATTACK LAB
# =======================================

with tab2:
    st.header("Model Poisoning Attack Lab")

    st.write(
        "Use the sidebar to switch between normal "
        "training, a poisoned update from Hospital 4, "
        "and the coordinate-wise median defence."
    )

    if security:
        normal = security["normal"][selected_round - 1]["accuracy"]
        attacked = security["attacked"][selected_round - 1]["accuracy"]
        defended = security["defended"][selected_round - 1]["accuracy"]

        a, b, c = st.columns(3)

        a.metric("Normal FedAvg", f"{normal:.2f}%")
        b.metric(
            "Under Attack",
            f"{attacked:.2f}%",
            delta=f"{attacked - normal:.2f} pp",
            delta_color="normal"
        )
        c.metric(
            "With Median Defence",
            f"{defended:.2f}%",
            delta=f"{defended - attacked:+.2f} pp",
            delta_color="normal"
        )

        st.subheader("Selected Scenario")

        if scenario == "Normal FedAvg":
            st.success(
                "All four hospitals submit legitimate "
                "model updates. FedAvg combines them."
            )

        elif scenario == "Hospital 4 Attack":
            st.error(
                "Hospital 4 sends a reversed, amplified "
                "model update. FedAvg is vulnerable "
                "to this tested attack."
            )

        else:
            st.info(
                "Hospital 4 still sends a poisoned update, "
                "but coordinate-wise median aggregation "
                "reduces its impact."
            )

        rows = []

        labels = {
            "normal": "Normal FedAvg",
            "attacked": "Poisoned FedAvg",
            "defended": "Median Defence"
        }

        for key, history in security.items():
            for item in history:
                rows.append({
                    "Round": item["round"],
                    "Accuracy": item["accuracy"],
                    "Scenario": labels.get(key, key)
                })

        attack_df = pd.DataFrame(rows)

        fig = px.line(
            attack_df,
            x="Round",
            y="Accuracy",
            color="Scenario",
            markers=True,
            title="Recorded Attack and Defence Results"
        )

        fig.update_xaxes(
            tickmode="array",
            tickvals=[1, 2, 3]
        )

        fig.update_yaxes(range=[0, 65])

        fig.add_vline(
            x=selected_round,
            line_dash="dash",
            line_color="gray"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("What did the defence achieve?")

        st.write(
            f"At round {selected_round}, the attack "
            f"reduced accuracy by "
            f"{normal - attacked:.2f} percentage points."
        )

        st.write(
            f"Median aggregation recovered "
            f"{defended - attacked:.2f} percentage points."
        )

        st.caption(
            "The defence does not fully restore "
            "normal accuracy and is not guaranteed "
            "against other poisoning attacks."
        )

    else:
        st.error(
            "security.json not found. Run "
            "python -m src.security first."
        )


# =======================================
# TAB 3: PRIVACY LAB
# =======================================

with tab3:
    st.header("Differential Privacy Lab")

    st.write(
        "DP-SGD clips per-example gradients "
        "and adds Gaussian noise during local training."
    )

    st.caption(
        "The private and non-private runs use "
        "different training configurations, so "
        "their accuracy gap is not a controlled "
        "estimate of the effect of DP alone."
    )

    if comparison and private:
        baseline_accuracy = comparison["federated"]
        dp_accuracy = private[-1]["accuracy"]

        p1, p2 = st.columns(2)

        p1.metric(
            "Earlier Non-Private FedAvg",
            f"{baseline_accuracy:.2f}%"
        )

        p2.metric(
            "DP-FedAvg",
            f"{dp_accuracy:.2f}%"
        )

        st.subheader("Recorded Training Accuracy")

        history_rows = []

        for item in private:
            history_rows.append({
                "Round": item["round"],
                "Accuracy": item["accuracy"]
            })

        history_df = pd.DataFrame(history_rows)

        fig = px.line(
            history_df,
            x="Round",
            y="Accuracy",
            markers=True,
            title="DP-FedAvg Training"
        )

        fig.update_xaxes(
            tickmode="array",
            tickvals=[1, 2, 3]
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Cumulative Privacy Budget")

        epsilon_rows = []

        for item in private:
            epsilons = item.get(
                "cumulative_client_epsilons",
                []
            )

            for i, epsilon in enumerate(epsilons):
                epsilon_rows.append({
                    "Round": item["round"],
                    "Hospital": f"Hospital {i + 1}",
                    "Epsilon": epsilon
                })

        if epsilon_rows:
            epsilon_df = pd.DataFrame(epsilon_rows)

            fig = px.line(
                epsilon_df,
                x="Round",
                y="Epsilon",
                color="Hospital",
                markers=True,
                title="Cumulative Epsilon by Hospital"
            )

            fig.update_xaxes(
                tickmode="array",
                tickvals=[1, 2, 3]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.caption(
                "Recorded RDP accountant estimates, "
                "delta = 1e-5. Secure RNG was disabled. "
                "These results are not a production "
                "privacy certification."
            )
        else:
            st.warning(
                "Cumulative epsilon values are missing. "
                "Check federated_dp.json."
            )

    else:
        st.warning(
            "Missing comparison.json or federated_dp.json."
        )

    st.subheader("Privacy Experiment Diagnostics")

    if isinstance(privacy, list):
        diagnostic_rows = []

        for item in privacy:
            diagnostic_rows.append({
                "Experiment": item.get("experiment", "Unknown"),
                "Accuracy": item.get("accuracy"),
                "Noise": item.get("noise_multiplier"),
                "Epsilon": item.get("epsilon")
            })

        st.dataframe(
            pd.DataFrame(diagnostic_rows),
            hide_index=True
        )

        st.caption(
            "These diagnostics used pooled training data. "
            "They are separate from the federated experiment."
        )


# =======================================
# TAB 4: AUDIT REPORT
# =======================================

with tab4:
    st.header("Experiment Audit Report")

    st.write(
        "Download a summary of the selected "
        "recorded scenario and the project limitations."
    )

    st.subheader("Current Selection")

    st.write(f"**Scenario:** {scenario}")
    st.write(f"**Round:** {selected_round}")

    if current_accuracy is not None:
        st.write(
            f"**Recorded accuracy:** "
            f"{current_accuracy:.2f}%"
        )

    st.write("**Simulated hospitals:** 4")
    st.write("**Training images:** 8,000")
    st.write("**Dataset:** PathMNIST")

    audit = {
        "project": "FedShield AI",
        "generated_at": datetime.now().isoformat(),
        "simulation": True,
        "dataset": "PathMNIST",
        "hospital_count": 4,
        "training_images": 8000,
        "selected_scenario": scenario,
        "selected_round": selected_round,
        "recorded_accuracy": current_accuracy,
        "attack_configuration": {
            "attacker": "Hospital 4",
            "attack_scale": -5.0
        },
        "security_results": security,
        "private_federated_results": private,
        "baseline_comparison": comparison,
        "limitations": [
            "Simulated clients, not real hospitals",
            "Public dataset, not private patient records",
            "Recorded replay, not live training",
            "No secure aggregation implemented",
            "Secure RNG disabled in DP experiment",
            "DP and poisoning experiments are separate",
            "No clinical validation",
            "Median defence is not universally robust"
        ]
    }

    report_text = json.dumps(
        audit,
        indent=2,
        default=str
    )

    st.download_button(
        label="Download Audit Report",
        data=report_text,
        file_name="fedshield_audit_report.json",
        mime="application/json"
    )

    st.subheader("Known Limitations")

    for limitation in audit["limitations"]:
        st.write(f"• {limitation}")


# =======================================
# TAB 5: TRAINING OPTIMIZATION LAB
# =======================================

with tab5:
    st.header("📈 Training Optimization Lab")

    st.write(
        "Compare FedAvg and FedProx across "
        "10 federated communication rounds."
    )

    if fedavg_history and fedprox_history:

        # Some training scripts save a list directly.
        # Others save a dictionary containing the list.
        if isinstance(fedavg_history, dict):
            fedavg_rounds = fedavg_history.get(
                "history",
                fedavg_history.get("rounds", [])
            )
        else:
            fedavg_rounds = fedavg_history

        if isinstance(fedprox_history, dict):
            fedprox_rounds = fedprox_history.get(
                "history",
                fedprox_history.get("rounds", [])
            )
        else:
            fedprox_rounds = fedprox_history

        if not fedavg_rounds or not fedprox_rounds:
            st.error(
                "Could not find round-by-round results. "
                "Check the structure of your JSON files."
            )
        else:
            # Turn saved results into a chart-friendly table.
            rows = []

            for item in fedavg_rounds:
                rows.append({
                    "Round": item["round"],
                    "Accuracy": item["accuracy"],
                    "Algorithm": "FedAvg"
                })

            for item in fedprox_rounds:
                rows.append({
                    "Round": item["round"],
                    "Accuracy": item["accuracy"],
                    "Algorithm": "FedProx"
                })

            df = pd.DataFrame(rows)

            # Select which round to inspect.
            max_round = int(df["Round"].max())

            inspect_round = st.slider(
                "Inspect communication round",
                min_value=1,
                max_value=max_round,
                value=max_round,
                key="optimization_round"
            )

            selected = df[
                df["Round"] == inspect_round
            ]

            fedavg_value = selected[
                selected["Algorithm"] == "FedAvg"
            ]["Accuracy"]

            fedprox_value = selected[
                selected["Algorithm"] == "FedProx"
            ]["Accuracy"]

            if not fedavg_value.empty and not fedprox_value.empty:
                avg = float(fedavg_value.iloc[0])
                prox = float(fedprox_value.iloc[0])

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "FedAvg Accuracy",
                    f"{avg:.2f}%"
                )

                c2.metric(
                    "FedProx Accuracy",
                    f"{prox:.2f}%"
                )

                c3.metric(
                    "FedAvg - FedProx",
                    f"{avg - prox:+.2f} pp"
                )

            # Draw the accuracy comparison chart.
            fig = px.line(
                df,
                x="Round",
                y="Accuracy",
                color="Algorithm",
                markers=True,
                title="FedAvg vs FedProx"
            )

            fig.add_vline(
                x=inspect_round,
                line_dash="dash",
                line_color="gray"
            )

            fig.update_xaxes(
                dtick=1
            )

            fig.update_yaxes(
                range=[0, 100]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.subheader("What is FedProx?")

            st.write(
                "FedProx adds a proximal penalty "
                "to local training. This discourages "
                "hospital models from drifting too far "
                "from the global model."
            )

            st.latex(
                r"L = L_{classification} + "
                r"\frac{\mu}{2}"
                r"\|w_{local}-w_{global}\|^2"
            )

            st.subheader("Experimental Interpretation")

            final_avg = float(
                df[df["Algorithm"] == "FedAvg"]
                .sort_values("Round")
                .iloc[-1]["Accuracy"]
            )

            final_prox = float(
                df[df["Algorithm"] == "FedProx"]
                .sort_values("Round")
                .iloc[-1]["Accuracy"]
            )

            if final_avg > final_prox:
                st.success(
                    f"FedAvg finished {final_avg - final_prox:.2f} "
                    "percentage points above FedProx "
                    "in these recorded runs."
                )
            elif final_prox > final_avg:
                st.success(
                    f"FedProx finished {final_prox - final_avg:.2f} "
                    "percentage points above FedAvg "
                    "in these recorded runs."
                )
            else:
                st.info(
                    "Both algorithms achieved the "
                    "same final accuracy."
                )

            st.caption(
                "A controlled algorithm comparison "
                "requires identical initialization, "
                "data partitions, training settings, "
                "and evaluation procedures."
            )

    else:
        st.warning(
            "Missing history.json or fedprox.json."
        )



st.divider()

st.caption(
    "FedShield AI — Experimental federated "
    "learning, privacy and security simulator."
)

st.sidebar.subheader("Security Experiment Replay")

selected_round = st.sidebar.slider(
    "Security experiment round",
    min_value=1,
    max_value=3,
    value=3
)