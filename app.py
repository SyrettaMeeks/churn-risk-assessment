import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Seller Pulse | Square",
    page_icon="🟦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #F7F8FA; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebar"] { background-color: #1A1A2E; }
    [data-testid="stSidebar"] * { color: #E2E8F0 !important; }
    h1 { color: #0F172A !important; font-weight: 700 !important; }
    h2, h3 { color: #1E293B !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_sellers(n=120, seed=42):
    np.random.seed(seed)
    rng = np.random
    business_types = ["Food & Beverage", "Retail", "Services", "Health & Beauty", "Home & Repair"]
    states = ["CA", "TX", "NY", "FL", "IL", "WA", "CO", "GA", "AZ", "NC"]
    interventions = {
        "declining_txn":   "📞 Schedule business review call",
        "low_adoption":    "🎓 Offer onboarding session for unused features",
        "support_spike":   "🛠️ Escalate to senior support specialist",
        "cash_flow":       "💰 Send Square Capital pre-approval offer",
        "hardware_issue":  "📦 Offer free hardware upgrade",
        "high_value_risk": "⭐ Assign dedicated account manager",
    }
    def make_risk_pattern(base_score):
        if base_score >= 75:
            return rng.choice(list(interventions.keys()), p=[.25,.15,.25,.15,.10,.10])
        elif base_score >= 50:
            return rng.choice(list(interventions.keys()), p=[.20,.20,.20,.15,.15,.10])
        else:
            return rng.choice(list(interventions.keys()), p=[.15,.25,.10,.20,.20,.10])
    data = []
    for i in range(n):
        score = int(np.clip(rng.beta(2,3)*100 + rng.normal(0,8), 5, 99))
        pattern = make_risk_pattern(score)
        monthly_rev = rng.choice([500,1000,2000,5000,10000,25000,50000], p=[.05,.10,.20,.25,.20,.12,.08])
        join_months_ago = rng.randint(6, 72)
        join_date = datetime.now() - timedelta(days=join_months_ago*30)
        data.append({
            "seller_id": f"SQ-{10000+i}",
            "business_name": f"{rng.choice(['Sunny','Metro','Peak','Harbor','Oak','Rise','Core','Bloom','True','Bright'])} {rng.choice(['Eats','Goods','Studio','Salon','Builds','Roast','Market','Wellness','Co.','Shop'])}",
            "type": rng.choice(business_types),
            "state": rng.choice(states),
            "monthly_revenue": monthly_rev,
            "months_on_square": join_months_ago,
            "join_date": join_date.strftime("%Y-%m-%d"),
            "risk_score": score,
            "risk_pattern": pattern,
            "intervention": interventions[pattern],
            "txn_trend_30d": round(rng.uniform(-35,5) if score > 50 else rng.uniform(-10,15), 1),
            "support_tickets_90d": int(rng.poisson(score/20)),
            "days_since_login": int(rng.exponential(score/8+1)),
            "features_adopted": int(rng.uniform(1,8)*(1-score/200)),
            "signal_txn": min(100, int(max(0, score+rng.normal(0,15)))),
            "signal_support": min(100, int(max(0, score+rng.normal(0,20)))),
            "signal_login": min(100, int(max(0, score+rng.normal(0,18)))),
            "signal_adoption": min(100, int(max(0, score+rng.normal(0,15)))),
            "contacted": rng.choice([True, False], p=[.25,.75]),
            "prev_score": int(np.clip(score+rng.normal(5,10), 5, 99)),
        })
    df = pd.DataFrame(data)
    def risk_label(s):
        if s >= 75: return "Critical"
        elif s >= 55: return "High"
        elif s >= 35: return "Medium"
        else: return "Low"
    df["risk_level"] = df["risk_score"].apply(risk_label)
    df["monthly_revenue_fmt"] = df["monthly_revenue"].apply(lambda x: f"${x:,.0f}")
    df["score_delta"] = df["prev_score"] - df["risk_score"]
    return df

df = generate_sellers()

with st.sidebar:
    st.markdown("### 🟦 Seller Pulse")
    st.markdown("*Square Merchant Success Tool*")
    st.markdown("---")
    st.markdown("**Filter Sellers**")
    risk_filter = st.multiselect("Risk Level", ["Critical","High","Medium","Low"], default=["Critical","High"])
    type_filter = st.multiselect("Business Type", df["type"].unique().tolist(), default=df["type"].unique().tolist())
    rev_min, rev_max = st.select_slider("Monthly Revenue", options=[500,1000,2000,5000,10000,25000,50000], value=(500,50000))
    st.markdown("---")
    st.markdown("**View**")
    page = st.radio("", ["📊 Dashboard", "👥 Seller Queue", "🔍 Seller Detail"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("*Prototype · Feb 2026*")
    st.markdown("*Built by Syretta Meeks*")

filtered = df[df["risk_level"].isin(risk_filter) & df["type"].isin(type_filter) & df["monthly_revenue"].between(rev_min, rev_max)].copy()

if page == "📊 Dashboard":
    st.markdown("# Seller Pulse")
    st.markdown("*Proactive churn risk monitoring for Square's Merchant Success team*")
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    critical_count = len(df[df["risk_level"]=="Critical"])
    high_count = len(df[df["risk_level"]=="High"])
    at_risk_rev = df[df["risk_level"].isin(["Critical","High"])]["monthly_revenue"].sum()
    contacted_pct = df[df["risk_level"].isin(["Critical","High"])]["contacted"].mean()*100
    with col1:
        st.metric("🔴 Critical Risk Sellers", critical_count, delta="+3 vs last week", delta_color="inverse")
    with col2:
        st.metric("🟡 High Risk Sellers", high_count, delta="-2 vs last week")
    with col3:
        st.metric("💸 At-Risk Monthly Revenue", f"${at_risk_rev:,.0f}", delta="Combined Critical + High")
    with col4:
        st.metric("📬 Outreach Coverage", f"{contacted_pct:.0f}%", delta="of at-risk sellers contacted", delta_color="off")
    st.markdown("---")
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### Risk Score Distribution")
        fig = px.histogram(df, x="risk_score", nbins=20, color_discrete_sequence=["#3B82F6"], labels={"risk_score":"Churn Risk Score"})
        fig.add_vrect(x0=75, x1=100, fillcolor="#FEE2E2", opacity=0.3, annotation_text="Critical Zone", annotation_position="top left")
        fig.add_vrect(x0=55, x1=75, fillcolor="#FEF3C7", opacity=0.3)
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=10,r=10,t=10,b=10), height=280)
        st.plotly_chart(fig, use_container_width=True)
    with col_right:
        st.markdown("#### At-Risk Revenue by Business Type")
        type_risk = df[df["risk_level"].isin(["Critical","High"])].groupby("type")["monthly_revenue"].sum().reset_index().sort_values("monthly_revenue", ascending=True)
        fig2 = px.bar(type_risk, x="monthly_revenue", y="type", orientation="h", color_discrete_sequence=["#6366F1"], labels={"monthly_revenue":"Monthly Revenue at Risk ($)","type":""})
        fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=10,r=10,t=10,b=10), height=280)
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown("#### 🚨 Top 5 Critical Sellers — Act Now")
    top5 = df[df["risk_level"]=="Critical"].sort_values("monthly_revenue", ascending=False).head(5)[["business_name","type","state","monthly_revenue_fmt","risk_score","txn_trend_30d","intervention","contacted"]].copy()
    top5["contacted"] = top5["contacted"].map({True:"✅ Yes", False:"❌ No"})
    top5.columns = ["Business","Type","State","Monthly Rev","Risk Score","Txn Trend 30d (%)","Recommended Action","Contacted"]
    st.dataframe(top5, use_container_width=True, hide_index=True)

elif page == "👥 Seller Queue":
    st.markdown("# Seller Queue")
    st.markdown(f"*Showing {len(filtered)} sellers · sorted by churn risk*")
    st.markdown("---")
    if filtered.empty:
        st.info("No sellers match your current filters.")
    else:
        sort_col = st.selectbox("Sort by", ["risk_score","monthly_revenue","txn_trend_30d"], format_func=lambda x: {"risk_score":"Risk Score (highest first)","monthly_revenue":"Monthly Revenue (highest first)","txn_trend_30d":"Transaction Trend (worst first)"}[x])
        asc = sort_col == "txn_trend_30d"
        display = filtered.sort_values(sort_col, ascending=asc).head(50)
        def color_risk(val):
            colors = {"Critical":"#FEE2E2","High":"#FEF3C7","Medium":"#FFF7ED","Low":"#DCFCE7"}
            return f"background-color: {colors.get(val,'white')}"
        show = display[["business_name","type","state","monthly_revenue_fmt","risk_level","risk_score","txn_trend_30d","support_tickets_90d","days_since_login","intervention"]].copy()
        show.columns = ["Business","Type","State","Monthly Rev","Risk Level","Score","Txn Trend 30d (%)","Support Tickets 90d","Days Since Login","Recommended Action"]
        st.dataframe(show.style.applymap(color_risk, subset=["Risk Level"]), use_container_width=True, hide_index=True, height=520)
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Sellers shown", len(display))
        c2.metric("Total monthly revenue", f"${display['monthly_revenue'].sum():,.0f}")
        c3.metric("Avg risk score", f"{display['risk_score'].mean():.0f}")

elif page == "🔍 Seller Detail":
    st.markdown("# Seller Detail View")
    st.markdown("---")
    seller_options = df.sort_values("risk_score", ascending=False)
    selected_name = st.selectbox("Select a seller", seller_options["business_name"].tolist(), format_func=lambda x: f"{x}  ·  Risk Score: {df[df['business_name']==x]['risk_score'].values[0]}")
    s = df[df["business_name"]==selected_name].iloc[0]
    risk_colors = {"Critical":"🔴","High":"🟡","Medium":"🟠","Low":"🟢"}
    st.markdown(f"## {risk_colors[s['risk_level']]} {s['business_name']}")
    st.markdown(f"`{s['seller_id']}` · {s['type']} · {s['state']} · On Square since {s['join_date']}")
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Churn Risk Score", s["risk_score"], delta=f"{s['score_delta']:+.0f} vs last week", delta_color="inverse")
    c2.metric("Monthly Revenue", s["monthly_revenue_fmt"])
    c3.metric("Txn Trend (30d)", f"{s['txn_trend_30d']:+.1f}%", delta_color="normal" if s["txn_trend_30d"] >= 0 else "inverse")
    c4.metric("Days Since Login", s["days_since_login"])
    st.markdown("---")
    col_signals, col_action = st.columns([1.2, 1])
    with col_signals:
        st.markdown("#### Signal Breakdown")
        st.caption("What's driving this seller's risk score")
        signals = {"Transaction Frequency": s["signal_txn"], "Support Contact Rate": s["signal_support"], "Login Recency": s["signal_login"], "Feature Adoption": s["signal_adoption"]}
        for label, val in signals.items():
            col_a, col_b = st.columns([2, 3])
            with col_a:
                st.write(label)
            with col_b:
                st.progress(int(val)/100)
        st.caption("Higher bar = stronger churn signal for that dimension")
    with col_action:
        st.markdown("#### Recommended Intervention")
        st.markdown(f"""<div style="background:white;border-radius:12px;padding:20px;border:2px solid #6366F1;">
    <div style="font-size:22px;margin-bottom:8px">{s['intervention']}</div>
    <div style="color:#64748B;font-size:14px;margin-bottom:16px">Based on pattern: <strong>{s['risk_pattern'].replace('_',' ').title()}</strong></div>
    <div style="font-size:13px;color:#374151"><strong>Why this action?</strong><br>This seller's dominant signal is <em>{s['risk_pattern'].replace('_',' ')}</em>. Historical data shows this intervention has a 28% success rate in reversing churn trajectory for similar seller profiles.</div>
</div>""", unsafe_allow_html=True)
        st.markdown("&nbsp;")
        st.markdown(f"**Outreach Status:** {'✅ Contacted' if s['contacted'] else '❌ Not yet contacted'}")
        if not s["contacted"]:
            if st.button("✉️ Mark as Contacted", type="primary"):
                st.success("Marked as contacted! Score will refresh in next weekly update.")
    st.markdown("---")
    st.markdown("#### Transaction Volume — Last 12 Weeks")
    np.random.seed(int(s["risk_score"])+7)
    weeks = [f"W{i}" for i in range(1,13)]
    base = s["monthly_revenue"]/4
    trend_factor = s["txn_trend_30d"]/100
    volumes = [max(0, base*(1+trend_factor*(i/12)+np.random.normal(0,0.08))) for i in range(12)]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=weeks, y=volumes, mode="lines+markers", line=dict(color="#6366F1",width=2.5), marker=dict(size=7), fill="tozeroy", fillcolor="rgba(99,102,241,0.08)"))
    fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white", xaxis_title="Week", yaxis_title="Transaction Volume ($)", margin=dict(l=10,r=10,t=10,b=10), height=220)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Features Adopted", f"{s['features_adopted']} / 8", help="Out of 8 core Square features")
    c2.metric("Support Tickets (90d)", s["support_tickets_90d"])
    c3.metric("Months on Square", s["months_on_square"])
