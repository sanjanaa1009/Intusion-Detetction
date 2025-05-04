import streamlit as st
import pandas as pd
import numpy as np
import time
import threading
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from models.anomaly_detection import AnomalyDetectionSystem
from models.unknownThreat_detector import UnknownThreatClassifier
from blockchain.core import LogBlockchain
from concurrent.futures import ThreadPoolExecutor
import re
from datetime import datetime
import os
import gdown
import joblib

# =============================================
# SETUP & CONFIGURATION
# =============================================
st.set_page_config(
    page_title="CyberShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dashboard
st.markdown("""
<style>
    :root {
        --primary: #4a6fa5;
        --secondary: #166088;
        --danger: #d64045;
        --success: #4cb944;
    }
    
    .main {
        background-color: #f8fafc;
    }
    
    .header {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white !important;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 4px solid var(--primary);
    }
    
    .attack-card {
        background: white;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid var(--danger);
    }
    
    .unknown-card {
        border-left: 4px solid #ff9f1c;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
    }
    
    .stProgress>div>div>div {
        background: linear-gradient(90deg, var(--primary), var(--secondary));
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# HELPER FUNCTIONS
# =============================================
def process_log_file(log_text: str) -> pd.DataFrame:
    """Convert raw log text into a structured DataFrame with comprehensive error handling"""
    log_entries = []
    
    patterns = {
        'apache_common': r'(?P<host>\S+) (?P<identity>\S+) (?P<user>\S+) \[(?P<timestamp>.+?)\] "(?P<request>.+?)" (?P<status>\d+) (?P<size>\S+)',
        'syslog': r'(?P<timestamp>\w{3} \d{2} \d{2}:\d{2}:\d{2}) (?P<host>\S+) (?P<process>\S+)(?:\[(?P<pid>\d+)\])?: (?P<message>.+)',
        'firewall': r'(?P<timestamp>.+?) (?P<action>\S+) (?P<protocol>\S+) (?P<src_ip>\S+):(?P<src_port>\d+) -> (?P<dst_ip>\S+):(?P<dst_port>\d+)',
        'custom': r'(?P<timestamp>.+?) (?P<source_ip>\S+) (?P<destination_ip>\S+) (?P<port>\d+) (?P<protocol>\S+) (?P<message>.+)'
    }

    for line in log_text.split('\n'):
        if not line.strip():
            continue
            
        entry = {}
        try:
            for pattern_name, pattern in patterns.items():
                match = re.search(pattern, line)
                if match:
                    entry.update(match.groupdict())
                    entry['log_type'] = pattern_name
                    break
            else:
                entry = {'raw': line, 'log_type': 'unparsed'}
                
            if 'timestamp' in entry:
                entry['timestamp'] = pd.to_datetime(entry['timestamp'], errors='coerce')
                    
            log_entries.append(entry)
            
        except Exception as e:
            continue
    
    df = pd.DataFrame(log_entries)
    
    column_mapping = {
        'host': 'source_ip',
        'src_ip': 'source_ip',
        'dst_ip': 'destination_ip',
        'request': 'http_request',
        'protocol': 'network_protocol'
    }
    df = df.rename(columns=column_mapping)
    
    for col in ['source_ip', 'destination_ip', 'timestamp', 'message']:
        if col not in df.columns:
            df[col] = None
            
    return df

def process_uploaded_file(uploaded_file):
    """Handle all supported file types with robust error handling"""
    try:
        if uploaded_file is None:
            st.error("🚨 No file uploaded")
            return None

        if uploaded_file.name.endswith(('.log', '.txt')):
            log_text = uploaded_file.getvalue().decode("utf-8")
            df = process_log_file(log_text)
        elif uploaded_file.name.endswith('.parquet'):
            df = pd.read_parquet(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        if df is None or df.empty:
            st.error("❌ Failed to parse file - no valid data")
            return None
            
        if 'message' not in df.columns:
            df['message'] = df.apply(lambda x: str(x.to_dict()), axis=1)
                
        return df
        
    except Exception as e:
        st.error(f"💥 File processing error: {str(e)}")
        return None

# =============================================
# MODEL LOADING
# =============================================
@st.cache_resource
def load_models():
    """Load both detection models with progress tracking"""
    progress_bar = st.progress(0, text="Loading AI engines...")
    
    def load_lgbm():
        
        file_id = "1Bjri7bIBhy7CKwUOcJWpDm7F3qmV2tkp/"
        url = f"https://drive.google.com/uc?id={file_id}"
        output = "anomaly_detection_models_multiclass.joblib"

# Only download if not already present
        if not os.path.exists(output):
             gdown.download(url, output, quiet=False)

        model = joblib.load(output)

        return model 
    
    def load_isoforest():
        return UnknownThreatClassifier.load('unknown_threat_model.joblib')
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        lgbm_future = executor.submit(load_lgbm)
        iso_future = executor.submit(load_isoforest)
        
        while not (lgbm_future.done() and iso_future.done()):
            completed = lgbm_future.done() + iso_future.done()
            progress_bar.progress(
                int(50 * completed),
                text=f"Loading models ({completed}/2)"
            )
            time.sleep(0.1)
    
    progress_bar.progress(100, text="Systems online!")
    time.sleep(0.3)
    progress_bar.empty()
    
    return lgbm_future.result(), iso_future.result()

# Initialize models
detector, threat_detector = load_models()

# =============================================
# DETECTION FUNCTIONS
# =============================================
def detect_threats(df):
    """Run parallel threat detection with error handling"""
    results = {'lgbm': None, 'iso': None}
    
    def run_lgbm():
        try:
            results['lgbm'] = {
                'binary': detector.predict(df, 'binary'),
                'multiclass': detector.predict(df, 'multiclass')
            }
        except Exception as e:
            st.error(f"LGBM detection failed: {str(e)}")
    
    def run_iso():
        try:
            if 'message' not in df.columns:
                df['message'] = df.apply(lambda x: str(x.to_dict()), axis=1)
            results['iso'] = threat_detector.detect_batch(df)
        except Exception as e:
            st.error(f"Isolation Forest detection failed: {str(e)}")
    
    with ThreadPoolExecutor() as executor:
        executor.submit(run_lgbm)
        executor.submit(run_iso)
    
    return results

# =============================================
# BLOCKCHAIN INTEGRATION
# =============================================
def log_to_blockchain(data, results):
    """Secure logging with comprehensive error handling"""
    with st.spinner("⛓️ Securing results on blockchain..."):
        try:
            blockchain = LogBlockchain()
            
            detection_record = {
                'timestamp': str(pd.Timestamp.now()),
                'detection_summary': {
                    'total_records': len(data),
                    'known_threats': sum(results['lgbm']['binary'] == 1),
                    'unknown_threats': sum(results['iso']['category'] != 'Normal'),
                    'critical_threats': sum(results['iso']['confidence'] > 2)
                },
                'raw_data_hash': hash(str(data.values.tolist())),
                'threat_details': {
                    'top_known_threat': pd.Series(results['lgbm']['multiclass']).mode()[0],
                    'top_unknown_threat': results['iso']['category'].value_counts().idxmax() 
                                           if not results['iso']['category'].value_counts().empty 
                                           else None
                }
            }
            
            new_block = blockchain.add_block(detection_record)
            
            return {
                'block_index': new_block.index,
                'block_hash': new_block.hash,
                'timestamp': new_block.timestamp
            }
            
        except Exception as e:
            st.error(f"🔗 Blockchain logging failed: {str(e)}")
            return None

# =============================================
# VISUALIZATIONS
# =============================================
def plot_threat_distribution(lgbm_results, iso_results):
    """Interactive threat distribution visualization"""
    fig = plt.figure(figsize=(12, 6))
    
    # Known threats
    known_counts = pd.Series(lgbm_results).value_counts()
    ax1 = fig.add_subplot(121)
    sns.barplot(y=known_counts.index, x=known_counts.values, palette="Blues_d", ax=ax1)
    ax1.set_title("Known Threat Distribution")
    ax1.set_xlabel("Count")
    
    # Unknown threats
    unknown_counts = iso_results['category'].value_counts()
    ax2 = fig.add_subplot(122)
    if not unknown_counts.empty:
        sns.barplot(y=unknown_counts.index, x=unknown_counts.values, palette="Oranges_d", ax=ax2)
    ax2.set_title("Unknown Anomalies")
    ax2.set_xlabel("Count")
    
    plt.tight_layout()
    st.pyplot(fig)

def create_timeline_plot(df, results):
    """Temporal threat visualization"""
    if 'timestamp' in df.columns:
        timeline_df = df.copy()
        timeline_df['Threat'] = np.where(
            results['iso']['category'] != 'Normal',
            'Unknown: ' + results['iso']['category'],
            results['lgbm']['multiclass']
        )
        
        fig, ax = plt.subplots(figsize=(14, 6))
        for threat_type in timeline_df['Threat'].unique():
            subset = timeline_df[timeline_df['Threat'] == threat_type]
            ax.scatter(
                subset['timestamp'],
                [threat_type] * len(subset),
                label=threat_type,
                s=100
            )
        
        ax.set_title("Threat Timeline Analysis")
        ax.set_xlabel("Timestamp")
        ax.set_yticks([])
        ax.legend(bbox_to_anchor=(1.05, 1))
        plt.xticks(rotation=45)
        st.pyplot(fig)

# =============================================
# MAIN APP
# =============================================
def main():
    st.markdown("""
    <div class="header">
        <h1 style="color:white; margin:0;">CyberShield AI</h1>
        <p style="color:white; margin:0;">Advanced Network Threat Detection System</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📤 Upload Network Data", expanded=True):
        uploaded_file = st.file_uploader(
            "Drag & drop network logs or datasets",
            type=["log", "csv", "parquet", "json"],
            help="Supports: CSV, Parquet, JSON, and raw log files"
        )
    
    if uploaded_file:
        with st.spinner("🔍 Analyzing data..."):
            df = process_uploaded_file(uploaded_file)
            
            if df is not None:
                try:
                    results = detect_threats(df)
                    
                    # Add results to DataFrame
                    df['Threat Status'] = np.where(
                        (results['lgbm']['binary'] == 1) | (results['iso']['category'] != 'Normal'),
                        'Threat Detected', 
                        'Normal'
                    )
                    df['Threat Type'] = np.where(
                        results['iso']['category'] != 'Normal',
                        'Unknown: ' + results['iso']['category'],
                        results['lgbm']['multiclass']
                    )
                    df['Confidence'] = results['iso']['confidence']
                    
                    st.success("✅ Analysis complete!")
                    show_dashboard(df, results)
                    
                except Exception as e:
                    st.error(f"🚨 Detection failed: {str(e)}")

def show_dashboard(df, results):
    """Main dashboard display with all visualizations"""
    # Metrics
    cols = st.columns(4)
    metrics = [
        ("Total Records", len(df), "var(--primary)"),
        ("Known Threats", sum(results['lgbm']['binary'] == 1), "var(--danger)"),
        ("Unknown Anomalies", sum(results['iso']['category'] != 'Normal'), "#ff9f1c"),
        ("Critical Threats", sum(results['iso']['confidence'] > 2), "#d64045")
    ]
    
    for col, (title, value, color) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{title}</h3>
                <h1 style='color: {color};'>{value:,}</h1>
            </div>
            """, unsafe_allow_html=True)
    
    # Visualizations
    st.markdown("---")
    st.header("Threat Intelligence Overview")
    plot_threat_distribution(results['lgbm']['multiclass'], results['iso'])
    
    if 'timestamp' in df.columns:
        st.header("Temporal Threat Analysis")
        create_timeline_plot(df, results)
    
    # Detailed Results
    st.header("Threat Breakdown")
    tab1, tab2, tab3 = st.tabs(["All Threats", "Known Attacks", "Unknown Anomalies"])
    
    with tab1:
        st.dataframe(
            df[['timestamp', 'Threat Type', 'Confidence']].sort_values('Confidence', ascending=False),
            height=400,
            use_container_width=True
        )
    
    with tab2:
        known_df = df[~df['Threat Type'].str.startswith('Unknown')]
        if not known_df.empty:
            st.dataframe(known_df, height=400, use_container_width=True)
        else:
            st.info("No known threats detected")
    
    with tab3:
        unknown_df = df[df['Threat Type'].str.startswith('Unknown')]
        if not unknown_df.empty:
            for _, row in unknown_df.iterrows():
                st.markdown(f"""
                <div class='attack-card unknown-card'>
                    <h4>{row['Threat Type']}</h4>
                    <p><strong>Confidence:</strong> {row['Confidence']:.2f}</p>
                    <p><strong>First Seen:</strong> {row.get('timestamp', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No unknown anomalies detected")
    
    # Blockchain Integration
    st.markdown("---")
    st.header("Blockchain Security")
    if st.button("⛓️ Secure Results on Blockchain"):
        tx_hash = log_to_blockchain(df, results)
        if tx_hash:
            st.success(f"Results secured! Transaction: {tx_hash['block_hash']}")
            st.json({
                "Block Index": tx_hash['block_index'],
                "Timestamp": tx_hash['timestamp']
            })

if __name__ == "__main__":
    main()