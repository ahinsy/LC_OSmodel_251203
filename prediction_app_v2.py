"""
肺癌手術予後予測モデル - 予測UIアプリケーション v4.0
Lung Cancer Surgery Prognosis Prediction UI

機能:
- 術前予測: OS, RFS, 合併症（個別モデル設定可能）
- 術後予測: OS, RFS（個別モデル設定可能）
- 5年生存率・無再発生存率 + 期待値表示
- 一括予測表示

必要なライブラリ:
pip install streamlit pandas numpy scikit-learn scikit-survival openpyxl

使用方法:
streamlit run prediction_app_v2.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

# ページ設定
st.set_page_config(
    page_title="肺癌手術予後予測システム",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 定数定義
# =============================================================================

VALUE_RANGES = {
    '年齢': (30, 85),
    '喫煙本数': (0, 100),
    '喫煙年間': (0, 70),
    '肺野全体腫瘍径': (0.0, 7.0),
    '充実性腫瘍径': (0.0, 7.0),
    '原発SUV': (0.0, 30.0),
    'CEA': (0.0, 80.0),
    '手術時間(分)': (1, 500),
    '出血量(ml)': (0, 3000),
}

CHOICES = {
    '性別': ['男', '女'],
    '喫煙': ['喫煙している', '喫煙していた', '吸った事なし'],
    '病側': ['右', '左'],
    '原発巣部位': ['末梢', '中枢'],
    '原発肺葉': ['右上', '右中', '右下', '左上', '左下'],
    '8th c-T': ['T1a', 'T1b', 'T1c', 'T2a'],
    '8th c-病期': ['IA1', 'IA2', 'IA3', 'IB'],
    'PET': ['あり', 'なし'],
    '術前診断': ['腺癌', '扁平上皮癌', '未確診', 'その他'],
    'あり_なし': ['なし', 'あり'],
    'アプローチ': ['cVATS', '開胸'],
    'LN郭清': ['ND2a-1', 'ND2a-2', 'ND2b'],
    '病理組織型': ['腺癌', '扁平上皮癌', 'その他'],
    '腺癌亜型': ['Lepidic', 'Acinar', 'Papillary', 'Solid', 'Micropapillary', 'Mucinous', 'Others'],
    '8th p-T': ['T1a', 'T1b', 'T1c', 'T2a', 'T2b', 'T3', 'T4'],
    '8th p-N': ['N0', 'N1', 'N2'],
    'Ly': ['Ly0', 'Ly1'],
    'V': ['V0', 'V1', 'V2'],
    'pl': ['pl0', 'pl1', 'pl2', 'pl3'],
    'STAS': ['なし', 'あり', '不明'],
    'EGFR変異': ['未検', '変異無', 'exon19 Del', 'exon21 L858R', 'その他変異'],
    'ALK変異': ['未検', '陰性', '陽性'],
    '術後補助治療': ['なし', 'あり'],
}

COMORBIDITIES = [
    '他悪性疾患既往', '肺気腫', '虚血心', '心不全', '末梢血管障害',
    '腎障害', '脳梗塞・出血', '片麻痺', '認知症', '肝障害',
    '肝硬変', '消化器潰瘍', '糖尿病', '膠原病'
]

# モデルパス設定
SCRIPT_DIR = Path(__file__).parent if '__file__' in dir() else Path('.')
MODEL_DIR = SCRIPT_DIR / 'model'

DEFAULT_MODELS = {
    'preop': {
        'os': MODEL_DIR / 'preop_os_best.pkl',
        'rfs': MODEL_DIR / 'preop_rfs_best.pkl',
        'complication': MODEL_DIR / 'preop_complication_best.pkl',
    },
    'postop': {
        'os': MODEL_DIR / 'postop_os_best.pkl',
        'rfs': MODEL_DIR / 'postop_rfs_best.pkl',
    }
}

# =============================================================================
# ユーティリティ関数
# =============================================================================

def check_value_range(value: float, key: str) -> Tuple[bool, str]:
    if key not in VALUE_RANGES:
        return True, ""
    min_val, max_val = VALUE_RANGES[key]
    if value is None or pd.isna(value):
        return True, ""
    if value < min_val or value > max_val:
        return False, f"**{key}**: 入力値 {value} が推奨範囲 ({min_val}〜{max_val}) 外です"
    return True, ""


def convert_to_model_features(input_data: Dict, mode: str) -> pd.DataFrame:
    """入力データをモデルの特徴量形式に変換"""
    features = {}
    
    features['年齢'] = input_data.get('年齢', 70)
    features['喫煙指数'] = input_data.get('喫煙指数', 0)
    features['heavy_smoker'] = 1 if features['喫煙指数'] >= 600 else 0
    
    ct = input_data.get('8th c-T', 'T1b')
    features['8th c-T_num'] = {'T1a': 0, 'T1b': 1, 'T1c': 2, 'T2a': 3}.get(ct, 1)
    
    features['肺野全体腫瘍径'] = input_data.get('肺野全体腫瘍径', 2.5)
    features['C/T比'] = input_data.get('C/T比', 1.0)
    
    ct_ratio = input_data.get('C/T比', 1.0)
    features['pure_GGO'] = 1 if ct_ratio <= 0.5 else 0
    features['solid_tumor'] = 1 if ct_ratio >= 1.0 else 0
    
    features['PET'] = 1 if input_data.get('PET') == 'あり' else 0
    features['原発SUV'] = input_data.get('原発SUV', 0.0)
    features['CEA'] = input_data.get('CEA', 3.0)
    
    cci_weights = {
        '他悪性疾患既往': 2, '肺気腫': 1, '虚血心': 1, '心不全': 1,
        '末梢血管障害': 1, '腎障害': 2, '脳梗塞・出血': 1, '片麻痺': 2,
        '認知症': 1, '肝障害': 1, '肝硬変': 3, '消化器潰瘍': 1,
        '糖尿病': 1, '膠原病': 1
    }
    
    cci = 0
    for comorbidity in COMORBIDITIES:
        val = 1 if input_data.get(comorbidity) == 'あり' else 0
        if comorbidity in ['肺気腫', '糖尿病', '虚血心']:
            features[comorbidity] = val
        cci += val * cci_weights.get(comorbidity, 1)
    features['CCI'] = cci
    
    # 術後モードの追加特徴量
    if mode == '術後予測':
        features['手術時間'] = input_data.get('手術時間(分)', 180)
        features['出血量'] = input_data.get('出血量(ml)', 50)
        features['アプローチ_0'] = 1 if input_data.get('アプローチ', 'cVATS') == 'cVATS' else 0
        
        pt = input_data.get('8th p-T', 'T1b')
        features['8th p-T_num'] = {'T1a': 0, 'T1b': 1, 'T1c': 2, 'T2a': 3, 'T2b': 4, 'T3': 5, 'T4': 6}.get(pt, 1)
        
        pn = input_data.get('8th p-N', 'N0')
        features['8th p-N_num'] = {'N0': 0, 'N1': 1, 'N2': 2}.get(pn, 0)
        
        features['Ly_num'] = 0 if input_data.get('Ly', 'Ly0') == 'Ly0' else 1
        features['V_num'] = {'V0': 0, 'V1': 1, 'V2': 2}.get(input_data.get('V', 'V0'), 0)
        features['pl_num'] = {'pl0': 0, 'pl1': 1, 'pl2': 2, 'pl3': 3}.get(input_data.get('pl', 'pl0'), 0)
        
        pathology = input_data.get('病理組織型', '腺癌')
        features['病理組織型_grouped_0'] = 1 if pathology == '腺癌' else 0
        features['病理組織型_grouped_1'] = 1 if pathology == '扁平上皮癌' else 0
    
    return pd.DataFrame([features])


def load_model(model_path: Path) -> Tuple[Any, Dict]:
    """モデルを読み込む"""
    try:
        if not model_path.exists():
            return None, {'error': f'ファイルが見つかりません'}
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        if isinstance(model_data, dict):
            return model_data, model_data
        return model_data, {}
    except Exception as e:
        return None, {'error': str(e)}


def predict_survival(model_data: Dict, X: pd.DataFrame) -> Dict:
    """生存解析予測を実行"""
    results = {}
    
    try:
        model = model_data.get('model')
        scaler = model_data.get('scaler')
        feature_names = model_data.get('feature_names', [])
        median_values = model_data.get('median_values', {})
        
        # 特徴量を整列
        X_aligned = pd.DataFrame(index=X.index)
        for feat in feature_names:
            if feat in X.columns:
                X_aligned[feat] = X[feat].values
            else:
                X_aligned[feat] = median_values.get(feat, 0)
        
        for col in X_aligned.columns:
            X_aligned[col] = X_aligned[col].fillna(median_values.get(col, 0))
        
        X_scaled = scaler.transform(X_aligned)
        
        # リスクスコア予測
        risk_score = model.predict(X_scaled)[0]
        results['risk_score'] = float(risk_score)
        
        # 生存関数から生存率と期待値を計算
        try:
            surv_func = model.predict_survival_function(X_scaled)
            times = surv_func[0].x
            probs = surv_func[0].y
            
            # 1年、3年、5年生存率
            for years, days in [(1, 365), (3, 1095), (5, 1825)]:
                idx = np.searchsorted(times, days)
                if idx >= len(times):
                    idx = len(times) - 1
                results[f'survival_{years}y'] = float(probs[idx])
            
            # 期待値（中央生存時間）を計算
            # S(t) = 0.5となる時点を探す
            median_idx = np.searchsorted(-probs, -0.5)  # probsは降順
            if median_idx < len(times):
                median_survival = times[median_idx]
            else:
                # 50%に達しない場合は最終観察時点
                median_survival = times[-1]
            results['median_survival_days'] = float(median_survival)
            results['median_survival_years'] = float(median_survival / 365)
            
            # 平均生存時間（曲線下面積）
            mean_survival = np.trapz(probs, times)
            results['mean_survival_days'] = float(mean_survival)
            results['mean_survival_years'] = float(mean_survival / 365)
            
        except Exception as e:
            # 生存関数が利用できない場合はリスクスコアから推定
            baseline = 0.90
            results['survival_5y'] = max(0.05, min(0.99, baseline * np.exp(-risk_score * 0.15)))
            results['survival_3y'] = max(0.05, min(0.99, baseline * np.exp(-risk_score * 0.10)))
            results['survival_1y'] = max(0.05, min(0.99, baseline * np.exp(-risk_score * 0.05)))
            results['median_survival_years'] = None
            results['mean_survival_years'] = None
        
        # リスク分類（モデルの閾値を使用）
        surv_5y = results.get('survival_5y', 0.9)
        thresholds = model_data.get('thresholds', {'low': 0.90, 'high': 0.75})
        low_thresh = thresholds.get('low', 0.90)
        high_thresh = thresholds.get('high', 0.75)
        
        if surv_5y >= low_thresh:
            results['risk_category'] = '低リスク'
        elif surv_5y >= high_thresh:
            results['risk_category'] = '中リスク'
        else:
            results['risk_category'] = '高リスク'
        
        results['thresholds'] = thresholds
        
        # リスク因子
        coefficients = model_data.get('coefficients', {})
        if coefficients:
            risk_factors = []
            for feat, coef in coefficients.items():
                if feat in X_aligned.columns and abs(coef) > 0.05:
                    val = X_aligned[feat].values[0]
                    risk_factors.append({
                        'feature': feat,
                        'coefficient': coef,
                        'value': val,
                        'direction': '↑リスク上昇' if coef > 0 else '↓リスク低下'
                    })
            risk_factors.sort(key=lambda x: abs(x['coefficient']), reverse=True)
            results['risk_factors'] = risk_factors[:5]
            
    except Exception as e:
        results['error'] = str(e)
    
    return results


def predict_classification(model_data: Dict, X: pd.DataFrame) -> Dict:
    """分類予測を実行"""
    results = {}
    
    try:
        model = model_data.get('model')
        scaler = model_data.get('scaler')
        feature_names = model_data.get('feature_names', [])
        median_values = model_data.get('median_values', {})
        
        X_aligned = pd.DataFrame(index=X.index)
        for feat in feature_names:
            if feat in X.columns:
                X_aligned[feat] = X[feat].values
            else:
                X_aligned[feat] = median_values.get(feat, 0)
        
        for col in X_aligned.columns:
            X_aligned[col] = X_aligned[col].fillna(median_values.get(col, 0))
        
        X_scaled = scaler.transform(X_aligned)
        
        prob = model.predict_proba(X_scaled)[:, 1][0]
        results['probability'] = float(prob)
        
        # リスク分類（モデルの閾値を使用）
        thresholds = model_data.get('thresholds', {'low': 0.10, 'high': 0.20})
        low_thresh = thresholds.get('low', 0.10)
        high_thresh = thresholds.get('high', 0.20)
        
        if prob <= low_thresh:
            results['risk_category'] = '低リスク'
        elif prob <= high_thresh:
            results['risk_category'] = '中リスク'
        else:
            results['risk_category'] = '高リスク'
        
        results['thresholds'] = thresholds
            
    except Exception as e:
        results['error'] = str(e)
    
    return results


# =============================================================================
# メインUI
# =============================================================================

st.title("🏥 肺癌手術予後予測システム")
st.markdown("---")

# サイドバー
st.sidebar.title("🔧 設定")

# 実行モード選択
mode = st.sidebar.radio(
    "**実行モード**",
    ["術前予測", "術後予測"],
    help="術前予測：OS, RFS, 合併症\n術後予測：OS, RFS"
)

mode_key = 'preop' if mode == '術前予測' else 'postop'

# モデル設定
st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 モデル設定")

models = {}
model_info = {}

# モデルごとに読み込み
targets = ['os', 'rfs', 'complication'] if mode_key == 'preop' else ['os', 'rfs']
target_names = {'os': 'OS予測', 'rfs': 'RFS予測', 'complication': '合併症予測'}

for target in targets:
    st.sidebar.markdown(f"**{target_names[target]}**")
    
    default_path = DEFAULT_MODELS[mode_key].get(target)
    
    # デフォルトモデル使用 or カスタム
    use_default = st.sidebar.checkbox(f"デフォルトモデル使用", value=True, key=f"default_{target}")
    
    if use_default and default_path and default_path.exists():
        data, meta = load_model(default_path)
        if data and 'error' not in meta:
            models[target] = data
            model_name = meta.get('model_name', 'Unknown')
            if target == 'complication':
                perf = f"AUC={meta.get('auc_test', 0):.3f}"
            else:
                perf = f"C-index={meta.get('c_index_test', 0):.3f}"
            model_info[target] = f"✅ {model_name} ({perf})"
        else:
            model_info[target] = "❌ 読み込みエラー"
    else:
        # カスタムモデルアップロード
        uploaded = st.sidebar.file_uploader(
            f"{target_names[target]}モデル (.pkl)", 
            type=['pkl'], 
            key=f"upload_{target}"
        )
        if uploaded:
            try:
                custom_data = pickle.loads(uploaded.read())
                models[target] = custom_data
                model_info[target] = f"✅ カスタムモデル"
            except Exception as e:
                model_info[target] = f"❌ {str(e)[:20]}"
        else:
            model_info[target] = "⏳ モデル未設定"
    
    st.sidebar.caption(model_info.get(target, ""))

# =============================================================================
# 入力フォーム
# =============================================================================

st.header(f"📋 患者情報入力（{mode}）")

input_data = {}
warnings = []

if mode == "術前予測":
    tabs = st.tabs(["基本データ", "腫瘍データ", "併存疾患"])
else:
    tabs = st.tabs(["基本データ", "腫瘍データ", "併存疾患", "手術データ", "病理データ"])

# 基本データ
with tabs[0]:
    st.subheader("● 基本データ")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        input_data['年齢'] = st.number_input("年齢", min_value=0, max_value=120, value=70, step=1)
        ok, msg = check_value_range(input_data['年齢'], '年齢')
        if not ok: warnings.append(msg)
        input_data['性別'] = st.selectbox("性別", CHOICES['性別'])
    
    with col2:
        input_data['喫煙'] = st.selectbox("喫煙歴", CHOICES['喫煙'])
        if input_data['喫煙'] == '吸った事なし':
            input_data['喫煙本数'] = 0
            input_data['喫煙年間'] = 0
            st.text_input("喫煙本数（本/日）", value="0", disabled=True)
            st.text_input("喫煙年数（年）", value="0", disabled=True)
        else:
            input_data['喫煙本数'] = st.number_input("喫煙本数（本/日）", min_value=0, max_value=200, value=20, step=1)
            input_data['喫煙年間'] = st.number_input("喫煙年数（年）", min_value=0, max_value=100, value=30, step=1)
    
    with col3:
        smoking_index = input_data['喫煙本数'] * input_data['喫煙年間']
        input_data['喫煙指数'] = smoking_index
        st.metric("喫煙指数（自動計算）", f"{smoking_index}")
        if smoking_index >= 600:
            st.warning("⚠️ 重喫煙者")

# 腫瘍データ
with tabs[1]:
    st.subheader("● 腫瘍データ")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        input_data['病側'] = st.selectbox("病側", CHOICES['病側'])
        input_data['原発巣部位'] = st.selectbox("原発巣部位", CHOICES['原発巣部位'])
        input_data['原発肺葉'] = st.selectbox("原発肺葉", CHOICES['原発肺葉'])
        input_data['8th c-T'] = st.selectbox("8th c-T", CHOICES['8th c-T'])
        input_data['8th c-病期'] = st.selectbox("8th c-Stage", CHOICES['8th c-病期'])
    
    with col2:
        input_data['肺野全体腫瘍径'] = st.number_input("腫瘤全体径（cm）", min_value=0.0, max_value=20.0, value=2.5, step=0.1, format="%.1f")
        ok, msg = check_value_range(input_data['肺野全体腫瘍径'], '肺野全体腫瘍径')
        if not ok: warnings.append(msg)
        
        input_data['充実性腫瘍径'] = st.number_input("腫瘤充実径（cm）", min_value=0.0, max_value=20.0, value=2.5, step=0.1, format="%.1f")
        if input_data['充実性腫瘍径'] > input_data['肺野全体腫瘍径']:
            warnings.append("**腫瘤充実径**: 腫瘤全体径より大きい値です")
        
        if input_data['肺野全体腫瘍径'] > 0:
            ct_ratio = min(input_data['充実性腫瘍径'] / input_data['肺野全体腫瘍径'], 1.0)
        else:
            ct_ratio = 1.0
        input_data['C/T比'] = ct_ratio
        st.metric("C/T比（自動計算）", f"{ct_ratio:.2f}")
    
    with col3:
        input_data['PET'] = st.selectbox("PET検査", CHOICES['PET'])
        if input_data['PET'] == 'あり':
            input_data['原発SUV'] = st.number_input("原発SUVmax", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
        else:
            input_data['原発SUV'] = 0.0
            st.text_input("原発SUVmax", value="N/A", disabled=True)
        
        input_data['CEA'] = st.number_input("CEA（ng/mL）", min_value=0.0, max_value=200.0, value=3.0, step=0.1)
        input_data['術前診断'] = st.selectbox("術前診断", CHOICES['術前診断'])

# 併存疾患
with tabs[2]:
    st.subheader("● 併存疾患")
    cols = st.columns(3)
    for i, comorbidity in enumerate(COMORBIDITIES):
        with cols[i % 3]:
            input_data[comorbidity] = st.selectbox(comorbidity, CHOICES['あり_なし'], key=f"c_{comorbidity}")
    
    cci = sum(
        {'他悪性疾患既往': 2, '肺気腫': 1, '虚血心': 1, '心不全': 1, '末梢血管障害': 1, '腎障害': 2,
         '脳梗塞・出血': 1, '片麻痺': 2, '認知症': 1, '肝障害': 1, '肝硬変': 3, '消化器潰瘍': 1,
         '糖尿病': 1, '膠原病': 1}.get(c, 1)
        for c in COMORBIDITIES if input_data.get(c) == 'あり'
    )
    st.metric("Charlson Comorbidity Index (CCI)", cci)

# 手術データ（術後のみ）
if mode == "術後予測":
    with tabs[3]:
        st.subheader("● 手術データ")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            input_data['手術時間(分)'] = st.number_input("手術時間（分）", min_value=0, max_value=2000, value=180, step=1)
            input_data['出血量(ml)'] = st.number_input("出血量（mL）", min_value=0, max_value=10000, value=50, step=10)
        
        with col2:
            input_data['アプローチ'] = st.selectbox("アプローチ", CHOICES['アプローチ'])
            input_data['LN郭清'] = st.selectbox("LN郭清", CHOICES['LN郭清'])
            input_data['他臓器合切'] = st.selectbox("他臓器合切", CHOICES['あり_なし'])
        
        with col3:
            st.selectbox("切除範囲", ["肺葉切除"], disabled=True, help="精度検証中")
            input_data['術中迅速病理'] = st.selectbox("術中迅速病理", CHOICES['あり_なし'])

    with tabs[4]:
        st.subheader("● 病理データ")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            input_data['病理組織型'] = st.selectbox("病理組織型", CHOICES['病理組織型'])
            if input_data['病理組織型'] == '腺癌':
                input_data['腺癌亜型'] = st.selectbox("腺癌亜型", CHOICES['腺癌亜型'])
            else:
                st.selectbox("腺癌亜型", ["該当なし"], disabled=True)
            input_data['8th p-T'] = st.selectbox("8th p-T", CHOICES['8th p-T'])
            input_data['8th p-N'] = st.selectbox("8th p-N", CHOICES['8th p-N'])
        
        with col2:
            st.markdown("**脈管侵襲・胸膜浸潤**")
            input_data['Ly'] = st.selectbox("Ly（リンパ管侵襲）", CHOICES['Ly'])
            input_data['V'] = st.selectbox("V（静脈侵襲）", CHOICES['V'])
            input_data['pl'] = st.selectbox("pl（胸膜浸潤）", CHOICES['pl'])
            input_data['STAS'] = st.selectbox("STAS", CHOICES['STAS'])
        
        with col3:
            st.markdown("**遺伝子変異・術後治療**")
            input_data['EGFR変異'] = st.selectbox("EGFR変異", CHOICES['EGFR変異'])
            input_data['ALK変異'] = st.selectbox("ALK変異", CHOICES['ALK変異'])
            input_data['術後補助治療'] = st.selectbox("術後補助治療", CHOICES['術後補助治療'])

# 警告表示
if warnings:
    st.markdown("---")
    st.warning("⚠️ **入力値の警告** - 予測精度が低下する可能性があります")
    for w in warnings:
        st.markdown(f"- {w}")

# =============================================================================
# 予測実行
# =============================================================================

st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔮 予測実行", type="primary", use_container_width=True)

if predict_button:
    if not models:
        st.error("⚠️ モデルが読み込まれていません")
    else:
        with st.spinner("予測中..."):
            X = convert_to_model_features(input_data, mode)
            
            st.markdown("---")
            st.header("📊 予測結果")
            
            if mode == "術前予測":
                result_cols = st.columns(3)
            else:
                result_cols = st.columns(2)
            
            os_result = {}
            rfs_result = {}
            
            # OS予測
            with result_cols[0]:
                st.subheader("🫁 全生存（OS）")
                if 'os' in models:
                    os_result = predict_survival(models['os'], X)
                    if 'error' not in os_result:
                        surv_5y = os_result.get('survival_5y', 0.9)
                        surv_3y = os_result.get('survival_3y', 0.95)
                        surv_1y = os_result.get('survival_1y', 0.98)
                        
                        st.metric("5年生存率", f"{surv_5y*100:.1f}%")
                        st.metric("3年生存率", f"{surv_3y*100:.1f}%")
                        st.metric("1年生存率", f"{surv_1y*100:.1f}%")
                        
                        # 期待値
                        mean_surv = os_result.get('mean_survival_years')
                        median_surv = os_result.get('median_survival_years')
                        if mean_surv:
                            st.metric("平均生存期間", f"{mean_surv:.1f}年")
                        if median_surv:
                            st.metric("中央生存期間", f"{median_surv:.1f}年")
                        
                        risk_cat = os_result.get('risk_category', '不明')
                        if risk_cat == '低リスク':
                            st.success(f"🟢 **{risk_cat}**")
                        elif risk_cat == '中リスク':
                            st.warning(f"🟡 **{risk_cat}**")
                        else:
                            st.error(f"🔴 **{risk_cat}**")
                        
                        st.caption(f"モデル: {models['os'].get('model_name', 'Unknown')}")
                        st.caption(f"C-index: {models['os'].get('c_index_test', 0):.3f}")
                    else:
                        st.error(f"エラー: {os_result['error']}")
                else:
                    st.warning("モデル未設定")
            
            # RFS予測
            with result_cols[1]:
                st.subheader("🔄 無再発生存（RFS）")
                if 'rfs' in models:
                    rfs_result = predict_survival(models['rfs'], X)
                    if 'error' not in rfs_result:
                        rfs_5y = rfs_result.get('survival_5y', 0.85)
                        rfs_3y = rfs_result.get('survival_3y', 0.90)
                        rfs_1y = rfs_result.get('survival_1y', 0.95)
                        
                        st.metric("5年無再発生存率", f"{rfs_5y*100:.1f}%")
                        st.metric("3年無再発生存率", f"{rfs_3y*100:.1f}%")
                        st.metric("1年無再発生存率", f"{rfs_1y*100:.1f}%")
                        
                        # 期待値
                        mean_rfs = rfs_result.get('mean_survival_years')
                        median_rfs = rfs_result.get('median_survival_years')
                        if mean_rfs:
                            st.metric("平均無再発生存期間", f"{mean_rfs:.1f}年")
                        if median_rfs:
                            st.metric("中央無再発生存期間", f"{median_rfs:.1f}年")
                        
                        # 再発確率
                        recurrence_prob = 1 - rfs_5y
                        st.metric("5年再発確率", f"{recurrence_prob*100:.1f}%")
                        
                        risk_cat = rfs_result.get('risk_category', '不明')
                        if risk_cat == '低リスク':
                            st.success(f"🟢 **{risk_cat}**")
                        elif risk_cat == '中リスク':
                            st.warning(f"🟡 **{risk_cat}**")
                        else:
                            st.error(f"🔴 **{risk_cat}**")
                        
                        st.caption(f"モデル: {models['rfs'].get('model_name', 'Unknown')}")
                        st.caption(f"C-index: {models['rfs'].get('c_index_test', 0):.3f}")
                    else:
                        st.error(f"エラー: {rfs_result['error']}")
                else:
                    st.warning("モデル未設定")
            
            # 合併症予測（術前のみ）
            if mode == "術前予測":
                with result_cols[2]:
                    st.subheader("⚠️ 術後合併症")
                    if 'complication' in models:
                        comp_result = predict_classification(models['complication'], X)
                        if 'error' not in comp_result:
                            prob = comp_result.get('probability', 0.1)
                            st.metric("合併症発生確率", f"{prob*100:.1f}%")
                            
                            risk_cat = comp_result.get('risk_category', '不明')
                            if risk_cat == '低リスク':
                                st.success(f"🟢 **{risk_cat}**")
                            elif risk_cat == '中リスク':
                                st.warning(f"🟡 **{risk_cat}**")
                            else:
                                st.error(f"🔴 **{risk_cat}**")
                            
                            st.caption(f"モデル: {models['complication'].get('model_name', 'Unknown')}")
                            st.caption(f"AUC: {models['complication'].get('auc_test', 0):.3f}")
                        else:
                            st.error(f"エラー: {comp_result['error']}")
                    else:
                        st.warning("モデル未設定")
            
            # リスク因子表示
            if 'os' in models and 'risk_factors' in os_result:
                st.markdown("---")
                st.subheader("📈 主要リスク因子（OS予測）")
                factors = os_result.get('risk_factors', [])
                if factors:
                    factor_df = pd.DataFrame([
                        {
                            '因子': str(f['feature']),
                            '係数': f"{float(f['coefficient']):.3f}",
                            '入力値': f"{float(f['value']):.2f}" if isinstance(f['value'], (int, float, np.integer, np.floating)) else str(f['value']),
                            '影響': str(f['direction'])
                        }
                        for f in factors
                    ])
                    st.dataframe(factor_df, use_container_width=True, hide_index=True)
            
            # 注意事項
            st.markdown("---")
            st.caption("""
            ⚠️ **注意事項**
            - この予測結果は参考値です。臨床判断は必ず医師が行ってください。
            - 期待値（平均/中央生存期間）は生存関数から算出した推定値です。
            """)

# フッター
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'>肺癌手術予後予測システム v4.0</div>", unsafe_allow_html=True)
