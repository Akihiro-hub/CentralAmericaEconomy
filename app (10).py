import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np
import time
import io
import scipy.stats as stats
import warnings

warnings.filterwarnings('ignore')

# アプリケーション設定
st.set_page_config(
    page_title="World Bank Data Analysis",
    page_icon="📊",
    layout="wide"
)

# 言語翻訳辞書
TRANSLATIONS = {
    'JAP': {
        'title': "📊 中米データ分析アプリ",
        'subtitle': "**中米・グアテマラを中心とした経済・社会データの総合分析プラットフォーム**",
        'sidebar_analysis': "🔧 分析機能",
        'analysis_mode': "分析機能を選択",
        'multi_country': "📈 多国間比較分析",
        'single_country': "🏛️ 単一国詳細分析",
        'lang_select': "言語選択 (Language)",
        'start_year': "開始年",
        'end_year': "終了年",
        'analyze_btn': "📈 分析実行",
        'loading': "データを取得中...",
        'error_no_data': "データを取得できませんでした。",
        'ca_countries': "中米9か国",
        'major_countries': "主要9か国",
        'regional_aggregates': "地域・所得分類平均",
        'select_indicator': "分析指標を選択",
        'compare_countries': "比較対象国の選択",
        'data_table': "📊 データテーブル",
        'footer_source': "**データソース**: 世界銀行 World Bank Open Data API",
        'footer_note': "**注意**: このアプリケーションは関係者の調査補助での使用を想定しています。",
        'pop_trend': "人口推移分析",
        'gdp_comp': "GDP構成分析",
        'industry_comp': "産業別GDP構成分析",
        'basic_indicators': "基本経済指標",
        'analysis_package': "分析パッケージ",
        'packages': {
            "基本経済プロファイル": "基本経済プロファイル",
            "社会発展指標": "社会発展指標",
            "持続可能性評価": "持続可能性評価"
        }
    },
    'ESP': {
        'title': "📊 App de Análisis de Datos de Centroamérica",
        'subtitle': "**Plataforma de análisis integral de datos económicos y sociales centrada en Centroamérica y Guatemala**",
        'sidebar_analysis': "🔧 Funciones de Análisis",
        'analysis_mode': "Seleccionar Función",
        'multi_country': "📈 Análisis Comparativo Multinacional",
        'single_country': "🏛️ Análisis Detallado por País",
        'lang_select': "Seleccionar Idioma",
        'start_year': "Año de Inicio",
        'end_year': "Año de Fin",
        'analyze_btn': "📈 Ejecutar Análisis",
        'loading': "Obteniendo datos...",
        'error_no_data': "No se pudieron obtener los datos.",
        'ca_countries': "9 Países de Centroamérica",
        'major_countries': "9 Países Principales",
        'regional_aggregates': "Promedios Regionales/Ingresos",
        'select_indicator': "Seleccionar Indicador",
        'compare_countries': "Seleccionar Países para Comparar",
        'data_table': "📊 Tabla de Datos",
        'footer_source': "**Fuente de datos**: API del Banco Mundial",
        'footer_note': "**Nota**: Esta aplicación está destinada a asistir en la investigación técnica.",
        'pop_trend': "Análisis de Tendencia Demográfica",
        'gdp_comp': "Análisis de Composición del PIB",
        'industry_comp': "Composición del PIB por Industria",
        'basic_indicators': "Indicadores Económicos Básicos",
        'analysis_package': "Paquete de Análisis",
        'packages': {
            "基本経済プロファイル": "Perfil Económico Básico",
            "社会発展指標": "Indicadores de Desarrollo Social",
            "持続可能性評価": "Evaluación de Sostenibilidad"
        }
    }
}

# サイドバーでの言語選択
if 'lang' not in st.session_state:
    st.session_state.lang = 'JAP'

selected_lang = st.sidebar.selectbox("Language / 言語", ["JAP", "ESP"], index=0 if st.session_state.lang == 'JAP' else 1)
st.session_state.lang = selected_lang
t = TRANSLATIONS[st.session_state.lang]

# 国コードマッピング
CENTRAL_AMERICA_COUNTRIES = {
    'Guatemala' if st.session_state.lang == 'ESP' else 'グアテマラ': 'GT',
    'Honduras' if st.session_state.lang == 'ESP' else 'ホンジュラス': 'HN', 
    'El Salvador' if st.session_state.lang == 'ESP' else 'エルサルバドル': 'SV',
    'Costa Rica' if st.session_state.lang == 'ESP' else 'コスタリカ': 'CR',
    'Nicaragua' if st.session_state.lang == 'ESP' else 'ニカラグア': 'NI',
    'Panama' if st.session_state.lang == 'ESP' else 'パナマ': 'PA',
    'Belize' if st.session_state.lang == 'ESP' else 'ベリーズ': 'BZ',
    'Dominican Republic' if st.session_state.lang == 'ESP' else 'ドミニカ共和国': 'DO',
    'Mexico' if st.session_state.lang == 'ESP' else 'メキシコ': 'MX'
}

MAJOR_COUNTRIES = {
    'Japan' if st.session_state.lang == 'ESP' else '日本': 'JP',
    'South Korea' if st.session_state.lang == 'ESP' else '韓国': 'KR',
    'USA' if st.session_state.lang == 'ESP' else '米国': 'US',
    'China' if st.session_state.lang == 'ESP' else '中国': 'CN',
    'India' if st.session_state.lang == 'ESP' else 'インド': 'IN',
    'UK' if st.session_state.lang == 'ESP' else '英国': 'GB',
    'Germany' if st.session_state.lang == 'ESP' else 'ドイツ': 'DE',
    'France' if st.session_state.lang == 'ESP' else 'フランス': 'FR',
    'Italy' if st.session_state.lang == 'ESP' else 'イタリア': 'IT'
}

REGIONAL_AGGREGATES = {
    'Middle Income Total' if st.session_state.lang == 'ESP' else '中所得国全体': 'MIC',
    'Latin America & Caribbean Total' if st.session_state.lang == 'ESP' else '中南米・カリブ全体': 'LCN'
}

# 分析指標の定義
INDICATORS = {
    'GDP成長率（%）' if st.session_state.lang == 'JAP' else 'Crecimiento del PIB (%)': 'NY.GDP.MKTP.KD.ZG',
    '一人当たりGDP（名目）' if st.session_state.lang == 'JAP' else 'PIB per cápita (nominal)': 'NY.GDP.PCAP.CD',
    '一人当たりGDP（実質2015USD）' if st.session_state.lang == 'JAP' else 'PIB per cápita (real 2015 USD)': 'NY.GDP.PCAP.KD',
    '一人当たりGDP（PPPベース）' if st.session_state.lang == 'JAP' else 'PIB per cápita (PPA)': 'NY.GDP.PCAP.PP.CD',
    'GDP（名目USD）' if st.session_state.lang == 'JAP' else 'PIB (nominal USD)': 'NY.GDP.MKTP.CD',
    'GDP（実質、2015USD）' if st.session_state.lang == 'JAP' else 'PIB (real, 2015 USD)': 'NY.GDP.MKTP.KD',
    'インフレ率（%）' if st.session_state.lang == 'JAP' else 'Tasa de inflación (%)': 'FP.CPI.TOTL.ZG',
    '輸出（GDP比%）' if st.session_state.lang == 'JAP' else 'Exportaciones (% del PIB)': 'NE.EXP.GNFS.ZS',
    '輸入（GDP比%）' if st.session_state.lang == 'JAP' else 'Importaciones (% del PIB)': 'NE.IMP.GNFS.ZS',
    '政府支出（GDP比%）' if st.session_state.lang == 'JAP' else 'Gasto público (% del PIB)': 'NE.CON.GOVT.ZS',
    '外国直接投資（USD）' if st.session_state.lang == 'JAP' else 'FDI (USD)': 'BX.KLT.DINV.CD.WD',
    '個人送金額（USD）' if st.session_state.lang == 'JAP' else 'Remesas personales (USD)': 'BX.TRF.PWKR.CD.DT',
    '送金流入（GDP比%）' if st.session_state.lang == 'JAP' else 'Remesas (% del PIB)': 'BX.TRF.PWKR.DT.GD.ZS',
    '政府債務（GDP比%）' if st.session_state.lang == 'JAP' else 'Deuda pública (% del PIB)': 'GC.DOD.TOTL.GD.ZS',
    '財政収入（GDP比%）' if st.session_state.lang == 'JAP' else 'Ingresos fiscales (% del PIB)': 'GC.REV.XGRT.GD.ZS',
    '対外債務残高（GNI比%）' if st.session_state.lang == 'JAP' else 'Deuda externa (% del INB)': 'DT.DOD.DECT.GN.ZS',
    '総資本形成（GDP比%）' if st.session_state.lang == 'JAP' else 'Formación bruta de capital (% del PIB)': 'NE.GDI.TOTL.ZS',
    '貯蓄・投資ギャップ（%）' if st.session_state.lang == 'JAP' else 'Brecha ahorro-inversión (%)': 'NY.GNS.ICTR.ZS',
    '金融口座保有率（%）' if st.session_state.lang == 'JAP' else 'Titularidad de cuenta financiera (%)': 'FX.OWN.TOTL.ZS',
    '金融深化度（%）' if st.session_state.lang == 'JAP' else 'Profundidad financiera (%)': 'FS.AST.PRVT.GD.ZS',
    '外国直接投資（GDP比%）' if st.session_state.lang == 'JAP' else 'FDI (% del PIB)': 'BX.KLT.DINV.WD.GD.ZS',
    '貧困率（%）' if st.session_state.lang == 'JAP' else 'Tasa de pobreza (%)': 'SI.POV.NAHC',
    '所得格差（ジニ係数）' if st.session_state.lang == 'JAP' else 'Desigualdad de ingresos (Gini)': 'SI.POV.GINI',
    '失業率（%）' if st.session_state.lang == 'JAP' else 'Tasa de desempleo (%)': 'SL.UEM.TOTL.ZS',
    '労働参加率（%）' if st.session_state.lang == 'JAP' else 'Tasa de participación laboral (%)': 'SL.TLF.CACT.ZS',
    '人口' if st.session_state.lang == 'JAP' else 'Población': 'SP.POP.TOTL',
    '人口成長率（%）' if st.session_state.lang == 'JAP' else 'Crecimiento poblacional (%)': 'SP.POP.GROW',
    '都市人口率（%）' if st.session_state.lang == 'JAP' else 'Población urbana (%)': 'SP.URB.TOTL.IN.ZS',
    '純移民数' if st.session_state.lang == 'JAP' else 'Migración neta': 'SM.POP.NETM',
    '乳児死亡率' if st.session_state.lang == 'JAP' else 'Tasa de mortalidad infantil': 'SP.DYN.IMRT.IN',
    '平均寿命' if st.session_state.lang == 'JAP' else 'Esperanza de vida': 'SP.DYN.LE00.IN',
    '保健支出（GDP比%）' if st.session_state.lang == 'JAP' else 'Gasto en salud (% del PIB)': 'SH.XPD.CHEX.GD.ZS',
    '教育支出（GDP比%）' if st.session_state.lang == 'JAP' else 'Gasto en educación (% del PIB)': 'SE.XPD.TOTL.GD.ZS',
    '識字率（%）' if st.session_state.lang == 'JAP' else 'Tasa de alfabetización (%)': 'SE.ADT.LITR.ZS',
    'CO2排出量（1人当たり）' if st.session_state.lang == 'JAP' else 'Emisiones de CO2 (per cápita)': 'EN.ATM.CO2E.PC',
    '再生可能エネルギー比率（%）' if st.session_state.lang == 'JAP' else 'Energía renovable (%)': 'EG.FEC.RNEW.ZS',
    '電力普及率（%）' if st.session_state.lang == 'JAP' else 'Acceso a la electricidad (%)': 'EG.ELC.ACCS.ZS',
    'インターネット普及率（%）' if st.session_state.lang == 'JAP' else 'Uso de Internet (%)': 'IT.NET.USER.ZS'
}

# キャッシュされたデータ取得関数
@st.cache_data(ttl=3600)
def fetch_world_bank_data(country_codes, indicator_code, start_year=2000, end_year=2023):
    """世界銀行APIからデータを取得"""
    try:
        countries_str = ';'.join(country_codes)
        url = f"http://api.worldbank.org/v2/countries/{countries_str}/indicators/{indicator_code}"
        params = {
            'date': f'{start_year}:{end_year}',
            'format': 'json',
            'per_page': '1000'
        }
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code != 200:
            return None
            
        data = response.json()
        
        if len(data) < 2 or not data[1]:
            return None
            
        records = []
        for item in data[1]:
            if item['value'] is not None:
                records.append({
                    'country': item['country']['value'],
                    'country_code': item['countryiso3code'],
                    'year': int(item['date']),
                    'value': float(item['value']),
                    'indicator': item['indicator']['value']
                })
        
        if not records:
            return None
            
        df = pd.DataFrame(records)
        return df.sort_values(['country', 'year'])
        
    except Exception as e:
        return None

def multi_country_comparison_analysis():
    """多国間比較分析"""
    st.header(t['multi_country'])
    
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.selectbox(t['start_year'], range(2000, 2024), index=10)
    with col2:
        end_year = st.selectbox(t['end_year'], range(start_year, 2024), index=len(range(start_year, 2024))-1)
    
    selected_indicator = st.selectbox(t['select_indicator'], list(INDICATORS.keys()))
    indicator_code = INDICATORS[selected_indicator]
    
    st.subheader(t['compare_countries'])
    
    # 中米9か国
    st.write(f"**{t['ca_countries']}**")
    central_america_selected = []
    cols = st.columns(3)
    for i, country in enumerate(CENTRAL_AMERICA_COUNTRIES.keys()):
        with cols[i % 3]:
            if st.checkbox(country, key=f"ca_{country}"):
                central_america_selected.append(CENTRAL_AMERICA_COUNTRIES[country])
    
    # 主要9か国
    st.write(f"**{t['major_countries']}**")
    major_countries_selected = []
    cols = st.columns(3)
    for i, country in enumerate(MAJOR_COUNTRIES.keys()):
        with cols[i % 3]:
            if st.checkbox(country, key=f"major_{country}"):
                major_countries_selected.append(MAJOR_COUNTRIES[country])
    
    # 地域平均
    st.write(f"**{t['regional_aggregates']}**")
    regional_selected = []
    cols = st.columns(2)
    for i, (region_name, region_code) in enumerate(REGIONAL_AGGREGATES.items()):
        with cols[i % 2]:
            if st.checkbox(region_name, key=f"region_{region_code}"):
                regional_selected.append(region_code)
    
    all_selected_countries = central_america_selected + major_countries_selected + regional_selected
    
    if not all_selected_countries:
        st.warning(t['compare_countries'])
        return
    
    if st.button(t['analyze_btn'], key="multi_country_analyze"):
        with st.spinner(t['loading']):
            df = fetch_world_bank_data(all_selected_countries, indicator_code, start_year, end_year)
            
            if df is None or df.empty:
                st.error(t['error_no_data'])
                return
        
        fig = go.Figure()
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        color_index = 0
        
        available_countries = df['country_code'].unique().tolist()
        
        for country_code in available_countries:
            country_data = df[df['country_code'] == country_code]
            if not country_data.empty:
                country_name = country_data['country'].iloc[0]
                
                if country_code == 'GT':
                    color = 'red'; line_width = 3; marker_size = 8
                elif country_code in ['MIC', 'LCN']:
                    color = 'darkgreen'; line_width = 3; marker_size = 6
                else:
                    color = colors[color_index % len(colors)]
                    color_index += 1
                    line_width = 2; marker_size = 6
                
                country_data_sorted = country_data.sort_values('year')
                fig.add_trace(go.Scatter(
                    x=country_data_sorted['year'],
                    y=country_data_sorted['value'],
                    mode='lines+markers',
                    name=country_name,
                    line=dict(color=color, width=line_width),
                    marker=dict(size=marker_size)
                ))
        
        fig.update_layout(
            title=f"{selected_indicator} ({start_year}-{end_year})",
            xaxis_title=t['end_year'],
            yaxis_title=selected_indicator,
            height=600,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader(t['data_table'])
        pivot_df = df.pivot(index='year', columns='country', values='value')
        st.dataframe(pivot_df)

def single_country_detailed_analysis():
    """単一国詳細分析"""
    st.header(t['single_country'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        start_year = st.selectbox(t['start_year'], range(2000, 2024), index=10, key="single_start")
    with col2:
        end_year = st.selectbox(t['end_year'], range(start_year, 2024), index=len(range(start_year, 2024))-1, key="single_end")
    with col3:
        selected_country = st.selectbox(t['single_country'], list(CENTRAL_AMERICA_COUNTRIES.keys()))
    
    country_code = CENTRAL_AMERICA_COUNTRIES[selected_country]
    
    analysis_package = st.selectbox(t['analysis_package'], list(t['packages'].values()))
    
    # 指標定義（内部キーで管理）
    package_indicators_map = {
        t['packages']["基本経済プロファイル"]: {
            'GDP成長率（%）' if st.session_state.lang == 'JAP' else 'Crecimiento del PIB (%)': 'NY.GDP.MKTP.KD.ZG',
            '一人当たりGDP（名目）' if st.session_state.lang == 'JAP' else 'PIB per cápita (nominal)': 'NY.GDP.PCAP.CD',
            'インフレ率（%）' if st.session_state.lang == 'JAP' else 'Tasa de inflación (%)': 'FP.CPI.TOTL.ZG',
            '失業率（%）' if st.session_state.lang == 'JAP' else 'Tasa de desempleo (%)': 'SL.UEM.TOTL.ZS'
        },
        t['packages']["社会発展指標"]: {
            '平均寿命' if st.session_state.lang == 'JAP' else 'Esperanza de vida': 'SP.DYN.LE00.IN',
            '貧困率（%）' if st.session_state.lang == 'JAP' else 'Tasa de pobreza (%)': 'SI.POV.NAHC',
            '教育支出（GDP比%）' if st.session_state.lang == 'JAP' else 'Gasto en educación (% del PIB)': 'SE.XPD.TOTL.GD.ZS'
        },
        t['packages']["持続可能性評価"]: {
            'CO2排出量' if st.session_state.lang == 'JAP' else 'Emisiones de CO2': 'EN.ATM.CO2E.PC',
            '再生可能エネルギー（%）' if st.session_state.lang == 'JAP' else 'Energía renovable (%)': 'EG.FEC.RNEW.ZS'
        }
    }
    
    indicators = package_indicators_map.get(analysis_package, {})
    
    if st.button(t['analyze_btn'], key="single_country_analyze"):
        st.subheader(f"{selected_country} - {analysis_package}")
        
        cols = st.columns(2)
        for i, (indicator_name, indicator_code) in enumerate(indicators.items()):
            with cols[i % 2]:
                df = fetch_world_bank_data([country_code], indicator_code, start_year, end_year)
                if df is not None and not df.empty:
                    latest_value = df['value'].iloc[-1]
                    st.metric(indicator_name, f"{latest_value:.2f}")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df['year'], y=df['value'], mode='lines+markers', name=indicator_name, line=dict(color='red')))
                    fig.update_layout(title=indicator_name, height=300)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"{indicator_name}: {t['error_no_data']}")

# メイン画面
st.title(t['title'])
st.markdown(t['subtitle'])

# サイドバー設定
st.sidebar.header(t['sidebar_analysis'])

analysis_mode = st.sidebar.radio(
    t['analysis_mode'],
    [
        t['multi_country'],
        t['single_country']
    ]
)

# 分析機能の実行
if analysis_mode == t['multi_country']:
    multi_country_comparison_analysis()
elif analysis_mode == t['single_country']:
    single_country_detailed_analysis()

# フッター
st.markdown("---")
st.markdown(t['footer_source'])
st.markdown(t['footer_note'])

