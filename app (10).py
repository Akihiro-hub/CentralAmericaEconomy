import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np
import time
import io
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# アプリケーション設定
st.set_page_config(
    page_title="中米データ分析アプリ",
    page_icon="📊",
    layout="wide"
)

# 国コードマッピング
CENTRAL_AMERICA_COUNTRIES = {
    'グアテマラ': 'GT',
    'ホンジュラス': 'HN', 
    'エルサルバドル': 'SV',
    'コスタリカ': 'CR',
    'ニカラグア': 'NI',
    'パナマ': 'PA',
    'ベリーズ': 'BZ',
    'ドミニカ共和国': 'DO',
    'メキシコ': 'MX'
}

MAJOR_COUNTRIES = {
    '日本': 'JP',
    '韓国': 'KR',
    '米国': 'US',
    '中国': 'CN',
    'インド': 'IN',
    '英国': 'GB',
    'ドイツ': 'DE',
    'フランス': 'FR',
    'イタリア': 'IT'
}

# 中南米・カリブ33か国
LATIN_AMERICA_CARIBBEAN = {
    'アルゼンチン': 'AR', 'ボリビア': 'BO', 'ブラジル': 'BR', 'チリ': 'CL', 'コロンビア': 'CO',
    'エクアドル': 'EC', 'ガイアナ': 'GY', 'パラグアイ': 'PY', 'ペルー': 'PE', 'スリナム': 'SR',
    'ウルグアイ': 'UY', 'ベネズエラ': 'VE', 'アンティグア・バーブーダ': 'AG', 'バハマ': 'BS',
    'バルバドス': 'BB', 'ベリーズ': 'BZ', 'コスタリカ': 'CR', 'キューバ': 'CU', 'ドミニカ': 'DM',
    'ドミニカ共和国': 'DO', 'エルサルバドル': 'SV', 'グレナダ': 'GD', 'グアテマラ': 'GT',
    'ハイチ': 'HT', 'ホンジュラス': 'HN', 'ジャマイカ': 'JM', 'メキシコ': 'MX', 'ニカラグア': 'NI',
    'パナマ': 'PA', 'セントキッツ・ネイビス': 'KN', 'セントルシア': 'LC', 'セントビンセント': 'VC',
    'トリニダード・トバゴ': 'TT'
}

# 高位・中位所得国（簡略版 - 実際にはWorld Bank分類に基づく）
UPPER_MIDDLE_INCOME = {
    'アルゼンチン': 'AR', 'ブラジル': 'BR', 'チリ': 'CL', 'コロンビア': 'CO', 'コスタリカ': 'CR',
    'メキシコ': 'MX', 'パナマ': 'PA', 'ペルー': 'PE', 'ドミニカ共和国': 'DO', 'エクアドル': 'EC',
    'グアテマラ': 'GT', 'ジャマイカ': 'JM', 'パラグアイ': 'PY', 'エルサルバドル': 'SV'
}

# 世銀地域・所得分類コード
REGIONAL_AGGREGATES = {
    '中所得国全体': 'MIC',
    '中南米・カリブ全体': 'LCN'
}

# 分析指標の定義（元アプリと同様）
INDICATORS = {
    'GDP成長率（%）': 'NY.GDP.MKTP.KD.ZG',
    '一人当たりGDP（名目）': 'NY.GDP.PCAP.CD',
    '一人当たりGDP（実質2015USD）': 'NY.GDP.PCAP.KD',
    '一人当たりGDP（PPPベース）': 'NY.GDP.PCAP.PP.CD',
    '生産人口一人当たりPPP GDP（合成処）': 'COMPOSITE_WORKING_AGE_PPP_GDP',
    'GDP（名目USD）': 'NY.GDP.MKTP.CD',
    'GDP（実質、2015USD）': 'NY.GDP.MKTP.KD',
    'インフレ率（%）': 'FP.CPI.TOTL.ZG',
    '輸出（GDP比%）': 'NE.EXP.GNFS.ZS',
    '輸入（GDP比%）': 'NE.IMP.GNFS.ZS',
    '政府支出（GDP比%）': 'NE.CON.GOVT.ZS',
    '外国直接投資（USD）': 'BX.KLT.DINV.CD.WD',
    '個人送金額（USD）': 'BX.TRF.PWKR.CD.DT',
    '送金流入（GDP比%）': 'BX.TRF.PWKR.DT.GD.ZS',
    '政府債務（GDP比%）': 'GC.DOD.TOTL.GD.ZS',
    '財政収入（GDP比%）': 'GC.REV.XGRT.GD.ZS',
    '対外債務残高（GNI比%）': 'DT.DOD.DECT.GN.ZS',
    '総資本形成（GDP比%）': 'NE.GDI.TOTL.ZS',
    '貯蓄・投資ギャップ（%）': 'NY.GNS.ICTR.ZS',
    '金融口座保有率（%）': 'FX.OWN.TOTL.ZS',
    '金融深化度（%）': 'FS.AST.PRVT.GD.ZS',
    '外国直接投資（GDP比%）': 'BX.KLT.DINV.WD.GD.ZS',
    '貧困率（%）': 'SI.POV.NAHC',
    '所得格差（ジニ係数）': 'SI.POV.GINI',
    '失業率（%）': 'SL.UEM.TOTL.ZS',
    '労働参加率（%）': 'SL.TLF.CACT.ZS',
    '女性労働参加率（%）': 'SL.TLF.CACT.FE.ZS',
    '女性管理者比率（%）': 'SL.EMP.SMGT.FE.ZS',
    '人口': 'SP.POP.TOTL',
    '人口成長率（%）': 'SP.POP.GROW',
    '都市人口率（%）': 'SP.URB.TOTL.IN.ZS',
    '純移民数': 'SM.POP.NETM',
    '乳児死亡率': 'SP.DYN.IMRT.IN',
    '医師数（人口千人当たり）': 'SH.MED.PHYS.ZS',
    '病床数（人口千人当たり）': 'SH.MED.BEDS.ZS',
    'HIV感染率': 'SH.DYN.AIDS.ZS',
    '平均寿命': 'SP.DYN.LE00.IN',
    '保健支出（GDP比%）': 'SH.XPD.CHEX.GD.ZS',
    '教育支出（GDP比%）': 'SE.XPD.TOTL.GD.ZS',
    '初等教育修了率（%）': 'SE.PRM.CMPT.ZS',
    '識字率（%）': 'SE.ADT.LITR.ZS',
    'CO2排出量（1人当たり）': 'EN.ATM.CO2E.PC',
    '再生可能エネルギー比率（%）': 'EG.FEC.RNEW.ZS',
    '森林面積（%）': 'AG.LND.FRST.ZS',
    '水資源アクセス（%）': 'SH.H2O.BASW.ZS',
    '電力普及率（%）': 'EG.ELC.ACCS.ZS',
    'インターネット普及率（%）': 'IT.NET.USER.ZS',
    '研究開発費（GDP比%）': 'GB.XPD.RSDV.GD.ZS'
}

def calculate_composite_working_age_ppp_gdp(country_codes, start_year, end_year):
    """生産人口一人当たり購買力平価GDPを計算"""
    
    # 間接的な呼び出しを防ぐため、直接APIを呼び出し
    def get_wb_data_direct(country_codes, indicator_code, start_year, end_year):
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
            
        except Exception:
            return None
    
    try:
        # PPP GDP取得
        ppp_gdp_df = get_wb_data_direct(country_codes, 'NY.GDP.MKTP.PP.CD', start_year, end_year)
        # 生産人口（15-64歳）取得
        working_age_df = get_wb_data_direct(country_codes, 'SP.POP.1564.TO', start_year, end_year)
        
        if ppp_gdp_df is None or working_age_df is None or ppp_gdp_df.empty or working_age_df.empty:
            return None
        
        # データを結合
        merged = pd.merge(ppp_gdp_df, working_age_df, 
                         on=['country_code', 'country', 'year'], 
                         suffixes=('_gdp', '_pop'))
        
        if merged.empty:
            return None
        
        # 生産人口一人当たりPPP GDPを計算
        merged['value'] = merged['value_gdp'] / merged['value_pop']
        merged['indicator'] = '生産人口一人当たり購買力平価GDP'
        
        return merged[['country_code', 'country', 'year', 'value', 'indicator']]
        
    except Exception as e:
        return None

def create_population_trend_chart(country_code, start_year=2000, end_year=2023):
    """人口推移グラフ作成（元アプリと同じロジック）"""
    try:
        # 総人口取得
        df_pop = fetch_world_bank_data([country_code], "SP.POP.TOTL", start_year, end_year)
        if df_pop is None or df_pop.empty:
            return go.Figure()
        
        # 生産人口取得
        df_working = fetch_world_bank_data([country_code], "SP.POP.1564.TO", start_year, end_year)
        if df_working is None or df_working.empty:
            return go.Figure()
        
        # データを結合（年をインデックスとして）
        df_pop_data = df_pop.set_index('year')['value'].sort_index()
        df_working_data = df_working.set_index('year')['value'].sort_index()
        
        df_pop_all = pd.concat([df_pop_data.rename("総人口"), df_working_data.rename("生産人口（15-64歳）")], axis=1)
        
        if df_pop_all.empty:
            return go.Figure()
        
        fig_pop = go.Figure()
        
        # 生産人口（青, 面グラフ, tozeroy）を手前に
        fig_pop.add_trace(go.Scatter(
            x=df_pop_all.index, y=df_pop_all["生産人口（15-64歳）"],
            mode='lines',
            name="生産人口（15-64歳, Working Age）",
            line=dict(color="#1f77b4", width=2),
            fill='tozeroy',
            fillcolor="rgba(31,119,180,0.5)",
            showlegend=True
        ))
        
        # 総人口（薄い青, 面グラフ, tozeroy）を奥に
        fig_pop.add_trace(go.Scatter(
            x=df_pop_all.index, y=df_pop_all["総人口"],
            mode='lines',
            name="総人口（Total Population）",
            line=dict(color="rgba(31,119,180,0.3)", width=0),
            fill='tozeroy',
            fillcolor="rgba(31,119,180,0.15)",
            showlegend=True
        ))
        
        fig_pop.update_layout(
            xaxis_title='年度',
            yaxis_title='人口（人）',
            title="人口推移（重ね合わせ面グラフ）",
            legend_title='人口区分',
            yaxis=dict(rangemode='tozero', tickformat=',d'),
            height=600
        )
        
        return fig_pop
    except Exception as e:
        st.error(f"人口推移グラフ作成エラー: {str(e)}")
        return go.Figure()

def create_gdp_composition_chart(country_code, start_year=2000, end_year=2023):
    """GDP構成（G, I, C, 純輸出）時系列積み上げ棒グラフ（元アプリと同じロジック）"""
    try:
        # 指標コード（元アプリと同じ）
        gdp_components = {
            '政府支出(G)': 'NE.CON.GOVT.CD',
            '投資(I)': 'NE.GDI.FTOT.CD',
            '消費(C)': 'NE.CON.PRVT.CD',
            '輸出(X)': 'NE.EXP.GNFS.CD',
            '輸入(M)': 'NE.IMP.GNFS.CD'
        }
        
        # データ取得
        df_g_list = []
        for label, code in gdp_components.items():
            df_comp = fetch_world_bank_data([country_code], code, start_year, end_year)
            if df_comp is not None and not df_comp.empty:
                df_comp_data = df_comp.set_index('year')['value'].sort_index()
                df_comp_data.name = label
                df_g_list.append(df_comp_data)
        
        if not df_g_list:
            return go.Figure()
        
        df_gdp = pd.concat(df_g_list, axis=1)
        
        # 純輸出(X-M)列作成
        if '輸出(X)' in df_gdp.columns and '輸入(M)' in df_gdp.columns:
            df_gdp['純輸出(X-M)'] = df_gdp['輸出(X)'] - df_gdp['輸入(M)']
        
        # 必要カラムのみ
        plot_cols = ['政府支出(G)', '投資(I)', '消費(C)', '純輸出(X-M)']
        # 4項目いずれかが欠損している年は除外
        available_cols = [col for col in plot_cols if col in df_gdp.columns]
        if not available_cols:
            return go.Figure()
        
        df_gdp_plot = df_gdp[available_cols].dropna(how='any')
        
        if df_gdp_plot.empty:
            return go.Figure()
        
        fig = go.Figure()
        for col in available_cols:
            fig.add_trace(go.Bar(
                x=df_gdp_plot.index,
                y=df_gdp_plot[col],
                name=col
            ))
        
        fig.update_layout(
            barmode='relative',
            xaxis_title='年度',
            yaxis_title='金額 (current US$)',
            title="GDP構成推移 (積み上げ棒グラフ)",
            legend_title='項目',
            yaxis=dict(tickformat=',d'),
            height=600
        )
        
        return fig
    except Exception as e:
        st.error(f"GDP構成グラフ作成エラー: {str(e)}")
        return go.Figure()

def create_industry_gdp_composition_chart(country_code, start_year=2000, end_year=2023):
    """産業別GDP構成（農業・工業・サービス）時系列積み上げ棒グラフ（元アプリと同じロジック）"""
    try:
        # 産業別指標コード（元アプリと同じ）
        industry_components = {
            '農業': 'NV.AGR.TOTL.ZS',
            '工業': 'NV.IND.TOTL.ZS',
            'サービス': 'NV.SRV.TETC.ZS'
        }
        
        # データ取得
        df_agri = fetch_world_bank_data([country_code], industry_components['農業'], start_year, end_year)
        df_ind = fetch_world_bank_data([country_code], industry_components['工業'], start_year, end_year)
        df_serv = fetch_world_bank_data([country_code], industry_components['サービス'], start_year, end_year)
        
        # データを時系列に変換
        agri_data = df_agri.set_index('year')['value'].sort_index() if df_agri is not None and not df_agri.empty else pd.Series(dtype=float)
        ind_data = df_ind.set_index('year')['value'].sort_index() if df_ind is not None and not df_ind.empty else pd.Series(dtype=float)
        serv_data = df_serv.set_index('year')['value'].sort_index() if df_serv is not None and not df_serv.empty else pd.Series(dtype=float)
        
        agri_data.name = '農業'
        ind_data.name = '工業'
        serv_data.name = 'サービス'
        
        # データ結合
        df_industry = pd.concat([agri_data, ind_data, serv_data], axis=1)
        
        # 常に「商業・サービス（推定）」として表示
        # サービスデータがある場合もそのまま「商業・サービス（推定）」として扱う
        if 'サービス' in df_industry.columns and not df_industry['サービス'].isnull().all():
            # サービスデータがある場合はそのまま使用
            df_industry['商業・サービス（推定）'] = df_industry['サービス']
        else:
            # サービスデータがない場合は農業・工業以外を計算
            df_industry['商業・サービス（推定）'] = 100 - df_industry[['農業', '工業']].sum(axis=1)
        
        plot_cols = ['農業', '工業', '商業・サービス（推定）']
        # 農業または工業が欠損している年は全て除外
        available_cols = [col for col in ['農業', '工業'] if col in df_industry.columns]
        if not available_cols:
            return go.Figure()
        df_ind_plot = df_industry[plot_cols].dropna(subset=available_cols, how='any')
        colors = ['#2ca02c', '#1f77b4', '#FFD700']  # 緑・青・黄色
        
        if df_ind_plot.empty:
            return go.Figure()
        
        fig2 = go.Figure()
        for col, color in zip(plot_cols, colors):
            if col in df_ind_plot.columns:
                fig2.add_trace(go.Bar(
                    x=df_ind_plot.index,
                    y=df_ind_plot[col],
                    name=col,
                    marker_color=color
                ))
        
        fig2.update_layout(
            barmode='relative',
            xaxis_title='年度',
            yaxis_title='GDP比 (%)',
            title="産業別GDP構成推移 (100%積み上げ棒グラフ)",
            legend_title='産業部門',
            yaxis=dict(range=[0, 100], tickformat=',d'),
            height=600
        )
        
        return fig2
    except Exception as e:
        st.error(f"産業別GDP構成グラフ作成エラー: {str(e)}")
        return go.Figure()

# キャッシュされたデータ取得関数
@st.cache_data(ttl=3600)
def fetch_world_bank_data(country_codes, indicator_code, start_year=2000, end_year=2023):
    """世界銀行APIからデータを取得"""
    try:
        # 合成指標の場合は特別処理
        if indicator_code == 'COMPOSITE_WORKING_AGE_PPP_GDP':
            return calculate_composite_working_age_ppp_gdp(country_codes, start_year, end_year)
        
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
        st.error(f"データ取得エラー: {str(e)}")
        return None

def multi_country_comparison_analysis():
    """多国間比較分析"""
    st.header("📈 多国間比較分析")
    
    # 共通設定
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.selectbox("開始年", range(2000, 2024), index=10)
    with col2:
        end_year = st.selectbox("終了年", range(start_year, 2024), index=len(range(start_year, 2024))-1)
    
    # 指標選択
    selected_indicator = st.selectbox("分析指標を選択", list(INDICATORS.keys()))
    indicator_code = INDICATORS[selected_indicator]
    
    # 国選択
    st.subheader("比較対象国の選択")
    
    # 中米9か国
    st.write("**中米9か国**")
    central_america_selected = []
    cols = st.columns(3)
    for i, country in enumerate(CENTRAL_AMERICA_COUNTRIES.keys()):
        with cols[i % 3]:
            if st.checkbox(country, key=f"ca_{country}"):
                central_america_selected.append(CENTRAL_AMERICA_COUNTRIES[country])
    
    # 主要9か国
    st.write("**主要9か国**")
    major_countries_selected = []
    cols = st.columns(3)
    for i, country in enumerate(MAJOR_COUNTRIES.keys()):
        with cols[i % 3]:
            if st.checkbox(country, key=f"major_{country}"):
                major_countries_selected.append(MAJOR_COUNTRIES[country])
    
    # 地域平均
    st.write("**地域・所得分類平均**")
    regional_selected = []
    cols = st.columns(2)
    for i, (region_name, region_code) in enumerate(REGIONAL_AGGREGATES.items()):
        with cols[i % 2]:
            if st.checkbox(region_name, key=f"region_{region_code}"):
                regional_selected.append(region_code)
    
    all_selected_countries = central_america_selected + major_countries_selected + regional_selected
    
    if not all_selected_countries:
        st.warning("比較する国を選択してください。")
        return
    
    # 分析ボタン
    if not st.button("📈 分析実行", key="multi_country_analyze"):
        st.info("上記の設定を完了したら、「分析実行」ボタンを押してください。")
        return
    
    # データ取得と表示
    with st.spinner("データを取得中..."):
        df = fetch_world_bank_data(all_selected_countries, indicator_code, start_year, end_year)
        
        if df is None or df.empty:
            st.error("データを取得できませんでした。")
            st.write(f"デバッグ情報: 選択された国: {all_selected_countries}, 指標: {indicator_code}")
            return
    
    # 線グラフで表示
    fig = go.Figure()
    
    # 色のリストを定義
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    color_index = 0
    
    # 実際に利用可能な国コードを使用
    available_countries = df['country_code'].unique().tolist()
    
    for country_code in available_countries:
        country_data = df[df['country_code'] == country_code]
        if not country_data.empty:
            country_name = country_data['country'].iloc[0]
            
            # グアテマラを赤色、地域平均を太い緑色、他を順番に別色で表示
            if country_code == 'GT':
                color = 'red'
                line_width = 3
                marker_size = 8
            elif country_code in ['MIC', 'LCN']:
                color = 'darkgreen'
                line_width = 3
                marker_size = 6
            else:
                color = colors[color_index % len(colors)]
                color_index += 1
                line_width = 2
                marker_size = 6
            
            # データをソートして追加
            country_data_sorted = country_data.sort_values('year')
            
            fig.add_trace(go.Scatter(
                x=country_data_sorted['year'],
                y=country_data_sorted['value'],
                mode='lines+markers',
                name=country_name,
                line=dict(color=color, width=line_width),
                marker=dict(size=marker_size)
            ))
    
    # グラフにデータが追加されているかチェック
    if len(fig.data) == 0:
        st.error("選択された国または指標のデータが見つかりませんでした。別の組み合わせをお試しください。")
        return
    
    fig.update_layout(
        title=f"{selected_indicator}の推移 ({start_year}-{end_year})",
        xaxis_title="年",
        yaxis_title=selected_indicator,
        height=600,
        hovermode='x unified',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # データテーブル表示
    if st.checkbox("データテーブルを表示"):
        if not df.empty:
            pivot_df = df.pivot(index='year', columns='country', values='value')
            st.dataframe(pivot_df)
        else:
            st.warning("表示するデータがありません。")

def single_country_detailed_analysis():
    """単一国詳細分析"""
    st.header("🏛️ 単一国詳細分析")
    
    # 共通設定
    col1, col2, col3 = st.columns(3)
    with col1:
        start_year = st.selectbox("開始年", range(2000, 2024), index=10, key="single_start")
    with col2:
        end_year = st.selectbox("終了年", range(start_year, 2024), index=len(range(start_year, 2024))-1, key="single_end")
    with col3:
        selected_country = st.selectbox("分析対象国", list(CENTRAL_AMERICA_COUNTRIES.keys()))
    
    country_code = CENTRAL_AMERICA_COUNTRIES[selected_country]
    
    # 分析パッケージ選択
    analysis_package = st.selectbox("分析パッケージ", [
        "基本経済プロファイル",
        "社会発展指標", 
        "持続可能性評価",
        "技術・イノベーション",
        "貿易・国際化"
    ])
    
    # 各分析パッケージの指標定義
    package_indicators = {
        "基本経済プロファイル": {
            'GDP成長率（%）': 'NY.GDP.MKTP.KD.ZG',
            '一人当たりGDP（名目）': 'NY.GDP.PCAP.CD',
            '一人当たりGDP（実質2015USD）': 'NY.GDP.PCAP.KD',
            'インフレ率（%）': 'FP.CPI.TOTL.ZG',
            '失業率（%）': 'SL.UEM.TOTL.ZS',
            '政府支出（GDP比%）': 'NE.CON.GOVT.ZS'
        },
        "社会発展指標": {
            '平均寿命': 'SP.DYN.LE00.IN',
            '識字率（%）': 'SE.ADT.LITR.ZS',
            '貧困率（%）': 'SI.POV.NAHC',
            '所得格差': 'SI.POV.GINI',
            '教育支出（GDP比%）': 'SE.XPD.TOTL.GD.ZS'
        },
        "持続可能性評価": {
            'CO2排出量': 'EN.ATM.CO2E.PC',
            '再生可能エネルギー（%）': 'EG.FEC.RNEW.ZS',
            '電力普及率（%）': 'EG.ELC.ACCS.ZS',
            '森林面積（%）': 'AG.LND.FRST.ZS',
            '水資源アクセス（%）': 'SH.H2O.BASW.ZS'
        },
        "技術・イノベーション": {
            'インターネット普及率（%）': 'IT.NET.USER.ZS',
            '研究開発費（%）': 'GB.XPD.RSDV.GD.ZS',
            '高技術製品輸出（%）': 'TX.VAL.TECH.MF.ZS',
            'モバイル普及率': 'IT.CEL.SETS.P2',
            '特許申請': 'IP.PAT.RESD'
        },
        "貿易・国際化": {
            '輸出（GDP比%）': 'NE.EXP.GNFS.ZS',
            '輸入（GDP比%）': 'NE.IMP.GNFS.ZS',
            '外国直接投資GDP比（%）': 'BX.KLT.DINV.WD.GD.ZS',
            '貿易収支（GDP比%）': 'NE.RSB.GNFS.ZS',
            '観光収入': 'ST.INT.RCPT.CD'
        }
    }
    
    indicators = package_indicators[analysis_package]
    
    # 分析ボタン
    if not st.button("📈 分析実行", key="single_country_analyze"):
        st.info("上記の設定を完了したら、「分析実行」ボタンを押してください。")
        return
    
    # データ取得と表示
    st.subheader(f"{selected_country}の{analysis_package}")
    
    # 基本経済プロファイルの場合は特別な表示
    if analysis_package == "基本経済プロファイル":
        # 人口推移グラフ
        st.subheader("人口推移分析")
        pop_chart = create_population_trend_chart(country_code, start_year, end_year)
        if pop_chart.data:
            st.plotly_chart(pop_chart, use_container_width=True)
        else:
            st.warning("人口データが利用できません")
        
        # GDP構成分析
        st.subheader("GDP構成分析")
        gdp_chart = create_gdp_composition_chart(country_code, start_year, end_year)
        if gdp_chart.data:
            st.plotly_chart(gdp_chart, use_container_width=True)
        else:
            st.warning("GDP構成データが利用できません")
        
        # 産業別GDP構成分析
        st.subheader("産業別GDP構成分析")
        industry_chart = create_industry_gdp_composition_chart(country_code, start_year, end_year)
        if industry_chart.data:
            st.plotly_chart(industry_chart, use_container_width=True)
        else:
            st.warning("産業別GDPデータが利用できません")
        
        # 基本指標の表示
        st.subheader("基本経済指標")
        cols = st.columns(2)
        
        for i, (indicator_name, indicator_code) in enumerate(indicators.items()):
            with cols[i % 2]:
                with st.spinner(f"{indicator_name}のデータを取得中..."):
                    df = fetch_world_bank_data([country_code], indicator_code, start_year, end_year)
                    
                    if df is not None and not df.empty:
                        # 簡単な統計情報
                        latest_value = df['value'].iloc[-1] if len(df) > 0 else None
                        if latest_value is not None:
                            st.metric(indicator_name, f"{latest_value:.2f}")
                        
                        # 小さなグラフ
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df['year'], 
                            y=df['value'],
                            mode='lines+markers',
                            name=indicator_name,
                            line=dict(color='red', width=2)
                        ))
                        fig.update_layout(
                            title=indicator_name,
                            height=300,
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"{indicator_name}: データが利用できません")
    
    else:
        # 他の分析パッケージは従来通り
        cols = st.columns(2)
        
        for i, (indicator_name, indicator_code) in enumerate(indicators.items()):
            with cols[i % 2]:
                with st.spinner(f"{indicator_name}のデータを取得中..."):
                    df = fetch_world_bank_data([country_code], indicator_code, start_year, end_year)
                    
                    if df is not None and not df.empty:
                        # 簡単な統計情報
                        latest_value = df['value'].iloc[-1] if len(df) > 0 else None
                        if latest_value is not None:
                            st.metric(indicator_name, f"{latest_value:.2f}")
                        
                        # 小さなグラフ
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df['year'], 
                            y=df['value'],
                            mode='lines+markers',
                            name=indicator_name,
                            line=dict(color='red', width=2)
                        ))
                        fig.update_layout(
                            title=indicator_name,
                            height=300,
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"{indicator_name}: データが利用できません")

def composite_indicator_analysis():
    """合成指標分析"""
    st.header("📊 合成指標分析")
    
    # 共通設定
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.selectbox("開始年", range(2000, 2024), index=15, key="composite_start")
    with col2:
        end_year = st.selectbox("終了年", range(start_year, 2024), index=len(range(start_year, 2024))-1, key="composite_end")
    
    # 合成指標選択
    composite_indicators = [
        "経済発展指数",
        "持続可能性指数", 
        "社会包摂指数",
        "技術革新指数",
        "経済安定性指数",
        "貿易競争力指数",
        "投融資活性化指数",
        "構造調整型マクロ安定指数"
    ]
    
    selected_composite = st.selectbox("合成指標を選択", composite_indicators)
    
    # 地域フィルター
    region_filter = st.selectbox("地域・国フィルター", [
        "中米9か国",
        "中南米・カリブ33か国", 
        "世銀Middle Income諸国全体"
    ])
    
    # 地域に応じた国の選択
    if region_filter == "中米9か国":
        countries_dict = CENTRAL_AMERICA_COUNTRIES
    elif region_filter == "中南米・カリブ33か国":
        countries_dict = LATIN_AMERICA_CARIBBEAN
    else:  # 世銀Middle Income諸国全体
        # World Bank Middle Income諸国の実際の国リスト
        countries_dict = {
            'Albania': 'AL', 'Algeria': 'DZ', 'American Samoa': 'AS', 'Argentina': 'AR', 'Armenia': 'AM',
            'Azerbaijan': 'AZ', 'Belarus': 'BY', 'Belize': 'BZ', 'Bosnia and Herzegovina': 'BA', 'Botswana': 'BW',
            'Brazil': 'BR', 'Bulgaria': 'BG', 'China': 'CN', 'Colombia': 'CO', 'Costa Rica': 'CR',
            'Croatia': 'HR', 'Cuba': 'CU', 'Dominica': 'DM', 'Dominican Republic': 'DO', 'Ecuador': 'EC',
            'Egypt, Arab Rep.': 'EG', 'El Salvador': 'SV', 'Equatorial Guinea': 'GQ', 'Fiji': 'FJ', 'Gabon': 'GA',
            'Georgia': 'GE', 'Grenada': 'GD', 'Guatemala': 'GT', 'Guyana': 'GY', 'Honduras': 'HN',
            'Hungary': 'HU', 'India': 'IN', 'Indonesia': 'ID', 'Iran, Islamic Rep.': 'IR', 'Iraq': 'IQ',
            'Jamaica': 'JM', 'Jordan': 'JO', 'Kazakhstan': 'KZ', 'Kosovo': 'XK', 'Lebanon': 'LB',
            'Libya': 'LY', 'Malaysia': 'MY', 'Maldives': 'MV', 'Marshall Islands': 'MH', 'Mauritius': 'MU',
            'Mexico': 'MX', 'Micronesia, Fed. Sts.': 'FM', 'Moldova': 'MD', 'Montenegro': 'ME', 'Namibia': 'NA',
            'Nicaragua': 'NI', 'North Macedonia': 'MK', 'Pakistan': 'PK', 'Palau': 'PW', 'Panama': 'PA',
            'Paraguay': 'PY', 'Peru': 'PE', 'Philippines': 'PH', 'Poland': 'PL', 'Romania': 'RO',
            'Russian Federation': 'RU', 'Samoa': 'WS', 'Serbia': 'RS', 'South Africa': 'ZA', 'Sri Lanka': 'LK',
            'St. Lucia': 'LC', 'St. Vincent and the Grenadines': 'VC', 'Suriname': 'SR', 'Thailand': 'TH',
            'Tonga': 'TO', 'Turkey': 'TR', 'Turkmenistan': 'TM', 'Tuvalu': 'TV', 'Ukraine': 'UA',
            'Venezuela, RB': 'VE'
        }
    
    # 表示形式選択
    display_type = st.selectbox("表示形式", ["偏差値", "Zスコア", "元の値"])
    
    # 合成指標の説明（構成比率付き）
    st.info(f"**{selected_composite}の構成要素と比率:**")
    composite_components = {
        "経済発展指数": [('GDP成長率（%） (NY.GDP.MKTP.KD.ZG)', 40, 1), ('一人当たりGDP（名目） (NY.GDP.PCAP.CD)', 40, 1), ('政府支出（GDP比%） (NE.CON.GOVT.ZS)', 20, 1)],
        "持続可能性指数": [('CO2排出量 (EN.ATM.CO2E.PC)', 40, -1), ('再生可能エネルギー（%） (EG.FEC.RNEW.ZS)', 35, 1), ('森林面積（%） (AG.LND.FRST.ZS)', 25, 1)],
        "社会包摂指数": [('所得格差 (SI.POV.GINI)', 35, -1), ('識字率（%） (SE.ADT.LITR.ZS)', 35, 1), ('平均寿命 (SP.DYN.LE00.IN)', 30, 1)],
        "技術革新指数": [('インターネット普及率（%） (IT.NET.USER.ZS)', 30, 1), ('研究開発費（%） (GB.XPD.RSDV.GD.ZS)', 40, 1), ('高技術製品輸出（%） (TX.VAL.TECH.MF.ZS)', 30, 1)],
        "経済安定性指数": [('インフレ率（%） (FP.CPI.TOTL.ZG)', 40, -1), ('失業率（%） (SL.UEM.TOTL.ZS)', 35, -1), ('貿易収支（GDP比%） (NE.RSB.GNFS.ZS)', 25, 1)],
        "貿易競争力指数": [('輸出（GDP比%） (NE.EXP.GNFS.ZS)', 30), ('高技術製品輸出（%） (TX.VAL.TECH.MF.ZS)', 40), ('外国直接投資GDP比（%） (BX.KLT.DINV.WD.GD.ZS)', 30)],
        "投融資活性化指数": [('外国直接投資（USD） (BX.KLT.DINV.CD.WD)', 40, 1), ('金融深化度（%） (FS.AST.PRVT.GD.ZS)', 35, 1), ('金融口座保有率（%） (FX.OWN.TOTL.ZS)', 25, 1)],
        "構造調整型マクロ安定指数": [('GDP成長率（%） (NY.GDP.MKTP.KD.ZG)', 30, 1), ('インフレ率（%） (FP.CPI.TOTL.ZG)', 20, -1), ('対外債務残高（GNI比%） (DT.DOD.DECT.GN.ZS)', 20, -1), ('財政収支 (GC.BAL.CASH.GD.ZS)', 30, 1)]
    }
    component_list = composite_components.get(selected_composite, [])
    for component, weight, direction in component_list:
        direction_str = "+" if direction == 1 else "-"
        st.write(f"• {component} {direction_str}{weight}%")
    
    # 分析ボタン
    if not st.button("📊 分析実行", key="composite_analyze"):
        st.info("上記の設定を完了したら、「分析実行」ボタンを押してください。")
        return
    
    # 合成指標の構成要素（実際のWorld Bank指標コード）
    composite_indicator_codes = {
        "経済発展指数": [('NY.GDP.MKTP.KD.ZG', 40, 1), ('NY.GDP.PCAP.CD', 40, 1), ('NE.CON.GOVT.ZS', 20, 1)],
        "持続可能性指数": [('EN.ATM.CO2E.PC', 40, -1), ('EG.FEC.RNEW.ZS', 35, 1), ('AG.LND.FRST.ZS', 25, 1)],
        "社会包摂指数": [('SI.POV.GINI', 35, -1), ('SE.ADT.LITR.ZS', 35, 1), ('SP.DYN.LE00.IN', 30, 1)],
        "技術革新指数": [('IT.NET.USER.ZS', 30, 1), ('GB.XPD.RSDV.GD.ZS', 40, 1), ('TX.VAL.TECH.MF.ZS', 30, 1)],
        "経済安定性指数": [('FP.CPI.TOTL.ZG', 40, -1), ('SL.UEM.TOTL.ZS', 35, -1), ('NE.RSB.GNFS.ZS', 25, 1)],
        "貿易競争力指数": [('NE.EXP.GNFS.ZS', 30, 1), ('TX.VAL.TECH.MF.ZS', 40, 1), ('BX.KLT.DINV.WD.GD.ZS', 30, 1)],
        "投融資活性化指数": [('BX.KLT.DINV.CD.WD', 40, 1), ('FS.AST.PRVT.GD.ZS', 35, 1), ('FX.OWN.TOTL.ZS', 25, 1)],
        "構造調整型マクロ安定指数": [('NY.GDP.MKTP.KD.ZG', 30, 1), ('FP.CPI.TOTL.ZG', 20, -1), ('DT.DOD.DECT.GN.ZS', 20, -1), ('GC.BAL.CASH.GD.ZS', 30, 1)]
    }
    
    components = composite_indicator_codes.get(selected_composite, [('NY.GDP.MKTP.KD.ZG', 100, 1)])
    
    # データ取得と合成指標計算
    with st.spinner("データを計算中..."):
        composite_data = []
        country_codes = list(countries_dict.values())
        
        for country_code in country_codes:
            country_values = []
            
            for component, weight, direction in components:
                df = fetch_world_bank_data([country_code], component, start_year, end_year)
                if df is not None and not df.empty:
                    # 指定された期間の平均値を使用（年による変動を考慮）
                    avg_value = df['value'].mean()
                    # 方向性を考慮（小さいほど良い指標は逆数で処理）
                    if direction == -1 and not np.isnan(avg_value):
                        avg_value = -avg_value
                    country_values.append(avg_value)
                else:
                    country_values.append(np.nan)
            
            # 欠損値でない場合のみ合成指標を計算
            if not all(np.isnan(country_values)):
                # 構成比率を考慮した加重平均による合成指標計算
                component_list = composite_components.get(selected_composite, [])
                if len(component_list) == len(country_values):
                    weights = [weight/100 for _, weight, _ in component_list]  # パーセントを小数に変換
                    # 加重平均を計算（欠損値を除外）
                    valid_indices = [i for i, val in enumerate(country_values) if not np.isnan(val)]
                    if valid_indices:
                        valid_values = [country_values[i] for i in valid_indices]
                        valid_weights = [weights[i] for i in valid_indices]
                        # 重みを再正規化
                        total_weight = sum(valid_weights)
                        if total_weight > 0:
                            normalized_weights = [w/total_weight for w in valid_weights]
                            composite_value = sum(val * weight for val, weight in zip(valid_values, normalized_weights))
                        else:
                            composite_value = np.nanmean(country_values)
                    else:
                        composite_value = np.nan
                else:
                    # フォールバック：単純平均
                    composite_value = np.nanmean(country_values)
                
                # 国名を英語で取得（APIから）
                sample_df = fetch_world_bank_data([country_code], 'SP.POP.TOTL', end_year, end_year)
                if sample_df is not None and not sample_df.empty:
                    english_country_name = sample_df['country'].iloc[0]
                else:
                    english_country_name = country_code  # フォールバック
                
                composite_data.append({
                    'country': english_country_name,
                    'country_code': country_code,
                    'composite_value': composite_value
                })
        
        if not composite_data:
            st.error("データを取得できませんでした。")
            return
        
        composite_df = pd.DataFrame(composite_data)
        
        # 表示形式に応じてデータを変換
        if display_type == "偏差値":
            mean_val = composite_df['composite_value'].mean()
            std_val = composite_df['composite_value'].std()
            composite_df['display_value'] = 50 + (composite_df['composite_value'] - mean_val) / std_val * 10
            y_title = f"{selected_composite} (偏差値)"
        elif display_type == "Zスコア":
            composite_df['display_value'] = stats.zscore(composite_df['composite_value'])
            y_title = f"{selected_composite} (Zスコア)"
        else:
            composite_df['display_value'] = composite_df['composite_value']
            y_title = selected_composite
        
        # データを値の降順でソート（最高値が左から順に表示）
        composite_df = composite_df.sort_values('display_value', ascending=False)
        
        # 棒グラフで表示
        fig = go.Figure()
        
        colors = ['red' if code == 'GT' else 'blue' for code in composite_df['country_code']]
        
        fig.add_trace(go.Bar(
            x=composite_df['country'],
            y=composite_df['display_value'],
            marker_color=colors,
            name=selected_composite
        ))
        
        fig.update_layout(
            title=f"{selected_composite} - {region_filter}",
            xaxis_title="国",
            yaxis_title=y_title,
            height=600,
            font=dict(color='black', size=12),
            xaxis=dict(tickangle=45, tickfont=dict(color='black', size=12)),
            yaxis=dict(tickfont=dict(color='black', size=12))
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # データテーブル表示
        if st.checkbox("データテーブルを表示", key="composite_table"):
            display_df = composite_df[['country', 'display_value']].copy()
            st.dataframe(display_df.sort_values('display_value', ascending=False))

def sdgs_achievement_analysis():
    """SDGs目標達成度分析"""
    st.header("🎯 SDGs目標達成度分析")
    
    # 共通設定
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.selectbox("開始年", range(2000, 2024), index=15, key="sdgs_start")
    with col2:
        end_year = st.selectbox("終了年", range(start_year, 2024), index=len(range(start_year, 2024))-1, key="sdgs_end")
    
    # SDGs指標選択
    sdgs_indicators = [
        "経済的貧困度緩和指数",
        "保健指数",
        "教育充実度指数",
        "ジェンダー代表性指数",
        "気候レジリエンス指数",
        "社会安定性指数"
    ]
    
    selected_sdgs = st.selectbox("SDGs指標を選択", sdgs_indicators)
    
    # 地域フィルター
    region_filter = st.selectbox("地域・国フィルター", [
        "中米9か国",
        "中南米・カリブ33か国",
        "世銀Middle Income諸国全体"
    ], key="sdgs_region")
    
    # 地域に応じた国の選択
    if region_filter == "中米9か国":
        countries_dict = CENTRAL_AMERICA_COUNTRIES
    elif region_filter == "中南米・カリブ33か国":
        countries_dict = LATIN_AMERICA_CARIBBEAN
    else:  # 世銀Middle Income諸国全体
        # World Bank Middle Income諸国の実際の国リスト
        countries_dict = {
            'Albania': 'AL', 'Algeria': 'DZ', 'American Samoa': 'AS', 'Argentina': 'AR', 'Armenia': 'AM',
            'Azerbaijan': 'AZ', 'Belarus': 'BY', 'Belize': 'BZ', 'Bosnia and Herzegovina': 'BA', 'Botswana': 'BW',
            'Brazil': 'BR', 'Bulgaria': 'BG', 'China': 'CN', 'Colombia': 'CO', 'Costa Rica': 'CR',
            'Croatia': 'HR', 'Cuba': 'CU', 'Dominica': 'DM', 'Dominican Republic': 'DO', 'Ecuador': 'EC',
            'Egypt, Arab Rep.': 'EG', 'El Salvador': 'SV', 'Equatorial Guinea': 'GQ', 'Fiji': 'FJ', 'Gabon': 'GA',
            'Georgia': 'GE', 'Grenada': 'GD', 'Guatemala': 'GT', 'Guyana': 'GY', 'Honduras': 'HN',
            'Hungary': 'HU', 'India': 'IN', 'Indonesia': 'ID', 'Iran, Islamic Rep.': 'IR', 'Iraq': 'IQ',
            'Jamaica': 'JM', 'Jordan': 'JO', 'Kazakhstan': 'KZ', 'Kosovo': 'XK', 'Lebanon': 'LB',
            'Libya': 'LY', 'Malaysia': 'MY', 'Maldives': 'MV', 'Marshall Islands': 'MH', 'Mauritius': 'MU',
            'Mexico': 'MX', 'Micronesia, Fed. Sts.': 'FM', 'Moldova': 'MD', 'Montenegro': 'ME', 'Namibia': 'NA',
            'Nicaragua': 'NI', 'North Macedonia': 'MK', 'Pakistan': 'PK', 'Palau': 'PW', 'Panama': 'PA',
            'Paraguay': 'PY', 'Peru': 'PE', 'Philippines': 'PH', 'Poland': 'PL', 'Romania': 'RO',
            'Russian Federation': 'RU', 'Samoa': 'WS', 'Serbia': 'RS', 'South Africa': 'ZA', 'Sri Lanka': 'LK',
            'St. Lucia': 'LC', 'St. Vincent and the Grenadines': 'VC', 'Suriname': 'SR', 'Thailand': 'TH',
            'Tonga': 'TO', 'Turkey': 'TR', 'Turkmenistan': 'TM', 'Tuvalu': 'TV', 'Ukraine': 'UA',
            'Venezuela, RB': 'VE'
        }
    
    # 表示形式選択
    display_type = st.selectbox("表示形式", ["偏差値", "Zスコア", "元の値"], key="sdgs_display")
    
    # SDGs指標の構成要素説明（構成比率と方向性付き）
    sdgs_goals = {
        "経済的貧困度緩和指数": "SDGs1 貧困削減",
        "保健指数": "SDGs3 保健・福祉",
        "教育充実度指数": "SDGs4 質の高い教育",
        "ジェンダー代表性指数": "SDGs5 ジェンダー平等",
        "気候レジリエンス指数": "SDGs13 気候変動",
        "社会安定性指数": "SDGs16 平和・公正"
    }
    
    selected_goal = sdgs_goals.get(selected_sdgs, "")
    st.info(f"**{selected_sdgs} ({selected_goal})**")
    st.info(f"**構成要素と比率:**")
    
    sdgs_display_components = {
        "経済的貧困度緩和指数": [('失業率 (SL.UEM.TOTL.ZS)', 40, -1), ('インフレ率 (FP.CPI.TOTL.ZG)', 20, -1), ('GDP成長率 (NY.GDP.MKTP.KD.ZG)', 20, 1), ('貧困率 (SI.POV.NAHC)', 20, -1)],
        "保健指数": [('平均寿命 (SP.DYN.LE00.IN)', 35, 1), ('安全な水アクセス (SH.H2O.BASW.ZS)', 35, 1), ('乳児死亡率 (SP.DYN.IMRT.IN)', 30, -1)],
        "教育充実度指数": [('識字率 (SE.ADT.LITR.ZS)', 40, 1), ('初等教育就学率 (SE.PRM.NENR)', 35, 1), ('中等教育就学率 (SE.SEC.NENR)', 30, 1)],
        "ジェンダー代表性指数": [('女性国会議員比率 (SG.GEN.PARL.ZS)', 30, 1), ('女性労働参加率 (SL.TLF.CACT.FE.ZS)', 35, 1), ('中等教育の男女比率 (SE.ENR.SECO.FM.ZS)', 35, 1)],
        "気候レジリエンス指数": [('CO2排出量 (EN.ATM.CO2E.PC)', 40, -1), ('森林面積率 (AG.LND.FRST.ZS)', 40, 1), ('エネルギー使用量 (EG.USE.PCAP.KG.OE)', 20, -1)],
        "社会安定性指数": [('故意の殺人率 (VC.IHR.PSRC.P5)', 25, -1), ('統計パフォーマンス指数 (IQ.SCI.PRDC)', 25, 1), ('教育支出（GDP比） (SE.XPD.TOTL.GD.ZS)', 25, 1), ('保健支出（GDP比） (SH.XPD.CHEX.GD.ZS)', 25, 1)]
    }
    component_list = sdgs_display_components.get(selected_sdgs, [])
    for component, weight, direction in component_list:
        direction_str = "+" if direction == 1 else "-"
        st.write(f"• {component} {direction_str}{weight}%")
    
    # 分析ボタン
    if not st.button("🎯 分析実行", key="sdgs_analyze"):
        st.info("上記の設定を完了したら、「分析実行」ボタンを押してください。")
        return
    
    # SDGs指標の構成要素（方向性考慮）
    sdgs_components = {
        "経済的貧困度緩和指数": [('SL.UEM.TOTL.ZS', 40, -1), ('FP.CPI.TOTL.ZG', 20, -1), ('NY.GDP.MKTP.KD.ZG', 20, 1), ('SI.POV.NAHC', 20, -1)],
        "保健指数": [('SP.DYN.LE00.IN', 35, 1), ('SH.H2O.BASW.ZS', 35, 1), ('SP.DYN.IMRT.IN', 30, -1)],
        "教育充実度指数": [('SE.ADT.LITR.ZS', 40, 1), ('SE.PRM.NENR', 35, 1), ('SE.SEC.NENR', 30, 1)],
        "ジェンダー代表性指数": [('SG.GEN.PARL.ZS', 30, 1), ('SL.TLF.CACT.FE.ZS', 35, 1), ('SE.ENR.SECO.FM.ZS', 35, 1)],
        "気候レジリエンス指数": [('EN.ATM.CO2E.PC', 40, -1), ('AG.LND.FRST.ZS', 40, 1), ('EG.USE.PCAP.KG.OE', 20, -1)],
        "社会安定性指数": [('VC.IHR.PSRC.P5', 25, -1), ('IQ.SCI.PRDC', 25, 1), ('SE.XPD.TOTL.GD.ZS', 25, 1), ('SH.XPD.CHEX.GD.ZS', 25, 1)]
    }
    
    components = sdgs_components.get(selected_sdgs, [('SP.DYN.LE00.IN', 100, 1)])
    
    # データ取得とSDGs指標計算
    with st.spinner("SDGsデータを計算中..."):
        sdgs_data = []
        country_codes = list(countries_dict.values())
        
        for country_code in country_codes:
            country_values = []
            
            for component, weight, direction in components:
                df = fetch_world_bank_data([country_code], component, start_year, end_year)
                if df is not None and not df.empty:
                    # 指定された期間の平均値を使用（年による変動を考慮）
                    avg_value = df['value'].mean()
                    # 方向性を考慮（小さいほど良い指標は逆数で処理）
                    if direction == -1 and not np.isnan(avg_value):
                        avg_value = -avg_value
                    country_values.append(avg_value)
                else:
                    country_values.append(np.nan)
            
            if not all(np.isnan(country_values)):
                # 構成比率を考慮した加重平均によるSDGs指標計算
                component_list = sdgs_display_components.get(selected_sdgs, [])
                if len(component_list) == len(country_values):
                    weights = [weight/100 for _, weight, _ in component_list]  # パーセントを小数に変換
                    # 加重平均を計算（欠損値を除外）
                    valid_indices = [i for i, val in enumerate(country_values) if not np.isnan(val)]
                    if valid_indices:
                        valid_values = [country_values[i] for i in valid_indices]
                        valid_weights = [weights[i] for i in valid_indices]
                        # 重みを再正規化
                        total_weight = sum(valid_weights)
                        if total_weight > 0:
                            normalized_weights = [w/total_weight for w in valid_weights]
                            sdgs_value = sum(val * weight for val, weight in zip(valid_values, normalized_weights))
                        else:
                            sdgs_value = np.nanmean(country_values)
                    else:
                        sdgs_value = np.nan
                else:
                    # フォールバック：単純平均
                    sdgs_value = np.nanmean(country_values)
                
                # 国名を英語で取得（APIから）
                sample_df = fetch_world_bank_data([country_code], 'SP.POP.TOTL', end_year, end_year)
                if sample_df is not None and not sample_df.empty:
                    english_country_name = sample_df['country'].iloc[0]
                else:
                    english_country_name = country_code  # フォールバック
                
                sdgs_data.append({
                    'country': english_country_name,
                    'country_code': country_code,
                    'sdgs_value': sdgs_value
                })
        
        if not sdgs_data:
            st.error("データを取得できませんでした。")
            return
        
        sdgs_df = pd.DataFrame(sdgs_data)
        
        # 表示形式に応じてデータを変換
        if display_type == "偏差値":
            mean_val = sdgs_df['sdgs_value'].mean()
            std_val = sdgs_df['sdgs_value'].std()
            sdgs_df['display_value'] = 50 + (sdgs_df['sdgs_value'] - mean_val) / std_val * 10
            y_title = f"{selected_sdgs} (偏差値)"
        elif display_type == "Zスコア":
            sdgs_df['display_value'] = stats.zscore(sdgs_df['sdgs_value'])
            y_title = f"{selected_sdgs} (Zスコア)"
        else:
            sdgs_df['display_value'] = sdgs_df['sdgs_value']
            y_title = selected_sdgs
        
        # データを値の降順でソート（最高値が左から順に表示）
        sdgs_df = sdgs_df.sort_values('display_value', ascending=False)
        
        # 棒グラフで表示
        fig = go.Figure()
        
        colors = ['red' if code == 'GT' else 'blue' for code in sdgs_df['country_code']]
        
        fig.add_trace(go.Bar(
            x=sdgs_df['country'],
            y=sdgs_df['display_value'],
            marker_color=colors,
            name=selected_sdgs
        ))
        
        fig.update_layout(
            title=f"{selected_sdgs} - {region_filter}",
            xaxis_title="国",
            yaxis_title=y_title,
            height=600,
            font=dict(color='black', size=12),
            xaxis=dict(tickangle=45, tickfont=dict(color='black', size=12)),
            yaxis=dict(tickfont=dict(color='black', size=12))
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # データテーブル表示
        if st.checkbox("データテーブルを表示", key="sdgs_table"):
            display_df = sdgs_df[['country', 'display_value']].copy()
            st.dataframe(display_df.sort_values('display_value', ascending=False))

def pca_analysis():
    """主成分分析（PCA）- 中米9か国限定"""
    st.header("🔄 主成分分析（PCA）")
    
    # 共通設定
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.selectbox("開始年", range(2000, 2024), index=15, key="pca_start")
    with col2:
        end_year = st.selectbox("終了年", range(start_year, 2024), index=len(range(start_year, 2024))-1, key="pca_end")
    
    # 指標選択（元アプリと同様）
    selected_indicators = st.multiselect(
        "分析指標を選択（複数選択可）",
        list(INDICATORS.keys()),
        default=['GDP成長率（%）', 'インフレ率（%）', '教育支出（GDP比%）', '平均寿命']
    )
    
    if len(selected_indicators) < 2:
        st.warning("2つ以上の指標を選択してください。")
        return
    
    # 中米9か国のデータを取得
    selected_countries = list(CENTRAL_AMERICA_COUNTRIES.values())
    
    st.info("**注：各特徴量についてはStandardScalerによる標準化を行っています。**")
    
    # 分析ボタン
    if not st.button("🔄 分析実行", key="pca_analyze"):
        st.info("上記の設定を完了したら、「分析実行」ボタンを押してください。")
        return
    
    with st.spinner("PCA分析を実行中..."):
        # データ収集（全年度を使用）
        data_rows = []
        
        for country_code in selected_countries:
            # 各指標のデータを取得
            indicator_dfs = {}
            all_indicators_available = True
            
            for indicator in selected_indicators:
                indicator_code = INDICATORS[indicator]
                df = fetch_world_bank_data([country_code], indicator_code, start_year, end_year)
                if df is not None and not df.empty:
                    indicator_dfs[indicator] = df.set_index('year')['value']
                else:
                    all_indicators_available = False
                    break
            
            if not all_indicators_available:
                continue
                
            # 年ごとにデータポイントを作成
            for year in range(start_year, end_year + 1):
                year_data = {'country_code': country_code, 'year': year}
                has_all_year_indicators = True
                
                for indicator in selected_indicators:
                    if year in indicator_dfs[indicator].index and not pd.isna(indicator_dfs[indicator][year]):
                        year_data[indicator] = indicator_dfs[indicator][year]
                    else:
                        has_all_year_indicators = False
                        break
                
                if has_all_year_indicators:
                    data_rows.append(year_data)
        
        if len(data_rows) < 3:
            st.error("PCA分析に必要な最低限のデータが不足しています。")
            return
        
        # データフレーム作成
        pca_df = pd.DataFrame(data_rows)
        
        # 特徴量行列を作成
        feature_columns = [col for col in pca_df.columns if col not in ['country_code', 'year']]
        data_matrix = pca_df[feature_columns].values
        
        # 国名リストを作成（表示用）
        country_year_labels = [f"{[k for k, v in CENTRAL_AMERICA_COUNTRIES.items() if v == row['country_code']][0]}_{row['year']}" 
                              for _, row in pca_df.iterrows()]
        
        if len(data_matrix) < 3:
            st.error("PCA分析に必要な最低限のデータが不足しています。")
            return
        
        data_matrix = np.array(data_matrix)
        
        # 標準化
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_matrix)
        
        # PCA実行
        pca = PCA(n_components=2)
        pca_transformed = pca.fit_transform(scaled_data)
        
        # PCA結果をデータフレームに追加
        pca_results_df = pca_df.copy()
        pca_results_df['component_1'] = pca_transformed[:, 0]
        pca_results_df['component_2'] = pca_transformed[:, 1]
        
        # 英語国名対応表（2文字コードに修正）
        country_english_names = {
            'GT': 'Guatemala',
            'HN': 'Honduras', 
            'SV': 'El Salvador',
            'CR': 'Costa Rica',
            'NI': 'Nicaragua',
            'PA': 'Panama',
            'BZ': 'Belize',
            'DO': 'Dominican Republic',
            'MX': 'Mexico'
        }
        
        # 国ごとに主成分得点を平均化
        country_averages = pca_results_df.groupby('country_code').agg({
            'component_1': 'mean',
            'component_2': 'mean'
        }).reset_index()
        
        # 散布図表示
        fig = go.Figure()
        
        for _, row in country_averages.iterrows():
            country_code = str(row['country_code'])
            country_english = country_english_names.get(country_code, country_code)
            color = 'red' if country_code == 'GT' else 'blue'
            marker_size = 12 if country_code == 'GT' else 10
            
            fig.add_trace(go.Scatter(
                x=[row['component_1']],
                y=[row['component_2']],
                mode='markers+text',
                name=country_english,
                text=country_english,
                textposition="top center",
                marker=dict(color=color, size=marker_size),
                textfont=dict(color=color, size=12),
                showlegend=False
            ))
        
        fig.update_layout(
            title="主成分分析結果（中米9か国）",
            xaxis_title=f"第1主成分 (寄与率: {pca.explained_variance_ratio_[0]:.1%})",
            yaxis_title=f"第2主成分 (寄与率: {pca.explained_variance_ratio_[1]:.1%})",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 主成分の解釈
        st.subheader("主成分の構成")
        components_df = pd.DataFrame(
            pca.components_.T,
            columns=['第1主成分', '第2主成分'],
            index=selected_indicators
        )
        st.dataframe(components_df.round(3))
        
        # 寄与率
        st.subheader("寄与率")
        st.write(f"第1主成分: {pca.explained_variance_ratio_[0]:.1%}")
        st.write(f"第2主成分: {pca.explained_variance_ratio_[1]:.1%}")
        st.write(f"累積寄与率: {sum(pca.explained_variance_ratio_):.1%}")

def machine_learning_models():
    """機械学習モデル"""
    st.header("🤖 機械学習予測モデル")
    
    # 共通設定
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.selectbox("開始年", range(2000, 2024), index=15, key="ml_start")
    with col2:
        end_year = st.selectbox("終了年", range(start_year, 2024), index=len(range(start_year, 2024))-1, key="ml_end")
    
    # 目的変数選択
    target_variables = [
        'GDP成長率（%）', '失業率（%）', 'インフレ率（%）', '一人当たりGDP（名目）',
        '一人当たりPPP（購買力平価）ベースでのGDP', '総資本形成（GDP比%）',
        '貯蓄・投資ギャップ（%）', '金融口座保有率（%）', '金融深化度（%）', '外国直接投資GDP比（%）',
        '労働参加率（%）', '女性労働参加率（%）', '女性管理者比率（%）', '平均寿命', '純移民数'
    ]
    
    selected_target = st.selectbox("予測する目的変数", target_variables, index=target_variables.index('純移民数') if '純移民数' in target_variables else 0)
    
    # 特徴量選択
    available_features = [k for k in INDICATORS.keys() if k != selected_target]
    selected_features = st.multiselect(
        "特徴量（説明変数）を選択",
        available_features,
        default=['一人当たりGDP（実質2015USD）', '失業率（%）', '金融深化度（%）', '所得格差（ジニ係数）', '総資本形成（GDP比%）'] if all(f in available_features for f in ['一人当たりGDP（実質2015USD）', '失業率（%）', '金融深化度（%）', '所得格差（ジニ係数）', '総資本形成（GDP比%）']) else available_features[:5] if len(available_features) >= 5 else available_features
    )
    
    if len(selected_features) < 2:
        st.warning("2つ以上の特徴量を選択してください。")
        return
    
    # 対象地域選択
    region_scope = st.selectbox("対象地域", [
        "中米9か国",
        "中南米カリブ33か国",
        "高位及び低位中所得国全体"
    ])
    
    if region_scope == "中米9か国":
        countries_dict = CENTRAL_AMERICA_COUNTRIES
    elif region_scope == "中南米カリブ33か国":
        countries_dict = LATIN_AMERICA_CARIBBEAN
    else:
        countries_dict = UPPER_MIDDLE_INCOME
    
    # モデル選択
    model_type = st.selectbox("予測モデル", ["ランダムフォレスト", "XGBoost", "リッジ回帰"])
    
    st.info("**注：各特徴量についてはStandardScalerによる標準化を行っています。**")
    
    # 分析ボタン
    if not st.button("🤖 分析実行", key="ml_analyze"):
        st.info("上記の設定を完了したら、「分析実行」ボタンを押してください。")
        return
    
    with st.spinner("機械学習モデルを実行中..."):
        # データ収集
        target_code = INDICATORS[selected_target]
        feature_codes = [INDICATORS[feature] for feature in selected_features]
        
        country_codes = list(countries_dict.values())
        
        # データ収集（全年度を使用）
        data_rows = []
        
        for country_code in country_codes:
            # 目的変数取得
            target_df = fetch_world_bank_data([country_code], target_code, start_year, end_year)
            if target_df is None or target_df.empty:
                continue
                
            # 特徴量取得
            feature_dfs = {}
            all_features_available = True
            
            for i, feature_code in enumerate(feature_codes):
                feature_df = fetch_world_bank_data([country_code], feature_code, start_year, end_year)
                if feature_df is not None and not feature_df.empty:
                    feature_dfs[f'feature_{i}'] = feature_df.set_index('year')['value']
                else:
                    all_features_available = False
                    break
            
            if not all_features_available:
                continue
                
            # 年ごとにデータポイントを作成
            target_indexed = target_df.set_index('year')['value']
            
            for year in range(start_year, end_year + 1):
                if year in target_indexed.index and not pd.isna(target_indexed[year]):
                    # この年のデータが全特徴量で利用可能かチェック
                    year_data = {'country_code': country_code, 'year': year}
                    has_all_year_features = True
                    
                    for feature_name, feature_series in feature_dfs.items():
                        if year in feature_series.index and not pd.isna(feature_series[year]):
                            year_data[feature_name] = feature_series[year]
                        else:
                            has_all_year_features = False
                            break
                    
                    if has_all_year_features:
                        year_data['target'] = target_indexed[year]
                        data_rows.append(year_data)
        
        if len(data_rows) < 5:
            st.error("機械学習に必要な最低限のデータが不足しています。")
            return
        
        # データフレーム作成
        ml_df = pd.DataFrame(data_rows)
        
        # 特徴量とターゲットを分離
        X = ml_df[[col for col in ml_df.columns if col.startswith('feature_')]].values
        y = ml_df['target'].values
        
        # 標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 学習・テストデータ分割
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        
        # モデル訓練
        if model_type == "ランダムフォレスト":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "XGBoost":
            model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        else:  # リッジ回帰
            model = Ridge(alpha=1.0)
        
        model.fit(X_train, y_train)
        
        # 予測
        y_pred = model.predict(X_test)
        
        # 評価指標
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 結果表示
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² スコア", f"{r2:.3f}")
        with col2:
            st.metric("平均二乗誤差", f"{mse:.3f}")
        with col3:
            st.metric("平均絶対誤差", f"{mae:.3f}")
        
        # 予測 vs 実測値のプロット
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            name='予測結果',
            marker=dict(size=8)
        ))
        
        # 理想線
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='理想線',
            line=dict(dash='dash', color='red')
        ))
        
        fig.update_layout(
            title=f"{model_type}による{selected_target}の予測結果",
            xaxis_title="実測値",
            yaxis_title="予測値",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 特徴量重要度
        st.subheader("特徴量重要度")
        
        if hasattr(model, 'feature_importances_'):
            # ランダムフォレスト、XGBoostの場合
            importance_values = model.feature_importances_
            feature_labels = selected_features
            is_ridge = False
        elif hasattr(model, 'coef_'):
            # リッジ回帰の場合は係数の絶対値を使用
            importance_values = abs(model.coef_)
            # 係数の符号を特徴量名に追加
            feature_labels = []
            for i, feature in enumerate(selected_features):
                sign = "（＋）" if model.coef_[i] > 0 else "（ー）"
                feature_labels.append(f"{feature}{sign}")
            is_ridge = True
        else:
            st.warning("このモデルでは特徴量重要度を計算できません。")
            importance_values = None
        
        if importance_values is not None:
            importance_df = pd.DataFrame({
                '特徴量': feature_labels,
                '重要度': importance_values
            }).sort_values('重要度', ascending=True)  # 重要度の高いものを上に
            
            fig_importance = go.Figure(go.Bar(
                x=importance_df['重要度'],
                y=importance_df['特徴量'],
                orientation='h'
            ))
            fig_importance.update_layout(
                title="特徴量重要度",
                height=400,
                yaxis=dict(tickfont=dict(color='black', size=12))  # y軸ラベルを黒色に
            )
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # 重要度の合計に関する説明
            if not is_ridge:
                st.info("📊 **各特徴量の重要度を合計すると1になる**")
            else:
                st.info("📊 **リッジ回帰では係数の絶対値を重要度として表示しています。（＋）は正の影響、（ー）は負の影響を示します。**")

# メイン画面
st.title("📊 中米データ分析アプリ")
st.markdown("**中米・グアテマラを中心とした経済・社会データの総合分析プラットフォーム**")

# サイドバー設定
st.sidebar.header("🔧 分析機能")

# 分析モード選択
analysis_mode = st.sidebar.radio(
    "分析機能を選択",
    [
        "📈 多国間比較分析",
        "🏛️ 単一国詳細分析", 
        "📊 合成指標分析",
        "🎯 SDGs目標達成度分析",
        "🔄 主成分分析（PCA）",
        "🤖 機械学習予測モデル"
    ]
)

# 分析機能の実行
if analysis_mode == "📈 多国間比較分析":
    multi_country_comparison_analysis()
elif analysis_mode == "🏛️ 単一国詳細分析":
    single_country_detailed_analysis()
elif analysis_mode == "📊 合成指標分析":
    composite_indicator_analysis()
elif analysis_mode == "🎯 SDGs目標達成度分析":
    sdgs_achievement_analysis()
elif analysis_mode == "🔄 主成分分析（PCA）":
    pca_analysis()
elif analysis_mode == "🤖 機械学習予測モデル":
    machine_learning_models()

# フッター
st.markdown("---")
st.markdown("**データソース**: 世界銀行 World Bank Open Data API")
st.markdown("**注意**: このアプリケーションは教育・研究目的での使用を想定しています。")
