import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import re

# ==============================================================================
# 1. CONFIGURAÇÃO E ESTILO
# ==============================================================================
st.set_page_config(
    page_title="Datathon | Passos Mágicos",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS CUSTOMIZADO
st.markdown("""
    <style>
    .stApp {background-color: #ffffff;}
    .analysis-box { 
        background-color: #e8f4f8; 
        border-left: 5px solid #00a6ce; 
        padding: 15px 20px !important; 
        border-radius: 5px; 
        margin-bottom: 20px; 
        color: #2c3e50; 
        font-size: 0.85rem !important; 
        line-height: 1.3 !important; 
        height: auto !important; 
        min-height: fit-content; 
        overflow-wrap: break-word; 
    }
    .analysis-title { font-weight: bold; color: #007bb5; font-size: 0.95rem !important; }
    .stMetric { background-color: #ffffff; padding: 10px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 2. CARREGAMENTO DE DADOS E MODELO (DINÂMICO PARA DEPLOY)
# ==============================================================================

# Definição do diretório base (raiz do projeto) para caminhos absolutos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def buscar_arquivo(nome_arquivo):
    """Procura arquivo em /data, /app ou na raiz usando BASE_DIR."""
    possiveis_caminhos = [
        os.path.join(BASE_DIR, 'data', nome_arquivo),
        os.path.join(BASE_DIR, 'app', nome_arquivo),
        os.path.join(BASE_DIR, nome_arquivo),
        os.path.join(os.getcwd(), nome_arquivo), # fallback diretório atual
        nome_arquivo # fallback relativo simples
    ]
    for caminho in possiveis_caminhos:
        if os.path.exists(caminho):
            return caminho
    return None

@st.cache_data
def carregar_dados():
    caminho_csv = buscar_arquivo('dados_passos_magicos_limpos.csv')
    
    if not caminho_csv:
        st.error(f"""
        ### 🚨 Erro de Carregamento
        O arquivo **'dados_passos_magicos_limpos.csv'** não foi encontrado.
        
        **Caminhos verificados:**
        1. `{os.path.join(BASE_DIR, 'data', 'dados_passos_magicos_limpos.csv')}`
        2. `{os.path.join(BASE_DIR, 'app', 'dados_passos_magicos_limpos.csv')}`
        3. Root: `{os.path.join(BASE_DIR, 'dados_passos_magicos_limpos.csv')}`
        """)
        st.stop()
        return pd.DataFrame()

    try:
        df = pd.read_csv(caminho_csv)
        df.columns = [c.upper() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()].copy()

        if 'ANO_PEDE' in df.columns:
            df['ANO_PEDE'] = pd.to_numeric(df['ANO_PEDE'], errors='coerce').fillna(0).astype(int)

        # PEDRA: normalizar variantes de acentuação, manter NaN real
        if 'PEDRA' in df.columns:
            mapa_pedra = {
                'quartzo': 'Quartzo', 'quarzo': 'Quartzo',
                'agata': 'Ágata', 'ágata': 'Ágata',
                'ametista': 'Ametista',
                'topazio': 'Topázio', 'topázio': 'Topázio',
            }
            df['PEDRA'] = df['PEDRA'].apply(
                lambda v: np.nan if pd.isna(v) or str(v).strip() in ('', '0', 'nan', 'NaN')
                else mapa_pedra.get(str(v).strip().lower(), str(v).strip())
            )

        # FASE: normalizar para 'Fase X' ou 'ALFA', manter NaN real
        if 'FASE' in df.columns:
            def _norm_fase(val):
                if pd.isna(val) or str(val).strip() in ('', '0', '0.0', 'nan', 'NaN'):
                    return np.nan
                s = str(val).strip()
                if 'ALFA' in s.upper(): return 'ALFA'
                s_num = re.sub(r'^FASE\s*', '', s.upper()).strip()
                s_num = re.sub(r'[A-Z]+$', '', s_num).strip()
                s_num = re.sub(r'\.0$', '', s_num)
                try: return f'Fase {int(float(s_num))}'
                except: return s
            df['FASE'] = df['FASE'].apply(_norm_fase)

        if 'PEDRA' in df.columns:
            ordem_pedras = ['Quartzo', 'Ágata', 'Ametista', 'Topázio']
            df['PEDRA'] = pd.Categorical(df['PEDRA'], categories=ordem_pedras, ordered=True)

        return df
    except Exception as e:
        st.error(f'Falha crítica ao ler o CSV: {e}')
        st.stop()
        return pd.DataFrame()

@st.cache_resource
def carregar_modelo():
    # Tentar nomes comuns: risco ou simulador
    nomes = ['modelo_risco_passos_magicos.pkl', 'modelo_simulador.pkl']
    caminho_modelo = None
    
    for nome in nomes:
        caminho_modelo = buscar_arquivo(nome)
        if caminho_modelo: break
        
    if not caminho_modelo:
        st.warning("⚠️ Modelo preditivo (.pkl) não encontrado. Algumas funcionalidades podem estar limitadas.")
        return None

    try:
        return joblib.load(caminho_modelo)
    except Exception as e:
        st.info(f"O simulador está temporariamente indisponível (Erro: {e})")
        return None

df = carregar_dados()
modelo = carregar_modelo()

# ==============================================================================
# 3. SIDEBAR (FILTROS INTELIGENTES)
# ==============================================================================
with st.sidebar:
    st.image("https://passosmagicos.org.br/wp-content/uploads/2020/10/Passos-magicos-icon-cor.png", width=180)
    st.markdown("### 🔍 Central de Filtros")
    st.divider()
    
    busca_ra = st.text_input("Ficha do Aluno (Busca por RA)", placeholder="Ex: RA-1").strip()
    st.divider()
    
    # Filtros Dinâmicos Dependentes do RA
    # RA preenchido → exibe e pré-seleciona só os valores reais do aluno
    # RA vazio      → comportamento padrão com "Todos"/"Todas"
    df_opcoes = df.copy()
    if busca_ra:
        df_opcoes = df_opcoes[df_opcoes['RA'].astype(str).str.upper() == busca_ra.upper()]

    anos_disp = sorted([int(a) for a in df_opcoes['ANO_PEDE'].unique() if a > 0], reverse=True)
    if busca_ra and anos_disp:
        ano_sel = st.multiselect(
            "Ano de Referência",
            options=anos_disp,
            default=anos_disp,
            help="Mostrando apenas anos com dados para este aluno."
        )
    else:
        ano_sel = st.multiselect("Ano de Referência", ["Todos"] + anos_disp, default=["Todos"])

    # Pedras disponíveis: excluir NaN, exibir só valores reais
    pedras_raw = df_opcoes['PEDRA'].dropna().unique()
    pedras_disp = sorted(
        [str(p) for p in pedras_raw if str(p) not in ('nan', 'NaN', '')],
        key=lambda x: ['Quartzo', 'Ágata', 'Ametista', 'Topázio'].index(x)
                      if x in ['Quartzo', 'Ágata', 'Ametista', 'Topázio'] else 99
    )
    if busca_ra and pedras_disp:
        pedra_sel = st.multiselect(
            "Fase (Pedra)",
            options=pedras_disp,
            default=pedras_disp,
            help="Mostrando apenas pedras atribuídas a este aluno."
        )
    else:
        pedra_sel = st.multiselect("Fase (Pedra)", ["Todas"] + pedras_disp, default=["Todas"])
    
    # Aplicação Sequencial dos Filtros (Cumulativos)
    df_f = df.copy()
    if busca_ra:
        df_f = df_f[df_f['RA'].astype(str).str.upper() == busca_ra.upper()]

    # Salva o df_base pós-RA para cálculos de Deltas na Visão Geral
    df_base = df_f.copy()

    # Filtro de Ano: funciona com e sem "Todos" (compatível com modo dinâmico)
    if "Todos" not in ano_sel and ano_sel:
        df_f = df_f[df_f['ANO_PEDE'].isin([int(a) for a in ano_sel])]

    # Filtro de Pedra: funciona com e sem "Todas" (compatível com modo dinâmico)
    if "Todas" not in pedra_sel and pedra_sel:
        df_f = df_f[df_f['PEDRA'].isin(pedra_sel)]
        df_base = df_base[df_base['PEDRA'].isin(pedra_sel)]

    st.divider()
    # Botão de Download (Valor Agregado)
    csv = df_f.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Baixar Dados Filtrados", data=csv, file_name='dados_filtrados.csv', mime='text/csv')

# ==============================================================================
# 4. DASHBOARD (ABAS E STORYTELLING)
# ==============================================================================
if df.empty:
    st.warning("Base de dados não carregada. Verifique o arquivo CSV.")
    st.stop()

st.title(f"Monitoramento de Impacto • {'Todos os Anos' if 'Todos' in ano_sel else ', '.join(map(str, ano_sel))}")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Visão Geral", "🎓 Acadêmico", "🧠 Psicossocial", "📈 Laboratório EDA", "🤖 Simulador Preditivo", "📚 Glossário"
])

# --- TAB 1: VISÃO GERAL ---
with tab1:
    if busca_ra and 'RA' in df_f.columns and df_f['RA'].nunique() == 1:
        aluno_data = df_f.sort_values('ANO_PEDE', ascending=False).iloc[0] if 'ANO_PEDE' in df_f.columns else df_f.iloc[0]
        st.subheader(f"🎓 Ficha Individual: {aluno_data.get('NOME', 'Aluno')} ({aluno_data['RA']}) (Ano Mais Recente)")
        
        c1, c2, c3, c4, c5 = st.columns(5)
        fase_val = aluno_data.get('FASE', None)
        pedra_val = aluno_data.get('PEDRA', None)
        c1.metric("Fase", fase_val if fase_val and str(fase_val) not in ('nan', 'NaN', 'Não Informado', '') else 'Não registrado')
        c2.metric("Pedra", pedra_val if pedra_val and str(pedra_val) not in ('nan', 'NaN', 'Não Informado', '') else 'Não registrado')
        c3.metric("INDE", f"{aluno_data.get('INDE', 0):.2f}")
        c4.metric("Recomendação", aluno_data.get('REC AV1', 'N/A'))
        c5.metric("Status de Risco", "Alto Risco" if aluno_data.get('RISCO', 0) == 1 else "Fora de Risco")
        
        st.divider()
        
        c_radar_ind, c_radar_comp = st.columns(2)
        with c_radar_ind:
            st.markdown("**Desempenho do Aluno vs Média da Fase**")
            ind_radar = ['IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV']
            aluno_vals = [aluno_data.get(c, 0) for c in ind_radar]
            
            pedra_aluno = aluno_data.get('PEDRA', 'N/A')
            # Fetch average for this Pedra globally
            media_pedra = df_f[df_f['PEDRA'] == pedra_aluno][ind_radar].mean().fillna(0).values if pd.notna(pedra_aluno) else [0]*6
            
            fig_radari = go.Figure()
            fig_radari.add_trace(go.Scatterpolar(r=aluno_vals, theta=ind_radar, fill='toself', name='Aluno'))
            fig_radari.add_trace(go.Scatterpolar(r=media_pedra, theta=ind_radar, fill='toself', name=f'Média {pedra_aluno}'))
            fig_radari.update_layout(polar={"radialaxis": {"visible": True, "range": [0, 10]}}, height=350)
            st.plotly_chart(fig_radari, use_container_width=True)
            
        with c_radar_comp:
            st.markdown("**Síntese das Avaliações**")
            
            def formata_destaque(valor):
                if pd.isna(valor) or valor == '' or str(valor).strip() == '0' or str(valor).strip() == '0.0':
                    return 'Informações de acompanhamento pedagógico não registradas para este período.'
                return str(valor)

            st.info(f"**Engajamento (IEG):** {formata_destaque(aluno_data.get('DESTAQUE IEG', 'N/A'))}")
            st.success(f"**Desempenho (IDA):** {formata_destaque(aluno_data.get('DESTAQUE IDA', 'N/A'))}")
            st.warning(f"**Automação (IPV):** {formata_destaque(aluno_data.get('DESTAQUE IPV', 'N/A'))}")
            
    else:
        st.subheader("🎯 KPIs Principais")
        if df_f.empty:
            st.info('ℹ️ Sem registros para esta combinação de Filtros.')
            
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        
        delta_alunos = 0
        delta_inde = 0.0
        
        # Lógica de Deltas (Ano Anterior)
        total_alunos_atual = df_f['RA'].nunique() if 'RA' in df_f.columns else len(df_f)
        
        # Cálculos de Métricas Reativas
        media_inde_atual = pd.Series([df_f['INDE'].mean()]).fillna(0)[0]
        
        risco_perc_atual = pd.Series([df_f['RISCO'].mean()]).fillna(0)[0] if 'RISCO' in df_f.columns else 0.0
        ipv_high_atual = df_f[df_f['IPV'] > 7]['RA'].nunique() if 'IPV' in df_f.columns and 'RA' in df_f.columns else 0
        
        anos_selecionados = [int(a) for a in ano_sel] if "Todos" not in ano_sel else [int(a) for a in df_base['ANO_PEDE'].unique() if a > 0]
        if anos_selecionados:
            ano_max = max(anos_selecionados)
            df_ano_anterior = df_base[df_base['ANO_PEDE'] == (ano_max - 1)]
            
            if not df_ano_anterior.empty:
                total_alunos_ant = df_ano_anterior['RA'].nunique() if 'RA' in df_ano_anterior.columns else len(df_ano_anterior)
                media_inde_ant = pd.Series([df_ano_anterior['INDE'].mean()]).fillna(0)[0]
                
                delta_alunos = total_alunos_atual - total_alunos_ant
                delta_inde = media_inde_atual - media_inde_ant
        
        # Garante a formatação 0.00 no display
        col_kpi1.metric("Alunos Monitorados", total_alunos_atual, delta=delta_alunos if delta_alunos != 0 else None)
        col_kpi2.metric("Média INDE", f"{media_inde_atual:.2f}", delta=f"{delta_inde:.2f}" if delta_inde != 0 else None)
        col_kpi3.metric("Taxa de Risco", f"{risco_perc_atual:.1%}", delta=f"{risco_perc_atual*100:.1f}%", delta_color="inverse")
        col_kpi4.metric("Destaques (IPV > 7)", ipv_high_atual)
    
        st.divider()
        
        # --- Novos Gráficos (Visão 360º) ---
        st.subheader("🌐 Visão 360º")
        c_hist, c_dist = st.columns(2)
        
        with c_hist:
            if not df_f.empty and 'ANO_PEDE' in df_f.columns and 'INDE' in df_f.columns:
                df_hist = df_f[df_f['ANO_PEDE'] > 0].groupby('ANO_PEDE', as_index=False)['INDE'].mean()
                fig_hist = px.line(df_hist, x='ANO_PEDE', y='INDE', text=[f'{v:.2f}' for v in df_hist['INDE']],
                                   markers=True, title="Evolução Histórica da Média do INDE")
                fig_hist.update_traces(textposition="top center")
                # Fix x-axis to show integer years nicely
                fig_hist.update_xaxes(type='category')
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("Dados insuficientes para histórico.")
                
        with c_dist:
            if not df_f.empty and 'PEDRA' in df_f.columns:
                # Excluir NaN/não informado — só exibir pedras reais
                df_pedra_graf = df_f['PEDRA'].dropna().astype(str)
                df_pedra_graf = df_pedra_graf[df_pedra_graf.isin(['Quartzo','Ágata','Ametista','Topázio'])]
                if not df_pedra_graf.empty:
                    df_pedra = df_pedra_graf.value_counts().reset_index()
                    df_pedra.columns = ['Pedra', 'Quantidade']
                    # Ordem lógica por desempenho
                    ordem = ['Quartzo','Ágata','Ametista','Topázio']
                    df_pedra['Pedra'] = pd.Categorical(df_pedra['Pedra'], categories=ordem, ordered=True)
                    df_pedra = df_pedra.sort_values('Pedra')
                    cores_pedra = {'Quartzo':'#78909C','Ágata':'#42A5F5','Ametista':'#AB47BC','Topázio':'#FFB300'}
                    fig_dist = px.bar(df_pedra, x='Quantidade', y='Pedra', orientation='h',
                                      title="Distribuição por Pedra (Classificação INDE)",
                                      text='Quantidade', color='Pedra',
                                      color_discrete_map=cores_pedra)
                    fig_dist.update_traces(textposition='outside')
                    st.plotly_chart(fig_dist, use_container_width=True)
                else:
                    st.info("Sem classificações de Pedra para os filtros selecionados.")
            else:
                st.info("Distribuição por pedra não disponível.")
                
        st.divider()
        
        c_ian, c_radar = st.columns([1.5, 1])
        with c_ian:
            st.subheader("Adequação Idade-Série (IAN)")
            def cat_ian(x): return "Crítico (<5)" if x<5 else "Atenção (5-7)" if x<7 else "Adequado (7+)"
            df_f['IAN_CAT'] = df_f['IAN'].apply(cat_ian)
            fig_ian = px.bar(df_f['IAN_CAT'].value_counts().reset_index(), x='IAN_CAT', y='count', 
                             labels={'IAN_CAT':'Status','count':'Qtd'}, color='IAN_CAT',
                             color_discrete_map={"Crítico (<5)":"#e53935", "Atenção (5-7)":"#fb8c00", "Adequado (7+)":"#43a047"})
            st.plotly_chart(fig_ian, use_container_width=True)
            
            # NLG Automático
            if not df_f.empty and 'IAN' in df_f.columns and 'RA' in df_f.columns:
                media_ian = df_f['IAN'].mean()
                qtd_criticos = df_f[df_f['IAN_CAT'] == "Crítico (<5)"]['RA'].nunique()
                pct_critico = (qtd_criticos / total_alunos_atual) * 100 if total_alunos_atual > 0 else 0
                st.markdown(f"""
                <div class="analysis-box">
                    <span class="analysis-title">🤖 Diagnóstico:</span><br>
                    O nível médio de adequação idade-série está em <b>{media_ian:.1f}</b>.<br>
                    Cerca de <b>{pct_critico:.1f}%</b> dos nossos alunos filtrados estão em nível crítico, o que representa um forte desafio de alfabetização base para o grupo.
                </div>
                """, unsafe_allow_html=True)
    
        with c_radar:
            st.subheader("Média Multidimensional")
            ind_radar = ['IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV']
            if df_f.empty:
                media_radar = pd.Series([0.0]*len(ind_radar), index=ind_radar)
            else:
                media_radar = df_f[[c for c in ind_radar if c in df_f.columns]].mean().fillna(0)
            fig_radar = go.Figure(go.Scatterpolar(r=media_radar.values, theta=media_radar.index, fill='toself', name='Média Geral'))
            fig_radar.update_layout(polar={"radialaxis": {"visible": True, "range": [0, 10]}}, height=350)
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # NLG Automático Radar
            if not media_radar.empty:
                ind_forte = media_radar.idxmax()
                val_forte = media_radar.max()
                ind_fraco = media_radar.idxmin()
                val_fraco = media_radar.min()
                
                st.markdown(f"""
                <div class="analysis-box">
                    <span class="analysis-title">🤖 Diagnóstico Radial:</span><br>
                    <b>Ponto Forte:</b> O indicador de melhor desempenho coletivo é o <b>{ind_forte}</b> ({val_forte:.1f}).<br>
                    <b>Atenção:</b> O indicador mais fraco desta seleção é o <b>{ind_fraco}</b> ({val_fraco:.1f}).
                </div>
                """, unsafe_allow_html=True)

# --- TAB 2: ACADÊMICO ---
with tab2:
    st.subheader("🎓 Diagnóstico Pedagógico")

    # ──────────────────────────────────────────────────────────────
    # MODO INDIVIDUAL: exibição personalizada quando RA está filtrado
    # ──────────────────────────────────────────────────────────────
    _modo_individual = busca_ra and 'RA' in df_f.columns and df_f['RA'].nunique() == 1

    if _modo_individual:
        aluno_data_ac = df_f.sort_values('ANO_PEDE', ascending=False).iloc[0]
        ra_label = aluno_data_ac.get('RA', busca_ra)
        ind_labels = {'IDA':'Aprendizagem','IEG':'Engajamento','IAA':'Autoavaliação',
                      'IPS':'Psicossocial','IPP':'Psicopedagógico','IPV':'Ponto de Virada','IAN':'Adequação Nível'}

        st.markdown(f"#### 📋 Análise Individual — {ra_label}")

        # — Bloco 1: Radar do aluno vs média geral da mesma Pedra ——————————
        c_rad_al, c_ev = st.columns(2)
        with c_rad_al:
            ind_radar = ['IDA','IEG','IAA','IPS','IPP','IPV']
            # Corrigido: nan or 0 retorna nan porque NaN é truthy em Python
            vals_aluno = [0.0 if pd.isna(aluno_data_ac.get(c)) else float(aluno_data_ac.get(c, 0))
                          for c in ind_radar]
            # média da mesma pedra no ano do aluno (referência de comparação)
            pedra_ref = aluno_data_ac.get('PEDRA', None)
            ano_ref   = aluno_data_ac.get('ANO_PEDE', None)
            df_ref = df[(df['ANO_PEDE']==ano_ref) & (df['PEDRA']==pedra_ref)] if ano_ref and pedra_ref else pd.DataFrame()
            vals_ref = [float(df_ref[c].mean()) if not df_ref.empty and c in df_ref.columns else 0 for c in ind_radar]

            fig_rad_al = go.Figure()
            fig_rad_al.add_trace(go.Scatterpolar(r=vals_aluno, theta=ind_radar, fill='toself',
                                                  name='Aluno', line_color='#00a6ce'))
            fig_rad_al.add_trace(go.Scatterpolar(r=vals_ref, theta=ind_radar, fill='toself',
                                                  name=f'Média {pedra_ref or "Grupo"}',
                                                  line_color='#ff9f43', opacity=0.5))
            fig_rad_al.update_layout(polar={'radialaxis':{'visible':True,'range':[0,10]}},
                                      height=380, title='Perfil vs Média da Pedra')
            st.plotly_chart(fig_rad_al, use_container_width=True)

        with c_ev:
            # — Bloco 2: Evolução temporal dos indicadores por ano ————————
            anos_al = sorted(df_f['ANO_PEDE'].unique().tolist())
            if len(anos_al) > 1:
                df_evo = df_f.sort_values('ANO_PEDE')[['ANO_PEDE'] + ind_radar].melt(
                    id_vars='ANO_PEDE', var_name='Indicador', value_name='Nota')
                fig_evo = px.line(df_evo, x='ANO_PEDE', y='Nota', color='Indicador',
                                  markers=True, title='Evolução por Ano',
                                  labels={'ANO_PEDE':'Ano'})
                fig_evo.update_xaxes(type='category')
                fig_evo.update_layout(height=380)
                st.plotly_chart(fig_evo, use_container_width=True)
            else:
                # 1 único ano → mostrar barras de cada indicador com referência
                df_bar = pd.DataFrame({
                    'Indicador': [ind_labels.get(c, c) for c in ind_radar],
                    'Aluno': vals_aluno,
                    'Referência': vals_ref
                })
                fig_bar = go.Figure()
                fig_bar.add_bar(name='Aluno', x=df_bar['Indicador'], y=df_bar['Aluno'],
                                marker_color='#00a6ce', text=[f'{v:.1f}' for v in df_bar['Aluno']],
                                textposition='outside')
                fig_bar.add_bar(name=f'Média {pedra_ref or "Grupo"}', x=df_bar['Indicador'],
                                y=df_bar['Referência'], marker_color='#ff9f43',
                                text=[f'{v:.1f}' for v in df_bar['Referência']],
                                textposition='outside', opacity=0.75)
                fig_bar.update_layout(barmode='group', height=380,
                                       title=f'Indicadores {ra_label} vs Média do Grupo ({int(ano_ref or 0)})',
                                       yaxis_range=[0, 10.5])
                st.plotly_chart(fig_bar, use_container_width=True)

        # — Bloco 3: Diagnóstico textual personalizado ————————————————
        st.divider()
        ind_vals_dict = {c: float(aluno_data_ac.get(c, 0) or 0) for c in ind_radar if pd.notna(aluno_data_ac.get(c))}
        if ind_vals_dict:
            ind_min = min(ind_vals_dict, key=ind_vals_dict.get)
            ind_max = max(ind_vals_dict, key=ind_vals_dict.get)
            # Posição relativa: percentil do INDE do aluno na base completa do mesmo ano
            inde_aluno = float(aluno_data_ac.get('INDE', 0) or 0)
            inde_base = df[df['ANO_PEDE']==ano_ref]['INDE'].dropna() if ano_ref else pd.Series(dtype=float)
            percentil = int((inde_base < inde_aluno).mean() * 100) if not inde_base.empty else None

            recom_map = {
                'IDA': 'reforço acadêmico em conteúdo base e revisão de disciplinas com menor rendimento',
                'IEG': 'estratégias para aumentar o engajamento: projetos práticos, tutoria e metas de curto prazo',
                'IAA': 'acompanhamento da autopercepção do aluno e alinhamento com avaliadores',
                'IPS': 'suporte à saúde emocional e redes de apoio psicossocial',
                'IPP': 'acompanhamento psicopedagógico intensificado e plano de desenvolvimento',
                'IPV': 'estímulo ao desenvolvimento de metas e Ponto de Virada com orientação vocacional',
                'IAN': 'adequação do plano pedagógico ao nível real do aluno para reduzir a defasagem',
            }
            recom_txt = recom_map.get(ind_min, 'intervenção pedagógica personalizada')

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown(f"""
                <div class="analysis-box">
                    <span class="analysis-title">🤖 Diagnóstico Personalizado — {ra_label}</span><br>
                    <b>Ponto Forte:</b> {ind_labels.get(ind_max, ind_max)} ({ind_vals_dict[ind_max]:.1f})<br>
                    <b>Ponto de Atenção:</b> {ind_labels.get(ind_min, ind_min)} ({ind_vals_dict[ind_min]:.1f})<br>
                    <b>INDE:</b> {inde_aluno:.2f}
                    {f' — <b>Top {100-percentil}%</b> da turma de {int(ano_ref)}' if percentil is not None else ''}<br><br>
                    <b>Recomendação:</b> Priorizar {recom_txt}.
                </div>
                """, unsafe_allow_html=True)
            with col_d2:
                quad_aluno = 'N/A'
                ieg_a = float(aluno_data_ac.get('IEG', 0) or 0)
                ida_a = float(aluno_data_ac.get('IDA', 0) or 0)
                limiar = 7.0
                if ieg_a >= limiar and ida_a >= limiar: quad_aluno = '✅ Potencializado'
                elif ieg_a >= limiar and ida_a < limiar: quad_aluno = '⚠️ Esforçado com Dificuldade'
                elif ieg_a < limiar and ida_a >= limiar: quad_aluno = '🟡 Desmotivado'
                elif ieg_a > 0 or ida_a > 0: quad_aluno = '🔴 Crítico'
                st.markdown(f"""
                <div class="analysis-box">
                    <span class="analysis-title">📊 Posição na Matriz</span><br>
                    <b>IEG (Engajamento):</b> {ieg_a:.1f}<br>
                    <b>IDA (Aprendizagem):</b> {ida_a:.1f}<br>
                    <b>Quadrante:</b> {quad_aluno}<br><br>
                    <b>Pedra:</b> {aluno_data_ac.get('PEDRA', 'N/A')} &nbsp;|&nbsp;
                    <b>Fase:</b> {aluno_data_ac.get('FASE', 'N/A')}
                </div>
                """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────
    # MODO COLETIVO: análise normal para múltiplos alunos
    # ──────────────────────────────────────────────────────────────
    else:
        df_plot = df_f.dropna(subset=['IEG', 'IDA'])

        if df_plot.empty:
            st.info('ℹ️ Sem registros para esta combinação de Filtros.')
        else:
            # Correlação (protegida contra NaN com < 2 pontos)
            corr_ieg_ida = df_plot['IEG'].corr(df_plot['IDA']) if len(df_plot) >= 2 else float('nan')

            limiar = 7.0
            def calc_quadrante(row):
                if pd.isna(row['IEG']) or pd.isna(row['IDA']): return 'Desconhecido'
                if row['IEG'] >= limiar and row['IDA'] >= limiar: return 'Potencializado'
                if row['IEG'] >= limiar and row['IDA'] < limiar: return 'Esforçado com Dificuldade'
                if row['IEG'] < limiar and row['IDA'] >= limiar: return 'Desmotivado'
                return 'Crítico'
            df_plot = df_plot.copy()
            df_plot['Quadrante'] = df_plot.apply(calc_quadrante, axis=1)

            c_q, c_b = st.columns(2)
            with c_q:
                color_quad = {'Potencializado':'#28a745','Esforçado com Dificuldade':'#fd7e14',
                              'Desmotivado':'#ffc107','Crítico':'#dc3545','Desconhecido':'#cccccc'}
                fig_quad = px.scatter(df_plot, x='IEG', y='IDA', color='Quadrante',
                                      color_discrete_map=color_quad, title='Matriz de Desempenho',
                                      hover_data=['NOME'] if 'NOME' in df_plot.columns else [])
                fig_quad.add_hline(y=limiar, line_dash='dash', line_color='gray', annotation_text='Limiar (7.0)')
                fig_quad.add_vline(x=limiar, line_dash='dash', line_color='gray')
                st.plotly_chart(fig_quad, use_container_width=True)

                total_validos = df_plot['RA'].nunique() if 'RA' in df_plot.columns else len(df_plot)
                qtd_esf = df_plot[df_plot['Quadrante']=='Esforçado com Dificuldade']['RA'].nunique() \
                          if 'RA' in df_plot.columns else len(df_plot[df_plot['Quadrante']=='Esforçado com Dificuldade'])
                pct_esf = qtd_esf / total_validos * 100 if total_validos > 0 else 0
                st.markdown(f"""
                <div class="analysis-box">
                    <span class="analysis-title">🤖 Diagnóstico:</span><br>
                    <b>{pct_esf:.1f}%</b> dos alunos têm alto engajamento (IEG ≥ 7) mas desempenho abaixo do esperado (IDA &lt; 7).<br>
                    <b>Recomendação:</b> Reforço focado em conteúdo base para este grupo.
                </div>""", unsafe_allow_html=True)

            with c_b:
                c_m1, c_m2 = st.columns(2)
                corr_str = f"{corr_ieg_ida:.2f}" if not np.isnan(corr_ieg_ida) else "N/A (1 ponto)"
                c_m1.metric("Correlação IEG vs IDA", corr_str)
                c_m2.metric("Alunos no radar", total_validos)

                if 'IAA' in df_plot.columns and 'PEDRA' in df_plot.columns:
                    df_comp = df_plot.dropna(subset=['PEDRA']).melt(
                        id_vars=['PEDRA','RA'], value_vars=['IDA','IAA'], var_name='Métrica', value_name='Nota')
                    if not df_comp.empty:
                        fig_comp = px.box(df_comp, x='PEDRA', y='Nota', color='Métrica',
                                          title='Realidade (IDA) vs Percepção (IAA)',
                                          color_discrete_map={'IDA':'#00a6ce','IAA':'#ff9f43'})
                        st.plotly_chart(fig_comp, use_container_width=True)
                else:
                    st.info("Dados de IAA/PEDRA indisponíveis.")

# --- TAB 3: PSICOSSOCIAL ---
with tab3:
    st.subheader("🧠 Fatores Psicossociais e Preditivos")

    _modo_psico_ind = busca_ra and 'RA' in df_f.columns and df_f['RA'].nunique() == 1

    # ─── MODO INDIVIDUAL ──────────────────────────────────────────────────────
    if _modo_psico_ind:
        aluno_ps = df_f.sort_values('ANO_PEDE', ascending=False).iloc[0]
        ra_lbl   = aluno_ps.get('RA', busca_ra)
        ano_ps   = aluno_ps.get('ANO_PEDE', None)
        pedra_ps = aluno_ps.get('PEDRA', None)

        st.markdown(f"#### 🧩 Análise Psicossocial Individual — {ra_lbl}")

        # Indicadores psicossociais (corrigido: pd.isna ao invés de or 0)
        ind_psico = ['IPS', 'IPP', 'IPV', 'IAN']
        lbl_psico = {'IPS':'Psicossocial','IPP':'Psicopedag\u00f3gico','IPV':'Ponto de Virada','IAN':'Adequa\u00e7\u00e3o N\u00edvel'}
        vals_ps   = {c: (0.0 if pd.isna(aluno_ps.get(c)) else float(aluno_ps.get(c, 0)))
                     for c in ind_psico if pd.notna(aluno_ps.get(c))}

        df_ref_ps = df[(df['ANO_PEDE']==ano_ps) & (df['PEDRA']==pedra_ps)] \
                    if ano_ps and pedra_ps and not df.empty else pd.DataFrame()
        vals_ref_ps = {c: float(df_ref_ps[c].mean()) if not df_ref_ps.empty and c in df_ref_ps.columns else 0
                       for c in ind_psico}

        # C\u00e1lculo do risco antes do layout
        prob_risco = None
        if modelo:
            FEATURES_ML = ['IDA','IEG','IAA','IPS','IPP','IPV','IAN',
                            'ENGAJAMENTO_ACADEMICO','SUPORTE_PSICO','SCORE_GERAL',
                            'EVOLUCAO_IDA','EVOLUCAO_IEG']
            try:
                row_ml = {f: 0.0 if pd.isna(aluno_ps.get(f)) else float(aluno_ps.get(f, 0)) for f in FEATURES_ML}
                if row_ml.get('ENGAJAMENTO_ACADEMICO', 0) == 0:
                    row_ml['ENGAJAMENTO_ACADEMICO'] = row_ml['IEG'] * row_ml['IDA']
                if row_ml.get('SUPORTE_PSICO', 0) == 0:
                    row_ml['SUPORTE_PSICO'] = (row_ml['IPS'] + row_ml['IPP']) / 2
                if row_ml.get('SCORE_GERAL', 0) == 0:
                    row_ml['SCORE_GERAL'] = (row_ml['IDA']+row_ml['IEG']+row_ml['IAA']+
                                              row_ml['IPS']+row_ml['IPP']) / 5
                X_pred = pd.DataFrame([row_ml])[FEATURES_ML]
                prob_risco = float(modelo.predict_proba(X_pred)[0][1])
            except Exception:
                prob_risco = None

        val_gauge = round((prob_risco or 0) * 100, 1)
        cor_gauge = '#dc3545' if prob_risco and prob_risco > 0.5 else '#28a745'

        # N\u00edvel de risco e recomenda\u00e7\u00e3o
        if prob_risco is not None:
            if prob_risco < 0.30:
                nivel_txt, nivel_desc = '\U0001f7e2 Baixo', 'Bom suporte psicossocial. Manter acompanhamento de rotina.'
            elif prob_risco < 0.60:
                nivel_txt, nivel_desc = '\U0001f7e1 Moderado', 'Alguns indicadores merecem aten\u00e7\u00e3o. Acompanhamento pr\u00f3ximo recomendado.'
            else:
                nivel_txt, nivel_desc = '\U0001f534 Alto', 'Situa\u00e7\u00e3o de vulnerabilidade. A\u00e7\u00e3o imediata recomendada.'
        else:
            nivel_txt, nivel_desc = 'N/D', 'Modelo n\u00e3o dispon\u00edvel para c\u00e1lculo.'

        ind_ps_min = min(vals_ps, key=vals_ps.get) if vals_ps else None
        ind_ps_max = max(vals_ps, key=vals_ps.get) if vals_ps else None
        recom_ps_map = {
            'IPS': 'atividades de apoio emocional, grupos de pertencimento e redes de suporte familiar',
            'IPP': 'acompanhamento psicopedag\u00f3gico intensificado e avalia\u00e7\u00e3o de dificuldades de aprendizagem',
            'IPV': 'orienta\u00e7\u00e3o vocacional, defini\u00e7\u00e3o de metas e est\u00edmulo ao Ponto de Virada',
            'IAN': 'revis\u00e3o do plano pedag\u00f3gico e adequa\u00e7\u00e3o ao n\u00edvel real do aluno',
        }
        recom_txt_ps = recom_ps_map.get(ind_ps_min, 'acompanhamento multidisciplinar') if ind_ps_min else 'acompanhamento geral'

        # Layout: coluna esquerda = gauge + diagn\u00f3stico | coluna direita = barras + evolu\u00e7\u00e3o
        col_esq, col_dir = st.columns(2)

        with col_esq:
            fig_gauge = go.Figure(go.Indicator(
                mode='number+gauge',
                value=val_gauge,
                number={'suffix': '%', 'font': {'size': 28}, 'valueformat': '.1f'},
                gauge={
                    'shape': 'bullet',
                    'axis': {'range': [0, 100]},
                    'bar': {'color': cor_gauge, 'thickness': 0.5},
                    'steps': [
                        {'range': [0, 30],   'color': '#d4edda'},
                        {'range': [30, 60],  'color': '#fff3cd'},
                        {'range': [60, 100], 'color': '#f8d7da'},
                    ],
                    'threshold': {'line': {'color': '#333', 'width': 3},
                                  'thickness': 0.75, 'value': 50}
                },
                title={'text': f'Probabilidade de Risco \u2014 {ra_lbl}', 'font': {'size': 13}},
                domain={'x': [0, 1], 'y': [0.6, 1]}
            ))
            fig_gauge.update_layout(height=200, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown(f"""
            <div class="analysis-box">
                <span class="analysis-title">\U0001f916 Diagn\u00f3stico \u2014 {ra_lbl}</span><br>
                <b>N\u00edvel de Risco:</b> {nivel_txt} ({f"{val_gauge:.0f}%" if prob_risco is not None else "N/D"})<br>
                {nivel_desc}<br><br>
                <b>Ponto Forte:</b> {lbl_psico.get(ind_ps_max, ind_ps_max or "N/D")}
                    ({f"{vals_ps[ind_ps_max]:.1f}" if ind_ps_max else "N/D"})<br>
                <b>Ponto de Aten\u00e7\u00e3o:</b> {lbl_psico.get(ind_ps_min, ind_ps_min or "N/D")}
                    ({f"{vals_ps[ind_ps_min]:.1f}" if ind_ps_min else "N/D"})<br><br>
                <b>Recomenda\u00e7\u00e3o:</b> Priorizar {recom_txt_ps}.
            </div>
            """, unsafe_allow_html=True)

        with col_dir:
            df_psbar = pd.DataFrame({
                'Indicador':  [lbl_psico.get(c, c) for c in ind_psico if c in vals_ps],
                'Aluno':      [vals_ps[c] for c in ind_psico if c in vals_ps],
                'Refer\u00eancia': [vals_ref_ps[c] for c in ind_psico if c in vals_ps],
            })
            fig_psbar = go.Figure()
            fig_psbar.add_bar(name='Aluno', x=df_psbar['Indicador'], y=df_psbar['Aluno'],
                              marker_color='#7b2ff7',
                              text=[f'{v:.1f}' for v in df_psbar['Aluno']],
                              textposition='outside')
            fig_psbar.add_bar(name=f'M\u00e9dia {pedra_ps or "Grupo"}',
                              x=df_psbar['Indicador'], y=df_psbar['Refer\u00eancia'],
                              marker_color='#ff9f43', opacity=0.7,
                              text=[f'{v:.1f}' for v in df_psbar['Refer\u00eancia']],
                              textposition='outside')
            fig_psbar.update_layout(barmode='group', height=300, yaxis_range=[0, 10.5],
                                    title='Indicadores Psicossociais vs M\u00e9dia do Grupo')
            st.plotly_chart(fig_psbar, use_container_width=True)

            if len(df_f) > 1:
                df_ps_evo = df_f.sort_values('ANO_PEDE')[['ANO_PEDE'] + ind_psico].melt(
                    id_vars='ANO_PEDE', var_name='Indicador', value_name='Nota')
                df_ps_evo['Indicador'] = df_ps_evo['Indicador'].map(lbl_psico).fillna(df_ps_evo['Indicador'])
                fig_ps_evo = px.line(df_ps_evo, x='ANO_PEDE', y='Nota', color='Indicador',
                                     markers=True, title='Evolu\u00e7\u00e3o Psicossocial por Ano',
                                     labels={'ANO_PEDE': 'Ano'})
                fig_ps_evo.update_xaxes(type='category')
                st.plotly_chart(fig_ps_evo, use_container_width=True)
            else:
                ian_v = vals_ps.get('IAN', 0)
                ian_cat = 'Cr\u00edtico (<5)' if ian_v < 5 else 'Aten\u00e7\u00e3o (5-7)' if ian_v < 7 else 'Adequado (7+)'
                cor_ian = '#dc3545' if ian_v < 5 else '#fd7e14' if ian_v < 7 else '#28a745'
                st.markdown(f"""
                <div class="analysis-box" style="border-left-color:{cor_ian}">
                    <span class="analysis-title">\U0001f4ca Adequa\u00e7\u00e3o Idade-S\u00e9rie (IAN)</span><br>
                    <b style="font-size:1.6rem;color:{cor_ian}">{ian_v:.1f}</b> &nbsp;\u2014&nbsp;
                    <b style="color:{cor_ian}">{ian_cat}</b><br><br>
                    Aluno com registro de <b>1 ano</b>.
                    O acompanhamento longitudinal estar\u00e1 dispon\u00edvel a partir do pr\u00f3ximo ciclo.
                </div>
                """, unsafe_allow_html=True)

    # ─── MODO COLETIVO ────────────────────────────────────────────────────────
    else:
        c_x, c_y = st.columns(2)

        with c_x:
            st.markdown("**IPP vs Nível de Defasagem (IAN)**")
            if 'IAN' in df_f.columns and 'IAN_CAT' not in df_f.columns:
                def cat_ian(x): return "Crítico (<5)" if x < 5 else "Atenção (5-7)" if x < 7 else "Adequado (7+)"
                df_f = df_f.copy()
                df_f['IAN_CAT'] = df_f['IAN'].apply(cat_ian)

            if 'IPP' in df_f.columns and 'IAN_CAT' in df_f.columns and df_f['IAN_CAT'].notna().sum() >= 2:
                fig_box2 = px.box(df_f, x='IAN_CAT', y='IPP', color='IAN_CAT',
                                  color_discrete_map={
                                      'Crítico (<5)': '#dc3545',
                                      'Atenção (5-7)': '#fd7e14',
                                      'Adequado (7+)': '#28a745'
                                  },
                                  category_orders={'IAN_CAT': ['Crítico (<5)', 'Atenção (5-7)', 'Adequado (7+)']})
                st.plotly_chart(fig_box2, use_container_width=True)
                st.markdown("""
                <div class="analysis-box">
                    <span class="analysis-title">📝 Análise Psicopedagógica:</span><br>
                    Verifica se o IPP (avaliação psicopedagógica) acompanha o nível de defasagem.
                    Alunos <b>Críticos</b> com IPP baixo indicam coerência entre necessidade e avaliação.
                </div>""", unsafe_allow_html=True)
            else:
                st.info("Dados insuficientes para o boxplot com os filtros selecionados.")

        with c_y:
            st.markdown("**Fatores Determinantes de Risco (Modelo ML)**")
            if modelo and hasattr(modelo, 'feature_importances_'):
                try:
                    feat_imp = pd.DataFrame({
                        'Indicador': modelo.feature_names_in_,
                        'Importância': modelo.feature_importances_
                    }).sort_values(by='Importância', ascending=True)

                    fig_imp = px.bar(feat_imp, x='Importância', y='Indicador', orientation='h',
                                     title='Importância das Features no Modelo de Risco',
                                     color='Importância', color_continuous_scale='Purp')
                    st.plotly_chart(fig_imp, use_container_width=True)

                    top_fat = feat_imp.iloc[-1]['Indicador']
                    st.markdown(f"""
                    <div class="analysis-box">
                        <span class="analysis-title">📝 Inteligência Artificial:</span><br>
                        O modelo identificou <b>{top_fat}</b> como o principal preditor de risco.
                        Monitorar este indicador é a estratégia preventiva mais eficaz.
                    </div>""", unsafe_allow_html=True)
                except Exception:
                    st.info("Dados de importância não disponíveis.")
            else:
                st.info("O modelo atual não possui exportação de Feature Importances nativa.")

# --- TAB 4: LABORATÓRIO EDA ---
with tab4:
    st.subheader("📈 Análise Personalizada")
    metricas_numericas = ['INDE', 'IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV', 'IAN']
    m_disp = [m for m in metricas_numericas if m in df_f.columns]
    
    c_x, c_y, c_z = st.columns(3)
    sel_x = c_x.selectbox("Eixo X", m_disp, index=0)
    sel_y = c_y.selectbox("Eixo Y", m_disp, index=1 if len(m_disp)>1 else 0)
    
    cat_cols_disp = [c for c in ["PEDRA", "FASE", "ANO_PEDE", "GÊNERO"] if c in df_f.columns]
    sel_color = c_z.selectbox("Agrupar por", cat_cols_disp) if cat_cols_disp else None
    
    tipo_viz = st.radio("Selecione a Visualização", ["Dispersão (Scatter)", "Distribuição (Boxplot)", "Matriz de Correlação"], horizontal=True)
    
    if df_f.empty:
        st.info('ℹ️ Sem registros para esta combinação de Filtros.')
        
    try:
        if df_f.empty:
            pass
        elif tipo_viz == "Dispersão (Scatter)":
            fig_eda = px.scatter(df_f, x=sel_x, y=sel_y, color=sel_color, 
                             marginal_x="box", marginal_y="violin", trendline="ols",
                             title=f"Relação entre {sel_y} e {sel_x}")
            st.plotly_chart(fig_eda, use_container_width=True)
            st.markdown(f"""<div class="analysis-box"><b>Interpretando a Dispersão:</b><br>
            A linha de tendência indica a correlação média global entre {sel_y} e {sel_x}. 
            Se inclinada para cima, eles crescem juntos. Os gráficos nas bordas indicam onde está o "grosso" dos dados.</div>""", unsafe_allow_html=True)

        elif tipo_viz == "Distribuição (Boxplot)":
            if sel_color:
                fig_eda = px.box(df_f, x=sel_color, y=sel_y, color=sel_color, points="all",
                                 title=f"Boxplot: {sel_y} segmentado por {sel_color}")
            else:
                fig_eda = px.box(df_f, y=sel_y, points="all", title=f"Boxplot: {sel_y} Geral")
            st.plotly_chart(fig_eda, use_container_width=True)
            st.markdown(f"""<div class="analysis-box"><b>Interpretando a Distribuição:</b><br>
            A linha no meio da "caixa" é a mediana. As bordas da caixa indicam onde estão 50% dos alunos do grupo. 
            Pontos distantes são exceções/outliers e devem ser investigados pela equipe de pedagogia.</div>""", unsafe_allow_html=True)
            
        else:
            fig_eda = px.imshow(df_f[m_disp].corr(), text_auto=".2f", color_continuous_scale='Blues',
                                title="Mapa de Calor: Correlações Lineares de Pearson")
            st.plotly_chart(fig_eda, use_container_width=True)
            st.markdown("""<div class="analysis-box"><b>Interpretando a Correlação:</b><br>
            Valores próximos de +1.00 indicam que os indicadores crescem perfeitamente juntos de forma positiva.
            Valores próximos de 0.00 informam ausência de correlação estatística entre essas colunas.</div>""", unsafe_allow_html=True)
            
    except Exception as e:
        st.warning(f"Ocorreu um erro ao gerar a visualização selecionada: {e}")

# --- TAB 5: SIMULADOR PREDITIVO (PENÚLTIMA) ---
with tab5:
    st.subheader("🤖 Simulador de Risco Preditivo")
    if modelo:
        with st.form("form_simul"):
            c_in, c_out = st.columns([1, 1])
            with c_in:
                st.write("**Preencha os indicadores do aluno:**")
                s_ida = st.slider("IDA (Desempenho Acadêmico)", 0.0, 10.0, 6.0)
                s_ieg = st.slider("IEG (Engajamento)", 0.0, 10.0, 7.0)
                s_iaa = st.slider("IAA (Autoavaliação)", 0.0, 10.0, 8.0)
                s_ips = st.slider("IPS (Psicossocial)", 0.0, 10.0, 7.0)
                s_ipp = st.slider("IPP (Psicopedagógico)", 0.0, 10.0, 7.0)
                s_ipv = st.slider("IPV (Ponto de Virada)", 0.0, 10.0, 6.0)
                s_ian = st.slider("IAN (Adequação Série/Idade)", 0.0, 10.0, 7.0)
                btn_simular = st.form_submit_button("Gerar Diagnóstico", type="primary", use_container_width=True)
            
            with c_out:
                if btn_simular:
                    # Calcular features derivadas
                    engajamento_academico = s_ieg * s_ida
                    suporte_psico = (s_ips + s_ipp) / 2
                    score_geral = (s_ida + s_ieg + s_iaa + s_ips + s_ipp) / 5
                    evolucao_ida = 0.0  # sem histórico disponível na simulação
                    evolucao_ieg = 0.0
                    
                    # Montar DataFrame com as 12 features na ordem correta
                    features_modelo = ['IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV', 'IAN',
                                       'ENGAJAMENTO_ACADEMICO', 'SUPORTE_PSICO', 'SCORE_GERAL',
                                       'EVOLUCAO_IDA', 'EVOLUCAO_IEG']
                    valores = [s_ida, s_ieg, s_iaa, s_ips, s_ipp, s_ipv, s_ian,
                               engajamento_academico, suporte_psico, score_geral,
                               evolucao_ida, evolucao_ieg]
                    
                    entrada = pd.DataFrame([valores], columns=features_modelo)
                    
                    try:
                        prob = modelo.predict_proba(entrada)[0][1]
                    except Exception as err:
                        st.error(f"Erro na predição: {err}")
                        prob = None
                    
                    if prob is not None:
                        st.write("### Diagnóstico da IA")
                        st.metric("Probabilidade de Risco", f"{prob:.1%}")
                        
                        if prob > 0.5:
                            st.error("🚨 **ALTO RISCO DETECTADO**")
                            st.markdown("Recomenda-se acompanhamento psicopedagógico intensivo e revisão do engajamento.")
                        else:
                            st.success("✅ **SITUAÇÃO SOB CONTROLE**")
                            st.markdown("O aluno apresenta indicadores condizentes com a permanência saudável no programa.")
                            
                        st.divider()
                        
                        # Recomendação prescritiva baseada no menor indicador
                        st.markdown("#### Recomendação Pedagógica Automática")
                        indicadores_input = {'IDA': s_ida, 'IEG': s_ieg, 'IAA': s_iaa, 
                                             'IPS': s_ips, 'IPP': s_ipp, 'IPV': s_ipv, 'IAN': s_ian}
                        menor_ind = min(indicadores_input.items(), key=lambda x: x[1])[0]
                        
                        recomenda_prescritiva = {
                            'IDA': "O Desempenho Acadêmico (IDA) está baixo. Focar em aulas de reforço em Matemática e Português.",
                            'IEG': "O Engajamento (IEG) está baixo. Investigar motivos de ausência e propor atividades lúdicas e extracurriculares.",
                            'IAA': "A Autoavaliação (IAA) está baixa. Sugere-se mentorias para trabalhar a autoestima e confiança do aluno.",
                            'IPS': "O Indicador Psicossocial (IPS) requer atenção. Necessário acompanhamento do assistente social junto à família.",
                            'IPP': "O Indicador Psicopedagógico (IPP) dimensiona dificuldades de aprendizagem. Avaliar distúrbios ou adaptar materiais.",
                            'IPV': "O Ponto de Virada (IPV) é o menor indicador. Estimular projetos práticos que promovam autonomia e projeto de vida.",
                            'IAN': "A Adequação de Série/Idade (IAN) indica defasagem escolar. Avaliar possibilidade de aceleração ou nívelamento."
                        }
                        
                        st.info(f"💡 **Foco Prioritário: Módulo {menor_ind}**\n\n{recomenda_prescritiva[menor_ind]}")
    else:
        st.error("Modelo ML não carregado.")

# --- TAB 6: GLOSSÁRIO ---
with tab6:
    st.subheader("📚 Glossário de Termos")
    col_d1, col_d2 = st.columns(2)
    dicionario_p = {
        "INDE": "Índice de Desenvolvimento Educacional (Nota Final, de 0 a 10).",
        "IAN": "Adequação de Nível (Relação entre série escolar e idade).",
        "IDA": "Desempenho Acadêmico (Média das notas escolares).",
        "IEG": "Engajamento (Participação em aulas e entregas de tarefas).",
        "IAA": "Autoavaliação (Percepção do próprio aluno sobre seu progresso).",
        "IPS": "Psicossocial (Avaliação emocional e de ambiente familiar).",
        "IPP": "Psicopedagógico (Capacidade de aprendizado e concentração).",
        "IPV": "Ponto de Virada (Nível de autonomia e maturidade do aluno)."
    }
    with col_d1:
        for k, v in list(dicionario_p.items())[:4]:
            st.markdown(f"**{k}:** {v}")
    with col_d2:
        for k, v in list(dicionario_p.items())[4:]:
            st.markdown(f"**{k}:** {v}")