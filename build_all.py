"""
build_all.py · Emotional Receipts · DTU 02806

Generates the website's interactive Plotly files, the static V3 SVG, and the
JSON payload, then injects the payload into index.html. Re-run any time.

The explainer notebook is a separate file; you run it yourself in Jupyter.

NUMERICAL CONSISTENCY WITH THE NOTEBOOK
---------------------------------------
This script mirrors the explainer notebook exactly on every shared parameter:

  FEATS          ←  notebook cell 11           ·  10 raw audio features
  CORR_FEATS     ←  notebook cell 21 loop      ·  same 10 raw features
                                                  (coping_ratio reported
                                                  separately as a derived
                                                  metric; see §3.3)
  WHR_2019 dict  ←  notebook cell 11           ·  34 country scores
  CONTINENTS     ←  notebook cell 11           ·  17 / 10 / 7 split
  coping_ratio   =  energy / (valence + 0.01)  ·  identical formula
  palette        ←  notebook cell 3            ·  identical hex codes

If you ever change one of these in the notebook, change it here too — the
website prose hard-codes the headline r-values (acousticness, speechiness),
so the script's output and the notebook's output must agree to three decimals.
"""

import os, json, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats as sps

# 0 · visual identity (cyber-noir; consistent with notebook cell 3 + site CSS)
BG, FG     = '#05050a', '#e6e6f0'
CYAN       = '#00e5ff'
MAGENTA    = '#ff2ec4'
AMBER      = '#ffb400'
GRID       = 'rgba(230,230,240,0.06)'
BG_CARD    = '#0d0d18'

mpl.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG,
    'axes.edgecolor': '#3a3a4f', 'axes.labelcolor': FG,
    'xtick.color': FG, 'ytick.color': FG, 'text.color': FG,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Inter', 'Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 11, 'axes.titlesize': 14, 'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False,
})

# 1 · load and aggregate
print('=' * 60)
print('Emotional Receipts · website asset build')
print('=' * 60)

os.makedirs('visualizations', exist_ok=True)
os.makedirs('images',         exist_ok=True)

df = pd.read_csv('final_spotify_dataset.csv')
print(f'\n[load] rows = {len(df):,} · cols = {len(df.columns)} · '
      f"countries = {df['Country'].nunique()}")

# FEATS — must equal the FEATS list in notebook cell 11.
FEATS = ['valence', 'energy', 'danceability', 'acoustics', 'loudness', 'tempo',
         'speechiness', 'instrumentalness', 'liveliness', 'mode']

cf = df.groupby('Country')[FEATS].mean().reset_index()
cf['n_tracks']     = df.groupby('Country').size().values
cf['coping_ratio'] = cf['energy'] / (cf['valence'] + 0.01)
GLOBAL_RATIO = float(cf.loc[cf['Country'] == 'Global', 'coping_ratio'].iloc[0])
print(f'[derive] global coping ratio = {GLOBAL_RATIO:.3f}')

# WHR_2019 — must equal the dictionary in notebook cell 11.
WHR_2019 = {
    'Finland': 7.769, 'Denmark': 7.600, 'Norway': 7.554, 'Netherlands': 7.488,
    'Switzerland': 7.480, 'Sweden': 7.343, 'New Zealand': 7.307, 'Canada': 7.278,
    'Austria': 7.246, 'Australia': 7.228, 'Costa Rica': 7.167, 'UK': 7.054,
    'Ireland': 7.021, 'Germany': 6.985, 'Belgium': 6.923, 'USA': 6.892,
    'Mexico': 6.595, 'France': 6.592, 'Taiwan': 6.446, 'Chile': 6.444,
    'Spain': 6.354, 'Brazil': 6.300, 'Singapore': 6.262, 'Italy': 6.223,
    'Poland': 6.182, 'Colombia': 6.125, 'Argentina': 6.086, 'Ecuador': 6.028,
    'Peru': 5.697, 'Portugal': 5.693, 'Philippines': 5.631, 'Turkey': 5.373,
    'Malaysia': 5.339, 'Indonesia': 5.192,
}

# CONTINENTS — must equal the dictionary in notebook cell 11.
CONTINENTS = {
    'Europe': ['Austria', 'Belgium', 'Denmark', 'Finland', 'France', 'Germany',
               'Ireland', 'Italy', 'Netherlands', 'Norway', 'Poland', 'Portugal',
               'Spain', 'Sweden', 'Switzerland', 'Turkey', 'UK'],
    'Americas': ['Argentina', 'Brazil', 'Canada', 'Chile', 'Colombia', 'Costa Rica',
                 'Ecuador', 'Mexico', 'Peru', 'USA'],
    'Asia/Oceania': ['Australia', 'Indonesia', 'Malaysia', 'New Zealand',
                     'Philippines', 'Singapore', 'Taiwan'],
}
cont_map = {c: k for k, v in CONTINENTS.items() for c in v}
cf['continent'] = cf['Country'].map(cont_map)
cf['whr']       = cf['Country'].map(WHR_2019)

cf_nat = cf[cf['Country'] != 'Global'].copy()
cf_nat['whr_rank'] = cf_nat['whr'].rank(ascending=False, method='min').astype(int)
cf.to_csv('country_features_full.csv', index=False)

# 2 · Pearson correlations vs WHR — mirror notebook §3.2 (cell 21) exactly.
#
# CORR_FEATS is the same 10 raw audio features used in notebook cell 21's
# loop. We test raw features only; coping_ratio is a *derived* compositional
# metric and is reported separately (see notebook §3.3). Display-name rename
# matches the website prose: 'acoustics' is shown as 'acousticness'.

print('\n[stats] Pearson r vs WHR 2019  ·  n = 34  ·  10 raw audio features')
CORR_FEATS = ['valence', 'energy', 'danceability', 'acoustics', 'loudness',
              'tempo', 'speechiness', 'instrumentalness', 'liveliness', 'mode']
DISPLAY_RENAME = {'acoustics': 'acousticness'}  # liveliness left as-is to
                                                # match notebook cell 21 output

corr_rows = []
for c in CORR_FEATS:
    r, p = sps.pearsonr(cf_nat[c], cf_nat['whr'])
    corr_rows.append({'feat': DISPLAY_RENAME.get(c, c), 'r': r, 'p': p})
    sig = '  *' if p < 0.05 else ''
    print(f'   {c:18s}  r = {r:+.3f}   p = {p:.4f}{sig}')

# Derived metric — coping_ratio reported as one extra line, matching the
# notebook cell 21 footer ("[derived] coping_ratio  r = +0.176  p = 0.3180").
r_cr, p_cr = sps.pearsonr(cf_nat['coping_ratio'], cf_nat['whr'])
print(f'   {"[derived] coping_ratio":21s} r = {r_cr:+.3f}   p = {p_cr:.4f}')
corr_derived = {'feat': 'coping_ratio', 'r': r_cr, 'p': p_cr}

# 3 · plotly visualisations (4 standalone html files)
PLOTLY_CDN = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>'

def plotly_html(div_id, data, layout, extra_html='', extra_js=''):
    cfg = {'responsive': True, 'displayModeBar': False}
    return (
        f'<!DOCTYPE html><html><head><meta charset="utf-8"/><style>'
        f'html,body{{margin:0;background:{BG};overflow:hidden;height:100%;'
        f'font-family:Inter,sans-serif;color:{FG}}}'
        f'#{div_id}{{height:100vh;width:100vw}}</style>{PLOTLY_CDN}</head><body>'
        f'{extra_html}<div id="{div_id}"></div><script>'
        f'Plotly.newPlot("{div_id}",{json.dumps(data)},{json.dumps(layout)},{json.dumps(cfg)});'
        f'{extra_js}</script></body></html>'
    )

# v1 · choropleth (orthographic globe with zoom buttons)
print('\n[viz]  V1 · choropleth')
v1_data = [{
    'type': 'choropleth',
    'locations': cf_nat['Country'].tolist(),
    'locationmode': 'country names',
    'z': cf_nat['coping_ratio'].round(3).tolist(),
    'text': cf_nat['Country'].tolist(),
    'customdata': np.column_stack([
        cf_nat['valence'].round(3), cf_nat['energy'].round(3),
        cf_nat['whr'].round(2),     cf_nat['whr_rank'], cf_nat['n_tracks']
    ]).tolist(),
    'colorscale': [[0, MAGENTA], [0.5, AMBER], [1, CYAN]],
    'zmin': 1.10, 'zmax': 1.40,
    'marker': {'line': {'color': BG, 'width': 0.5}},
    'colorbar': {
        'title': {'text': 'COPING RATIO<br>energy / (valence + 0.01)',
                  'font': {'color': FG, 'size': 11, 'family': 'Inter'}},
        'tickfont': {'color': FG, 'size': 11},
        'tickvals': [1.10, GLOBAL_RATIO, 1.40],
        'ticktext': [f'{1.10:.2f}  quiet-cope',
                 f'{GLOBAL_RATIO:.3f}  baseline',
                 f'{1.40:.2f}  loud-cope'],
        'len': 0.7, 'thickness': 16, 'x': 0.95,
    },
    'hovertemplate': (
        '<b>%{text}</b><br>'
        'Coping ratio: <b>%{z:.3f}</b><br>'
        'WHR rank: #%{customdata[3]} (score %{customdata[2]:.2f})<br>'
        'Valence: %{customdata[0]:.3f}  ·  Energy: %{customdata[1]:.3f}<br>'
        'Tracks: %{customdata[4]:,}<extra></extra>'
    ),
}]
v1_layout = {
    'paper_bgcolor': BG, 'plot_bgcolor': BG,
    'geo': {
        'bgcolor': BG, 'lakecolor': BG, 'landcolor': 'rgba(255,255,255,0.04)',
        'showcountries': True, 'countrycolor': 'rgba(255,255,255,0.12)',
        'projection': {'type': 'orthographic',
                       'rotation': {'lon': 10, 'lat': 30, 'roll': 0}},
        'showframe': False, 'showcoastlines': True,
        'coastlinecolor': 'rgba(0,229,255,0.25)',
    },
    'font': {'color': FG, 'family': 'Inter'}, 'margin': {'l': 0, 'r': 0, 't': 20, 'b': 0},
}
zoom_html = (
    '<div style="position:absolute;top:14px;right:14px;z-index:50;'
    'display:flex;flex-direction:column;gap:6px">'
    f'<button id="zin" style="width:34px;height:34px;background:{BG_CARD};'
    f'border:1px solid {CYAN};color:{CYAN};border-radius:3px;cursor:pointer;'
    'font-size:18px;line-height:1">+</button>'
    f'<button id="zout" style="width:34px;height:34px;background:{BG_CARD};'
    f'border:1px solid {CYAN};color:{CYAN};border-radius:3px;cursor:pointer;'
    'font-size:18px;line-height:1">−</button></div>'
)
zoom_js = (
    'var z=1;'
    'document.getElementById("zin").onclick=function(){z*=1.4;'
    'Plotly.relayout("v1",{"geo.projection.scale":z});};'
    'document.getElementById("zout").onclick=function(){z=Math.max(1,z/1.4);'
    'Plotly.relayout("v1",{"geo.projection.scale":z});};'
)
open('visualizations/v1_choropleth.html', 'w').write(
    plotly_html('v1', v1_data, v1_layout, extra_html=zoom_html, extra_js=zoom_js))

# v2 · valence × energy scatter
print('[viz]  V2 · valence x energy scatter')
v2_data = []
for cont, color in [('Europe', CYAN), ('Americas', MAGENTA), ('Asia/Oceania', AMBER)]:
    sub = cf_nat[cf_nat['continent'] == cont]
    v2_data.append({
        'type': 'scatter', 'mode': 'markers+text', 'name': cont,
        'x': sub['valence'].round(3).tolist(),
        'y': sub['energy'].round(3).tolist(),
        'text': sub['Country'].tolist(),
        'textposition': 'middle center',
        'textfont': {'size': 7, 'color': BG, 'family': 'Inter', 'weight': 700},
        'customdata': np.column_stack([
            sub['coping_ratio'].round(3), sub['whr'].round(2), sub['whr_rank']
        ]).tolist(),
        'marker': {'size': 30, 'color': color,
                   'line': {'color': BG, 'width': 1.5}, 'opacity': 0.92},
        'hovertemplate': (
            f'<b>%{{text}}</b>  ({cont})<br>'
            'WHR rank: #%{customdata[2]}  (score %{customdata[1]:.2f})<br>'
            'Valence: %{x:.3f}  ·  Energy: %{y:.3f}<br>'
            'Coping ratio: %{customdata[0]:.3f}<extra></extra>'
        ),
    })
v2_layout = {
    'paper_bgcolor': BG, 'plot_bgcolor': BG, 'font': {'color': FG, 'family': 'Inter'},
    'xaxis': {'range': [0.40, 0.64], 'showgrid': True, 'gridcolor': GRID, 'zeroline': False,
              'title': {'text': 'VALENCE  (musical positiveness)  →',
                        'font': {'color': FG, 'size': 12}}},
    'yaxis': {'range': [0.55, 0.73], 'showgrid': True, 'gridcolor': GRID, 'zeroline': False,
              'title': {'text': 'ENERGY  (intensity)  →',
                        'font': {'color': FG, 'size': 12}}},
    'legend': {'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02,
               'xanchor': 'right', 'x': 1,
               'font': {'color': FG, 'size': 11},
               'bgcolor': 'rgba(0,0,0,0)', 'bordercolor': 'rgba(0,0,0,0)'},
    'margin': {'l': 60, 'r': 30, 't': 50, 'b': 60}, 'hovermode': 'closest',
}
open('visualizations/v2_scatter.html', 'w').write(plotly_html('v2', v2_data, v2_layout))

# v3 · static correlations bar (matplotlib SVG for the website)
# Now plots 10 raw features — matching notebook Figure 4 (§3.2).
# Figure is slightly taller (5.5") to accommodate the two extra bars and
# keep the y-label spacing readable.
print('[viz]  V3 · correlation bars (static SVG · 10 raw features)')
cdf = pd.DataFrame(corr_rows).sort_values('r').reset_index(drop=True)
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=140)
colors = [CYAN if (p < 0.05 and r > 0) else (MAGENTA if p < 0.05 else '#5a5a6c')
          for r, p in zip(cdf['r'], cdf['p'])]
ax.barh(cdf['feat'], cdf['r'], color=colors, height=0.62,
        edgecolor=BG, linewidth=0.4)
for i, (r, p) in enumerate(zip(cdf['r'], cdf['p'])):
    ax.text(r + (0.018 if r > 0 else -0.018), i,
            f'{r:+.2f}{" *" if p < 0.05 else ""}',
            va='center', ha='left' if r > 0 else 'right',
            fontsize=11, color=FG, weight='bold' if p < 0.05 else 'normal')
ax.axvline(0, color=FG, linewidth=1.2)
ax.set_xlabel('Pearson r  vs  World Happiness 2019  (n = 34)',
              labelpad=10, color=FG)
ax.set_title('What audio features actually predict national happiness?\n* = p < 0.05',
             pad=18, fontsize=14, color=FG)
ax.set_xlim(-0.75, 0.6)
plt.tight_layout()
plt.savefig('images/v3_correlations.svg', bbox_inches='tight', facecolor=BG)
plt.close()

# v4 · genre over/under-indexing heatmap
print('[viz]  V4 · genre heatmap (continent dropdown · cell values · clear gaps)')
top_genres = df['Genre_new'].value_counts().head(10).index.tolist()
gmix       = df.groupby(['Country', 'Genre_new']).size().unstack(fill_value=0)
gmix_pct   = gmix.div(gmix.sum(axis=1), axis=0) * 100
global_pct = gmix_pct.loc['Global']
ratio_full = gmix_pct[top_genres].div(global_pct[top_genres], axis=1)
log2_full  = np.log2(ratio_full.replace(0, np.nan)).fillna(-4)
log2_full  = log2_full.drop('Global', errors='ignore')
log2_full  = log2_full.loc[[c for c in log2_full.index if c in cont_map]]

cont_payload = {}
for cont in CONTINENTS:
    members = [c for c in CONTINENTS[cont] if c in log2_full.index]
    sub = log2_full.loc[members].sort_index()
    cont_payload[cont] = {
        'z': sub.values.round(2).tolist(),
        'y': sub.index.tolist(),
        'x': sub.columns.tolist(),
    }
initial = 'Europe'
v4_data = [{
    'type': 'heatmap',
    'z': cont_payload[initial]['z'],
    'y': cont_payload[initial]['y'],
    'x': cont_payload[initial]['x'],
    'colorscale': [[0, CYAN], [0.45, '#15151f'], [0.55, '#15151f'], [1, MAGENTA]],
    'zmin': -3, 'zmax': 3, 'xgap': 4, 'ygap': 4,
    'texttemplate': '%{z:.1f}',
    'textfont': {'family': 'JetBrains Mono, monospace', 'size': 11, 'color': FG},
    'colorbar': {
        'title': {'text': 'log₂ ratio<br>P(genre|country)<br>P(genre|global)',
                  'font': {'color': FG, 'size': 10}},
        'tickfont': {'color': FG, 'size': 10},
        'tickvals': [-3, -1, 0, 1, 3],
        'ticktext': ['−3  much less', '−1', '0  baseline', '+1', '+3  much more'],
        'len': 0.85, 'thickness': 14, 'x': 1.02,
    },
    'hovertemplate': '<b>%{y}</b><br>%{x}: log₂ = %{z:.2f}<extra></extra>',
}]
v4_buttons = [{
    'method': 'restyle', 'label': cont,
    'args': [{
        'z': [cont_payload[cont]['z']],
        'y': [cont_payload[cont]['y']],
        'x': [cont_payload[cont]['x']],
    }],
} for cont in ['Europe', 'Americas', 'Asia/Oceania']]
v4_layout = {
    'paper_bgcolor': BG, 'plot_bgcolor': BG,
    'font': {'color': FG, 'family': 'Inter'},
    'title': {'text': 'Genre over- and under-indexing  ·  hover for log₂ ratio',
              'x': 0.5, 'xanchor': 'center', 'font': {'color': FG, 'size': 15},
              'y': 0.97},
    'xaxis': {'tickangle': -30, 'tickfont': {'color': FG, 'size': 11},
              'side': 'bottom', 'ticks': '', 'showgrid': False},
    'yaxis': {'tickfont': {'color': FG, 'size': 11},
              'autorange': 'reversed', 'ticks': '', 'showgrid': False},
    'updatemenus': [{
        'buttons': v4_buttons, 'direction': 'down', 'showactive': True,
        'x': 0.02, 'y': 1.10, 'xanchor': 'left', 'yanchor': 'top',
        'bgcolor': BG_CARD, 'bordercolor': CYAN, 'borderwidth': 1,
        'font': {'color': FG, 'family': 'Inter', 'size': 12},
        'pad': {'l': 10, 'r': 10, 't': 6, 'b': 6},
    }],
    'annotations': [{
        'x': 0.02, 'y': 1.18, 'xref': 'paper', 'yref': 'paper',
        'xanchor': 'left', 'yanchor': 'bottom',
        'text': 'CONTINENT', 'showarrow': False,
        'font': {'color': FG, 'family': 'Inter', 'size': 10},
    }],
    'margin': {'l': 130, 'r': 100, 't': 130, 'b': 80},
}
open('visualizations/v4_genre_heatmap.html', 'w').write(
    plotly_html('v4', v4_data, v4_layout))

# v5 · country radar (no internal controls; parent-controlled)
print('[viz]  V5 · country profile radar (parent-controlled via postMessage)')
RFEATS = ['valence', 'energy', 'danceability', 'acousticness', 'speechiness', 'coping_ratio']
RCOL_MAP = {
    'valence': 'valence', 'energy': 'energy', 'danceability': 'danceability',
    'acousticness': 'acoustics', 'speechiness': 'speechiness',
    'coping_ratio': 'coping_ratio',
}
ranges = {}
for k in RFEATS:
    col = cf_nat[RCOL_MAP[k]]
    ranges[k] = (float(col.min()), float(col.max()))

def radar_raw(name):
    row = cf[cf['Country'] == name].iloc[0]
    return {k: float(row[RCOL_MAP[k]]) for k in RFEATS}

def radar_norm(name):
    raw = radar_raw(name)
    out = []
    for k in RFEATS:
        lo, hi = ranges[k]
        out.append(0.5 if hi == lo else (raw[k] - lo) / (hi - lo))
    return out

global_norm  = radar_norm('Global')
country_norm = {c: radar_norm(c) for c in cf_nat['Country']}
country_raw  = {c: radar_raw(c)  for c in cf_nat['Country']}

v5_payload = json.dumps({
    'feats': RFEATS, 'norm': country_norm, 'raw': country_raw,
    'ranges': ranges, 'global_norm': global_norm,
}, ensure_ascii=False)

v5_html = f'''<!DOCTYPE html><html><head><meta charset="utf-8"/>
<style>
html,body{{margin:0;padding:0;background:{BG};color:{FG};
font-family:Inter,Helvetica Neue,Arial,sans-serif;height:100%;overflow:hidden}}
#radar{{height:100vh;width:100vw}}
</style>
{PLOTLY_CDN}</head><body>
<div id="radar"></div>
<script>
const D = {v5_payload};
const FEATS = D.feats;
let currentCountry = 'Finland';

function buildTraces(c){{
  const r = D.norm[c].concat(D.norm[c][0]);
  const theta = FEATS.concat(FEATS[0]);
  const raw = D.raw[c];
  const customRaw = FEATS.map(f => raw[f].toFixed(3)).concat(raw[FEATS[0]].toFixed(3));
  const gr = D.global_norm.concat(D.global_norm[0]);
  return [
    {{type:'scatterpolar', r:gr, theta:theta, name:'Global baseline',
      line:{{color:'{AMBER}', width:1.5, dash:'dot'}},
      fill:'none', hoverinfo:'skip', showlegend:true}},
    {{type:'scatterpolar', r:r, theta:theta, customdata:customRaw,
      fill:'toself', name:c,
      line:{{color:'{MAGENTA}', width:3}},
      fillcolor:'rgba(255,46,196,0.18)',
      hovertemplate:'<b>'+c+'</b><br>%{{theta}}: %{{customdata}}  '+
                    '(norm %{{r:.2f}})<extra></extra>'}}
  ];
}}
function layout(c){{
  return {{
    paper_bgcolor:'{BG}', plot_bgcolor:'{BG}',
    font:{{color:'{FG}', family:'Inter'}},
    polar:{{bgcolor:'rgba(0,0,0,0)',
      radialaxis:{{visible:true, range:[0,1], showgrid:false,
        tickfont:{{color:'{FG}', size:10}},
        tickvals:[0,0.25,0.5,0.75,1], showline:false}},
      angularaxis:{{showgrid:false,
        tickfont:{{color:'{FG}', size:12, family:'Inter'}}}}}},
    showlegend:true,
    legend:{{orientation:'h', yanchor:'top', y:-0.04,
      xanchor:'center', x:0.5,
      font:{{color:'{FG}', size:11}}, bgcolor:'rgba(0,0,0,0)'}},
    title:{{text: c + '  ·  audio fingerprint',
      x:0.5, xanchor:'center',
      font:{{color:'{FG}', family:'Fraunces, Georgia, serif', size:20}}}},
    margin:{{l:60, r:60, t:70, b:60}}
  }};
}}
function render(c){{
  if (!D.norm[c]) return;
  currentCountry = c;
  Plotly.react('radar', buildTraces(c), layout(c),
               {{responsive:true, displayModeBar:false}});
}}
window.addEventListener('message', function(e){{
  if (e.data && e.data.type === 'set-country' && e.data.country) {{
    render(e.data.country);
  }}
}});
render(currentCountry);
if (window.parent && window.parent !== window) {{
  window.parent.postMessage({{type:'radar-ready'}}, '*');
}}
</script>
</body></html>'''
open('visualizations/v5_profile.html', 'w').write(v5_html)

# 4 · JSON payload (inlined into index.html)
#
# `corr` carries the 10 raw-feature correlations from §3.2 (matches notebook
# Figure 4). `corr_derived` carries the single coping_ratio correlation from
# §3.3 (matches the [derived] line in notebook cell 21). Both are exposed so
# the website prose has both available to quote from.
print('\n[payload] building JSON for index.html')

genre_payload = {}
for c in gmix_pct.index:
    top6 = gmix_pct.loc[c].sort_values(ascending=False).head(6)
    genre_payload[c] = [{'genre': str(g), 'pct': round(float(p), 1)}
                        for g, p in top6.items() if p > 0]

countries_payload = []
for _, row in cf.iterrows():
    rank = (int(cf_nat[cf_nat['Country'] == row['Country']]['whr_rank'].iloc[0])
            if (row['Country'] != 'Global' and pd.notna(row['whr'])) else None)
    countries_payload.append({
        'country':      row['Country'],
        'continent':    row['continent'] if pd.notna(row['continent']) else None,
        'n_tracks':     int(row['n_tracks']),
        'valence':      round(float(row['valence']), 3),
        'energy':       round(float(row['energy']), 3),
        'coping_ratio': round(float(row['coping_ratio']), 3),
        'acousticness': round(float(row['acoustics']), 3),
        'danceability': round(float(row['danceability']), 3),
        'speechiness':  round(float(row['speechiness']), 3),
        'whr':          round(float(row['whr']), 3) if pd.notna(row['whr']) else None,
        'whr_rank':     rank,
    })
global_row = next(c for c in countries_payload if c['country'] == 'Global')

payload = {
    'countries':    countries_payload,
    'genre_mix':    genre_payload,
    'global':       global_row,
    'global_ratio': round(GLOBAL_RATIO, 3),
    'continents':   CONTINENTS,
    # 10 raw-feature correlations from §3.2 — drives the V3 SVG.
    'corr':         [{'feat': r['feat'], 'r': round(r['r'], 3),
                      'p': round(r['p'], 4)} for r in corr_rows],
    # Derived metric — coping_ratio correlation from §3.3.
    'corr_derived': {'feat': corr_derived['feat'],
                     'r': round(corr_derived['r'], 3),
                     'p': round(corr_derived['p'], 4)},
    'meta': {
        'n_tracks':    int(len(df)),
        'n_unique':    int(df['Uri'].nunique()),
        'n_countries': 34,
        'n_features':  10,
        'year_range':  '2017-2020',
    },
}

def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    if hasattr(o, 'item'):  return o.item()
    return o
payload = clean(payload)
with open('web_payload.json', 'w') as f:
    json.dump(payload, f, ensure_ascii=False)

# inject payload into index.html — idempotent: replaces whatever is currently
# inside <script id="payload">…</script> (placeholder or stale data, both fine)

print('[site]    injecting payload into index.html')
if os.path.exists('index.html'):
    h = open('index.html').read()
    new_payload_str = (json.dumps(payload, ensure_ascii=False)
                       .replace('</', '<\\/'))   # safe inside <script>
    def _repl(m): return m.group(1) + new_payload_str + m.group(2)
    new_h, n = re.subn(
        r'(<script id="payload" type="application/json">).*?(</script>)',
        _repl, h, flags=re.DOTALL, count=1,
    )
    if n > 0:
        open('index.html', 'w').write(new_h)
        print('           -> payload injected ({} bytes)'.format(len(new_payload_str)))
    else:
        print('           -> WARNING: <script id="payload"> tag not found in index.html')
        print('           -> the file may be corrupted; restore from the original')
else:
    print('           -> index.html not present; skipping')

print(f"\n{'=' * 60}\nDONE\n{'=' * 60}")
for p in ['visualizations/v1_choropleth.html',
          'visualizations/v2_scatter.html',
          'visualizations/v4_genre_heatmap.html',
          'visualizations/v5_profile.html',
          'images/v3_correlations.svg',
          'country_features_full.csv', 'web_payload.json']:
    if os.path.exists(p):
        print(f'  {p:36s}  ({os.path.getsize(p) // 1024} KB)')
print('\nNow run the explainer notebook in Jupyter to generate inline figures.')
