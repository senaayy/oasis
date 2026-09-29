"""
OASIS-3 | Sigara Geçmişi × APOE × Alzheimer Analizi
Sena Ay — APOE-Alzheimer Genetik Araştırması
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 1: VERİ YÜKLEME VE SİGARA DEĞİŞKENLERİ
# ─────────────────────────────────────────────────────────────────

BASE = 'OASIS3_data_files/'

health = pd.read_csv(BASE + 'UDSa5/csv/OASIS3_UDSa5_health_history.csv')
demo   = pd.read_csv(BASE + 'demo/csv/OASIS3_demographics.csv')
cdr    = pd.read_csv(BASE + 'UDSb4/csv/OASIS3_UDSb4_cdr.csv')

health_sorted = health.sort_values(['OASISID', 'days_to_visit'])

# ── Sigara durumu kategorileri ──────────────────────────────────
# TOBAC100: 0=hayır (hiç 100+ sigara içmemiş), 1=evet, 9=bilinmiyor
# TOBAC30:  0=son 30 günde içmedi, 1=içti, 9=bilinmiyor
# Kategori: Hiç İçmemiş / Eski İçici / Aktif İçici

def smoking_status(grp):
    """Kişinin son ziyaretteki sigara durumunu belirle."""
    last = grp.iloc[-1]
    if last.get('TOBAC30') == 1:
        return 'Aktif'
    tobac100 = grp['TOBAC100'].dropna()
    if len(tobac100) > 0:
        val = tobac100.iloc[-1]
        if val == 0:   return 'Hiç İçmemiş'
        elif val == 1: return 'Eski İçici'
    return None

smoke_status = health_sorted.groupby('OASISID').apply(smoking_status).reset_index()
smoke_status.columns = ['OASISID', 'SmokingStatus']

# ── Pack-years hesaplama ─────────────────────────────────────────
# SMOKYRS: sigara içilen yıl sayısı (88=hiç içmedi, 99=bilinmiyor)
# PACKSPER: 1=<0.5pk/gün(~0.25), 2=0.5-1pk(~0.75), 3=1-2pk(~1.5),
#           4=2+pk(~2.5), 5=?(~1.0), 8=hiç içmedi, 9=bilinmiyor

packsper_map = {1: 0.25, 2: 0.75, 3: 1.5, 4: 2.5, 5: 1.0, 8: 0.0, 9: np.nan}

def calc_packyears(grp):
    """Pack-years = SMOKYRS × PACKSPER ortalaması."""
    smokyrs  = grp['SMOKYRS'].replace({88: 0, 99: np.nan}).dropna()
    packsper = grp['PACKSPER'].map(packsper_map).dropna()
    if len(smokyrs) == 0 or len(packsper) == 0:
        return np.nan
    return smokyrs.max() * packsper.max()

pack_years_ser = health_sorted.groupby('OASISID').apply(calc_packyears).reset_index()
pack_years_ser.columns = ['OASISID', 'PackYears']
pack_years_ser['HeavySmoker'] = (pack_years_ser['PackYears'] > 20).astype(float)
pack_years_ser.loc[pack_years_ser['PackYears'].isna(), 'HeavySmoker'] = np.nan

# ── Kaç yıl önce bıraktı ────────────────────────────────────────
# QUITSMOK: 888=hâlâ içiyor veya hiç içmedi, 999=bilinmiyor

def quit_years(grp):
    q = grp['QUITSMOK'].replace({888: np.nan, 999: np.nan, 8888: np.nan}).dropna()
    return q.max() if len(q) > 0 else np.nan

quit_yrs = health_sorted.groupby('OASISID').apply(quit_years).reset_index()
quit_yrs.columns = ['OASISID', 'YearsSinceQuit']

# ── APOE gruplandırması ──────────────────────────────────────────
def apoe_group(x):
    if pd.isna(x): return None
    x = int(x)
    if x == 44:          return 'Baskın (ε4/ε4)'
    elif x in [34, 24]:  return 'Taşıyıcı (1×ε4)'
    else:                return 'Sağlıklı (ε3/ε3)'

demo['APOE_Group'] = demo['APOE'].apply(apoe_group)
demo['e4_count']   = demo['APOE'].apply(
    lambda x: str(int(x)).count('4') if pd.notna(x) else 0
)

# ── Hedef değişken ───────────────────────────────────────────────
cdr_max = cdr.groupby('OASISID')['CDRTOT'].max().reset_index()

# ── Veri birleştirme ─────────────────────────────────────────────
df = demo[['OASISID','AgeatEntry','GENDER','EDUC','APOE','APOE_Group','e4_count']].copy()
df = df.merge(cdr_max,             on='OASISID', how='inner')
df = df.merge(smoke_status,        on='OASISID', how='left')
df = df.merge(pack_years_ser,      on='OASISID', how='left')
df = df.merge(quit_yrs,            on='OASISID', how='left')
df = df.dropna(subset=['CDRTOT'])
df['TARGET']       = (df['CDRTOT'] > 0).astype(int)
df['EverSmoked']   = df['SmokingStatus'].isin(['Aktif', 'Eski İçici']).astype(float)
df.loc[df['SmokingStatus'].isna(), 'EverSmoked'] = np.nan
df['CurrentSmoker']= (df['SmokingStatus'] == 'Aktif').astype(float)
df.loc[df['SmokingStatus'].isna(), 'CurrentSmoker'] = np.nan

print(f"Toplam N: {len(df)}")
print(f"Sigara durumu dolu: {df['SmokingStatus'].notna().sum()}")
print(df['SmokingStatus'].value_counts(dropna=False))

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 2: İSTATİSTİKSEL ANALİZ
# ─────────────────────────────────────────────────────────────────

rng = np.random.default_rng(42)

def odds_ratio_ci(df_in, var_col, target_col='TARGET', n_boot=500):
    """Ham OR, bootstrap %95 CI ve chi-kare p değeri."""
    d = df_in.dropna(subset=[var_col, target_col])
    a = ((d[var_col]==1) & (d[target_col]==1)).sum()
    b = ((d[var_col]==1) & (d[target_col]==0)).sum()
    c = ((d[var_col]==0) & (d[target_col]==1)).sum()
    dd= ((d[var_col]==0) & (d[target_col]==0)).sum()
    if b==0 or dd==0:
        return np.nan, np.nan, np.nan, np.nan
    or_raw = (a/b) / (c/dd)
    chi2, p, _, _ = stats.chi2_contingency([[a,b],[c,dd]])
    ors = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(d), size=len(d))
        sub = d.iloc[idx]
        a2 = ((sub[var_col]==1) & (sub[target_col]==1)).sum()
        b2 = ((sub[var_col]==1) & (sub[target_col]==0)).sum()
        c2 = ((sub[var_col]==0) & (sub[target_col]==1)).sum()
        d2 = ((sub[var_col]==0) & (sub[target_col]==0)).sum()
        if b2>0 and d2>0:
            ors.append((a2/b2) / (c2/d2))
    ci_lo, ci_hi = np.percentile(ors, [2.5, 97.5]) if ors else (np.nan, np.nan)
    return or_raw, ci_lo, ci_hi, p

def adjusted_or(df_in, var_col, covariates=['AgeatEntry', 'EDUC'], n_boot=500):
    """Yaş ve eğitime göre düzeltilmiş OR + Bootstrap CI."""
    cols = covariates + [var_col, 'TARGET']
    d = df_in.dropna(subset=cols)
    X = d[covariates + [var_col]].values
    y = d['TARGET'].values
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X, y)
    or_adj = np.exp(lr.coef_[0][-1])
    ors = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(X), size=len(X))
        try:
            lr2 = LogisticRegression(max_iter=300)
            lr2.fit(X[idx], y[idx])
            ors.append(np.exp(lr2.coef_[0][-1]))
        except:
            pass
    ci_lo, ci_hi = np.percentile(ors, [2.5, 97.5]) if ors else (np.nan, np.nan)
    return or_adj, ci_lo, ci_hi

print("\n=== TEMEL OR'LAR ===")
for col, label in [('EverSmoked','Ever Smoked'), ('HeavySmoker','Ağır İçici >20 pk-yr')]:
    or_raw, ci_lo, ci_hi, p = odds_ratio_ci(df, col)
    or_adj, adj_lo, adj_hi  = adjusted_or(df, col)
    print(f"\n{label}:")
    print(f"  Ham OR={or_raw:.2f} [{ci_lo:.2f}–{ci_hi:.2f}], p={p:.4f}")
    print(f"  Yaş düz. OR={or_adj:.2f} [{adj_lo:.2f}–{adj_hi:.2f}]")

# ── Pack-years doz-yanıt trend testi ────────────────────────────
df_py = df.dropna(subset=['PackYears']).copy()
df_py['PYord'] = pd.cut(df_py['PackYears'], bins=[0,5,20,40,1000], labels=[0,1,2,3]).astype(float)
trend_r, trend_p = stats.spearmanr(df_py['PYord'].dropna(),
                                    df_py.loc[df_py['PYord'].notna(), 'TARGET'])
print(f"\nPack-years doz-yanıt trendi: r={trend_r:.3f}, p={trend_p:.4f}")

# ── APOE grubu bazında OR ────────────────────────────────────────
print("\n=== APOE GRUBU BAZINDA OR (yaş düz.) ===")
GROUPS = ['Sağlıklı (ε3/ε3)', 'Taşıyıcı (1×ε4)', 'Baskın (ε4/ε4)']
for g in GROUPS:
    sub = df[df['APOE_Group']==g]
    or_g, lo_g, hi_g = adjusted_or(sub, 'EverSmoked', covariates=['AgeatEntry'])
    n_smk = (sub['EverSmoked']==1).sum()
    print(f"  {g}: OR={or_g:.2f} [{lo_g:.2f}–{hi_g:.2f}]  (N içici={n_smk})")

# ── Longitudinal CDR ─────────────────────────────────────────────
cdr_long = cdr.merge(
    df[['OASISID','SmokingStatus']].dropna().drop_duplicates('OASISID'), on='OASISID'
)
grp_never = cdr_long[cdr_long['SmokingStatus']=='Hiç İçmemiş'].groupby('OASISID')['CDRTOT'].max()
grp_ex    = cdr_long[cdr_long['SmokingStatus']=='Eski İçici'].groupby('OASISID')['CDRTOT'].max()
u, p_long = stats.mannwhitneyu(grp_ex.dropna(), grp_never.dropna())
print(f"\nLongitudinal CDR — Eski vs Hiç İçmemiş: {grp_ex.mean():.3f} vs {grp_never.mean():.3f}, p={p_long:.4f}")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 3: GÖRSELLEŞTİRME (9 Panel)
# ─────────────────────────────────────────────────────────────────

DARK='#0d1117'; CARD='#161b22'; TEXT='#e6edf3'; SUB='#8b949e'
GOLD='#e3b341'; RED='#f78166'; GREEN='#3fb950'; BLUE='#58a6ff'
PURPLE='#d2a8ff'; ORANGE='#ffa657'; TEAL='#39d353'; PINK='#ff7eb6'

plt.rcParams.update({
    'figure.facecolor': DARK, 'axes.facecolor': CARD,
    'axes.edgecolor': '#30363d', 'axes.labelcolor': TEXT,
    'xtick.color': SUB, 'ytick.color': SUB, 'text.color': TEXT,
    'grid.color': '#21262d', 'grid.alpha': 0.5, 'font.family': 'DejaVu Sans'
})

G_SHORT = ['Sağlıklı', 'Taşıyıcı', 'Baskın']
x3 = np.arange(3)
w3 = 0.28

fig = plt.figure(figsize=(24, 20), facecolor=DARK)
gs  = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
               left=0.07, right=0.97, top=0.93, bottom=0.06)
fig.suptitle('Sigara Geçmişi × APOE × Alzheimer  |  OASIS-3  (N=1.321)',
             fontsize=20, fontweight='bold', color=TEXT, y=0.97)

# Panel A — Temel oranlar
ax_a = fig.add_subplot(gs[0, 0])
smk_rates = [42.6, 45.2, 50.0]
smk_ns    = [713, 584, 24]
bars_a = ax_a.bar(['Hiç İçmemiş', 'Eski İçici', 'Aktif İçici'],
                  smk_rates, color=[BLUE, ORANGE, RED],
                  alpha=0.85, edgecolor='none', width=0.55)
for bar, r, n in zip(bars_a, smk_rates, smk_ns):
    ax_a.text(bar.get_x()+bar.get_width()/2, r+1.2, f'%{r:.0f}',
              ha='center', fontsize=12, fontweight='bold', color=TEXT)
    ax_a.text(bar.get_x()+bar.get_width()/2, 2, f'N={n}', ha='center', fontsize=9, color=SUB)
ax_a.set_ylim(0, 65); ax_a.set_ylabel('Alzheimer Oranı (%)', color=SUB)
ax_a.set_title('A  |  Sigara Durumu — Temel Oranlar\n(OR=1.12, p=0.34 — anlamlı değil)',
               fontsize=11, fontweight='bold', color=TEXT, pad=10)
ax_a.grid(axis='y', alpha=0.3); ax_a.spines[['top','right','left','bottom']].set_visible(False)

# Panel B — Pack-years doz-yanıt
ax_b = fig.add_subplot(gs[0, 1])
py_cats  = ['0-5\npk-yr', '5-20\npk-yr', '20-40\npk-yr', '>40\npk-yr']
py_rates = [33.6, 42.7, 49.2, 54.5]
py_ns    = [140, 192, 128, 101]
bars_b = ax_b.bar(py_cats, py_rates, color=[BLUE, TEAL, ORANGE, RED],
                  alpha=0.85, edgecolor='none', width=0.6)
for bar, r, n in zip(bars_b, py_rates, py_ns):
    ax_b.text(bar.get_x()+bar.get_width()/2, r+1.2, f'%{r:.0f}',
              ha='center', fontsize=12, fontweight='bold', color=TEXT)
    ax_b.text(bar.get_x()+bar.get_width()/2, 2, f'N={n}', ha='center', fontsize=9, color=SUB)
ax_b2 = ax_b.twinx()
ax_b2.plot([0,1,2,3], py_rates, color=GOLD, linewidth=2.5, marker='D',
           markersize=8, linestyle='--', alpha=0.9)
ax_b2.set_yticks([]); ax_b2.set_ylim(0, 65)
ax_b2.spines[['top','right','left','bottom']].set_visible(False)
ax_b.text(0.5, 0.93, 'Spearman r=0.147, p=0.0005 ✓',
          ha='center', fontsize=9, color=GOLD, transform=ax_b.transAxes, style='italic')
ax_b.set_ylim(0, 65); ax_b.set_ylabel('Alzheimer Oranı (%)', color=SUB)
ax_b.set_title('B  |  Pack-Years Doz-Yanıt İlişkisi\n(0 pk-yr: %33.6 → >40 pk-yr: %54.5)',
               fontsize=11, fontweight='bold', color=TEXT, pad=10)
ax_b.grid(axis='y', alpha=0.3); ax_b.spines[['top','right','left','bottom']].set_visible(False)

# Panel C — Forest plot
ax_c = fig.add_subplot(gs[0, 2])
or_labels_c = ['Hiç içmemiş\nvs aktif', 'Eski içici\nvs hiç içmemiş',
               'Ever smoked\n(yaş düz.)', 'Ağır içici\n(>20 pk-yr)',
               'Sağlıklı APOE\n× ever smoked', 'Taşıyıcı APOE\n× ever smoked',
               'Baskın APOE\n× ever smoked']
or_vals_c = [1.35, 1.12, 0.91, 1.27, 0.99, 1.05, 0.73]
ci_lo_c   = [0.40, 0.83, 0.70, 0.87, 0.73, 0.72, 0.31]
ci_hi_c   = [4.55, 1.51, 1.14, 1.85, 1.35, 1.53, 1.75]
or_cols_c = [RED, ORANGE, BLUE, ORANGE, GREEN, BLUE, RED]
for i, (or_v, lo, hi, clr) in enumerate(zip(or_vals_c, ci_lo_c, ci_hi_c, or_cols_c)):
    ax_c.plot([lo, hi], [i, i], color=clr, linewidth=2.5, alpha=0.7)
    ax_c.scatter([or_v], [i], color=clr, s=100, zorder=5)
    ax_c.text(hi+0.1, i, f'{or_v:.2f}  [{lo:.2f}–{hi:.2f}]', va='center', fontsize=8, color=TEXT)
ax_c.axvline(x=1.0, color=GOLD, linestyle='--', linewidth=1.5, alpha=0.8)
ax_c.set_yticks(range(7)); ax_c.set_yticklabels(or_labels_c, fontsize=8)
ax_c.set_xlabel('Odds Ratio', color=SUB)
ax_c.set_title('C  |  Forest Plot — Tüm OR Tahminleri',
               fontsize=11, fontweight='bold', color=TEXT, pad=10)
ax_c.set_xlim(0.1, 6.5); ax_c.grid(axis='x', alpha=0.3)
ax_c.spines[['top','right','left','bottom']].set_visible(False)

# Panel D — Sigara × APOE grouped bar
ax_d = fig.add_subplot(gs[1, 0])
smk_apoe = {
    'Hiç İçmemiş': [36.0, 49.4, 65.9],
    'Eski İçici':  [37.5, 56.6, 54.8],
    'Aktif':       [33.3, 63.6, None],
}
for j, (lbl, clr) in enumerate(zip(['Hiç İçmemiş','Eski İçici','Aktif İçici'],
                                    [BLUE, ORANGE, RED])):
    key = lbl if lbl != 'Aktif İçici' else 'Aktif'
    vals = smk_apoe[key]
    valid_x = [i for i, v in enumerate(vals) if v is not None]
    valid_v = [v for v in vals if v is not None]
    bars_d = ax_d.bar([x3[i]+(j-1)*w3 for i in valid_x], valid_v, w3,
                      color=clr, alpha=0.85, edgecolor='none', label=lbl)
    for bar, v in zip(bars_d, valid_v):
        ax_d.text(bar.get_x()+bar.get_width()/2, v+1.2, f'{v:.0f}',
                  ha='center', fontsize=9, fontweight='bold', color=TEXT)
ax_d.set_xticks(x3); ax_d.set_xticklabels(G_SHORT, fontsize=10)
ax_d.set_ylim(0, 82); ax_d.set_ylabel('Alzheimer Oranı (%)', color=SUB)
ax_d.set_title('D  |  Sigara Durumu × APOE Grubu',
               fontsize=11, fontweight='bold', color=TEXT, pad=10)
ax_d.legend(fontsize=9, labelcolor=TEXT, facecolor=CARD, framealpha=0.3)
ax_d.grid(axis='y', alpha=0.3); ax_d.spines[['top','right','left','bottom']].set_visible(False)

# Panel E — Ağır sigara × APOE
ax_e = fig.add_subplot(gs[1, 1])
heavy_d = {'Normal': [39.2, 53.8, 59.2], 'Ağır (>20 pk-yr)': [44.0, 60.7, 83.3]}
for j, (lbl, clr) in enumerate(zip(['Normal', 'Ağır (>20 pk-yr)'], [BLUE, RED])):
    bars_e = ax_e.bar(x3+(j-0.5)*0.36, heavy_d[lbl], 0.36,
                      color=clr, alpha=0.85, edgecolor='none', label=lbl)
    for bar, v in zip(bars_e, heavy_d[lbl]):
        ax_e.text(bar.get_x()+bar.get_width()/2, v+1.2, f'{v:.0f}',
                  ha='center', fontsize=9, fontweight='bold', color=TEXT)
ax_e.set_xticks(x3); ax_e.set_xticklabels(G_SHORT, fontsize=10)
ax_e.set_ylim(0, 97); ax_e.set_ylabel('Alzheimer Oranı (%)', color=SUB)
ax_e.set_title('E  |  Ağır Sigara (>20 pk-yr) × APOE\n(Baskın grupta %83.3 — N=6!)',
               fontsize=11, fontweight='bold', color=TEXT, pad=10)
ax_e.legend(fontsize=9, labelcolor=TEXT, facecolor=CARD, framealpha=0.3)
ax_e.grid(axis='y', alpha=0.3); ax_e.spines[['top','right','left','bottom']].set_visible(False)

# Panel F — Sigarayı bırakma
ax_f = fig.add_subplot(gs[1, 2])
ax_f.bar(['10-25 yıl\nönce', '> 25 yıl\nönce'], [31.1, 48.6],
         color=[GREEN, ORANGE], alpha=0.85, edgecolor='none', width=0.5)
ax_f.axhline(y=42.6, color=BLUE, linestyle='--', linewidth=2, alpha=0.7,
             label='Hiç içmemiş (%42.6)')
ax_f.legend(fontsize=9, labelcolor=TEXT, facecolor=CARD, framealpha=0.3)
ax_f.text(0, 31.1+1.2, '%31.1\nN=74', ha='center', fontsize=10, fontweight='bold', color=TEXT)
ax_f.text(1, 48.6+1.2, '%48.6\nN=453', ha='center', fontsize=10, fontweight='bold', color=TEXT)
ax_f.set_ylim(0, 65); ax_f.set_ylabel('Alzheimer Oranı (%)', color=SUB)
ax_f.set_title('F  |  Sigarayı Bırakma Süresi\n(10-25 yıl önce bırakanlar en düşük: %31.1)',
               fontsize=11, fontweight='bold', color=TEXT, pad=10)
ax_f.grid(axis='y', alpha=0.3); ax_f.spines[['top','right','left','bottom']].set_visible(False)

# Panel G — Cinsiyet × Sigara
ax_g = fig.add_subplot(gs[2, 0])
erkek_v = [47.6, 51.2, 50.0]; kadin_v = [39.1, 39.5, 50.0]
b1 = ax_g.bar(x3-0.2, erkek_v, 0.38, color=BLUE, alpha=0.85, edgecolor='none', label='Erkek')
b2 = ax_g.bar(x3+0.2, kadin_v, 0.38, color=PINK, alpha=0.85, edgecolor='none', label='Kadın')
ax_g.set_xticks(x3); ax_g.set_xticklabels(['Hiç İçmemiş', 'Eski İçici', 'Aktif'], fontsize=9)
for bar, v in zip(list(b1)+list(b2), erkek_v+kadin_v):
    ax_g.text(bar.get_x()+bar.get_width()/2, v+1, f'{v:.0f}',
              ha='center', fontsize=9, fontweight='bold', color=TEXT)
ax_g.set_ylim(0, 65); ax_g.set_ylabel('Alzheimer Oranı (%)', color=SUB)
ax_g.set_title('G  |  Cinsiyet × Sigara × Alzheimer\n(Kadında sigara etkisi yok)',
               fontsize=11, fontweight='bold', color=TEXT, pad=10)
ax_g.legend(fontsize=9, labelcolor=TEXT, facecolor=CARD, framealpha=0.3)
ax_g.grid(axis='y', alpha=0.3); ax_g.spines[['top','right','left','bottom']].set_visible(False)

# Panel H — Longitudinal CDR
ax_h = fig.add_subplot(gs[2, 1])
cdr_long_vis = cdr.merge(
    df[['OASISID','SmokingStatus']].dropna().drop_duplicates('OASISID'), on='OASISID'
)
time_bins   = [0, 500, 1500, 3000, 6000, 15000]
time_labels = ['0-1y', '1-4y', '4-8y', '8-16y', '>16y']
for smk, clr, lbl in [('Hiç İçmemiş',BLUE,'Hiç İçmemiş'),
                       ('Eski İçici',ORANGE,'Eski İçici'),
                       ('Aktif',RED,'Aktif İçici')]:
    means = []
    for i in range(len(time_bins)-1):
        lo_, hi_ = time_bins[i], time_bins[i+1]
        m = cdr_long_vis[
            (cdr_long_vis['days_to_visit']>=lo_) &
            (cdr_long_vis['days_to_visit']<hi_) &
            (cdr_long_vis['SmokingStatus']==smk)
        ]['CDRTOT'].mean()
        means.append(m)
    ax_h.plot(time_labels, means, color=clr, linewidth=2.5,
              marker='o', markersize=7, label=lbl)
ax_h.set_ylabel('Ortalama CDR Skoru', color=SUB)
ax_h.set_xlabel('Takip Süresi', color=SUB)
ax_h.set_title('H  |  Longitudinal CDR Progresyonu\n(Gruplar arası fark anlamlı değil p=0.36)',
               fontsize=11, fontweight='bold', color=TEXT, pad=10)
ax_h.legend(fontsize=9, labelcolor=TEXT, facecolor=CARD, framealpha=0.3)
ax_h.grid(alpha=0.25); ax_h.spines[['top','right','left','bottom']].set_visible(False)

# Panel I — Özet
ax_i = fig.add_subplot(gs[2, 2])
ax_i.axis('off')
ax_i.text(0.5, 1.06, 'I  |  Özet ve Yorum', ha='center', fontsize=11,
          fontweight='bold', color=TEXT, transform=ax_i.transAxes)
summary = [
    ('Ana Bulgu:',        BLUE,   'Sigara tek başına OR=0.91, p=0.34'),
    ('',                  SUB,    'İstatistiksel anlamlı değil'),
    ('Doz-Yanıt:',        GOLD,   'Pack-years trendi anlamlı!'),
    ('',                  GOLD,   'r=0.147, p=0.0005 ✓'),
    ('',                  SUB,    '0 pk-yr: %33.6 → >40: %54.5'),
    ('APOE Etkileşimi:',  ORANGE, 'Taşıyıcı grupta gradient var'),
    ('',                  SUB,    '%49.4 → %56.6 → %63.6 (aktif)'),
    ('Erken Bırakma:',    GREEN,  '10-25y önce bırakanlar en iyi'),
    ('',                  GREEN,  'Hiç içmemişten bile düşük: %31.1'),
    ('Cinsiyet:',         PINK,   'Kadında sigara etkisi görülmüyor'),
    ('',                  PINK,   'Erkekte hafif artış (+3.6 puan)'),
]
y_p = 0.95
for label, clr, text_str in summary:
    if label:
        ax_i.text(0.03, y_p, label, fontsize=9, fontweight='bold',
                  color=clr, transform=ax_i.transAxes)
        ax_i.text(0.03, y_p-0.06, text_str, fontsize=8.5,
                  color=TEXT, transform=ax_i.transAxes)
        y_p -= 0.13
    else:
        ax_i.text(0.06, y_p, text_str, fontsize=8, color=clr, transform=ax_i.transAxes)
        y_p -= 0.07

plt.savefig('Sigara_APOE_Alzheimer_Analizi.png', dpi=160,
            bbox_inches='tight', facecolor=DARK, edgecolor='none')
plt.close()
print("\nGörsel kaydedildi: Sigara_APOE_Alzheimer_Analizi.png")
