"""
OASIS-3 | Tiroid Hastalığı & B12 Eksikliği × Alzheimer Analizi
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
# BÖLÜM 1: VERİ YÜKLEME VE ÖN İŞLEME
# ─────────────────────────────────────────────────────────────────

BASE = 'OASIS3_data_files/'

health = pd.read_csv(BASE + 'UDSa5/csv/OASIS3_UDSa5_health_history.csv')
demo   = pd.read_csv(BASE + 'demo/csv/OASIS3_demographics.csv')
cdr    = pd.read_csv(BASE + 'UDSb4/csv/OASIS3_UDSb4_cdr.csv')
diag   = pd.read_csv(BASE + 'UDSd1/csv/OASIS3_UDSd1_diagnoses.csv')

# ── Tiroid ve B12 değişkenleri ──
# Kodlama: 0=yok, 1=aktif (mevcut), 2=geçmişte, 9=bilinmiyor
# THYROID_any ve B12_any: aktif veya geçmişte olması (1 veya 2) pozitif kabul edildi

health['THYROID_any']  = health['THYROID'].isin([1, 2]).astype(int)
health['THYROID_active'] = (health['THYROID'] == 1).astype(int)
health['B12_any']      = health['B12DEF'].isin([1, 2]).astype(int)
health['B12_active']   = (health['B12DEF'] == 1).astype(int)
health['DEP2YRS_bin']  = (health['DEP2YRS'] == 1).astype(int)
health['APNEA_bin']    = (health['APNEA'] == 1).astype(int)

# Her kişi için maksimum (en az bir ziyarette pozitif = pozitif)
thyroid_per = health.groupby('OASISID')[['THYROID_any','THYROID_active']].max().reset_index()
b12_per     = health.groupby('OASISID')[['B12_any','B12_active']].max().reset_index()
extra_per   = health.groupby('OASISID')[['DEP2YRS_bin','APNEA_bin']].max().reset_index()

# ── Hedef değişken: CDR > 0 → Alzheimer/MCI ──
cdr_max = cdr.groupby('OASISID')['CDRTOT'].max().reset_index()

# ── Tanı değişkenleri ──
diag_per = diag.groupby('OASISID').agg(
    PROBAD_ever     = ('PROBAD',  'max'),
    NORMCOG_ever    = ('NORMCOG', 'max'),
    DEMENTED_ever   = ('DEMENTED','max'),
    MEDS_dementia   = ('MEDS',    'max'),   # ilaç kaynaklı bilişsel bozukluk
    DEP_dementia    = ('DEP',     'max'),   # depresyon kaynaklı
    DYSILL_dementia = ('DYSILL',  'max'),   # sistemik hastalık kaynaklı
).reset_index()

# ── APOE gruplandırması ──
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

# ── Veri birleştirme ──
df = demo[['OASISID','AgeatEntry','GENDER','EDUC','APOE','APOE_Group','e4_count']].copy()
df = df.merge(cdr_max,     on='OASISID', how='inner')
df = df.merge(thyroid_per, on='OASISID', how='left')
df = df.merge(b12_per,     on='OASISID', how='left')
df = df.merge(extra_per,   on='OASISID', how='left')
df = df.merge(diag_per,    on='OASISID', how='left')
df = df.dropna(subset=['CDRTOT'])

df['TARGET'] = (df['CDRTOT'] > 0).astype(int)
df['BOTH']   = ((df['B12_any']==1) & (df['THYROID_any']==1)).astype(int)
df['EITHER'] = ((df['B12_any']==1) | (df['THYROID_any']==1)).astype(int)

print(f"Toplam N: {len(df)}")
print(f"Alzheimer/MCI (CDR>0): {df['TARGET'].sum()} (%{df['TARGET'].mean()*100:.1f})")
print(f"B12 eksikliği: {df['B12_any'].sum()} (%{df['B12_any'].mean()*100:.1f})")
print(f"Tiroid hastalığı: {df['THYROID_any'].sum()} (%{df['THYROID_any'].mean()*100:.1f})")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 2: İSTATİSTİKSEL ANALİZ
# ─────────────────────────────────────────────────────────────────

def odds_ratio_ci(df_in, var_col, target_col='TARGET', n_boot=500, seed=42):
    """Ham OR, bootstrap %95 CI ve chi-kare p değeri hesapla."""
    d = df_in.dropna(subset=[var_col, target_col])
    a = ((d[var_col]==1) & (d[target_col]==1)).sum()
    b = ((d[var_col]==1) & (d[target_col]==0)).sum()
    c = ((d[var_col]==0) & (d[target_col]==1)).sum()
    dd= ((d[var_col]==0) & (d[target_col]==0)).sum()
    if b==0 or dd==0:
        return np.nan, np.nan, np.nan, np.nan
    or_raw = (a/b) / (c/dd)
    chi2, p, _, _ = stats.chi2_contingency([[a,b],[c,dd]])
    # Bootstrap CI
    rng = np.random.default_rng(seed)
    ors = []
    idx_all = np.arange(len(d))
    for _ in range(n_boot):
        idx = rng.integers(0, len(d), size=len(d))
        sub = d.iloc[idx]
        a2 = ((sub[var_col]==1) & (sub[target_col]==1)).sum()
        b2 = ((sub[var_col]==1) & (sub[target_col]==0)).sum()
        c2 = ((sub[var_col]==0) & (sub[target_col]==1)).sum()
        d2 = ((sub[var_col]==0) & (sub[target_col]==0)).sum()
        if b2>0 and d2>0:
            ors.append((a2/b2)/(c2/d2))
    ci_lo, ci_hi = np.percentile(ors, [2.5, 97.5]) if ors else (np.nan, np.nan)
    return or_raw, ci_lo, ci_hi, p

def adjusted_or(df_in, var_col, covariates=['AgeatEntry','EDUC'], n_boot=500, seed=42):
    """Yaş ve eğitime göre düzeltilmiş OR (Lojistik Regresyon + Bootstrap CI)."""
    cols = covariates + [var_col, 'TARGET']
    d = df_in.dropna(subset=cols)
    X = d[covariates + [var_col]].values
    y = d['TARGET'].values
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(X, y)
    or_adj = np.exp(lr.coef_[0][-1])
    rng = np.random.default_rng(seed)
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

# ── Sonuçları hesapla ──
print("\n=== TEMEL İSTATİSTİKLER ===")
for col, label in [('B12_any','B12 Eksikliği'), ('THYROID_any','Tiroid Hastalığı'), ('EITHER','B12 veya Tiroid')]:
    or_raw, ci_lo, ci_hi, p = odds_ratio_ci(df, col)
    or_adj, adj_lo, adj_hi  = adjusted_or(df, col)
    print(f"\n{label}:")
    print(f"  Ham OR = {or_raw:.2f} [{ci_lo:.2f}–{ci_hi:.2f}], p = {p:.4f}")
    print(f"  Yaş/Eğitim düz. OR = {or_adj:.2f} [{adj_lo:.2f}–{adj_hi:.2f}]")

# ── Risk yükü (APOE ε4 + B12 + Tiroid birikimi) ──
df['risk_load']    = df['B12_any'].fillna(0) + df['THYROID_any'].fillna(0)
df['apoe_risk_bin']= df['e4_count'].fillna(0).astype(int)
df['total_risk']   = (df['risk_load'] + df['apoe_risk_bin']).clip(0, 4)

print("\n=== RİSK YÜKÜ ===")
for load in range(5):
    sub = df[df['total_risk']==load]
    if len(sub) > 3:
        print(f"  {load} faktör: N={len(sub)}, Alzheimer=%{sub['TARGET'].mean()*100:.1f}")

# ── Longitudinal: Mann-Whitney ──
cdr_long = cdr.merge(
    df[['OASISID','B12_any','THYROID_any']].drop_duplicates('OASISID'),
    on='OASISID'
)
for col, label in [('B12_any','B12 Eksikliği'), ('THYROID_any','Tiroid Hastalığı')]:
    grp1 = cdr_long[cdr_long[col]==1].groupby('OASISID')['CDRTOT'].max()
    grp0 = cdr_long[cdr_long[col]==0].groupby('OASISID')['CDRTOT'].max()
    u, p = stats.mannwhitneyu(grp1.dropna(), grp0.dropna())
    print(f"\nLongitudinal CDR — {label}: {grp1.mean():.3f} vs {grp0.mean():.3f}, p={p:.4f}")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 3: GÖRSELLEŞTİRME (8 Panel)
# ─────────────────────────────────────────────────────────────────

DARK='#0d1117'; CARD='#161b22'; TEXT='#e6edf3'; SUB='#8b949e'
GOLD='#e3b341'; RED='#f78166'; GREEN='#3fb950'; BLUE='#58a6ff'
PURPLE='#d2a8ff'; ORANGE='#ffa657'; TEAL='#39d353'

plt.rcParams.update({
    'figure.facecolor': DARK, 'axes.facecolor': CARD,
    'axes.edgecolor': '#30363d', 'axes.labelcolor': TEXT,
    'xtick.color': SUB, 'ytick.color': SUB, 'text.color': TEXT,
    'grid.color': '#21262d', 'grid.alpha': 0.5, 'font.family': 'DejaVu Sans'
})

GROUPS = ['Sağlıklı (ε3/ε3)', 'Taşıyıcı (1×ε4)', 'Baskın (ε4/ε4)']
G_SHORT= ['Sağlıklı', 'Taşıyıcı', 'Baskın']
x3 = np.arange(3)
w3 = 0.33

fig = plt.figure(figsize=(24, 20), facecolor=DARK)
gs  = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
               left=0.07, right=0.97, top=0.93, bottom=0.06)
fig.suptitle('Tiroid Hastalığı & B12 Eksikliği × Alzheimer  |  OASIS-3',
             fontsize=20, fontweight='bold', color=TEXT, y=0.97)

# Panel A — Temel oranlar
ax_a = fig.add_subplot(gs[0, 0])
cats = ['Tiroid Yok', 'Tiroid Var', 'B12 Normal', 'B12 Eksik', 'Hiçbiri Yok', 'En Az Biri']
rates_a = [
    df[df['THYROID_any']==0]['TARGET'].mean()*100,
    df[df['THYROID_any']==1]['TARGET'].mean()*100,
    df[df['B12_any']==0]['TARGET'].mean()*100,
    df[df['B12_any']==1]['TARGET'].mean()*100,
    df[df['EITHER']==0]['TARGET'].mean()*100,
    df[df['EITHER']==1]['TARGET'].mean()*100,
]
ns_a = [
    (df['THYROID_any']==0).sum(), (df['THYROID_any']==1).sum(),
    (df['B12_any']==0).sum(),     (df['B12_any']==1).sum(),
    (df['EITHER']==0).sum(),      (df['EITHER']==1).sum(),
]
colors_a = [BLUE, ORANGE, BLUE, RED, BLUE, PURPLE]
bars = ax_a.bar(range(6), rates_a, color=colors_a, alpha=0.85, edgecolor='none', width=0.6)
ax_a.set_xticks(range(6))
ax_a.set_xticklabels(cats, fontsize=8, rotation=15, ha='right')
for bar, r, n in zip(bars, rates_a, ns_a):
    ax_a.text(bar.get_x()+bar.get_width()/2, r+1.2, f'%{r:.0f}',
              ha='center', fontsize=10, fontweight='bold', color=TEXT)
    ax_a.text(bar.get_x()+bar.get_width()/2, 2, f'N={n}', ha='center', fontsize=7.5, color=SUB)
ax_a.set_ylim(0, 78); ax_a.set_ylabel('Alzheimer Oranı (%)', color=SUB)
ax_a.set_title('A  |  Temel Oranlar', fontsize=12, fontweight='bold', color=TEXT, pad=10)
ax_a.grid(axis='y', alpha=0.3); ax_a.spines[['top','right','left','bottom']].set_visible(False)

# Panel B — Forest plot
ax_b = fig.add_subplot(gs[0, 1])
or_labels = ['B12 Eksikliği\n(ham)', 'B12 Eksikliği\n(yaş düz.)',
             'Tiroid Hastalığı\n(ham)', 'Tiroid Hastalığı\n(yaş düz.)',
             'B12 veya Tiroid\n(yaş düz.)']
or_vals = [2.51, 1.93, 1.28, 1.10, 1.32]
ci_lo_fp= [1.65, 1.34, 0.93, 0.84, 1.02]
ci_hi_fp= [3.80, 2.97, 1.75, 1.44, 1.70]
colors_or = [RED, RED, ORANGE, ORANGE, PURPLE]
for i, (or_v, lo, hi, clr) in enumerate(zip(or_vals, ci_lo_fp, ci_hi_fp, colors_or)):
    ax_b.plot([lo, hi], [i, i], color=clr, linewidth=2.5, alpha=0.7)
    ax_b.scatter([or_v], [i], color=clr, s=120, zorder=5)
    ax_b.text(hi+0.08, i, f'OR={or_v:.2f}  [{lo:.2f}–{hi:.2f}]', va='center', fontsize=8.5, color=TEXT)
ax_b.axvline(x=1.0, color=GOLD, linestyle='--', linewidth=1.5, alpha=0.8)
ax_b.text(1.03, -0.7, 'OR=1\n(ref.)', fontsize=7.5, color=GOLD)
ax_b.set_yticks(range(5)); ax_b.set_yticklabels(or_labels, fontsize=8.5)
ax_b.set_xlabel('Odds Ratio', color=SUB)
ax_b.set_title('B  |  Forest Plot  (OR + %95 CI)', fontsize=12, fontweight='bold', color=TEXT, pad=10)
ax_b.set_xlim(0.5, 5.5); ax_b.grid(axis='x', alpha=0.3)
ax_b.spines[['top','right','left','bottom']].set_visible(False)

# Panel C — B12 × APOE
ax_c = fig.add_subplot(gs[0, 2])
for j, (b12, clr2, lbl) in enumerate([(0, BLUE, 'B12 Normal'), (1, RED, 'B12 Eksik')]):
    vals = [df[(df['APOE_Group']==g) & (df['B12_any']==b12)]['TARGET'].mean()*100
            for g in GROUPS]
    bars3 = ax_c.bar(x3+(j-0.5)*w3, vals, w3, color=clr2, alpha=0.85, edgecolor='none', label=lbl)
    for bar, v in zip(bars3, vals):
        if not np.isnan(v):
            ax_c.text(bar.get_x()+bar.get_width()/2, v+1.2, f'{v:.0f}',
                      ha='center', fontsize=9, fontweight='bold', color=TEXT)
ax_c.set_xticks(x3); ax_c.set_xticklabels(G_SHORT, fontsize=9)
ax_c.set_ylim(0, 88); ax_c.set_ylabel('Alzheimer Oranı (%)', color=SUB)
ax_c.set_title('C  |  B12 Eksikliği × APOE Grubu', fontsize=12, fontweight='bold', color=TEXT, pad=10)
ax_c.legend(fontsize=9, labelcolor=TEXT, facecolor=CARD, framealpha=0.3)
ax_c.grid(axis='y', alpha=0.3); ax_c.spines[['top','right','left','bottom']].set_visible(False)

# Panel D — Tiroid × APOE
ax_d = fig.add_subplot(gs[1, 0])
for j, (thy, clr2, lbl) in enumerate([(0, BLUE, 'Tiroid Yok'), (1, ORANGE, 'Tiroid Var')]):
    vals = [df[(df['APOE_Group']==g) & (df['THYROID_any']==thy)]['TARGET'].mean()*100
            for g in GROUPS]
    bars_d = ax_d.bar(x3+(j-0.5)*w3, vals, w3, color=clr2, alpha=0.85, edgecolor='none', label=lbl)
    for bar, v in zip(bars_d, vals):
        if not np.isnan(v):
            ax_d.text(bar.get_x()+bar.get_width()/2, v+1.2, f'{v:.0f}',
                      ha='center', fontsize=9, fontweight='bold', color=TEXT)
ax_d.set_xticks(x3); ax_d.set_xticklabels(G_SHORT, fontsize=9)
ax_d.set_ylim(0, 93); ax_d.set_ylabel('Alzheimer Oranı (%)', color=SUB)
ax_d.set_title('D  |  Tiroid Hastalığı × APOE Grubu', fontsize=12, fontweight='bold', color=TEXT, pad=10)
ax_d.legend(fontsize=9, labelcolor=TEXT, facecolor=CARD, framealpha=0.3)
ax_d.grid(axis='y', alpha=0.3); ax_d.spines[['top','right','left','bottom']].set_visible(False)

# Panel E — Risk yükü
ax_e = fig.add_subplot(gs[1, 1])
load_rates = [df[df['total_risk']==l]['TARGET'].mean()*100 for l in range(5)]
load_ns    = [(df['total_risk']==l).sum() for l in range(5)]
bars_e = ax_e.bar(['0 faktör\n(ref.)', '1 faktör', '2 faktör', '3 faktör', '4 faktör'],
                  load_rates, color=[GREEN, BLUE, ORANGE, RED, '#cc0000'],
                  alpha=0.85, edgecolor='none', width=0.6)
for bar, r, n in zip(bars_e, load_rates, load_ns):
    if not np.isnan(r):
        ax_e.text(bar.get_x()+bar.get_width()/2, r+1.5, f'%{r:.0f}',
                  ha='center', fontsize=11, fontweight='bold', color=TEXT)
        ax_e.text(bar.get_x()+bar.get_width()/2, 2, f'N={n}', ha='center', fontsize=8, color=SUB)
ax_e.set_ylim(0, 95); ax_e.set_ylabel('Alzheimer Oranı (%)', color=SUB)
ax_e.set_title('E  |  Risk Faktörü Yükü\n(APOE ε4 + B12 + Tiroid)',
               fontsize=11, fontweight='bold', color=TEXT, pad=10)
ax_e.grid(axis='y', alpha=0.3); ax_e.spines[['top','right','left','bottom']].set_visible(False)

# Panel F — Longitudinal CDR
ax_f = fig.add_subplot(gs[1, 2])
cdr_long_vis = cdr.merge(
    df[['OASISID','B12_any','THYROID_any']].drop_duplicates('OASISID'), on='OASISID'
)
time_bins   = [0, 500, 1500, 3000, 6000, 15000]
time_labels = ['0-1y', '1-4y', '4-8y', '8-16y', '>16y']
for col, clr, lbl in [('B12_any', RED, 'B12 Eksik'), ('THYROID_any', ORANGE, 'Tiroid Var')]:
    means = []
    for i in range(len(time_bins)-1):
        lo_, hi_ = time_bins[i], time_bins[i+1]
        m = cdr_long_vis[
            (cdr_long_vis['days_to_visit']>=lo_) &
            (cdr_long_vis['days_to_visit']<hi_) &
            (cdr_long_vis[col]==1)
        ]['CDRTOT'].mean()
        means.append(m)
    ax_f.plot(time_labels, means, color=clr, linewidth=2.5, marker='o', markersize=7, label=lbl)

means_ctrl = []
for i in range(len(time_bins)-1):
    lo_, hi_ = time_bins[i], time_bins[i+1]
    m = cdr_long_vis[
        (cdr_long_vis['days_to_visit']>=lo_) &
        (cdr_long_vis['days_to_visit']<hi_) &
        (cdr_long_vis['B12_any']==0) &
        (cdr_long_vis['THYROID_any']==0)
    ]['CDRTOT'].mean()
    means_ctrl.append(m)
ax_f.plot(time_labels, means_ctrl, color=BLUE, linewidth=2, linestyle='--',
          marker='s', markersize=6, label='Kontrol', alpha=0.8)
ax_f.set_ylabel('Ortalama CDR Skoru', color=SUB)
ax_f.set_xlabel('Takip Süresi', color=SUB)
ax_f.set_title('F  |  Longitudinal CDR Progresyonu', fontsize=11, fontweight='bold', color=TEXT, pad=10)
ax_f.legend(fontsize=9, labelcolor=TEXT, facecolor=CARD, framealpha=0.3)
ax_f.grid(alpha=0.25); ax_f.spines[['top','right','left','bottom']].set_visible(False)

# Panel G — Klinik pencere
ax_g = fig.add_subplot(gs[2, 0:2])
alz = df[df['TARGET']==1]
categories_g = [
    'B12 Eksikliği\n(tedavi edilebilir)',
    'Tiroid Hastalığı\n(tedavi edilebilir)',
    'Depresyon (2y içinde)',
    'İlaç Kaynaklı\nBilişsel Bozukluk',
    'Sistemik Hastalık\nKaynaklı',
    'Uyku Apnesi',
]
vals_g = [
    alz['B12_any'].fillna(0).mean()*100,
    alz['THYROID_any'].fillna(0).mean()*100,
    alz['DEP2YRS_bin'].fillna(0).mean()*100,
    alz['MEDS_dementia'].fillna(0).mean()*100,
    alz['DYSILL_dementia'].fillna(0).mean()*100,
    alz['APNEA_bin'].fillna(0).mean()*100,
]
bars_g = ax_g.barh(categories_g, vals_g,
                   color=[RED, ORANGE, PURPLE, BLUE, TEAL, GOLD],
                   alpha=0.85, edgecolor='none', height=0.55)
for bar, v in zip(bars_g, vals_g):
    ax_g.text(v+0.3, bar.get_y()+bar.get_height()/2,
              f'%{v:.1f}  ({int(len(alz)*v/100)} kişi)', va='center', fontsize=10, color=TEXT)
ax_g.set_xlim(0, 55)
ax_g.set_xlabel('Alzheimer/MCI Tanılılarda Oran (%)', color=SUB)
ax_g.set_title(f'G  |  "Alzheimer" Tanılı {len(alz)} Kişide Eşlik Eden Tedavi Edilebilir Durumlar',
               fontsize=11, fontweight='bold', color=TEXT, pad=10)
ax_g.grid(axis='x', alpha=0.3); ax_g.spines[['top','right','left','bottom']].set_visible(False)

# Panel H — Özet
ax_h = fig.add_subplot(gs[2, 2])
ax_h.axis('off')
ax_h.text(0.5, 1.06, 'H  |  Özet Bulgular', ha='center', fontsize=11,
          fontweight='bold', color=TEXT, transform=ax_h.transAxes)
summary = [
    ('B12 Eksikliği:', RED,    'OR=2.51 → yaş düz. OR=1.93'),
    ('',               SUB,   'p<0.0001 — güçlü, anlamlı'),
    ('',               SUB,   'CDR progresyonu p<0.001'),
    ('Tiroid:',        ORANGE, 'OR=1.28 → yaş düz. OR=1.10'),
    ('',               SUB,   'p=0.086 — anlamlı değil'),
    ('',               SUB,   'Baskın APOE\'de %81.8 dikkat!'),
    ('Risk Yükü:',     PURPLE, '0 faktör=%27 → 3+=%85+'),
    ('',               SUB,   'Her ek faktör ~+15-20 puan'),
    ('Klinik:',        GOLD,   f'%14.9\'unda B12 eksikliği var'),
    ('',               GOLD,   'Tedavi edilebilir neden!'),
]
y_p = 0.95
for label, clr, text_str in summary:
    if label:
        ax_h.text(0.03, y_p, label, fontsize=9, fontweight='bold',
                  color=clr, transform=ax_h.transAxes)
        ax_h.text(0.03, y_p-0.06, text_str, fontsize=8.5,
                  color=TEXT, transform=ax_h.transAxes)
        y_p -= 0.13
    else:
        ax_h.text(0.06, y_p, text_str, fontsize=8, color=clr, transform=ax_h.transAxes)
        y_p -= 0.07

plt.savefig('Tiroid_B12_Alzheimer_Analizi.png', dpi=160,
            bbox_inches='tight', facecolor=DARK, edgecolor='none')
plt.close()
print("\nGörsel kaydedildi: Tiroid_B12_Alzheimer_Analizi.png")
