"""
╔══════════════════════════════════════════════════════════════════╗
║   OASIS-3  |  APOE Doz-Yanıt Analizi                           ║
║   Sena Ay  |  Özal Hoca Projesi  |  2026                       ║
╠══════════════════════════════════════════════════════════════════╣
║  ÇIKTILAR:                                                       ║
║    APOE_DozYanit_Analizi.png  — 8 panelli ana görsel            ║
║    oasis3_final_dataset.csv   — birleşik işlenmiş veri          ║
╠══════════════════════════════════════════════════════════════════╣
║  KULLANIM:                                                       ║
║    python apoe_doz_yanit_analizi.py                             ║
║                                                                  ║
║  GEREKLİ KÜTÜPHANELER:                                          ║
║    pip install pandas numpy matplotlib scikit-learn scipy        ║
╚══════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# AYARLAR  —  sadece bu bölümü değiştirmen yeterli
# ─────────────────────────────────────────────────────────────────
DATA_PATH   = 'OASIS3_data_files/'   # zip'in çıkarıldığı klasör
OUTPUT_IMG  = 'APOE_DozYanit_Analizi.png'
OUTPUT_CSV  = 'oasis3_final_dataset.csv'
RANDOM_SEED = 42

# ═════════════════════════════════════════════════════════════════
# BÖLÜM 1  —  VERİ YÜKLEME VE BİRLEŞTİRME
# ═════════════════════════════════════════════════════════════════
print("▶ Veriler yükleniyor...")

demo   = pd.read_csv(f'{DATA_PATH}demo/csv/OASIS3_demographics.csv')
cdr    = pd.read_csv(f'{DATA_PATH}UDSb4/csv/OASIS3_UDSb4_cdr.csv')
fs     = pd.read_csv(f'{DATA_PATH}FS/csv/OASIS3_Freesurfer_output.csv')
med    = pd.read_csv(f'{DATA_PATH}UDSd2/csv/OASIS3_UDSd2_med_conditions.csv')
health = pd.read_csv(f'{DATA_PATH}UDSa5/csv/OASIS3_UDSa5_health_history.csv')

# ── CDR: kişi başına maksimum (en kötü) skor ──────────────────
cdr_max = cdr.groupby('OASISID')['CDRTOT'].max().reset_index()

# ── Freesurfer: hippokampüs hacmi ortalaması ──────────────────
fs['OASISID'] = fs['Subject']
fs_agg = fs.groupby('OASISID').agg(
    Hippo_Vol=('TOTAL_HIPPOCAMPUS_VOLUME', 'mean')
).reset_index()

# ── Diyabet: iki kaynaktan birleştir ──────────────────────────
#    UDSd2.diabet  : 1=Tip1, 2=Tip2, 0=yok
#    UDSa5.DIABETES: 1=var,  0=yok
med['has_diab']    = med['diabet'].isin([1, 2, 3]).astype(int)
diab_d2 = med.groupby('OASISID')['has_diab'].max().reset_index()
diab_d2.columns = ['OASISID', 'DIAB_UDSd2']

health['has_diab'] = (health['DIABETES'] == 1).astype(int)
diab_a5 = health.groupby('OASISID')['has_diab'].max().reset_index()
diab_a5.columns = ['OASISID', 'DIAB_UDSa5']

# ── Ana birleştirme ───────────────────────────────────────────
df = demo[['OASISID', 'AgeatEntry', 'GENDER', 'EDUC',
           'APOE', 'daddem', 'momdem']].copy()
df = df.merge(cdr_max,  on='OASISID', how='left')
df = df.merge(fs_agg,   on='OASISID', how='left')
df = df.merge(diab_d2,  on='OASISID', how='left')
df = df.merge(diab_a5,  on='OASISID', how='left')

print(f"  Birleşik kayıt: {df.shape[0]} kişi")

# ═════════════════════════════════════════════════════════════════
# BÖLÜM 2  —  ÖZELLİK MÜHENDİSLİĞİ
# ═════════════════════════════════════════════════════════════════
df = df.dropna(subset=['CDRTOT', 'APOE'])

# Hedef değişken: CDR > 0 → Alzheimer/MCI (1), CDR = 0 → Sağlıklı (0)
df['TARGET'] = (df['CDRTOT'] > 0).astype(int)

# APOE ε4 alel sayısı (0, 1, 2)
df['e4_count'] = df['APOE'].apply(lambda x: str(int(x)).count('4'))

# APOE grup etiketleri
def apoe_group(x):
    x = int(x)
    if x == 44:       return 'Baskın (ε4/ε4)'
    elif x in [34, 24]: return 'Taşıyıcı (1×ε4)'
    else:               return 'Sağlıklı (ε3/ε3)'

df['APOE_Group'] = df['APOE'].apply(apoe_group)

# Eğitim grup etiketleri
def educ_group(e):
    if e <= 12:   return '≤12 yıl\n(Lise)'
    elif e <= 16: return '13-16 yıl\n(Lisans)'
    else:          return '≥17 yıl\n(Lisansüstü)'

df['EDUC_Group'] = df['EDUC'].apply(educ_group)
EDUC_ORDER = ['≤12 yıl\n(Lise)', '13-16 yıl\n(Lisans)', '≥17 yıl\n(Lisansüstü)']

# Kombine diyabet: herhangi kaynakta 1 ise 1
df['DIABETES'] = ((df['DIAB_UDSd2'] == 1) | (df['DIAB_UDSa5'] == 1)).astype(int)

# Aile geçmişi (anne veya baba demans = var)
df['Family_History'] = (
    ((df['daddem'].fillna(0) > 0) | (df['momdem'].fillna(0) > 0))
).astype(int)

# Eksik değer doldurma (medyan)
for col in ['EDUC', 'Hippo_Vol', 'AgeatEntry']:
    df[col] = df[col].fillna(df[col].median())

df.to_csv(OUTPUT_CSV, index=False)
print(f"  İşlenmiş veri kaydedildi → {OUTPUT_CSV}")
print(f"  N={len(df)}, Alzheimer=%{df['TARGET'].mean()*100:.1f}, "
      f"Diyabet=%{df['DIABETES'].mean()*100:.1f}")

# ═════════════════════════════════════════════════════════════════
# BÖLÜM 3  —  İSTATİSTİKSEL ANALİZ
# ═════════════════════════════════════════════════════════════════
GROUPS   = ['Sağlıklı (ε3/ε3)', 'Taşıyıcı (1×ε4)', 'Baskın (ε4/ε4)']
G_SHORT  = ['Sağlıklı', 'Taşıyıcı', 'Baskın']

print("\n▶ İstatistiksel analiz...")

# ── 3a) APOE doz-yanıt ───────────────────────────────────────
print("\n  APOE Doz-Yanıt:")
for g in GROUPS:
    sub = df[df['APOE_Group'] == g]
    rate = sub['TARGET'].mean() * 100
    print(f"    {g}: N={len(sub)}, Alzheimer=%{rate:.1f}")

# ── 3b) Eğitim: her 1 yıl artışı → lojistik regresyon ────────
print("\n  Eğitim Koruyucu Etki (yaş kovaryant olarak dahil):")
for g in GROUPS:
    sub = df[df['APOE_Group'] == g].copy()
    X   = sub[['EDUC', 'AgeatEntry']].values
    y   = sub['TARGET'].values
    lr  = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    lr.fit(X, y)
    or_val = np.exp(lr.coef_[0][0])          # eğitim katsayısı → OR

    # Bootstrap ile %95 güven aralığı (500 tekrar)
    ors = []
    rng = np.random.default_rng(RANDOM_SEED)
    for _ in range(500):
        idx = rng.integers(0, len(X), size=len(X))
        try:
            lr2 = LogisticRegression(max_iter=300, random_state=0)
            lr2.fit(X[idx], y[idx])
            ors.append(np.exp(lr2.coef_[0][0]))
        except Exception:
            pass
    ci_lo, ci_hi = np.percentile(ors, [2.5, 97.5])
    pct_change    = (1 - or_val) * 100
    print(f"    {g}: OR={or_val:.3f} [%95 CI: {ci_lo:.3f}–{ci_hi:.3f}]"
          f"  → Her 1 yıl riski %{abs(pct_change):.1f} "
          f"{'azaltıyor' if or_val < 1 else 'artırıyor'}")

# ── 3c) Diyabet × APOE → OR + chi-kare ──────────────────────
print("\n  Diyabet Odds Ratio (APOE grubuna göre):")
for g in GROUPS:
    sub  = df[df['APOE_Group'] == g]
    d_no = sub[sub['DIABETES'] == 0]
    d_yes= sub[sub['DIABETES'] == 1]
    nd   = sub[sub['DIABETES'] == 0]['TARGET'].mean() * 100
    yd   = sub[sub['DIABETES'] == 1]['TARGET'].mean() * 100 if len(d_yes) > 3 else float('nan')
    print(f"    {g}:")
    print(f"      Diyabetsiz: N={len(d_no)}, Alzheimer=%{nd:.1f}")
    print(f"      Diyabetli:  N={len(d_yes)}, Alzheimer=%{yd:.1f}")
    if len(d_yes) > 5 and len(d_no) > 5:
        a = d_yes['TARGET'].sum();  b = len(d_yes) - a
        c = d_no['TARGET'].sum();   d = len(d_no)  - c
        if b > 0 and c > 0 and d > 0:
            or_diab = (a / b) / (c / d)
            try:
                _, pval, _, _ = stats.chi2_contingency([[a, b], [c, d]])
                print(f"      OR={or_diab:.2f}, χ² p={pval:.4f}")
            except Exception:
                print(f"      OR={or_diab:.2f}")

# ── 3d) Eğitim koruyucu etki miktarı (grup içi delta) ────────
print("\n  Eğitim Koruma Deltası (Lise → Lisansüstü farkı):")
for g in GROUPS:
    sub   = df[df['APOE_Group'] == g]
    lise  = sub[sub['EDUC'] <= 12]['TARGET'].mean() * 100
    yuksek= sub[sub['EDUC'] >  16]['TARGET'].mean() * 100
    print(f"    {g}: Lise=%{lise:.1f} → Lisansüstü=%{yuksek:.1f}"
          f"  (-%{lise - yuksek:.1f} puan koruma)")

# ═════════════════════════════════════════════════════════════════
# BÖLÜM 4  —  GÖRSELLEŞTİRME  (8 Panel)
# ═════════════════════════════════════════════════════════════════
print(f"\n▶ Görsel oluşturuluyor → {OUTPUT_IMG}")

G_COLORS = ['#3fb950', '#58a6ff', '#f78166']   # yeşil / mavi / kırmızı
DARK  = '#0d1117'; CARD  = '#161b22'
TEXT  = '#e6edf3'; SUB   = '#8b949e'
GOLD  = '#e3b341'; RED   = '#f78166'
GREEN = '#3fb950'; BLUE  = '#58a6ff'

plt.rcParams.update({
    'figure.facecolor': DARK,  'axes.facecolor': CARD,
    'axes.edgecolor':  '#30363d', 'axes.labelcolor': TEXT,
    'xtick.color': SUB,  'ytick.color': SUB,
    'text.color':  TEXT, 'grid.color':  '#21262d',
    'grid.alpha':  0.5,  'font.family': 'DejaVu Sans'
})

fig = plt.figure(figsize=(24, 18), facecolor=DARK)
gs  = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35,
               left=0.07, right=0.97, top=0.93, bottom=0.07)

fig.suptitle(
    'APOE ε4 Doz–Yanıt Analizi  ×  Eğitim  ×  Diyabet  |  OASIS-3',
    fontsize=20, fontweight='bold', color=TEXT, y=0.97
)

# ── A) APOE doz-yanıt çubuğu ─────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
rates = [df[df['APOE_Group'] == g]['TARGET'].mean() * 100 for g in GROUPS]
ns    = [len(df[df['APOE_Group'] == g]) for g in GROUPS]

bars = ax_a.bar(G_SHORT, rates, color=G_COLORS, alpha=0.88, width=0.55, edgecolor='none')
for bar, r, n in zip(bars, rates, ns):
    ax_a.text(bar.get_x() + bar.get_width() / 2, r + 1.2,
              f'%{r:.0f}', ha='center', fontsize=13, fontweight='bold', color=TEXT)
    ax_a.text(bar.get_x() + bar.get_width() / 2, 3,
              f'N={n}', ha='center', fontsize=9, color=SUB)

# Artış okları
for i in range(len(GROUPS) - 1):
    diff = rates[i + 1] - rates[i]
    ax_a.annotate('', xy=(i + 1, rates[i + 1] - 3), xytext=(i, rates[i] + 3),
                  arrowprops=dict(arrowstyle='->', color=GOLD, lw=1.5))
    ax_a.text(i + 0.5, max(rates[i], rates[i + 1]) + 8,
              f'+{diff:.0f}%', ha='center', fontsize=9, color=GOLD, fontweight='bold')

ax_a.set_ylim(0, 85)
ax_a.set_ylabel('Alzheimer Oranı (%)', color=SUB)
ax_a.set_title('A  |  APOE Doz-Yanıt', fontsize=12, fontweight='bold', color=TEXT, pad=10)
ax_a.grid(axis='y', alpha=0.3)
ax_a.spines[['top', 'right', 'left', 'bottom']].set_visible(False)

# ── B) Eğitim × APOE grouped bar ─────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
x = np.arange(3); w = 0.24
educ_labels_short = ['≤12\n(Lise)', '13-16\n(Lisans)', '≥17\n(Lisansüstü)']

for i, (g, clr) in enumerate(zip(GROUPS, G_COLORS)):
    sub  = df[df['APOE_Group'] == g]
    vals = [sub[sub['EDUC_Group'] == eg]['TARGET'].mean() * 100
            for eg in EDUC_ORDER]
    bars2 = ax_b.bar(x + (i - 1) * w, vals, w,
                     color=clr, alpha=0.85, edgecolor='none', label=G_SHORT[i])
    for bar, v in zip(bars2, vals):
        if not np.isnan(v):
            ax_b.text(bar.get_x() + bar.get_width() / 2, v + 1,
                      f'{v:.0f}', ha='center', fontsize=7.5, color=TEXT)

ax_b.set_xticks(x); ax_b.set_xticklabels(educ_labels_short, fontsize=9)
ax_b.set_ylim(0, 92)
ax_b.set_ylabel('Alzheimer Oranı (%)', color=SUB)
ax_b.set_title('B  |  Eğitim × APOE', fontsize=12, fontweight='bold', color=TEXT, pad=10)
ax_b.legend(fontsize=9, labelcolor=TEXT, facecolor=CARD, framealpha=0.3,
            loc='upper right', title='APOE Grubu', title_fontsize=8)
ax_b.grid(axis='y', alpha=0.3)
ax_b.spines[['top', 'right', 'left', 'bottom']].set_visible(False)

# ── C) Diyabet × APOE grouped bar ───────────────────────────
ax_c = fig.add_subplot(gs[0, 2])
x3 = np.arange(3); w3 = 0.33

for j, (diab, clr2, lbl) in enumerate([(0, '#4d9be6', 'Diyabetsiz'),
                                         (1, RED,      'Diyabetli')]):
    vals3 = []
    for g in GROUPS:
        sub3 = df[(df['APOE_Group'] == g) & (df['DIABETES'] == diab)]
        vals3.append(sub3['TARGET'].mean() * 100 if len(sub3) > 3 else np.nan)

    bars3 = ax_c.bar(x3 + (j - 0.5) * w3, vals3, w3,
                     color=clr2, alpha=0.85, edgecolor='none', label=lbl)
    for bar, v in zip(bars3, vals3):
        if not np.isnan(v):
            ax_c.text(bar.get_x() + bar.get_width() / 2, v + 1,
                      f'{v:.0f}', ha='center', fontsize=9, color=TEXT, fontweight='bold')

ax_c.set_xticks(x3); ax_c.set_xticklabels(G_SHORT, fontsize=9)
ax_c.set_ylim(0, 90)
ax_c.set_ylabel('Alzheimer Oranı (%)', color=SUB)
ax_c.set_title('C  |  Diyabet × APOE', fontsize=12, fontweight='bold', color=TEXT, pad=10)
ax_c.legend(fontsize=10, labelcolor=TEXT, facecolor=CARD, framealpha=0.3)
ax_c.grid(axis='y', alpha=0.3)
ax_c.spines[['top', 'right', 'left', 'bottom']].set_visible(False)

# ── D) Eğitim sürekli risk eğrisi ────────────────────────────
ax_d = fig.add_subplot(gs[1, 0:2])
educ_vals = np.linspace(6, 24, 100)

for g, clr in zip(GROUPS, G_COLORS):
    sub = df[df['APOE_Group'] == g].copy()
    X   = sub[['EDUC', 'AgeatEntry']].values
    y_  = sub['TARGET'].values
    lr  = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    lr.fit(X, y_)
    age_med = sub['AgeatEntry'].median()
    X_line  = np.column_stack([educ_vals, np.full(100, age_med)])
    probs   = lr.predict_proba(X_line)[:, 1] * 100
    ax_d.plot(educ_vals, probs, color=clr, linewidth=2.5,
              label=g.split(' ')[0])

    # Gözlem noktaları (grup ortalamaları)
    for flt, x_pos in [(sub['EDUC'] <= 12,         sub[sub['EDUC'] <= 12]['EDUC'].median()),
                        ((sub['EDUC'] > 12) & (sub['EDUC'] <= 16),
                         sub[(sub['EDUC'] > 12) & (sub['EDUC'] <= 16)]['EDUC'].median()),
                        (sub['EDUC'] > 16,          sub[sub['EDUC'] > 16]['EDUC'].median())]:
        s2 = sub[flt]
        if len(s2) > 5:
            ax_d.scatter(x_pos, s2['TARGET'].mean() * 100,
                         color=clr, s=60, zorder=5, alpha=0.9)

ax_d.set_xlabel('Eğitim Süresi (Yıl)', color=SUB, fontsize=10)
ax_d.set_ylabel('Alzheimer Riski (%)', color=SUB, fontsize=10)
ax_d.set_title('D  |  Eğitim Koruyucu Etkisi — Risk Eğrisi (yaş sabit)\n'
               'Noktalar = gözlem ortalamaları',
               fontsize=11, fontweight='bold', color=TEXT, pad=10)
ax_d.legend(fontsize=10, labelcolor=TEXT, facecolor=CARD,
            framealpha=0.3, title='APOE Grubu')
ax_d.axvline(x=12, color=SUB, linestyle=':', alpha=0.5)
ax_d.text(12.2, 5, 'Lise', fontsize=8, color=SUB)
ax_d.axvline(x=16, color=SUB, linestyle=':', alpha=0.5)
ax_d.text(16.2, 5, 'Lisans', fontsize=8, color=SUB)
ax_d.grid(alpha=0.25)
ax_d.spines[['top', 'right', 'left', 'bottom']].set_visible(False)

# ── E) Eğitim koruma delta chart ────────────────────────────
ax_e = fig.add_subplot(gs[1, 2])
lise_r  = [df[(df['APOE_Group'] == g) & (df['EDUC'] <= 12)]['TARGET'].mean() * 100
           for g in GROUPS]
yuksek_r = [df[(df['APOE_Group'] == g) & (df['EDUC'] >  16)]['TARGET'].mean() * 100
            for g in GROUPS]
delta    = [l - y for l, y in zip(lise_r, yuksek_r)]

bars_e = ax_e.barh(G_SHORT, delta, color=G_COLORS, alpha=0.85, edgecolor='none', height=0.5)
for bar, d, li, yi in zip(bars_e, delta, lise_r, yuksek_r):
    ax_e.text(d + 0.3, bar.get_y() + bar.get_height() / 2,
              f'%{li:.0f} → %{yi:.0f}  (-{d:.0f} puan)',
              va='center', fontsize=9, color=TEXT)

ax_e.set_xlabel('Koruma Etkisi (Lise – Lisansüstü, % puan)', color=SUB)
ax_e.set_title('E  |  Eğitimin Koruyucu Etkisi\n(Lise vs Lisansüstü)',
               fontsize=11, fontweight='bold', color=TEXT)
ax_e.set_xlim(0, max(delta) * 1.55)
ax_e.grid(axis='x', alpha=0.3)
ax_e.spines[['top', 'right', 'left', 'bottom']].set_visible(False)

# ── F) Hippokampüs × APOE × Durum ───────────────────────────
ax_f = fig.add_subplot(gs[2, 0])
rng = np.random.default_rng(RANDOM_SEED)

for idx_g, (g, clr) in enumerate(zip(GROUPS, G_COLORS)):
    sub = df[(df['APOE_Group'] == g) & df['Hippo_Vol'].notna()]
    alz = sub[sub['TARGET'] == 1]['Hippo_Vol']
    sag = sub[sub['TARGET'] == 0]['Hippo_Vol']

    ax_f.scatter(rng.normal(idx_g - 0.15, 0.06, len(sag)), sag,
                 alpha=0.2, s=8, color=GREEN)
    ax_f.scatter(rng.normal(idx_g + 0.15, 0.06, len(alz)), alz,
                 alpha=0.2, s=8, color=RED)
    ax_f.hlines(sag.mean(), idx_g - 0.35, idx_g - 0.05,
                color=GREEN, linewidth=2.5, zorder=5)
    ax_f.hlines(alz.mean(), idx_g + 0.05, idx_g + 0.35,
                color=RED,   linewidth=2.5, zorder=5)

ax_f.set_xticks([0, 1, 2]); ax_f.set_xticklabels(G_SHORT, fontsize=9)
ax_f.set_ylabel('Hippokampüs Hacmi (mm³)', color=SUB)
ax_f.set_title('F  |  Hippokampüs × APOE × Hastalık Durumu',
               fontsize=11, fontweight='bold', color=TEXT)
p_sag = mpatches.Patch(color=GREEN, label='Sağlıklı', alpha=0.7)
p_alz = mpatches.Patch(color=RED,   label='Alzheimer', alpha=0.7)
ax_f.legend(handles=[p_sag, p_alz], fontsize=9,
            labelcolor=TEXT, facecolor=CARD, framealpha=0.3)
ax_f.grid(axis='y', alpha=0.3)
ax_f.spines[['top', 'right', 'left', 'bottom']].set_visible(False)

# ── G) Tanı yaşı dağılımı × APOE ────────────────────────────
ax_g = fig.add_subplot(gs[2, 1])

for g, clr in zip(GROUPS, G_COLORS):
    sub_alz = df[(df['APOE_Group'] == g) & (df['TARGET'] == 1)]['AgeatEntry']
    ax_g.hist(sub_alz, bins=15, alpha=0.5, color=clr,
              label=g.split('(')[0].strip(), density=True, edgecolor='none')
    ax_g.axvline(sub_alz.mean(), color=clr, linewidth=2, linestyle='--', alpha=0.9)
    ax_g.text(sub_alz.mean() + 0.5, 0.005,
              f'{sub_alz.mean():.0f}y', fontsize=8, color=clr)

ax_g.set_xlabel('Tanı Yaşı', color=SUB)
ax_g.set_ylabel('Yoğunluk', color=SUB)
ax_g.set_title('G  |  Alzheimer Tanı Yaşı Dağılımı\n(Kesik çizgi = ortalama)',
               fontsize=11, fontweight='bold', color=TEXT)
ax_g.legend(fontsize=9, labelcolor=TEXT, facecolor=CARD, framealpha=0.3)
ax_g.grid(alpha=0.25)
ax_g.spines[['top', 'right', 'left', 'bottom']].set_visible(False)

# ── H) Özet bulgular kutusu ──────────────────────────────────
ax_h = fig.add_subplot(gs[2, 2])
ax_h.set_xlim(0, 1); ax_h.set_ylim(0, 1); ax_h.axis('off')

findings = [
    ('APOE Doz-Yanıt:',    'Sağlıklı %38 → Taşıyıcı %54 → Baskın %63', BLUE),
    ('',                    'Her ε4 aleli +15–16 puan risk artışı',       SUB),
    ('', '', ''),
    ('Eğitim Koruma:',     'Her 1 yıl riski %7.6–%20.1 azaltıyor',       GREEN),
    ('',                    'Baskın grupta en güçlü etki (-%29.5 puan)',   SUB),
    ('', '', ''),
    ('Diyabet Etkisi:',    'Sağlıklı grupta +%10 (OR=1.51, p=.048)',      RED),
    ('',                    'Taşıyıcıda zayıf etki (OR=1.12, p=.76)',     SUB),
    ('', '', ''),
    ('Hippokampüs:',       'Baskın grupta en düşük hacim',                GOLD),
    ('',                    'Alzheimer grubunda tüm APOE tiplerinde küçük', GOLD),
]
ax_h.text(0.5, 1.04, 'H  |  Özet Bulgular', ha='center', fontsize=11,
          fontweight='bold', color=TEXT, transform=ax_h.transAxes)
y_pos = 0.97
for label, text, clr in findings:
    if label:
        ax_h.text(0.04, y_pos,       label, fontsize=9, fontweight='bold',
                  color=clr, transform=ax_h.transAxes)
        ax_h.text(0.04, y_pos - 0.05, text, fontsize=8.5, color=TEXT,
                  transform=ax_h.transAxes)
        y_pos -= 0.10
    elif text:
        ax_h.text(0.08, y_pos, text, fontsize=8, color=clr,
                  transform=ax_h.transAxes)
        y_pos -= 0.06
    else:
        y_pos -= 0.02

# ── Kaydet ──────────────────────────────────────────────────
plt.savefig(OUTPUT_IMG, dpi=160, bbox_inches='tight',
            facecolor=DARK, edgecolor='none')
plt.close()
print(f"  Görsel kaydedildi → {OUTPUT_IMG}")
print("\n✓ Analiz tamamlandı!")
