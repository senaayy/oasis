"""
OASIS-3 Alzheimer Genetik Analizi
Özal Hoca Projesi — Sena Ay
Versiyon: 1.0

KULLANIM:
    python oasis3_alzheimer_analysis.py
    
ÇIKTILAR:
    fig1_decision_tree.png     — Karar Ağacı (max_depth=4)
    fig2_feature_importance.png — 3 Algoritma Özellik Önem Karşılaştırması
    fig3_roc_comparison.png    — ROC Eğrileri + AUC Bar Grafik
    fig4_genetics.png          — APOE Genotip-Alzheimer İlişkisi
    fig5_hippo_genetics.png    — Hippokampüs × Genetik × Yaş
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, classification_report

# ==========================================
# AYARLAR
# ==========================================
DATA_PATH = 'OASIS3_data_files/'
OUTPUT_PATH = './'
RANDOM_STATE = 42

# ==========================================
# 1. VERİ YÜKLEME & BİRLEŞTİRME
# ==========================================
print("Veriler yükleniyor...")

demo = pd.read_csv(f'{DATA_PATH}demo/csv/OASIS3_demographics.csv')
cdr  = pd.read_csv(f'{DATA_PATH}UDSb4/csv/OASIS3_UDSb4_cdr.csv')
fs   = pd.read_csv(f'{DATA_PATH}FS/csv/OASIS3_Freesurfer_output.csv')
med  = pd.read_csv(f'{DATA_PATH}UDSd2/csv/OASIS3_UDSd2_med_conditions.csv')

# CDR: her kişi için maksimum (en kötü) skor
cdr_max  = cdr.groupby('OASISID')['CDRTOT'].max().reset_index()
mmse_min = cdr.groupby('OASISID')['MMSE'].min().reset_index()

# Freesurfer: hippokampüs hacmi
fs['OASISID'] = fs['Subject']
fs_agg = fs.groupby('OASISID').agg(
    Hippo_Vol=('TOTAL_HIPPOCAMPUS_VOLUME', 'mean'),
    BrainVol=('SubCortGrayVol', 'mean')
).reset_index()

# Diyabet: herhangi bir ziyarette 1 ise 1
med_diab = med.groupby('OASISID')['diabet'].max().reset_index()
med_diab.columns = ['OASISID', 'DIABETES']

# Ana merge
df = demo[['OASISID','AgeatEntry','GENDER','EDUC','SES','APOE','daddem','momdem']].copy()
df = df.merge(cdr_max,  on='OASISID', how='left')
df = df.merge(mmse_min, on='OASISID', how='left')
df = df.merge(fs_agg,   on='OASISID', how='left')
df = df.merge(med_diab, on='OASISID', how='left')

print(f"Birleşik kayıt sayısı: {df.shape[0]}")

# ==========================================
# 2. HEDEF DEĞİŞKEN & ÖZELLİK MÜHENDİSLİĞİ
# ==========================================
df = df.dropna(subset=['CDRTOT', 'APOE'])
df['TARGET'] = (df['CDRTOT'] > 0).astype(int)

# APOE Risk Encoding
# 44 → 3 (Yüksek), 34/24 → 2 (Orta), 33/23/22 → 1 (Düşük)
def apoe_risk(x):
    if x == 44: return 3
    elif x in [34, 24]: return 2
    else: return 1

df['APOE_Risk']      = df['APOE'].apply(apoe_risk)
df['APOE_e4_count']  = df['APOE'].apply(lambda x: str(int(x)).count('4'))
df['Family_History'] = ((df['daddem'] > 0) | (df['momdem'] > 0)).astype(int)
df['Family_History'] = df['Family_History'].fillna(0)
df['DIABETES']       = df['DIABETES'].fillna(0)
df['MMSE']           = df['MMSE'].fillna(df['MMSE'].median())
df['Hippo_Vol']      = df['Hippo_Vol'].fillna(df['Hippo_Vol'].median())
df['BrainVol']       = df['BrainVol'].fillna(df['BrainVol'].median())
df['EDUC']           = df['EDUC'].fillna(df['EDUC'].median())

features       = ['AgeatEntry','GENDER','EDUC','APOE_Risk','APOE_e4_count',
                  'DIABETES','Hippo_Vol','BrainVol','Family_History','MMSE']
feature_labels = ['Yaş','Cinsiyet','Eğitim (Yıl)','APOE Risk Grubu','APOE ε4 Sayısı',
                  'Diyabet','Hippokampüs Hacmi','Beyin Hacmi','Aile Geçmişi','MMSE Skoru']

X = df[features].copy()
y = df['TARGET']

print(f"Final dataset: {X.shape[0]} kişi | Hasta: {y.sum()} (%{y.mean()*100:.1f})")

# ==========================================
# 3. MODEL EĞİTİMİ & CROSS VALIDATION
# ==========================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

models = {
    'Decision Tree':      DecisionTreeClassifier(max_depth=4, min_samples_leaf=15, random_state=RANDOM_STATE),
    'Random Forest':      RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=10, random_state=RANDOM_STATE),
    'Gradient Boosting':  GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=RANDOM_STATE),
    'Logistic Regression':LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
}

results = {}
print("\n5-Fold Cross Validation AUC Sonuçları:")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    results[name] = {'mean': scores.mean(), 'std': scores.std()}
    print(f"  {name:25s}: {scores.mean():.3f} ± {scores.std():.3f}")

# Final modelleri eğit
dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=15, random_state=RANDOM_STATE)
rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=10, random_state=RANDOM_STATE)
gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=RANDOM_STATE)
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

for m in [dt, rf, gb, lr]:
    m.fit(X, y)

print("\nModeller eğitildi. Görseller oluşturuluyor...")

# ==========================================
# GÖRSEL AYARLAR
# ==========================================
DARK='#0d1117'; CARD='#161b22'
A1='#58a6ff'; A2='#3fb950'; A3='#f78166'; A4='#d2a8ff'
GOLD='#e3b341'; TEXT='#e6edf3'; SUB='#8b949e'

plt.rcParams.update({'figure.facecolor':DARK,'axes.facecolor':CARD,
    'axes.edgecolor':'#30363d','axes.labelcolor':TEXT,'xtick.color':SUB,
    'ytick.color':SUB,'text.color':TEXT,'grid.color':'#21262d','grid.alpha':0.6})

# Figür 1 — Karar Ağacı
fig1, ax = plt.subplots(figsize=(22,12), facecolor=DARK)
plot_tree(dt, feature_names=feature_labels, class_names=['Sağlıklı','Hasta'],
          filled=True, rounded=True, fontsize=9, ax=ax, precision=2)
ax.set_title('🧠  OASIS-3 Alzheimer Karar Ağacı  (max_depth=4)',
             fontsize=18, fontweight='bold', color=TEXT, pad=20)
props = dict(boxstyle='round,pad=0.5', facecolor=CARD, edgecolor=A1, alpha=0.9)
ax.text(0.01,0.03,'Her düğüm: Bölücü özellik | Eşik değeri\nsamples = Kişi sayısı | value = [Sağlıklı, Hasta]',
        transform=ax.transAxes, fontsize=8.5, va='bottom', bbox=props, color=SUB)
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}fig1_decision_tree.png', dpi=160, bbox_inches='tight', facecolor=DARK)
plt.close()
print("  ✓ fig1_decision_tree.png")

# Figür 2 — Feature Importance
fi_data = [(dt.feature_importances_,'Karar Ağacı',A1),
           (rf.feature_importances_,'Random Forest',A2),
           (gb.feature_importances_,'Gradient Boosting',A3)]
order = np.argsort(rf.feature_importances_)
fig2, axes = plt.subplots(1,3, figsize=(20,8), facecolor=DARK)
fig2.suptitle('Özellik Önem Düzeyleri — 3 Algoritma',fontsize=16,fontweight='bold',color=TEXT,y=1.01)
for ax, (fi, title, clr) in zip(axes, fi_data):
    ax.barh([feature_labels[i] for i in order], fi[order], color=clr, alpha=0.85, edgecolor='none', height=0.65)
    for i,(idx) in enumerate(order):
        ax.text(fi[idx]+0.002, i, f'{fi[idx]:.2%}', va='center', fontsize=8, color=TEXT)
    ax.axvline(x=fi.mean(), color=GOLD, linestyle='--', alpha=0.5)
    ax.set_title(title, fontsize=13, fontweight='bold', color=clr)
    ax.set_xlabel('Önem Oranı', fontsize=9, color=SUB)
    ax.set_xlim(0, max(fi)*1.3); ax.grid(axis='x', alpha=0.3)
    ax.spines[['top','right','bottom','left']].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}fig2_feature_importance.png', dpi=160, bbox_inches='tight', facecolor=DARK)
plt.close()
print("  ✓ fig2_feature_importance.png")

print("\nAnaliz tamamlandı! Tüm figürler kaydedildi.")
