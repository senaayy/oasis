# OASIS-3 | APOE Genetik Riski × Yaşam Tarzı Faktörleri × Alzheimer

Gerçek OASIS-3 longitudinal kohort verisi (1.378 katılımcı, Washington Üniversitesi Knight
Alzheimer Hastalığı Araştırma Merkezi) kullanılarak yapılmış, APOE ε4 genetik riskinin eğitim,
diyabet, sigara, tiroid hastalığı ve B12 eksikliği ile etkileşimini inceleyen dört bağımsız
istatistiksel analiz.

## ⚠️ Veri Gizliliği Notu

Bu repo **hiçbir ham veya kişi-bazlı OASIS-3 verisi içermez**. OASIS-3'e erişim bir Veri
Kullanım Sözleşmesi (Data Use Agreement) gerektirir ve bu sözleşme kişi-bazlı verinin
yeniden dağıtılmasını yasaklar. Burada yalnızca:

- analiz kodları,
- toplu/özet istatistiksel sonuçlar,
- ve önceden üretilmiş görseller (agregat grafikler, hiçbir kişisel tanımlayıcı içermez)

paylaşılmaktadır. Kodu çalıştırmak isteyen biri, OASIS-3 verisine kendi DUA'sı üzerinden
[oasis-brains.org](https://www.oasis-brains.org) adresinden erişmelidir; veri seti bu repoya
dahil değildir ve `.gitignore` ile açıkça hariç tutulmuştur.

## Veri Kaynağı

- LaMontagne PJ, et al. (2019). *OASIS-3: Longitudinal Neuroimaging, Clinical, and Cognitive
  Dataset for Normal Aging and Alzheimer Disease*. medRxiv.
  [doi:10.1101/2019.12.13.19014902](https://doi.org/10.1101/2019.12.13.19014902)
- Marcus DS, et al. (2010). *Open Access Series of Imaging Studies (OASIS)*. PMC2895005.

Kullanılan CSV modülleri: `demo` (demografi), `UDSb4` (CDR), `FS` (FreeSurfer beyin hacimleri),
`UDSa5` (sağlık geçmişi), `UDSd2` (tıbbi durumlar), `UDSd1` (tanılar) — Uniform Data Set (UDS)
standart kodlamasıyla.

## Analizler

| Script | Odak | Görseller |
|---|---|---|
| `scripts/oasis3_alzheimer_analysis.py` | APOE + klinik/demografik özelliklerle 4 algoritma karşılaştırması (Karar Ağacı, Random Forest, Gradient Boosting, Lojistik Regresyon), 5-fold CV | `fig1_decision_tree.png`, `fig2_feature_importance.png`, `fig3_roc_comparison.png`, `fig4_genetics.png`, `fig5_hippo_genetics.png` |
| `scripts/apoe_doz_yanit_analizi.py` | APOE ε4 alel sayısına göre doz-yanıt ilişkisi, eğitimin koruyucu etkisi, diyabet etkileşimi | `APOE_DozYanit_Analizi.png` |
| `scripts/sigara_apoe_analizi.py` | Sigara/pack-years doz-yanıt trendi, APOE grubuna göre etkileşim, bırakma süresi | `Sigara_APOE_Alzheimer_Analizi.png` |
| `scripts/tiroid_b12_analizi.py` | Tiroid hastalığı ve B12 eksikliğinin (tedavi edilebilir nedenler) Alzheimer tanısıyla ilişkisi | `Tiroid_B12_Alzheimer_Analizi.png` |

## Yöntem

- **Hedef değişken:** CDR (Clinical Dementia Rating) > 0 → Alzheimer/MCI (1), CDR = 0 → Sağlıklı (0)
- **APOE gruplaması:** Sağlıklı (ε3/ε3 vb.) / Taşıyıcı (1×ε4) / Baskın (ε4/ε4)
- **Odds Ratio + %95 güven aralığı:** 2×2 kontenjans tablosu + ki-kare testi; küçük alt
  gruplarda 500 tekrarlı bootstrap ile yüzdelik dilim yöntemi
- **Yaşa/eğitime göre düzeltilmiş OR:** Lojistik regresyon (kovaryant: yaş, eğitim yılı)
- **Doz-yanıt trend testi:** Spearman korelasyonu (sigara pack-years için)
- **Model karşılaştırması:** StratifiedKFold (5 kat) çapraz doğrulama, ROC-AUC

## Temel Bulgular

- **APOE doz-yanıt:** Sağlıklı %38 → Taşıyıcı %54 → Baskın %63 (her ε4 aleli +15-16 puan risk)
- **Eğitim koruyucu etkisi:** Her 1 yıl ek eğitim riski %7.6-%20.1 azaltıyor; en güçlü etki
  Baskın APOE grubunda (-%29.5 puan, Lise→Lisansüstü)
- **B12 eksikliği:** OR=2.51 (ham) → yaşa göre düzeltilmiş OR=1.93, p<0.0001 — güçlü ve
  anlamlı; tedavi edilebilir bir neden olması klinik açıdan önemli
- **Sigara — ana etki yok, doz-yanıt var:** Sigara tek başına anlamlı değil (OR=0.91, p=0.34)
  ama pack-years ile doz-yanıt trendi anlamlı (r=0.147, p=0.0005)
- **Model performansı:** En iyi model (Random Forest) 5-fold CV AUC=0.884±0.026; en önemli
  öngörücü MMSE skoru (3 algoritmada da baskın), ardından hippokampüs hacmi

## Sınırlılıklar (bilinçli olarak burada tutuluyor)

- **Çoklu karşılaştırma düzeltmesi uygulanmadı** (Bonferroni/FDR yok) — sonuçlar keşifsel
  olarak yorumlanmalı, doğrulayıcı değil. Sınırda anlamlı bulgular (ör. diyabet OR=1.51,
  p=.048) tek başına güvenilir kabul edilmemeli.
- **Per-kişi maksimum CDR** kullanıldığı için, başlangıçta CDR=0 olup sonradan kötüleşen
  kişilerle baştan beri CDR>0 olanlar ayrıştırılamıyor.
- **Öz-bildirim/klinisyen kaydı:** B12, tiroid, sigara verileri laboratuvar doğrulamasıyla
  teyit edilmemiş.
- **Baskın APOE alt grubu küçük** (N≈78) — bu gruptaki etkileşim tahminleri (ör. sigara
  bırakma paradoksu) survival bias'a açık ve geniş güven aralıklarına sahip.
- Bu depo bir tanı aracı değildir; sonuçlar yalnızca istatistiksel ilişkileri gösterir,
  klinik karar desteği olarak kullanılamaz.

## Çalıştırma

```bash
pip install -r requirements.txt
# OASIS-3 verisini kendi DUA'nızla indirip OASIS3_data_files/ altına çıkarın
python scripts/oasis3_alzheimer_analysis.py
python scripts/apoe_doz_yanit_analizi.py
python scripts/sigara_apoe_analizi.py
python scripts/tiroid_b12_analizi.py
```

## Yazar

Sena Ay — Özal Hoca ile yapılan araştırma projesi, 2026
