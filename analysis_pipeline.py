# -*- coding: utf-8 -*-
"""
=============================================================================
  PeakGuard-AI  |  단속 카메라 재배치 통합 분석 파이프라인
=============================================================================
  Phase 1  : 데이터 로드 & 공간 매칭
  Phase 2  : EDA
  Phase 3  : K-Means 군집화 (k=4)
  Phase 4  : 군집별 사고 유형 분석
  Phase 5  : 재배치 제안
  Phase 5-B: 과잉 군집 내 K-Means(k=3) 재군집화
  Phase 6  : 카메라 설치 실효성 증명
  Phase 7  : 군집별 심각도 분석
  Phase 8  : 재배치 우선순위 도출
  Phase 9  : 우선순위별 예방 효과 시각화
=============================================================================
  통합 이전 파일 목록:
    - camera_redistribution_v2.py  (Phase 1~5-B + CSV 저장)
    - camera_effectiveness_proof.py (Phase 6)
    - severity_analysis.py          (Phase 7)
    - camera_prioritization.py      (Phase 8)
    - visualize_proof.py            (Phase 9)
    - check_clusters.py             (진단 유틸 -> 삭제)
=============================================================================
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pyproj import Transformer
from scipy.spatial import cKDTree
from matplotlib.patches import Patch

mpl.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = 'output_v2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("  PeakGuard-AI  단속 카메라 재배치 통합 분석 파이프라인")
print("=" * 70)


# ╔═══════════════════════════════════════════════════════════════╗
# ║  PHASE 1: 데이터 로드 & 공간 매칭                               ║
# ╚═══════════════════════════════════════════════════════════════╝
print("\n[Phase 1] 데이터 로드 및 공간 매칭")

df_risk = pd.read_csv('RiskArea.csv', encoding='cp949')
df_risk.rename(columns={'츙사고건수': '총사고건수'}, inplace=True)

df_cam = pd.read_csv('경찰청_무인교통단속카메라_20260406.csv', encoding='utf-8')

# 카메라 위경도 → UTM-K
transformer = Transformer.from_crs("EPSG:4326", "EPSG:5178", always_xy=True)
cam_x, cam_y = transformer.transform(df_cam['경도'].values, df_cam['위도'].values)
df_cam['utmk_x'] = cam_x
df_cam['utmk_y'] = cam_y

# 시도 매핑
시도코드_map = {
    11: '서울', 26: '부산', 27: '대구', 28: '인천',
    29: '광주', 30: '대전', 31: '울산', 36: '세종',
    41: '경기', 42: '강원', 43: '충북', 44: '충남',
    45: '전북', 46: '전남', 47: '경북', 48: '경남', 50: '제주'
}
df_risk['시도코드'] = df_risk['시군구코드'] // 1000
df_risk['시도명'] = df_risk['시도코드'].map(시도코드_map)

# 단속구분 매핑
단속구분_map = {1: '과속', 2: '신호위반', 3: '과속+신호', 99: '기타'}
df_cam['단속구분명'] = df_cam['단속구분'].map(단속구분_map).fillna('기타')

print(f"  사고위험지역: {len(df_risk):,}건")
print(f"  카메라: {len(df_cam):,}대")

# ── 공간 매칭 (반경 100m, cKDTree) ───────────────────────────
RADIUS = 100
risk_coords = df_risk[['중심점utmkx좌표', '중심점utmky좌표']].dropna().values
cam_coords = df_cam[['utmk_x', 'utmk_y']].values
tree_cam = cKDTree(cam_coords)

cam_counts, cam_overspeed, cam_signal = [], [], []
for rx, ry in risk_coords:
    indices = tree_cam.query_ball_point([rx, ry], r=RADIUS)
    cam_counts.append(len(indices))
    if len(indices) > 0:
        subset = df_cam.iloc[indices]
        cam_overspeed.append((subset['단속구분'] == 1).sum())
        cam_signal.append((subset['단속구분'] == 2).sum())
    else:
        cam_overspeed.append(0)
        cam_signal.append(0)

df = df_risk.dropna(subset=['중심점utmkx좌표', '중심점utmky좌표']).copy()
df['반경내카메라수'] = cam_counts
df['반경내과속카메라'] = cam_overspeed
df['반경내신호카메라'] = cam_signal
df['카메라유무'] = (df['반경내카메라수'] > 0).astype(int)

# UTM-K → 위경도 (지도 시각화용)
transformer_inv = Transformer.from_crs("EPSG:5178", "EPSG:4326", always_xy=True)
df['경도'], df['위도'] = transformer_inv.transform(
    df['중심점utmkx좌표'].values, df['중심점utmky좌표'].values)

# 사고유형 파싱
def parse_types(type_str):
    if pd.isna(type_str):
        return []
    return [t.strip() for t in str(type_str).replace('/', ',').split(',') if t.strip()]

df['사고유형목록'] = df['사고분석유형명'].apply(parse_types)

print(f"  매칭 완료: {len(df):,}건")
print(f"  카메라 있는 지역: {df['카메라유무'].sum():,}건 ({100*df['카메라유무'].mean():.1f}%)")
print(f"  카메라 없는 지역: {(df['카메라유무']==0).sum():,}건 ({100*(1-df['카메라유무'].mean()):.1f}%)")


# ╔═══════════════════════════════════════════════════════════════╗
# ║  PHASE 2: 지점 단위 EDA                                        ║
# ╚═══════════════════════════════════════════════════════════════╝
print("\n[Phase 2] 지점 단위 EDA")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('사고위험지역 기초 현황 분석 (지점 단위)', fontsize=20, fontweight='bold', y=0.98)

# (a) 카메라 수 분포
ax = axes[0, 0]
cam_dist = df['반경내카메라수'].value_counts().sort_index()
ax.bar(cam_dist.index[:15], cam_dist.values[:15], color='#3498db', edgecolor='white', alpha=0.85)
ax.set_title(f'반경 100m 내 카메라 수 분포\n(평균: {df["반경내카메라수"].mean():.2f}대)', fontsize=13, fontweight='bold')
ax.set_xlabel('카메라 수 (대)')
ax.set_ylabel('지점 수')
ax.grid(axis='y', alpha=0.3)

# (b) 사고건수 분포
ax = axes[0, 1]
ax.hist(df['총사고건수'].clip(upper=100), bins=40, color='#e74c3c', edgecolor='white', alpha=0.85)
ax.set_title(f'총 사고건수 분포 (100건 이하)\n(평균: {df["총사고건수"].mean():.1f}건)', fontsize=13, fontweight='bold')
ax.set_xlabel('총 사고건수')
ax.set_ylabel('빈도')
ax.grid(axis='y', alpha=0.3)

# (c) 카메라 유무별 사고건수 박스플롯
ax = axes[1, 0]
groups = [df[df['카메라유무']==0]['총사고건수'], df[df['카메라유무']==1]['총사고건수']]
bp = ax.boxplot(groups, labels=['카메라 없음', '카메라 있음'], patch_artist=True, showfliers=False)
for patch, color in zip(bp['boxes'], ['#e74c3c', '#3498db']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title('카메라 유무별 사고건수 분포', fontsize=13, fontweight='bold')
ax.set_ylabel('총 사고건수')
ax.grid(axis='y', alpha=0.3)

# (d) 시도별 지점 수
ax = axes[1, 1]
sido_cnt = df['시도명'].value_counts().sort_values(ascending=True)
ax.barh(sido_cnt.index, sido_cnt.values, color=plt.cm.Blues(np.linspace(0.4, 0.9, len(sido_cnt))), edgecolor='white')
ax.set_title('시도별 사고위험지역 수', fontsize=13, fontweight='bold')
ax.set_xlabel('지점 수')
for i, (idx, val) in enumerate(sido_cnt.items()):
    ax.text(val + 20, i, f'{val:,}', va='center', fontsize=9)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, '01_EDA_기초현황.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → 01_EDA_기초현황.png 저장")

# ── EDA: 카메라-사고 상관관계 ──────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('카메라-사고 상관관계 분석', fontsize=20, fontweight='bold', y=0.98)

# (a) 카메라수 vs 사고건수 산점도
ax = axes[0, 0]
has_cam = df[df['카메라유무'] == 1]
no_cam_eda = df[df['카메라유무'] == 0]
ax.scatter(no_cam_eda['반경내카메라수'], no_cam_eda['총사고건수'], s=8, c='#e74c3c', alpha=0.3, label='카메라 없음')
ax.scatter(has_cam['반경내카메라수'], has_cam['총사고건수'], s=8, c='#3498db', alpha=0.3, label='카메라 있음')
ax.set_title('카메라 수 vs 사고건수', fontsize=13, fontweight='bold')
ax.set_xlabel('반경 100m 내 카메라 수')
ax.set_ylabel('총 사고건수')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# (b) 카메라 유무별 주요 지표 비교
ax = axes[0, 1]
metrics = ['총사고건수', '총사망자수', '총중상자수']
vals_no = [no_cam_eda[m].mean() for m in metrics]
vals_yes = [has_cam[m].mean() for m in metrics]
x_pos = np.arange(len(metrics))
bars1 = ax.bar(x_pos - 0.2, vals_no, 0.4, label=f'카메라 없음 ({len(no_cam_eda):,})', color='#e74c3c', alpha=0.85)
bars2 = ax.bar(x_pos + 0.2, vals_yes, 0.4, label=f'카메라 있음 ({len(has_cam):,})', color='#3498db', alpha=0.85)
ax.set_xticks(x_pos)
ax.set_xticklabels(['평균 사고건수', '평균 사망자수', '평균 중상자수'], fontsize=11)
ax.set_title('카메라 유무별 주요 지표 비교', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
for bars in [bars1, bars2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{bar.get_height():.2f}', ha='center', fontsize=9, fontweight='bold')

# (c) 카메라 수 구간별 평균 사고건수
ax = axes[1, 0]
df['카메라수_구간'] = pd.cut(df['반경내카메라수'], bins=[-1, 0, 1, 2, 3, 5, 100],
                            labels=['0대', '1대', '2대', '3대', '4-5대', '6대+'])
grp = df.groupby('카메라수_구간')['총사고건수'].mean()
ax.bar(grp.index, grp.values, color='#9b59b6', edgecolor='white', alpha=0.85)
ax.set_title('카메라 수 구간별 평균 사고건수', fontsize=13, fontweight='bold')
ax.set_xlabel('카메라 수')
ax.set_ylabel('평균 사고건수')
for i, v in enumerate(grp.values):
    ax.text(i, v + 0.2, f'{v:.1f}', ha='center', fontsize=10, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# (d) 사고유형 빈도
ax = axes[1, 1]
all_types = []
for types_list in df['사고유형목록']:
    all_types.extend(types_list)
type_counts = pd.Series(all_types).value_counts().head(10)
ax.barh(type_counts.index[::-1], type_counts.values[::-1], color='#27ae60', edgecolor='white', alpha=0.85)
ax.set_title('전체 사고 유형 빈도 (상위 10)', fontsize=13, fontweight='bold')
ax.set_xlabel('빈도')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, '02_EDA_상관관계.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → 02_EDA_상관관계.png 저장")


# ╔═══════════════════════════════════════════════════════════════╗
# ║  PHASE 3: 지점 단위 K-Means 군집화 (k=4)                       ║
# ╚═══════════════════════════════════════════════════════════════╝
print("\n[Phase 3] 지점 단위 K-Means 군집화")

feature_cols = ['반경내카메라수', '총사고건수', '총사망자수', '총중상자수']
X = df[feature_cols].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow + Silhouette
inertias, sil_scores = [], []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels_k = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels_k, sample_size=5000, random_state=42))
    print(f"    k={k}: Inertia={km.inertia_:.1f}, Silhouette={sil_scores[-1]:.4f}")

best_k_sil = list(K_range)[np.argmax(sil_scores)]
optimal_k = 4
print(f"\n  Silhouette 최적 k: {best_k_sil}, 채택 k: {optimal_k}")

# 최종 군집화
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 특성 기반 라벨 매핑
cluster_stats = df.groupby('cluster').agg(
    건수=('총사고건수', 'count'),
    평균카메라수=('반경내카메라수', 'mean'),
    평균사고건수=('총사고건수', 'mean'),
    평균사망자수=('총사망자수', 'mean'),
    평균중상자수=('총중상자수', 'mean'),
    카메라0비율=('카메라유무', lambda x: 1 - x.mean())
).reset_index()

label_order = [
    '카메라 설치지점',
    '사망위험',
    '부족',
    '사고다발'
]

death_idx = cluster_stats['평균사망자수'].idxmax()
death_cluster = int(cluster_stats.loc[death_idx, 'cluster'])
remaining = cluster_stats.drop(death_idx)

accident_idx = remaining['평균사고건수'].idxmax()
accident_cluster = int(remaining.loc[accident_idx, 'cluster'])
remaining = remaining.drop(accident_idx)

surplus_idx = remaining['평균카메라수'].idxmax()
surplus_cluster = int(remaining.loc[surplus_idx, 'cluster'])
remaining = remaining.drop(surplus_idx)

deficit_cluster = int(remaining.iloc[0]['cluster'])

label_map = {
    surplus_cluster: label_order[0],
    death_cluster:   label_order[1],
    deficit_cluster: label_order[2],
    accident_cluster: label_order[3],
}
df['군집라벨'] = df['cluster'].map(label_map)

print("\n  군집화 결과:")
for _, row in cluster_stats.iterrows():
    c = int(row['cluster'])
    label = label_map[c]
    print(f"    [{label}] {int(row['건수']):,}건 | 카메라 {row['평균카메라수']:.1f}대 | 사고 {row['평균사고건수']:.1f}건 | 사망 {row['평균사망자수']:.2f} | 중상 {row['평균중상자수']:.1f}")

cluster_colors_list = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
cluster_colors = {label_order[i]: cluster_colors_list[i] for i in range(4)}

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('K-Means 군집화 결과 (n={:,})'.format(len(df)), fontsize=20, fontweight='bold', y=0.98)

# (a) Elbow + Silhouette
ax = axes[0, 0]
ax2_twin = ax.twinx()
ax.plot(list(K_range), inertias, 'bo-', linewidth=2, markersize=8, label='Inertia')
ax2_twin.plot(list(K_range), sil_scores, 'gs-', linewidth=2, markersize=8, label='Silhouette')
ax.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'선택: k={optimal_k}')
ax.set_title('최적 k 결정 (Elbow + Silhouette)', fontsize=14, fontweight='bold')
ax.set_xlabel('클러스터 수 (k)')
ax.set_ylabel('Inertia', color='blue')
ax2_twin.set_ylabel('Silhouette Score', color='green')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')
ax.grid(alpha=0.3)

# (b) 산점도
ax = axes[0, 1]
for label, color in cluster_colors.items():
    subset = df[df['군집라벨'] == label]
    ax.scatter(subset['반경내카메라수'], subset['총사고건수'],
               s=12, c=color, alpha=0.35, label=f"{label.split('(')[0].strip()} ({len(subset):,})")
ax.set_title('카메라 수 vs 사고 건수 (군집별)', fontsize=14, fontweight='bold')
ax.set_xlabel('반경 100m 내 카메라 수')
ax.set_ylabel('총 사고 건수')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# (c) 군집별 평균 비교 (정규화)
ax = axes[1, 0]
plot_stats = df.groupby('군집라벨')[feature_cols].mean()
plot_stats_norm = plot_stats.copy()
for col in plot_stats.columns:
    mx = plot_stats[col].max()
    if mx > 0:
        plot_stats_norm[col] = plot_stats[col] / mx
x_pos = np.arange(len(feature_cols))
width = 0.2
for i, (label, row) in enumerate(plot_stats_norm.iterrows()):
    short = label.split('(')[0].strip()
    color = cluster_colors.get(label, '#95a5a6')
    ax.bar(x_pos + i * width, row.values, width, label=short, color=color, alpha=0.85)
ax.set_xticks(x_pos + width * 1.5)
ax.set_xticklabels(['카메라수', '사고건수', '사망자수', '중상자수'], fontsize=11)
ax.set_title('군집별 평균 지표 비교 (정규화)', fontsize=14, fontweight='bold')
ax.set_ylabel('정규화 값')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# (d) 시도별 군집 비율
ax = axes[1, 1]
ct = pd.crosstab(df['시도명'], df['군집라벨'], normalize='index') * 100
ct = ct.reindex(columns=label_order)
ct.plot(kind='barh', stacked=True, ax=ax,
        color=[cluster_colors[l] for l in ct.columns], edgecolor='white')
ax.set_title('시도별 군집 구성 비율 (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('비율 (%)')
ax.legend(fontsize=7, loc='lower right', labels=[l.split('(')[0].strip() for l in ct.columns])

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, '03_군집화결과.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → 03_군집화결과.png 저장")

# ── 1차 군집화 PCA 산점도 추가 (센트로이드 포함) ─────────────────────────────────────────
print("  -> 12_PCA_1차산점도.png 생성 중...")
from sklearn.decomposition import PCA
features_1 = ['반경내카메라수', '총사고건수', '총사망자수', '총중상자수']
df_log = np.log1p(df[features_1])
scaler_pca = StandardScaler()
X_scaled_pca = scaler_pca.fit_transform(df_log)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_pca)

np.random.seed(42)
jitter_strength = 0.15
pca_df = pd.DataFrame({
    'PCA1_jitter': X_pca[:, 0] + np.random.normal(0, jitter_strength, size=len(X_pca)),
    'PCA2_jitter': X_pca[:, 1] + np.random.normal(0, jitter_strength, size=len(X_pca)),
    'PCA1_real': X_pca[:, 0],
    'PCA2_real': X_pca[:, 1],
    '군집라벨': df['군집라벨']
})

plt.figure(figsize=(12, 9))
sns.scatterplot(
    x='PCA1_jitter', y='PCA2_jitter', hue='군집라벨', data=pca_df,
    palette=['#e74c3c', '#e67e22', '#2ecc71', '#3498db'], alpha=0.4, s=40, edgecolor=None
)

centroids_pca = pca_df.groupby('군집라벨')[['PCA1_real', 'PCA2_real']].mean().reset_index()
for idx, row in centroids_pca.iterrows():
    cluster_name = row['군집라벨']
    plt.scatter(row['PCA1_real'], row['PCA2_real'], marker='*', s=800, c='gold', edgecolor='black', linewidth=1.5, zorder=10)
    plt.text(row['PCA1_real'] + 0.1, row['PCA2_real'] + 0.1, f"Centroid\n{cluster_name.split('(')[0]}", 
             fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.3'), zorder=11)

plt.title('1차 군집화 시각화', fontsize=18, fontweight='bold')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=13)
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=13)
plt.legend(title='데이터 포인트', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '12_PCA_1차산점도.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  -> 12_PCA_1차산점도.png 저장")

# ── 1차 군집화 사고 지점 수 (포인트 수) 막대 그래프 추가 ─────────────────────────────────────────
print("  -> 14_1차군집별_사고지점수.png 생성 중...")
plt.figure(figsize=(10, 6))
cluster_counts = df['군집라벨'].value_counts().reindex(label_order)
ax = cluster_counts.plot(kind='bar', color=[cluster_colors.get(l, '#95a5a6') for l in cluster_counts.index], edgecolor='black', alpha=0.85)
plt.title('1차 군집별 사고 지점 수 (N)', fontsize=16, fontweight='bold')
plt.xlabel('군집 (Cluster)', fontsize=13)
plt.ylabel('사고 지점 수 (개)', fontsize=13)
plt.xticks(rotation=0, fontsize=11)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 9), 
                textcoords = 'offset points', fontsize=12, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '14_1차군집별_사고지점수.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> 14_1차군집별_사고지점수.png 저장")


# ╔═══════════════════════════════════════════════════════════════╗
# ║  PHASE 4: 군집별 사고 유형 분석                                  ║
# ╚═══════════════════════════════════════════════════════════════╝
print("\n[Phase 4] 군집별 사고 유형 분석")

단속가능 = {'과속', '신호위반'}
간접억제 = {'안전거리미확보', '중앙선침범'}

def classify_camera_effect(types_list):
    direct = sum(1 for t in types_list if t in 단속가능)
    indirect = sum(1 for t in types_list if t in 간접억제)
    total = max(len(types_list), 1)
    return direct, indirect, total

df[['단속가능수', '간접억제수', '유형총수']] = df['사고유형목록'].apply(
    lambda x: pd.Series(classify_camera_effect(x))
)
df['카메라효과비율'] = (df['단속가능수'] + df['간접억제수']) / df['유형총수'] * 100

all_types_by_cluster = {}
for label in label_order:
    subset = df[df['군집라벨'] == label]
    all_t = []
    for types_list in subset['사고유형목록']:
        all_t.extend(types_list)
    all_types_by_cluster[label] = pd.Series(all_t).value_counts()

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('군집별 사고 유형 분석\n(카메라 재배치 실효성 검증)', fontsize=20, fontweight='bold', y=0.98)

for idx, (label, type_cnt) in enumerate(all_types_by_cluster.items()):
    ax = axes[idx // 2, idx % 2]
    top7 = type_cnt.head(7)
    colors = ['#e74c3c' if t in 단속가능 else ('#f39c12' if t in 간접억제 else '#95a5a6')
              for t in top7.index]
    bars = ax.barh(range(len(top7)), top7.values, color=colors, edgecolor='white', height=0.6)
    ax.set_yticks(range(len(top7)))
    ax.set_yticklabels(top7.index, fontsize=11)
    ax.invert_yaxis()
    short = label.split('(')[0].strip()
    subset = df[df['군집라벨'] == label]
    cam_effect = subset['카메라효과비율'].mean()
    ax.set_title(f'{short} ({len(subset):,}건) | 카메라 단속 관련: {cam_effect:.1f}%',
                 fontsize=13, fontweight='bold', color=cluster_colors[label])
    ax.set_xlabel('빈도')
    for bar, val in zip(bars, top7.values):
        pct = val / type_cnt.sum() * 100
        ax.text(bar.get_width() + max(top7.values)*0.01,
                bar.get_y() + bar.get_height()/2,
                f'{val:,} ({pct:.1f}%)', va='center', fontsize=10)
    ax.grid(axis='x', alpha=0.3)

legend_elements = [
    Patch(facecolor='#e74c3c', label='🎯 직접 단속 (과속, 신호위반)'),
    Patch(facecolor='#f39c12', label='🔶 간접 억제 (안전거리미확보, 중앙선침범)'),
    Patch(facecolor='#95a5a6', label='⬜ 단속 어려움 (기타, U턴 등)')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12,
           bbox_to_anchor=(0.5, 0.01), frameon=True)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, '05_군집별_사고유형.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → 05_군집별_사고유형.png 저장")


# ╔═══════════════════════════════════════════════════════════════╗
# ║  PHASE 5: 재배치 제안                                           ║
# ╚═══════════════════════════════════════════════════════════════╝
print("\n[Phase 5] 재배치 제안")

no_cam = df[df['반경내카메라수'] == 0].copy()
no_cam['위험점수'] = no_cam['총사고건수'] + no_cam['총사망자수']*10 + no_cam['총중상자수']*5

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('단속 카메라 재배치 제안', fontsize=20, fontweight='bold', y=0.98)

# (a) 카메라 미설치 고위험 - 시도별
ax = axes[0, 0]
no_cam_sido = no_cam.groupby('시도명')['위험점수'].agg(['count', 'mean']).sort_values('count', ascending=True)
colors_nc = plt.cm.Reds(np.linspace(0.3, 0.9, len(no_cam_sido)))
ax.barh(no_cam_sido.index, no_cam_sido['count'], color=colors_nc, edgecolor='white')
ax.set_title(f'카메라 미설치 사고위험지역 ({len(no_cam):,}건)\n시도별 분포', fontsize=13, fontweight='bold')
ax.set_xlabel('지역 수')
for i, (idx, row) in enumerate(no_cam_sido.iterrows()):
    ax.text(row['count'] + 2, i, f"{int(row['count'])}건", va='center', fontsize=9)
ax.grid(axis='x', alpha=0.3)

# (b) 카메라 과밀 지역 시도별
ax = axes[0, 1]
over_cam = df[df['반경내카메라수'] >= 10].copy()
over_cam_sido = over_cam.groupby('시도명').agg(
    지역수=('총사고건수', 'count'),
    평균카메라=('반경내카메라수', 'mean')
).sort_values('지역수', ascending=True)
colors_oc = plt.cm.Blues(np.linspace(0.3, 0.9, len(over_cam_sido)))
ax.barh(over_cam_sido.index, over_cam_sido['지역수'], color=colors_oc, edgecolor='white')
ax.set_title(f'카메라 과밀(10대+) 사고위험지역 ({len(over_cam):,}건)\n시도별 분포', fontsize=13, fontweight='bold')
ax.set_xlabel('지역 수')
for i, (idx, row) in enumerate(over_cam_sido.iterrows()):
    ax.text(row['지역수'] + 1, i, f"{int(row['지역수'])}건 (평균{row['평균카메라']:.0f}대)", va='center', fontsize=9)
ax.grid(axis='x', alpha=0.3)

# (c) 군집별 카메라 효과 비율
ax = axes[1, 0]
effect_data = df.groupby('군집라벨').agg(
    직접=('단속가능수', 'sum'),
    간접=('간접억제수', 'sum'),
    총=('유형총수', 'sum')
).reindex(label_order)
effect_pct = pd.DataFrame({
    '직접 단속': effect_data['직접'] / effect_data['총'] * 100,
    '간접 억제': effect_data['간접'] / effect_data['총'] * 100,
    '단속 어려움': 100 - (effect_data['직접'] + effect_data['간접']) / effect_data['총'] * 100
})
short_labels = [l.split('(')[0].strip() for l in effect_pct.index]
effect_pct.index = short_labels
effect_pct.plot(kind='barh', stacked=True, ax=ax,
                color=['#e74c3c', '#f39c12', '#95a5a6'], edgecolor='white')
ax.set_title('군집별 사고 유형 구성 (카메라 단속 가능 비율)', fontsize=13, fontweight='bold')
ax.set_xlabel('비율 (%)')
ax.legend(fontsize=9, loc='lower right')
for i, (idx, row) in enumerate(effect_pct.iterrows()):
    total_effect = row['직접 단속'] + row['간접 억제']
    ax.text(total_effect + 1, i, f'{total_effect:.1f}%', va='center', fontsize=11, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# (d) 결론 요약
ax = axes[1, 1]
ax.axis('off')
부족_g = df[df['군집라벨'] == label_order[2]]
위험_g = df[df['군집라벨'] == label_order[3]]
과잉_g = df[df['군집라벨'] == label_order[0]]
사망_g = df[df['군집라벨'] == label_order[1]]
summary = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       단속 카메라 재배치 분석 결론
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 분석 규모
   · 사고위험지역: {len(df):,}건 (개별 지점)
   · 단속 카메라: {len(df_cam):,}대

📋 군집화 결과 (K-Means, k=4)
   🔵 과잉: {len(과잉_g):,}건 (카메라 {과잉_g['반경내카메라수'].mean():.1f}대)
   ⚫ 사망위험: {len(사망_g):,}건
   🔴 부족: {len(부족_g):,}건 (카메라 {부족_g['반경내카메라수'].mean():.1f}대)
   🟡 사고다발: {len(위험_g):,}건 (사고 {위험_g['총사고건수'].mean():.1f}건)

⚠️ 핵심 발견
   · 카메라 없는 사고위험지역: {len(no_cam):,}건
   · 카메라 10대+ 과밀 지역: {len(over_cam):,}건
   · 위험 구역 카메라 단속 관련 비율: {위험_g['카메라효과비율'].mean():.1f}%

💡 재배치 제안
   과잉 구역의 카메라를
   부족·위험 구역으로 이동하면
   과속·신호위반 사고를
   줄일 수 있는 개연성이 높음
"""
ax.text(0.05, 0.95, summary, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', fontfamily='Malgun Gothic',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', edgecolor='#dee2e6', alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, '06_재배치_제안.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → 06_재배치_제안.png 저장")


# ╔═══════════════════════════════════════════════════════════════╗
# ║  PHASE 5-B: 과잉 군집 내 K-Means(k=3) 재군집화                 ║
# ║  목적: 과잉 군집을 과잉/재배치가능/적정 3그룹으로 자동 분류    ║
# ╚═══════════════════════════════════════════════════════════════╝
print("\n[Phase 5-B] 과잉 군집 내 K-Means(k=3) 재군집화")

excess_df = df[df['군집라벨'] == label_order[0]].copy()
excess_df['위험점수'] = excess_df['총사고건수'] + excess_df['총사망자수']*10 + excess_df['총중상자수']*5
excess_df['카메라효율'] = excess_df['위험점수'] / excess_df['반경내카메라수']

print(f"  과잉 군집: {len(excess_df):,}개소 (카메라 1~{excess_df['반경내카메라수'].max()}대)")

# k별 실루엣 탐색
ex_k_range = range(2, 7)
ex_sil_scores, ex_inertias = [], []
X_ex = excess_df[['카메라효율', '카메라효과비율']].values
scaler_ex = StandardScaler()
X_ex_scaled = scaler_ex.fit_transform(X_ex)

for k in ex_k_range:
    km_ex = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    lab_ex = km_ex.fit_predict(X_ex_scaled)
    ex_inertias.append(km_ex.inertia_)
    ex_sil_scores.append(silhouette_score(X_ex_scaled, lab_ex))
    print(f"    k={k}: Silhouette={ex_sil_scores[-1]:.4f}, Inertia={ex_inertias[-1]:.1f}")

best_ex_k_idx = np.argmax(ex_sil_scores)
best_ex_k = list(ex_k_range)[best_ex_k_idx]
ADOPTED_K = 3
print(f"\n  실루엣 최적 k: {best_ex_k} (Silhouette={ex_sil_scores[best_ex_k_idx]:.4f})")
print(f"  채택 k: {ADOPTED_K} (과잉 / 재배치 가능 / 적정)")

# k=3 최종 재군집화
km3 = KMeans(n_clusters=ADOPTED_K, random_state=42, n_init=10, max_iter=300)
excess_df['sub3'] = km3.fit_predict(X_ex_scaled)
sil3 = silhouette_score(X_ex_scaled, excess_df['sub3'])

# 라벨 부여: 효율+효과 합산 오름차순 → 과잉/재배치가능/적정
sub_stats = []
for c in sorted(excess_df['sub3'].unique()):
    sub = excess_df[excess_df['sub3'] == c]
    sub_stats.append({'c': c, 'n': len(sub),
                      'eff': sub['카메라효율'].mean(),
                      'effect': sub['카메라효과비율'].mean(),
                      'risk': sub['위험점수'].mean(),
                      'cam': sub['반경내카메라수'].mean()})
sub_stats = sorted(sub_stats, key=lambda x: x['eff'] + x['effect'])

SUB_LABELS = ['과잉(재배치1순위)', '재배치가능(재배치2순위)', '적정(현행유지)']
sub_lbl_map = {sub_stats[i]['c']: SUB_LABELS[i] for i in range(3)}
excess_df['과잉세분화'] = excess_df['sub3'].map(sub_lbl_map)
df['과잉세분화'] = df.index.map(excess_df['과잉세분화'])

print(f"\n  재군집화 결과 (실루엣: {sil3:.4f}):")
for s in sub_stats:
    lbl = sub_lbl_map[s['c']]
    print(f"    [{lbl}] {s['n']:,}개소 | 효율: {s['eff']:.1f} | 효과비율: {s['effect']:.1f}% | 위험점수: {s['risk']:.1f} | 카메라: {s['cam']:.2f}")

n_supply = sub_stats[0]['n'] + sub_stats[1]['n']
n_proper = sub_stats[2]['n']
supply_cam = excess_df[excess_df['과잉세분화'].isin([SUB_LABELS[0], SUB_LABELS[1]])]['반경내카메라수'].sum()
print(f"\n  >> 재배치 공급원: {n_supply:,}개소 ({SUB_LABELS[0].split('(')[0]} {sub_stats[0]['n']} + {SUB_LABELS[1].split('(')[0]} {sub_stats[1]['n']})")
print(f"  >> 확보 가능 잉여 카메라: {int(supply_cam):,}대")
print(f"  >> 적정 (현행 유지): {n_proper:,}개소 ({n_proper/len(excess_df)*100:.1f}%)")

# ── 실루엣 및 엘보우 시각화 ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('과잉 군집 내 재군집화 검증 (K-Means, k=3)', fontsize=18, fontweight='bold', y=1.05)

# 1. 엘보우 메소드 & 실루엣 스코어 결합
ax1 = axes[0]
color1 = 'tab:green'
ax1.set_xlabel('k (군집 수)', fontsize=13)
ax1.set_ylabel('Inertia (Sum of Squared Distances)', color=color1, fontsize=13)
ax1.plot(list(ex_k_range), ex_inertias, 's-', color=color1, linewidth=2.5, markersize=8, label='Inertia (Elbow)')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
color2 = 'tab:blue'
ax2.set_ylabel('Silhouette Score', color=color2, fontsize=13)
ax2.plot(list(ex_k_range), ex_sil_scores, 'o-', color=color2, linewidth=2.5, markersize=9, label='Silhouette Score')
ax2.tick_params(axis='y', labelcolor=color2)

ax2.axvline(x=ADOPTED_K, color='red', linestyle='--', linewidth=2, label=f'채택 k={ADOPTED_K} (Sil={sil3:.4f})')
for k_val, sil_val in zip(ex_k_range, ex_sil_scores):
    ax2.annotate(f'{sil_val:.4f}', (k_val, sil_val), textcoords='offset points',
                 xytext=(0, 10), ha='center', fontsize=11, fontweight='bold', color=color2)

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right', fontsize=11)
ax1.set_title('Elbow Method & Silhouette Score\n(과잉 군집 내 재군집화 검증)', fontsize=15, fontweight='bold')

# 2. 재군집화 결과 산점도
ax = axes[1]
sub_colors = {SUB_LABELS[0]: '#c0392b', SUB_LABELS[1]: '#e67e22', SUB_LABELS[2]: '#27ae60'}
for lbl, color in sub_colors.items():
    sub = excess_df[excess_df['과잉세분화'] == lbl]
    ax.scatter(sub['카메라효율'], sub['카메라효과비율'],
               s=35, c=color, alpha=0.6, edgecolor='white', linewidth=0.5,
               label=f"{lbl} ({len(sub):,})")
ax.set_title(f'재군집화 결과 산점도\n(k=3, Silhouette={sil3:.4f})', fontsize=15, fontweight='bold')
ax.set_xlabel('카메라 효율 (위험점수 / 카메라수)', fontsize=13)
ax.set_ylabel('카메라 효과비율 (%)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_재군집화_실루엣검증.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> 07_재군집화_실루엣검증.png 저장")

# ── 3그룹 비교 시각화 ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('과잉 군집 세분화: 과잉 / 재배치가능 / 적정', fontsize=18, fontweight='bold', y=0.98)

# (a) 3그룹 평균 지표 비교
ax = axes[0, 0]
metrics_ex = ['카메라효율', '카메라효과비율', '위험점수', '반경내카메라수']
metric_labels = ['카메라 효율', '효과비율(%)', '위험점수', '카메라수']
sub_colors_list = ['#c0392b', '#e67e22', '#27ae60']
x_ex = np.arange(len(metrics_ex))
width_ex = 0.25
for i, (lbl, color) in enumerate(zip(SUB_LABELS, sub_colors_list)):
    sub = excess_df[excess_df['과잉세분화'] == lbl]
    vals = [sub['카메라효율'].mean(), sub['카메라효과비율'].mean(),
            sub['위험점수'].mean(), sub['반경내카메라수'].mean()]
    bars_ex = ax.bar(x_ex + (i - 1) * width_ex, vals, width_ex,
                     label=f"{lbl.split('(')[0]} ({len(sub)})", color=color, alpha=0.85)
ax.set_xticks(x_ex)
ax.set_xticklabels(metric_labels, fontsize=10)
ax.set_title('3그룹 평균 지표 비교', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# (b) 시도별 3그룹 비율
ax = axes[0, 1]
sido_sub = excess_df.groupby(['시도명', '과잉세분화']).size().unstack(fill_value=0)
for col in SUB_LABELS:
    if col not in sido_sub.columns:
        sido_sub[col] = 0
sido_sub = sido_sub[SUB_LABELS]
sido_sub_norm = sido_sub.div(sido_sub.sum(axis=1), axis=0) * 100
sido_sub_norm.plot(kind='barh', stacked=True, ax=ax,
                   color=sub_colors_list, edgecolor='white')
ax.set_title('시도별 과잉 군집 세분화 비율 (%)', fontsize=13, fontweight='bold')
ax.set_xlabel('비율 (%)')
ax.legend(fontsize=8, labels=[l.split('(')[0] for l in SUB_LABELS])

# (c) 재배치 공급원 vs 적정 카메라 수 비교
ax = axes[1, 0]
supply_df = excess_df[excess_df['과잉세분화'].isin([SUB_LABELS[0], SUB_LABELS[1]])]
proper_df = excess_df[excess_df['과잉세분화'] == SUB_LABELS[2]]
for grp, clr, lbl, offset in zip(
        [supply_df, proper_df],
        ['#e74c3c', '#2ecc71'],
        [f'재배치 공급원 ({n_supply}개)', f'적정 ({n_proper}개)'],
        [-0.15, 0.15]):
    cam_dist = grp['반경내카메라수'].value_counts().sort_index()
    ax.bar(cam_dist.index + offset, cam_dist.values, width=0.3,
           label=lbl, color=clr, alpha=0.8, edgecolor='white')
ax.set_title('재배치 공급원 vs 적정 카메라 수 분포', fontsize=13, fontweight='bold')
ax.set_xlabel('반경 100m 내 카메라 수 (대)')
ax.set_ylabel('지점 수')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# (d) 요약 텍스트
ax = axes[1, 1]
ax.axis('off')
summary_ex = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   과잉 군집 K-Means(k=3) 재군집화
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 실루엣 스코어: {sil3:.4f} (양호 ≥ 0.5)

⛔ 과잉 (재배치 1순위)
   {sub_stats[0]['n']:,}개소 ({sub_stats[0]['n']/len(excess_df)*100:.1f}%)
   · 카메라 효율: {sub_stats[0]['eff']:.1f}
   · 단속 효과: {sub_stats[0]['effect']:.1f}%

🔶 재배치 가능 (재배치 2순위)
   {sub_stats[1]['n']:,}개소 ({sub_stats[1]['n']/len(excess_df)*100:.1f}%)
   · 카메라 효율: {sub_stats[1]['eff']:.1f}
   · 단속 효과: {sub_stats[1]['effect']:.1f}%

🟢 적정 (현행 유지)
   {sub_stats[2]['n']:,}개소 ({n_proper/len(excess_df)*100:.1f}%)
   · 카메라 효율: {sub_stats[2]['eff']:.1f}
   · 단속 효과: {sub_stats[2]['effect']:.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
>> 재배치 공급원: {n_supply:,}개소
>> 확보 잉여 카메라: {int(supply_cam):,}대
"""
ax.text(0.05, 0.97, summary_ex, transform=ax.transAxes,
        fontsize=11.5, verticalalignment='top', fontfamily='Malgun Gothic',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#eaf2e3', edgecolor='#27ae60', alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, '04_밀집구역_재군집화.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> 04_밀집구역_재군집화.png 저장")

# ── 2차 군집화 산점도 추가 (Standardized + Jitter + 센트로이드) ─────────────────────────────────────────
print("  -> 13_2차산점도_센트로이드.png 생성 중...")
scaler_2nd = StandardScaler()
scaled_features_2nd = scaler_2nd.fit_transform(excess_df[['카메라효과비율', '카메라효율']])
excess_df_plot = excess_df.copy()
excess_df_plot['효과비율_scaled'] = scaled_features_2nd[:, 0]
excess_df_plot['효율_scaled'] = scaled_features_2nd[:, 1]

np.random.seed(42)
jitter_strength_2nd = 0.2
excess_df_plot['효과비율_jitter'] = excess_df_plot['효과비율_scaled'] + np.random.normal(0, jitter_strength_2nd, size=len(excess_df_plot))
excess_df_plot['효율_jitter'] = excess_df_plot['효율_scaled'] + np.random.normal(0, jitter_strength_2nd, size=len(excess_df_plot))

plt.figure(figsize=(12, 9))
sns.scatterplot(
    x='효과비율_jitter', y='효율_jitter', hue='과잉세분화', data=excess_df_plot,
    palette=['#c0392b', '#e67e22', '#27ae60'], alpha=0.6, s=70, edgecolor=None
)

centroids_2nd = excess_df_plot.groupby('과잉세분화')[['효과비율_scaled', '효율_scaled']].mean().reset_index()
for idx, row in centroids_2nd.iterrows():
    cluster_name = row['과잉세분화']
    plt.scatter(row['효과비율_scaled'], row['효율_scaled'], marker='*', s=1000, c='gold', edgecolor='black', linewidth=1.5, zorder=10)
    plt.text(row['효과비율_scaled'] + 0.1, row['효율_scaled'] + 0.1, f"Centroid\n{cluster_name.split('(')[0]}", 
             fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.3'), zorder=11)

plt.title('2차 군집화 시각화', fontsize=18, fontweight='bold')
plt.xlabel('카메라 효과 비율 (표준화 지수)', fontsize=13)
plt.ylabel('카메라 효율 (표준화 지수)', fontsize=13)
plt.legend(title='2차 군집 라벨', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '13_2차산점도_센트로이드.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  -> 13_2차산점도_센트로이드.png 저장")

# excess 상세 CSV 저장
excess_export = excess_df[['시도명', '사고위험지역명', '반경내카메라수',
                            '총사고건수', '총사망자수', '총중상자수',
                            '위험점수', '카메라효율', '카메라효과비율',
                            '과잉세분화', '위도', '경도']].copy()
excess_export = excess_export.sort_values(['과잉세분화', '카메라효율'])
excess_export.to_csv(os.path.join(OUTPUT_DIR, '결과_밀집구역_세분화데이터.csv'),
                     index=False, encoding='utf-8-sig')
print("  -> 결과_밀집구역_세분화데이터.csv 저장")


# ╔═══════════════════════════════════════════════════════════════╗
# ║  결과 CSV 저장 (결과_전체군집화데이터.csv)                         ║
# ╚═══════════════════════════════════════════════════════════════╝
print("\n[결과 저장]")

df['위험점수'] = df['총사고건수'] + df['총사망자수']*10 + df['총중상자수']*5

df_out = df[['연도코드', '시도명', '시군구코드', '사고위험지역명', '사고위험지역id',
             '총사고건수', '총사망자수', '총중상자수', '총경상자수',
             '사고분석유형명', '반경내카메라수', '반경내과속카메라', '반경내신호카메라',
             '카메라효과비율', 'cluster', '군집라벨', '과잉세분화', '위도', '경도']].copy()

df_out = df.drop(columns=['연도코드', '시군구코드', '사고위험지역id', 'cluster', '반경내과속카메라', '반경내신호카메라', '총경상자수'], errors='ignore')
if '카메라효과비율' in df_out.columns:
    df_out = df_out.drop(columns=['카메라효과비율'])
df_out.to_csv(os.path.join(OUTPUT_DIR, '결과_전체군집화데이터.csv'),

              index=False, encoding='utf-8-sig')
print(f"  → 결과_전체군집화데이터.csv ({len(df_out):,}건)")

no_cam_top = no_cam.nlargest(30, '위험점수')[
    ['연도코드', '시도명', '사고위험지역명', '총사고건수',
     '총사망자수', '총중상자수', '위험점수', '사고분석유형명']
]

summary_df = df.groupby('군집라벨').agg(
    지점수=('총사고건수', 'count'),
    평균카메라수=('반경내카메라수', 'mean'),
    평균사고건수=('총사고건수', 'mean'),
    평균사망자수=('총사망자수', 'mean'),
    평균중상자수=('총중상자수', 'mean'),
    카메라단속관련비율=('카메라효과비율', 'mean'),
    카메라없는비율=('카메라유무', lambda x: f"{(1-x.mean())*100:.1f}%")
).reset_index()

# Add extra columns to summary_df
try:
    summary_df['평균위험점수'] = [df[df['군집라벨'] == label]['위험점수'].mean() for label in label_order]
except:
    pass
cols_to_keep = ['군집라벨', '지점수', '평균카메라수', '평균사고건수', '평균사망자수', '평균중상자수', '평균위험점수']
summary_df_export = summary_df[[c for c in cols_to_keep if c in summary_df.columns]]
summary_df_export.to_csv(os.path.join(OUTPUT_DIR, '결과_군집화요약통계.csv'),

                  index=False, encoding='utf-8-sig')
print(f"  → 결과_군집화요약통계.csv")

type_summary = []
for label in label_order:
    tc = all_types_by_cluster[label]
    total = tc.sum()
    for t, c in tc.items():
        type_summary.append({
            '군집': label.split('(')[0].strip(),
            '사고유형': t, '건수': c,
            '비율(%)': round(c / total * 100, 1),
            '카메라단속': '직접' if t in 단속가능 else ('간접' if t in 간접억제 else '불가')
        })


# ╔═══════════════════════════════════════════════════════════════╗
# ║  PHASE 6: 카메라 설치 실효성 증명                                ║
# ╚═══════════════════════════════════════════════════════════════╝
print("\n[Phase 6] 카메라 설치 실효성 증명")

주요유형 = ['신호위반', '안전거리미확보', '과속', '중앙선침범', 'U턴중', '기타']

def get_type_rates(subset):
    all_t = []
    for _, row in subset.iterrows():
        all_t.extend(row['사고유형목록'])
    if not all_t:
        return {}
    tc = pd.Series(all_t).value_counts(normalize=True) * 100
    return tc.to_dict()

cam_yes = df[df['카메라유무'] == 1]
cam_no_p6 = df[df['카메라유무'] == 0]
rates_yes = get_type_rates(cam_yes)
rates_no  = get_type_rates(cam_no_p6)

print(f"\n  카메라 있음: {len(cam_yes):,}건 | 카메라 없음: {len(cam_no_p6):,}건\n")
print(f"  {'사고 유형':<15} {'카메라 없음':>10} {'카메라 있음':>10} {'차이':>10}  해석")
print(f"  {'─'*65}")
for t in 주요유형:
    r_no  = rates_no.get(t, 0)
    r_yes = rates_yes.get(t, 0)
    diff  = r_yes - r_no
    marker = "🎯" if t in 단속가능 else ("🔶" if t in 간접억제 else "  ")
    interpret = "← 카메라 효과 있음" if diff < -0.5 else ("← 카메라 있어도 증가" if diff > 0.5 else "")
    print(f"  {marker}{t:<13} {r_no:>9.1f}% {r_yes:>9.1f}% {diff:>+9.1f}%  {interpret}")

# 군집별 × 카메라 유무별 평균 사고건수
print(f"\n  군집별 × 카메라 유무별 평균 사고건수:")
label_order_short = ['카메라 설치지점', '사망위험', '부족', '사고다발']
for lbl_short, lbl_full in zip(label_order_short, label_order):
    subset = df[df['군집라벨'] == lbl_full]
    yes = subset[subset['카메라유무'] == 1]
    no  = subset[subset['카메라유무'] == 0]
    if len(no) == 0:
        print(f"  [{lbl_short}] 카메라 없는 지점 없음")
        continue
    print(f"  [{lbl_short}] 카메라 없음({len(no):,}건) → 사고 {no['총사고건수'].mean():.1f}건")
    print(f"          카메라 있음({len(yes):,}건) → 사고 {yes['총사고건수'].mean():.1f}건")

# 부족·위험 구역 카메라 미설치 지점 설치 제안
target_p6 = df[(df['군집라벨'].isin([label_order[2], label_order[3]])) & (df['카메라유무'] == 0)]
all_types_target = []
for tl in target_p6['사고유형목록']:
    all_types_target.extend(tl)
tc_target = pd.Series(all_types_target).value_counts()
total_target = tc_target.sum()
카메라관련_합 = sum(tc_target.get(t, 0) for t in list(단속가능) + list(간접억제))
카메라관련_비율 = 카메라관련_합 / total_target * 100
신호_비율 = tc_target.get('신호위반', 0) / total_target * 100
과속관련_비율 = sum(tc_target.get(t, 0) for t in ['과속', '안전거리미확보', '중앙선침범']) / total_target * 100

# 시각화
fig = plt.figure(figsize=(22, 20))
fig.suptitle('카메라 설치 실효성 최종 분석', fontsize=22, fontweight='bold', y=0.98)

# (1) 카메라 유무별 사고 유형 비율 비교
ax1 = fig.add_subplot(3, 2, (1, 2))
x = np.arange(len(주요유형))
width = 0.35
vals_no_p6  = [rates_no.get(t, 0) for t in 주요유형]
vals_yes_p6 = [rates_yes.get(t, 0) for t in 주요유형]
bars1 = ax1.bar(x - width/2, vals_no_p6,  width, label=f'카메라 없음 ({len(cam_no_p6):,}건)', color='#e74c3c', alpha=0.85)
bars2 = ax1.bar(x + width/2, vals_yes_p6, width, label=f'카메라 있음 ({len(cam_yes):,}건)', color='#3498db', alpha=0.85)
for i, (v_no, v_yes) in enumerate(zip(vals_no_p6, vals_yes_p6)):
    diff = v_yes - v_no
    ax1.annotate(f'{diff:+.1f}%p', xy=(i, max(v_no, v_yes) + 0.5),
                fontsize=11, fontweight='bold', ha='center',
                color='#27ae60' if diff < 0 else '#e74c3c')
for i, t in enumerate(주요유형):
    if t in 단속가능:
        ax1.axvspan(i - 0.45, i + 0.45, alpha=0.08, color='#e74c3c')
    elif t in 간접억제:
        ax1.axvspan(i - 0.45, i + 0.45, alpha=0.08, color='#f39c12')
ax1.set_xticks(x)
ax1.set_xticklabels(주요유형, fontsize=12)
ax1.set_title('카메라 유무별 사고 유형 비율 비교\n(카메라 설치 효과 검증)', fontsize=16, fontweight='bold')
ax1.set_ylabel('비율 (%)', fontsize=12)
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(axis='y', alpha=0.3)
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{bar.get_height():.1f}%', ha='center', fontsize=9)
for bar in bars2:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{bar.get_height():.1f}%', ha='center', fontsize=9)

# (2) 군집별 × 카메라 유무별 평균 사고건수
ax2 = fig.add_subplot(3, 2, 3)
cluster_cam_stats = []
for lbl_full in label_order:
    subset = df[df['군집라벨'] == lbl_full]
    yes = subset[subset['카메라유무'] == 1]
    no  = subset[subset['카메라유무'] == 0]
    cluster_cam_stats.append({
        '군집': lbl_full.split('(')[0].strip(),
        '카메라없음_사고': no['총사고건수'].mean() if len(no) > 0 else 0,
        '카메라있음_사고': yes['총사고건수'].mean() if len(yes) > 0 else 0,
        '카메라없음_N': len(no), '카메라있음_N': len(yes)
    })
ccs = pd.DataFrame(cluster_cam_stats)
x_ccs = np.arange(len(label_order))
ax2.bar(x_ccs - 0.175, ccs['카메라없음_사고'], 0.35, label='카메라 없음', color='#e74c3c', alpha=0.85)
ax2.bar(x_ccs + 0.175, ccs['카메라있음_사고'], 0.35, label='카메라 있음', color='#3498db', alpha=0.85)
ax2.set_xticks(x_ccs)
ax2.set_xticklabels([l.split('(')[0].strip() for l in label_order], fontsize=11)
ax2.set_title('군집별 × 카메라 유무별 평균 사고건수', fontsize=14, fontweight='bold')
ax2.set_ylabel('평균 사고건수')
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)
for i, row in ccs.iterrows():
    if row['카메라없음_사고'] > 0:
        ax2.text(i - 0.175, row['카메라없음_사고'] + 0.3,
                 f"{row['카메라없음_사고']:.1f}\n(n={int(row['카메라없음_N']):,})", ha='center', fontsize=8, fontweight='bold')
    ax2.text(i + 0.175, row['카메라있음_사고'] + 0.3,
             f"{row['카메라있음_사고']:.1f}\n(n={int(row['카메라있음_N']):,})", ha='center', fontsize=8, fontweight='bold')

# (3) 부족+위험 카메라 미설치 사고 유형
ax3 = fig.add_subplot(3, 2, 4)
colors_bar = ['#e74c3c' if t in 단속가능 else ('#f39c12' if t in 간접억제 else '#95a5a6') for t in tc_target.index]
bars_h = ax3.barh(range(len(tc_target)), tc_target.values, color=colors_bar, edgecolor='white', height=0.6)
ax3.set_yticks(range(len(tc_target)))
ax3.set_yticklabels(tc_target.index, fontsize=11)
ax3.invert_yaxis()
ax3.set_title(f'부족+사고다발 구역 카메라 미설치 지점\n사고 유형 ({len(target_p6):,}건)', fontsize=14, fontweight='bold')
ax3.set_xlabel('빈도')
for bar, val in zip(bars_h, tc_target.values):
    pct = val / total_target * 100
    ax3.text(bar.get_width() + max(tc_target.values)*0.01, bar.get_y() + bar.get_height()/2,
             f'{val:,} ({pct:.1f}%)', va='center', fontsize=10)
ax3.legend(handles=[
    Patch(facecolor='#e74c3c', label='직접 단속 가능'),
    Patch(facecolor='#f39c12', label='간접 억제 가능'),
    Patch(facecolor='#95a5a6', label='카메라 외 대책')
], fontsize=9, loc='lower right')
ax3.grid(axis='x', alpha=0.3)

# (4) 시도별 카메라 설치 유형 제안
ax4 = fig.add_subplot(3, 2, 5)
install_data = []
for 시도 in df['시도명'].unique():
    subset_sido = df[(df['군집라벨'].isin([label_order[2], label_order[3]])) &
                    (df['카메라유무'] == 0) & (df['시도명'] == 시도)]
    if len(subset_sido) < 5:
        continue
    all_t = []
    for tl in subset_sido['사고유형목록']:
        all_t.extend(tl)
    if not all_t:
        continue
    tc_sido = pd.Series(all_t)
    total_t = len(tc_sido)
    install_data.append({
        '시도': 시도, '미설치지점수': len(subset_sido),
        '신호위반': (tc_sido == '신호위반').sum() / total_t * 100,
        '과속관련': ((tc_sido == '과속') | (tc_sido == '안전거리미확보') | (tc_sido == '중앙선침범')).sum() / total_t * 100,
    })
if install_data:
    idf = pd.DataFrame(install_data).sort_values('미설치지점수', ascending=True)
    x_idf = np.arange(len(idf))
    ax4.barh(x_idf - 0.175, idf['신호위반'], 0.35, label='신호위반 → 신호위반 카메라', color='#e74c3c', alpha=0.85)
    ax4.barh(x_idf + 0.175, idf['과속관련'], 0.35, label='과속 관련 → 과속 카메라', color='#f39c12', alpha=0.85)
    ax4.set_yticks(x_idf)
    ax4.set_yticklabels([f"{r['시도']} ({r['미설치지점수']}곳)" for _, r in idf.iterrows()], fontsize=10)
    ax4.set_title('시도별 카메라 설치 유형 제안\n(부족·위험 구역 카메라 미설치 지점)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('사고 비율 (%)')
    ax4.legend(fontsize=9)
    ax4.grid(axis='x', alpha=0.3)

# (5) 결론
ax5 = fig.add_subplot(3, 2, 6)
ax5.axis('off')
conclusion_p6 = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     기존 카메라 배치의 실효성 한계 및 재배치 당위성
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 근거 1: 기존 배치의 한계점 노출
   카메라가 설치된 구역임에도 불구하고 '신호위반' 및
   '안전거리미확보' 사고 비중이 미설치 구역보다 오히려 높음.
   → 현재 설치된 위치가 사고 억제에 비효율적임을 시사.

📊 근거 2: 평균 사고건수 교차 검증
   사고다발/사망위험 구역 내에서도 카메라가 있는 지점의
   평균 사고건수가 없는 지점보다 높거나 비슷하게 나타남.
   → 단순 설치보다 '최적의 위치 선정'이 훨씬 중요함.

✅ 최종 결론
   단순히 단속 장비를 늘리는 것이 능사가 아니며,
   단속 효율이 떨어지는 '과잉 구역'의 잉여 자원을
   사고 억제 효과를 극대화할 수 있는 사각지대로
   전면 재배치(Relocation)하는 것이 시급함.
"""
ax5.text(0.05, 0.95, conclusion_p6, transform=ax5.transAxes,
         fontsize=11.5, verticalalignment='top', fontfamily='Malgun Gothic',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#fff5f5', edgecolor='#e74c3c', alpha=0.9))

# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig(os.path.join(OUTPUT_DIR, '08_실효성_증명.png'), dpi=150, bbox_inches='tight')
plt.close()
# print("  → 08_실효성_증명.png 저장 (폐기)")

# 카메라 설치 권장 CSV
def recommend_camera(types_list):
    has_signal = '신호위반' in types_list
    has_speed = any(t in types_list for t in ['과속', '안전거리미확보', '중앙선침범'])
    if has_signal and has_speed:
        return '과속+신호위반 겸용'
    elif has_signal:
        return '신호위반 카메라'
    elif has_speed:
        return '과속 카메라'
    else:
        return '카메라 외 대책 필요'

target_p6 = target_p6.copy()
target_p6['권장카메라유형'] = target_p6['사고유형목록'].apply(recommend_camera)
target_export = target_p6.nlargest(50, '위험점수')[
    ['시도명', '사고위험지역명', '총사고건수', '총사망자수', '총중상자수',
     '위험점수', '사고분석유형명', '군집라벨', '권장카메라유형']
]


# ╔═══════════════════════════════════════════════════════════════╗
# ║  PHASE 7: 군집별 심각도 분석                                    ║
# ╚═══════════════════════════════════════════════════════════════╝
print("\n[Phase 7] 군집별 심각도 분석")

df['사망률'] = df['총사망자수'] / df['총사고건수'].replace(0, np.nan)
df['중상률'] = df['총중상자수'] / df['총사고건수'].replace(0, np.nan)
df['경상률'] = df['총경상자수'] / df['총사고건수'].replace(0, np.nan)
df['총피해자수'] = df['총사망자수'] + df['총중상자수'] + df['총경상자수'] + df['총부상신고자수']
df['피해자비율'] = df['총피해자수'] / df['총사고건수'].replace(0, np.nan)

severity = df.groupby('군집라벨').agg(
    지점수=('총사고건수', 'count'),
    평균카메라수=('반경내카메라수', 'mean'),
    총사고=('총사고건수', 'sum'),
    총사망=('총사망자수', 'sum'),
    총중상=('총중상자수', 'sum'),
    총경상=('총경상자수', 'sum'),
    평균사고건수=('총사고건수', 'mean'),
    평균사망자수=('총사망자수', 'mean'),
    평균중상자수=('총중상자수', 'mean'),
    평균경상자수=('총경상자수', 'mean'),
    평균사망률=('사망률', 'mean'),
    평균중상률=('중상률', 'mean'),
    평균위험점수=('위험점수', 'mean'),
    평균피해자비율=('피해자비율', 'mean'),
).reindex(label_order)

emoji_map = {label_order[0]: '🔵', label_order[1]: '⚫', label_order[2]: '🔴', label_order[3]: '🟡'}
for label in label_order:
    row = severity.loc[label]
    e = emoji_map[label]
    print(f"\n{e} [{label}] ({int(row['지점수']):,}건 | 카메라 평균 {row['평균카메라수']:.1f}대)")
    print(f"   총 사고: {int(row['총사고']):,}건 | 평균 {row['평균사고건수']:.1f}건/지점")
    print(f"   총 사망: {int(row['총사망']):,}명 | 사망률 {row['평균사망률']*100:.2f}%")
    print(f"   총 중상: {int(row['총중상']):,}명 | 중상률 {row['평균중상률']*100:.1f}%")
    print(f"   위험점수: 평균 {row['평균위험점수']:.1f}")

fig, axes = plt.subplots(2, 3, figsize=(24, 14))
fig.suptitle('군집별 심각도 지표 분석', fontsize=22, fontweight='bold', y=0.98)

short_label_order = [l.split('(')[0].strip() for l in label_order]

# (1) 평균 사고·사망·중상
ax = axes[0, 0]
x_sv = np.arange(len(label_order))
vals_acc   = [severity.loc[l, '평균사고건수'] for l in label_order]
vals_death = [severity.loc[l, '평균사망자수'] * 100 for l in label_order]
vals_inj   = [severity.loc[l, '평균중상자수'] for l in label_order]
ax.bar(x_sv - 0.25, vals_acc,   0.25, label='평균 사고건수', color='#e74c3c', alpha=0.85)
ax.bar(x_sv,        vals_inj,   0.25, label='평균 중상자수', color='#f39c12', alpha=0.85)
ax.bar(x_sv + 0.25, vals_death, 0.25, label='평균 사망자수 (×100)', color='#2c3e50', alpha=0.85)
ax.set_xticks(x_sv)
ax.set_xticklabels(short_label_order, fontsize=12)
ax.set_title('군집별 평균 사고·피해 규모', fontsize=14, fontweight='bold')
ax.set_ylabel('건수 / 명')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# (2) 사망률 & 중상률
ax = axes[0, 1]
death_rates = [severity.loc[l, '평균사망률'] * 100 for l in label_order]
inj_rates   = [severity.loc[l, '평균중상률'] * 100 for l in label_order]
ax.bar(x_sv - 0.175, death_rates, 0.35, label='사망률 (%)', color='#2c3e50', alpha=0.85)
ax.bar(x_sv + 0.175, inj_rates,   0.35, label='중상률 (%)', color='#e67e22', alpha=0.85)
ax.set_xticks(x_sv)
ax.set_xticklabels(short_label_order, fontsize=12)
ax.set_title('군집별 사망률 & 중상률', fontsize=14, fontweight='bold')
ax.set_ylabel('비율 (%)')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
for bars_sv in [
    ax.containers[0] if hasattr(ax, 'containers') else []
]:
    pass  # 값 레이블은 생략

# (3) 위험점수 분포 박스플롯
ax = axes[0, 2]
bp_data = [df[df['군집라벨'] == l]['위험점수'] for l in label_order]
bp = ax.boxplot(bp_data, labels=short_label_order, patch_artist=True, showfliers=False)
for patch, label in zip(bp['boxes'], label_order):
    patch.set_facecolor(cluster_colors[label])
    patch.set_alpha(0.6)
ax.set_title('군집별 위험점수 분포\n(사고×1 + 사망×10 + 중상×5)', fontsize=14, fontweight='bold')
ax.set_ylabel('위험점수')
ax.grid(axis='y', alpha=0.3)
for i, l in enumerate(label_order):
    median = df[df['군집라벨'] == l]['위험점수'].median()
    ax.text(i + 1, median + 2, f'중앙값\n{median:.0f}', ha='center', fontsize=9, fontweight='bold')

# (4) 총 피해자 누적
ax = axes[1, 0]
damage = pd.DataFrame({
    '사망': [severity.loc[l, '총사망'] for l in label_order],
    '중상': [severity.loc[l, '총중상'] for l in label_order],
    '경상': [severity.loc[l, '총경상'] for l in label_order],
}, index=short_label_order)
damage.plot(kind='barh', stacked=True, ax=ax,
            color=['#2c3e50', '#e67e22', '#f1c40f'], edgecolor='white')
ax.set_title('군집별 총 피해자 수 (누적)', fontsize=14, fontweight='bold')
ax.set_xlabel('피해자 수 (명)')
ax.legend(fontsize=10)
ax.grid(axis='x', alpha=0.3)
for i, l in enumerate(short_label_order):
    total = damage.loc[l].sum()
    ax.text(total + 50, i, f'{int(total):,}명', va='center', fontsize=10, fontweight='bold')

# (5) 카메라 유무 × 심각도
ax = axes[1, 1]
cam_severity = df.groupby('카메라유무').agg(
    평균사고=('총사고건수', 'mean'),
    평균중상=('총중상자수', 'mean'),
    평균위험점수=('위험점수', 'mean'),
)
vals_no_sv  = [cam_severity.loc[0, '평균사고'], cam_severity.loc[0, '평균중상'], cam_severity.loc[0, '평균위험점수']/3]
vals_yes_sv = [cam_severity.loc[1, '평균사고'], cam_severity.loc[1, '평균중상'], cam_severity.loc[1, '평균위험점수']/3]
n_no_sv  = (df['카메라유무'] == 0).sum()
n_yes_sv = (df['카메라유무'] == 1).sum()
x_sv2 = np.arange(3)
ax.bar(x_sv2 - 0.175, vals_no_sv,  0.35, label=f'카메라 없음 ({n_no_sv:,}건)', color='#e74c3c', alpha=0.85)
ax.bar(x_sv2 + 0.175, vals_yes_sv, 0.35, label=f'카메라 있음 ({n_yes_sv:,}건)', color='#3498db', alpha=0.85)
ax.set_xticks(x_sv2)
ax.set_xticklabels(['평균 사고건수', '평균 중상자수', '평균 위험점수(÷3)'], fontsize=10)
ax.set_title('카메라 유무별 심각도 비교', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
for x_val, val_no, val_yes in zip(x_sv2, [cam_severity.loc[0, '평균사고'], cam_severity.loc[0, '평균중상'], cam_severity.loc[0, '평균위험점수']],
                                           [cam_severity.loc[1, '평균사고'], cam_severity.loc[1, '평균중상'], cam_severity.loc[1, '평균위험점수']]):
    ax.text(x_val - 0.175, (vals_no_sv[x_val] if x_val < 2 else cam_severity.loc[0,'평균위험점수']/3) + 0.2,
            f'{val_no:.1f}', ha='center', fontsize=9, fontweight='bold')
    ax.text(x_val + 0.175, (vals_yes_sv[x_val] if x_val < 2 else cam_severity.loc[1,'평균위험점수']/3) + 0.2,
            f'{val_yes:.1f}', ha='center', fontsize=9, fontweight='bold')

# (6) 결론 텍스트
ax = axes[1, 2]
ax.axis('off')
위험_row = severity.loc[label_order[3]]
부족_row = severity.loc[label_order[2]]
과잉_row = severity.loc[label_order[0]]
conclusion_p7 = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     군집별 심각도 핵심 요약
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 사고 규모
  과잉: 평균 {과잉_row['평균사고건수']:.1f}건/지점
  부족: 평균 {부족_row['평균사고건수']:.1f}건/지점
  사고다발: 평균 {위험_row['평균사고건수']:.1f}건/지점
  → 사고다발은 과잉의 {위험_row['평균사고건수']/과잉_row['평균사고건수']:.1f}배

📊 사망률
  과잉: {과잉_row['평균사망률']*100:.3f}%
  사망위험: {severity.loc[label_order[1],'평균사망률']*100:.3f}%
  부족: {부족_row['평균사망률']*100:.3f}%
  사고다발: {위험_row['평균사망률']*100:.3f}%

📊 위험점수 (사고+사망×10+중상×5)
  과잉: 평균 {과잉_row['평균위험점수']:.1f}
  부족: 평균 {부족_row['평균위험점수']:.1f}
  사고다발: 평균 {위험_row['평균위험점수']:.1f}
"""
ax.text(0.02, 0.98, conclusion_p7, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', fontfamily='Malgun Gothic',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#fdf2e9', edgecolor='#e67e22', alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, '09_군집별_심각도.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → 09_군집별_심각도.png 저장")

severity_out = severity.reset_index()


# ╔═══════════════════════════════════════════════════════════════╗
# ║  PHASE 8: 재배치 우선순위 도출                                   ║
# ╚═══════════════════════════════════════════════════════════════╝
print("\n[Phase 8] 재배치 우선순위 도출")

# 재배치 공급원: Phase 5-B 결과 재활용 (no re-clustering)
과잉_supply = excess_df[excess_df['과잉세분화'].isin([SUB_LABELS[0], SUB_LABELS[1]])]
공급원_카메라수 = int(과잉_supply['반경내카메라수'].sum())

# 재배치 대상: 카메라 0대인 곳
no_cam_p8 = df[df['반경내카메라수'] == 0].copy()

# 1순위: 사망위험 + 사고다발
p1 = no_cam_p8[no_cam_p8['군집라벨'].str.contains('사망위험|사고다발', na=False)].copy()

# 부족 군집 세분화
부족_all = no_cam_p8[no_cam_p8['군집라벨'].str.contains('부족', na=False)].sort_values('위험점수', ascending=False)
top_10_idx = int(len(부족_all) * 0.1)

p2 = 부족_all.iloc[:top_10_idx].copy()   # 2순위: 상위 10%
p3 = 부족_all.iloc[top_10_idx:].copy()  # 3순위: 나머지

p1['우선순위'] = '1순위 (사망/사고다발)'
p2['우선순위'] = '2순위 (일반부족 상위10%)'
p3['우선순위'] = '3순위 (일반부족 나머지)'

print(f"\n  [재배치 자원 - Phase 5-B 재활용]")
print(f"  공급원 (과잉+재배치가능): {len(과잉_supply):,}곳 (잉여 카메라 {공급원_카메라수:,}대)")
print(f"\n  [재배치 우선순위]")
print(f"  1순위 (사망위험+사고다발): {len(p1):,}곳")
print(f"  2순위 (일반 부족 상위 10%): {len(p2):,}곳")
print(f"  3순위 (일반 부족 하위 90%): {len(p3):,}곳")

final_targets = pd.concat([p1, p2, p3]).sort_values(['우선순위', '위험점수'], ascending=[True, False])

export_cols = ['우선순위', '시도명', '사고위험지역명', '군집라벨', '위험점수',
               '총사고건수', '총사망자수', '총중상자수', '사고분석유형명']
final_targets[export_cols].to_csv(
    os.path.join(OUTPUT_DIR, '결과_최종재배치대상.csv'),
    index=False, encoding='utf-8-sig')
print("  → 결과_최종재배치대상.csv 저장")

# 시각화
fig = plt.figure(figsize=(20, 14))
fig.suptitle('단속 카메라 재배치 3단계 전략 및 우선순위 세분화', fontsize=22, fontweight='bold', y=0.98)

# (1) 도넛 차트
ax1 = fig.add_subplot(2, 2, 1)
sizes_p8 = [len(p1), len(p2), len(p3)]
labels_p8 = [f'1순위 타겟\n(사망/사고다발)\n{len(p1):,}곳',
             f'2순위 타겟\n(부족 상위 10%)\n{len(p2):,}곳',
             f'3순위 후순위\n(부족 90%)\n{len(p3):,}곳']
wedges, texts, autotexts = ax1.pie(sizes_p8, explode=(0.05, 0.05, 0),
                                    labels=labels_p8, colors=['#e74c3c', '#f39c12', '#bdc3c7'],
                                    autopct='%1.1f%%', startangle=140, pctdistance=0.8)
plt.setp(autotexts, size=11, weight="bold", color="white")
plt.setp(texts, size=12, weight="bold")
centre_circle = plt.Circle((0, 0), 0.55, fc='white')
ax1.add_patch(centre_circle)
ax1.text(0, 0, f"카메라 없는\n위험지역\n총 {len(final_targets):,}곳",
         ha='center', va='center', fontsize=14, fontweight='bold')
ax1.set_title('카메라 미설치 위험지역 타겟 세분화', fontsize=15, fontweight='bold')

# (2) 우선순위별 평균 위험도
ax2 = fig.add_subplot(2, 2, 2)
priorities_p8 = [p1, p2, p3]
names_p8 = ['1순위 (사망+사고다발)', '2순위 (부족 상위 10%)', '3순위 (부족 하위 90%)']
avg_risk_p8 = [p['위험점수'].mean() for p in priorities_p8]
avg_acc_p8  = [p['총사고건수'].mean() for p in priorities_p8]
avg_inj_p8  = [p['총중상자수'].mean() for p in priorities_p8]
x_p8 = np.arange(len(names_p8))
bars_p8_1 = ax2.bar(x_p8 - 0.25, avg_risk_p8, 0.25, label='평균 위험점수', color='#2c3e50')
bars_p8_2 = ax2.bar(x_p8,         avg_acc_p8,  0.25, label='평균 사고건수', color='#e74c3c')
bars_p8_3 = ax2.bar(x_p8 + 0.25, avg_inj_p8,  0.25, label='평균 중상자수', color='#f39c12')
ax2.set_xticks(x_p8)
ax2.set_xticklabels(names_p8, fontsize=12, fontweight='bold')
ax2.set_title('우선순위 그룹별 평균 피해 규모 비교', fontsize=15, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3)
for bars_grp in [bars_p8_1, bars_p8_2, bars_p8_3]:
    for bar in bars_grp:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# (3) 시도별 1·2순위 분포
ax3 = fig.add_subplot(2, 1, 2)
sido_p1 = p1['시도명'].value_counts()
sido_p2 = p2['시도명'].value_counts()
sido_df = pd.DataFrame({'1순위 타겟': sido_p1, '2순위 타겟': sido_p2}).fillna(0)
sido_df['총합'] = sido_df['1순위 타겟'] + sido_df['2순위 타겟']
sido_df = sido_df.sort_values('총합', ascending=False).drop('총합', axis=1)
sido_df.plot(kind='bar', stacked=True, ax=ax3, color=['#e74c3c', '#f39c12'], edgecolor='white')
ax3.set_title('시도별 카메라 우선 설치 대상지 (1·2순위) 분포', fontsize=15, fontweight='bold')
ax3.set_ylabel('대상 지점 수')
ax3.tick_params(axis='x', rotation=0)
ax3.legend(fontsize=11)
ax3.grid(axis='y', alpha=0.3)
for i, (idx, row) in enumerate(sido_df.iterrows()):
    total = row['1순위 타겟'] + row['2순위 타겟']
    if total > 0:
        ax3.text(i, total + 5, f'{int(total)}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, '10_재배치_우선순위.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → 10_재배치_우선순위.png 저장")


# ╔═══════════════════════════════════════════════════════════════╗
# ║  PHASE 9: 우선순위별 카메라 예방 효과 시각화                      ║
# ╚═══════════════════════════════════════════════════════════════╝
print("\n[Phase 9] 우선순위별 카메라 예방 효과 시각화")

camera_preventable = ['신호위반', '안전거리미확보', '과속', '중앙선침범']
priorities_names = ['1순위 (사망/사고다발)', '2순위 (일반부족 상위10%)', '3순위 (일반부족 나머지)']
titles_p9 = ['[1순위 타겟] 구역 상세 사고 유형', '[2순위 타겟] 구역 상세 사고 유형', '[3순위 후순위] 구역 상세 사고 유형']

records_p9 = []
for p_df, p_name in zip([p1, p2, p3], priorities_names):
    for _, row in p_df.iterrows():
        if pd.isna(row['사고분석유형명']):
            continue
        types = [t.strip() for t in str(row['사고분석유형명']).split('/')]
        for t in types:
            records_p9.append({'우선순위': p_name, '사고유형': t})

df_types_p9 = pd.DataFrame(records_p9)
type_counts_p9 = df_types_p9.groupby(['우선순위', '사고유형']).size().unstack(fill_value=0)
type_ratios_p9 = type_counts_p9.div(type_counts_p9.sum(axis=1), axis=0) * 100

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('카메라 재배치 실효성 검증: 타겟 구역별 사고 예방 가능성 비교',
             fontsize=22, fontweight='bold', y=0.96)

prev_patch   = mpatches.Patch(color='#3498db', label='카메라로 예방/단속 가능')
unprev_patch = mpatches.Patch(color='#95a5a6', label='카메라로 예방 어려움')

for i, (ax, p_name, title) in enumerate(zip([axes[0, 0], axes[0, 1], axes[1, 0]], priorities_names, titles_p9)):
    if p_name not in type_ratios_p9.index:
        ax.axis('off')
        continue
    p_data = type_ratios_p9.loc[p_name].sort_values(ascending=True)
    colors_p9 = ['#3498db' if x in camera_preventable else '#95a5a6' for x in p_data.index]
    ax.barh(p_data.index, p_data.values, color=colors_p9, edgecolor='white')
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xlabel('사고 발생 비율 (%)')
    ax.set_xlim(0, 55)
    for j, v in enumerate(p_data.values):
        ax.text(v + 1.0, j, f"{v:.1f}%", va='center', fontweight='bold', fontsize=11)
    if i == 0:
        ax.legend(handles=[prev_patch, unprev_patch], loc='lower right', fontsize=11)

# (4) 예방 가능 비율 종합 비교
ax4 = axes[1, 1]
prev_ratios_p9 = []
for p_name in priorities_names:
    if p_name in type_ratios_p9.index:
        ratio = type_ratios_p9.loc[p_name, type_ratios_p9.columns.intersection(camera_preventable)].sum()
    else:
        ratio = 0
    prev_ratios_p9.append(ratio)
x_p9 = np.arange(len(priorities_names))
bars_p9 = ax4.bar(x_p9, prev_ratios_p9, color='#e74c3c', width=0.5, edgecolor='white')
ax4.set_xticks(x_p9)
ax4.set_xticklabels([l.replace(" ", "\n") for l in priorities_names], fontsize=13, fontweight='bold')
ax4.set_title('우선순위 그룹별 [카메라 예방 가능 사고] 총합 비교', fontsize=16, fontweight='bold')
ax4.set_ylabel('카메라 단속 가능 사고 비율 (%)')
ax4.set_ylim(0, 55)
for bar in bars_p9:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(OUTPUT_DIR, '11_우선순위별_실효성.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → 11_우선순위별_실효성.png 저장")


# ╔═══════════════════════════════════════════════════════════════╗
# ║  최종 요약                                                      ║
# ╚═══════════════════════════════════════════════════════════════╝
print("\n" + "=" * 70)
print("  PeakGuard-AI  통합 분석 완료!")
print("=" * 70)
print(f"\n  📁 출력 디렉토리: {os.path.abspath(OUTPUT_DIR)}")
print(f"\n  📊 생성된 차트 (10종):")
for png in ['01_EDA_기초현황', '02_EDA_상관관계', '03_군집화결과',
            '04_밀집구역_재군집화', '05_군집별_사고유형', '06_재배치_제안',
            '07_재군집화_실루엣검증', '09_군집별_심각도',
            '10_재배치_우선순위', '11_우선순위별_실효성']:
    print(f"    · {png}.png")

print(f"\n  📄 생성된 데이터 (7종):")
for csv_name in ['결과_전체군집화데이터', '결과_군집화요약통계', '',
                  '', '결과_밀집구역_세분화데이터',
                  '', '',
                  '결과_최종재배치대상']:
    print(f"    · {csv_name}.csv")

print(f"\n  📌 핵심 수치:")
for label in label_order:
    e = emoji_map[label]
    print(f"    {e} {label}: {len(df[df['군집라벨']==label]):,}건")
print(f"\n  ⛔ 과잉: {sub_stats[0]['n']:,}개소")
print(f"  🔶 재배치 가능: {sub_stats[1]['n']:,}개소")
print(f"  🟢 적정: {sub_stats[2]['n']:,}개소")
print(f"  >> 재배치 공급원 합계: {n_supply:,}개소 (잉여 카메라 {int(supply_cam):,}대)")
print(f"  >> 실루엣 스코어 (k=3): {sil3:.4f}")
print()
