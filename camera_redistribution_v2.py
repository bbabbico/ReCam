# -*- coding: utf-8 -*-
"""
=============================================================================
  단속 카메라 재배치 근거 분석 v2 - 통합본
  (EDA + 군집화 + 사고유형 + 재배치 제안 - 모두 지점 단위)
=============================================================================
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
print("  단속 카메라 재배치 분석 v2 통합본 (지점 단위)")
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

# ── 공간 매칭 ──────────────────────────────────────────────────
RADIUS = 100
risk_coords = df_risk[['중심점utmkx좌표', '중심점utmky좌표']].dropna().values
cam_coords = df_cam[['utmk_x', 'utmk_y']].values
tree_cam = cKDTree(cam_coords)

cam_counts = []
cam_overspeed = []
cam_signal = []

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

# ── 2-1. 사고위험지역 기본 분포 ────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('사고위험지역 EDA (지점 단위, n={:,})'.format(len(df)),
             fontsize=20, fontweight='bold', y=0.98)

# (a) 사고건수 분포 (히스토그램)
ax = axes[0, 0]
ax.hist(df['총사고건수'], bins=50, color='#e74c3c', alpha=0.7, edgecolor='white')
ax.axvline(df['총사고건수'].mean(), color='black', linestyle='--', linewidth=2,
           label=f'평균: {df["총사고건수"].mean():.1f}건')
ax.axvline(df['총사고건수'].median(), color='blue', linestyle='--', linewidth=2,
           label=f'중앙값: {df["총사고건수"].median():.1f}건')
ax.set_title('지점별 총 사고건수 분포', fontsize=14, fontweight='bold')
ax.set_xlabel('총 사고건수')
ax.set_ylabel('빈도 (지점 수)')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# (b) 반경 100m 내 카메라 수 분포
ax = axes[0, 1]
cam_hist = df['반경내카메라수'].value_counts().sort_index()
colors = ['#e74c3c' if x == 0 else '#3498db' for x in cam_hist.index]
ax.bar(cam_hist.index, cam_hist.values, color=colors, edgecolor='white')
ax.set_title('지점별 반경 100m 내 카메라 수 분포', fontsize=14, fontweight='bold')
ax.set_xlabel('카메라 수')
ax.set_ylabel('빈도 (지점 수)')
ax.annotate(f'카메라 0대:\n{cam_hist.get(0, 0):,}건 ({cam_hist.get(0,0)/len(df)*100:.1f}%)',
            xy=(0, cam_hist.get(0, 0)), xytext=(3, cam_hist.get(0, 0)*0.8),
            fontsize=11, fontweight='bold', color='#e74c3c',
            arrowprops=dict(arrowstyle='->', color='#e74c3c'))
ax.grid(axis='y', alpha=0.3)

# (c) 사고 심각도 분포 (박스플롯)
ax = axes[1, 0]
bp_data = [df['총사고건수'], df['총사망자수'], df['총중상자수'], df['총경상자수']]
bp_labels = ['사고건수', '사망자수', '중상자수', '경상자수']
bp_colors = ['#e74c3c', '#2c3e50', '#f39c12', '#3498db']
bp = ax.boxplot(bp_data, labels=bp_labels, patch_artist=True, showfliers=False)
for patch, color in zip(bp['boxes'], bp_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_title('지점별 사고 심각도 지표 분포', fontsize=14, fontweight='bold')
ax.set_ylabel('건수 / 명')
ax.grid(axis='y', alpha=0.3)

# (d) 사고 유형 빈도 (전체)
ax = axes[1, 1]
all_types = []
for types_list in df['사고유형목록']:
    all_types.extend(types_list)
type_counts = pd.Series(all_types).value_counts()

단속가능 = {'과속', '신호위반'}
간접억제 = {'안전거리미확보', '중앙선침범'}
colors_type = []
for t in type_counts.index:
    if t in 단속가능:
        colors_type.append('#e74c3c')
    elif t in 간접억제:
        colors_type.append('#f39c12')
    else:
        colors_type.append('#95a5a6')

bars = ax.barh(range(len(type_counts)), type_counts.values, color=colors_type, edgecolor='white')
ax.set_yticks(range(len(type_counts)))
ax.set_yticklabels(type_counts.index, fontsize=11)
ax.invert_yaxis()
ax.set_title('사고 유형 빈도 (전체)', fontsize=14, fontweight='bold')
ax.set_xlabel('빈도')
for i, (val, t) in enumerate(zip(type_counts.values, type_counts.index)):
    pct = val / type_counts.sum() * 100
    ax.text(val + type_counts.max()*0.01, i, f'{val:,} ({pct:.1f}%)', va='center', fontsize=10)

legend_elements = [
    Patch(facecolor='#e74c3c', label='카메라 직접 단속'),
    Patch(facecolor='#f39c12', label='카메라 간접 억제'),
    Patch(facecolor='#95a5a6', label='카메라 단속 어려움')
]
ax.legend(handles=legend_elements, fontsize=9, loc='lower right')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, '01_eda_point_basic.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → 01_eda_point_basic.png 저장")


# ── 2-2. 카메라-사고 연관성 (지점 단위) ────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('카메라-사고 연관성 분석 (지점 단위)', fontsize=20, fontweight='bold', y=0.98)

# (a) 카메라 수 vs 사고 건수 산점도
ax = axes[0, 0]
ax.scatter(df['반경내카메라수'], df['총사고건수'], s=10, alpha=0.2, c='#e74c3c')
# 카메라 수별 평균 사고건수 라인
cam_mean = df.groupby('반경내카메라수')['총사고건수'].mean()
ax.plot(cam_mean.index, cam_mean.values, 'b-o', linewidth=2, markersize=6,
        label='카메라 수별 평균 사고건수', zorder=5)
ax.set_title('반경 100m 카메라 수 vs 사고 건수', fontsize=14, fontweight='bold')
ax.set_xlabel('반경 100m 내 카메라 수')
ax.set_ylabel('총 사고 건수')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# (b) 카메라 유무별 사고 지표 비교
ax = axes[0, 1]
cam_yes = df[df['카메라유무'] == 1]
cam_no = df[df['카메라유무'] == 0]
metrics = ['총사고건수', '총사망자수', '총중상자수']
means_yes = [cam_yes[m].mean() for m in metrics]
means_no = [cam_no[m].mean() for m in metrics]

x = np.arange(len(metrics))
width = 0.35
bars1 = ax.bar(x - width/2, means_no, width, label=f'카메라 없음 ({len(cam_no):,}건)',
               color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x + width/2, means_yes, width, label=f'카메라 있음 ({len(cam_yes):,}건)',
               color='#3498db', alpha=0.8)
ax.set_title('카메라 유무별 평균 사고 지표', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['평균 사고건수', '평균 사망자수', '평균 중상자수'], fontsize=11)
ax.legend(fontsize=10)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{bar.get_height():.2f}', ha='center', fontsize=10, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{bar.get_height():.2f}', ha='center', fontsize=10, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# (c) 카메라 수 구간별 평균 사고 지표
ax = axes[1, 0]
df['카메라구간'] = pd.cut(df['반경내카메라수'],
                       bins=[-1, 0, 2, 5, 10, 100],
                       labels=['0대', '1~2대', '3~5대', '6~10대', '11대+'])
group_stats = df.groupby('카메라구간')[['총사고건수', '총중상자수']].mean()

x = np.arange(len(group_stats))
width = 0.35
ax.bar(x - width/2, group_stats['총사고건수'], width,
       label='평균 사고건수', color='#e74c3c', alpha=0.8)
ax.bar(x + width/2, group_stats['총중상자수'], width,
       label='평균 중상자수', color='#f39c12', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(group_stats.index, fontsize=11)
ax.set_title('카메라 수 구간별 평균 사고 지표', fontsize=14, fontweight='bold')
ax.set_ylabel('건수 / 명')
ax.legend(fontsize=10)
for i, (acc, inj) in enumerate(zip(group_stats['총사고건수'], group_stats['총중상자수'])):
    ax.text(i - width/2, acc + 0.2, f'{acc:.1f}', ha='center', fontsize=9, fontweight='bold')
    ax.text(i + width/2, inj + 0.2, f'{inj:.1f}', ha='center', fontsize=9, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# (d) 카메라 유무별 사고유형 비교
ax = axes[1, 1]
types_with_cam = []
types_without_cam = []
for _, row in df.iterrows():
    for t in row['사고유형목록']:
        if row['카메라유무'] == 1:
            types_with_cam.append(t)
        else:
            types_without_cam.append(t)

tc_with = pd.Series(types_with_cam).value_counts(normalize=True) * 100
tc_without = pd.Series(types_without_cam).value_counts(normalize=True) * 100

all_type_names = tc_with.index.tolist()
x = np.arange(len(all_type_names))
width = 0.35

vals_without = [tc_without.get(t, 0) for t in all_type_names]
vals_with = [tc_with.get(t, 0) for t in all_type_names]

ax.barh(x - width/2, vals_without, width, label='카메라 없음', color='#e74c3c', alpha=0.8)
ax.barh(x + width/2, vals_with, width, label='카메라 있음', color='#3498db', alpha=0.8)
ax.set_yticks(x)
ax.set_yticklabels(all_type_names, fontsize=11)
ax.invert_yaxis()
ax.set_title('카메라 유무별 사고 유형 비율 비교', fontsize=14, fontweight='bold')
ax.set_xlabel('비율 (%)')
ax.legend(fontsize=10)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, '02_eda_point_correlation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → 02_eda_point_correlation.png 저장")


# ╔═══════════════════════════════════════════════════════════════╗
# ║  PHASE 3: 지점 단위 군집화                                      ║
# ╚═══════════════════════════════════════════════════════════════╝
print("\n[Phase 3] 지점 단위 K-Means 군집화")

feature_cols = ['반경내카메라수', '총사고건수', '총사망자수', '총중상자수']
X = df[feature_cols].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow + Silhouette
inertias = []
sil_scores = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels, sample_size=5000, random_state=42))
    print(f"    k={k}: Inertia={km.inertia_:.1f}, Silhouette={sil_scores[-1]:.4f}")

best_k_sil = list(K_range)[np.argmax(sil_scores)]
optimal_k = 4
print(f"\n  Silhouette 최적 k: {best_k_sil}, 채택 k: {optimal_k}")

# 최종 군집화
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 라벨 부여 (실제 특성 기반)
cluster_stats = df.groupby('cluster').agg(
    건수=('총사고건수', 'count'),
    평균카메라수=('반경내카메라수', 'mean'),
    평균사고건수=('총사고건수', 'mean'),
    평균사망자수=('총사망자수', 'mean'),
    평균중상자수=('총중상자수', 'mean'),
    카메라0비율=('카메라유무', lambda x: 1 - x.mean())
).reset_index()

label_order = [
    '과잉 (카메라多 사고少)',
    '사망위험 (사망사고 집중)',
    '부족 (카메라少)',
    '사고다발 (사고多+중상多)'
]

# 특성 기반 라벨 매핑
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
    surplus_cluster: label_order[0],   # 과잉
    death_cluster: label_order[1],     # 사망위험
    deficit_cluster: label_order[2],   # 부족
    accident_cluster: label_order[3],  # 사고다발
}
df['군집라벨'] = df['cluster'].map(label_map)

print("\n  군집화 결과:")
for _, row in cluster_stats.iterrows():
    c = int(row['cluster'])
    label = label_map[c]
    print(f"    [{label}] {int(row['건수']):,}건 | 카메라 {row['평균카메라수']:.1f}대 | 사고 {row['평균사고건수']:.1f}건 | 사망 {row['평균사망자수']:.2f} | 중상 {row['평균중상자수']:.1f}")


# ── 군집화 시각화 ──────────────────────────────────────────────
cluster_colors_list = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
cluster_colors = {label_order[i]: cluster_colors_list[i] for i in range(4)}

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('K-Means 군집화 결과 (n={:,})'.format(len(df)),
             fontsize=20, fontweight='bold', y=0.98)

# (a) Elbow + Silhouette
ax = axes[0, 0]
ax2 = ax.twinx()
ax.plot(list(K_range), inertias, 'bo-', linewidth=2, markersize=8, label='Inertia')
ax2.plot(list(K_range), sil_scores, 'gs-', linewidth=2, markersize=8, label='Silhouette')
ax.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'선택: k={optimal_k}')
ax.set_title('최적 k 결정 (Elbow + Silhouette)', fontsize=14, fontweight='bold')
ax.set_xlabel('클러스터 수 (k)')
ax.set_ylabel('Inertia', color='blue')
ax2.set_ylabel('Silhouette Score', color='green')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
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
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)

# (c) 군집별 평균 비교
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
    ax.bar(x_pos + i * width, row.values, width,
           label=short, color=color, alpha=0.85)
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
ax.legend(fontsize=7, loc='lower right',
          labels=[l.split('(')[0].strip() for l in ct.columns])

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, '03_clustering.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → 03_clustering.png 저장")


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

# 군집별 유형
all_types_by_cluster = {}
for label in label_order:
    subset = df[df['군집라벨'] == label]
    all_t = []
    for types_list in subset['사고유형목록']:
        all_t.extend(types_list)
    all_types_by_cluster[label] = pd.Series(all_t).value_counts()

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('군집별 사고 유형 분석\n(카메라 재배치 실효성 검증)', fontsize=20, fontweight='bold', y=0.98)

for idx, (label, type_counts) in enumerate(all_types_by_cluster.items()):
    ax = axes[idx // 2, idx % 2]
    top7 = type_counts.head(7)
    colors = []
    for t in top7.index:
        if t in 단속가능:
            colors.append('#e74c3c')
        elif t in 간접억제:
            colors.append('#f39c12')
        else:
            colors.append('#95a5a6')
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
        pct = val / type_counts.sum() * 100
        ax.text(bar.get_width() + max(top7.values)*0.01, bar.get_y() + bar.get_height()/2,
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
plt.savefig(os.path.join(OUTPUT_DIR, '04_accident_type_by_cluster.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → 04_accident_type_by_cluster.png 저장")


# ╔═══════════════════════════════════════════════════════════════╗
# ║  PHASE 5: 재배치 제안                                           ║
# ╚═══════════════════════════════════════════════════════════════╝
print("\n[Phase 5] 재배치 제안")

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('단속 카메라 재배치 제안', fontsize=20, fontweight='bold', y=0.98)

# (a) 카메라 미설치 고위험 - 시도별
ax = axes[0, 0]
no_cam = df[df['반경내카메라수'] == 0].copy()
no_cam['위험점수'] = no_cam['총사고건수'] + no_cam['총사망자수']*10 + no_cam['총중상자수']*5
no_cam_sido = no_cam.groupby('시도명')['위험점수'].agg(['count', 'mean']).sort_values('count', ascending=True)
colors_nc = plt.cm.Reds(np.linspace(0.3, 0.9, len(no_cam_sido)))
ax.barh(no_cam_sido.index, no_cam_sido['count'], color=colors_nc, edgecolor='white')
ax.set_title(f'카메라 미설치 사고위험지역 ({len(no_cam):,}건)\n시도별 분포', fontsize=13, fontweight='bold')
ax.set_xlabel('지역 수')
for i, (idx, row) in enumerate(no_cam_sido.iterrows()):
    ax.text(row['count'] + 2, i, f"{int(row['count'])}건", va='center', fontsize=9)
ax.grid(axis='x', alpha=0.3)

# (b) 카메라 과밀 (10대+) - 시도별
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

부족 = df[df['군집라벨']==label_order[2]]
위험 = df[df['군집라벨']==label_order[3]]
과잉 = df[df['군집라벨']==label_order[0]]

summary = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       단속 카메라 재배치 분석 결론
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 분석 규모
   · 사고위험지역: {len(df):,}건 (개별 지점)
   · 단속 카메라: {len(df_cam):,}대

📋 군집화 결과 (K-Means, k=4)
   🔵 과잉: {len(과잉):,}건 (카메라 {과잉['반경내카메라수'].mean():.1f}대)
   ⚫ 사망위험: {len(df[df['군집라벨']==label_order[1]]):,}건
   🔴 부족: {len(부족):,}건 (카메라 {부족['반경내카메라수'].mean():.1f}대)
   🟡 사고다발: {len(위험):,}건 (사고 {위험['총사고건수'].mean():.1f}건)

⚠️ 핵심 발견
   · 카메라 없는 사고위험지역: {len(no_cam):,}건
   · 카메라 10대+ 과밀 지역: {len(over_cam):,}건
   · 위험 구역 카메라 단속 관련 비율: {위험['카메라효과비율'].mean():.1f}%

💡 재배치 제안
   과잉 구역의 카메라를
   부족·위험 구역으로 이동하면
   과속·신호위반 사고를
   줄일 수 있는 개연성이 높음
"""
ax.text(0.05, 0.95, summary, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', fontfamily='Malgun Gothic',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa',
                  edgecolor='#dee2e6', alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, '05_redistribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → 05_redistribution.png 저장")


# ╔═══════════════════════════════════════════════════════════════╗
# ║  결과 CSV 저장                                                 ║
# ╚═══════════════════════════════════════════════════════════════╝
print("\n[결과 저장]")

df_out = df[['연도코드', '시도명', '시군구코드', '사고위험지역명', '사고위험지역id',
             '총사고건수', '총사망자수', '총중상자수', '총경상자수',
             '사고분석유형명', '반경내카메라수', '반경내과속카메라', '반경내신호카메라',
             '카메라효과비율', 'cluster', '군집라벨']].copy()
df_out.to_csv(os.path.join(OUTPUT_DIR, 'cluster_result_v2.csv'),
              index=False, encoding='utf-8-sig')
print(f"  → cluster_result_v2.csv ({len(df_out):,}건)")

no_cam_top = no_cam.nlargest(30, '위험점수')[
    ['연도코드', '시도명', '사고위험지역명', '총사고건수',
     '총사망자수', '총중상자수', '위험점수', '사고분석유형명']
]
no_cam_top.to_csv(os.path.join(OUTPUT_DIR, 'high_risk_no_camera_top30.csv'),
                  index=False, encoding='utf-8-sig')
print(f"  → high_risk_no_camera_top30.csv")

summary_df = df.groupby('군집라벨').agg(
    지점수=('총사고건수', 'count'),
    평균카메라수=('반경내카메라수', 'mean'),
    평균사고건수=('총사고건수', 'mean'),
    평균사망자수=('총사망자수', 'mean'),
    평균중상자수=('총중상자수', 'mean'),
    카메라단속관련비율=('카메라효과비율', 'mean'),
    카메라없는비율=('카메라유무', lambda x: f"{(1-x.mean())*100:.1f}%")
).reset_index()
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'cluster_summary_v2.csv'),
                  index=False, encoding='utf-8-sig')
print(f"  → cluster_summary_v2.csv")

# 사고유형별 군집 CSV
type_summary = []
for label in label_order:
    tc = all_types_by_cluster[label]
    total = tc.sum()
    for t, c in tc.items():
        type_summary.append({
            '군집': label.split('(')[0].strip(),
            '사고유형': t,
            '건수': c,
            '비율(%)': round(c / total * 100, 1),
            '카메라단속': '직접' if t in 단속가능 else ('간접' if t in 간접억제 else '불가')
        })
pd.DataFrame(type_summary).to_csv(
    os.path.join(OUTPUT_DIR, 'accident_type_by_cluster.csv'),
    index=False, encoding='utf-8-sig')
print(f"  → accident_type_by_cluster.csv")

# 최종 출력
print("\n" + "=" * 70)
print("  v2 통합 분석 완료!")
print("=" * 70)
print(f"\n  📁 출력: {os.path.abspath(OUTPUT_DIR)}")
print(f"  📊 차트: 5개 PNG")
print(f"  📄 데이터: 4개 CSV")
for label in label_order:
    subset = df[df['군집라벨'] == label]
    emoji = {'과잉': '🔵', '사망위험': '⚫', '부족': '🔴', '사고다발': '🟡'}
    e = [v for k, v in emoji.items() if k in label][0]
    print(f"  {e} {label}: {len(subset):,}건")
print()
