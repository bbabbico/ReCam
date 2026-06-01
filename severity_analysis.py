"""
군집별 심각도 지표 분석
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pyproj import Transformer
from scipy.spatial import cKDTree

mpl.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = 'output_v2'

# ── 데이터 로드 & 군집화 ──────────────────────────────────────
print("데이터 로드 및 군집화 중...")
df_risk = pd.read_csv('RiskArea.csv', encoding='cp949')
df_risk.rename(columns={'츙사고건수': '총사고건수'}, inplace=True)
df_cam = pd.read_csv('경찰청_무인교통단속카메라_20260406.csv', encoding='utf-8')

transformer = Transformer.from_crs("EPSG:4326", "EPSG:5178", always_xy=True)
cam_x, cam_y = transformer.transform(df_cam['경도'].values, df_cam['위도'].values)
df_cam['utmk_x'] = cam_x
df_cam['utmk_y'] = cam_y

시도코드_map = {
    11: '서울', 26: '부산', 27: '대구', 28: '인천', 29: '광주',
    30: '대전', 31: '울산', 36: '세종', 41: '경기', 42: '강원',
    43: '충북', 44: '충남', 45: '전북', 46: '전남', 47: '경북',
    48: '경남', 50: '제주'
}
df_risk['시도코드'] = df_risk['시군구코드'] // 1000
df_risk['시도명'] = df_risk['시도코드'].map(시도코드_map)

RADIUS = 100
risk_coords = df_risk[['중심점utmkx좌표', '중심점utmky좌표']].dropna().values
cam_coords = df_cam[['utmk_x', 'utmk_y']].values
tree_cam = cKDTree(cam_coords)

cam_counts = []
for rx, ry in risk_coords:
    cam_counts.append(len(tree_cam.query_ball_point([rx, ry], r=RADIUS)))

df = df_risk.dropna(subset=['중심점utmkx좌표', '중심점utmky좌표']).copy()
df['반경내카메라수'] = cam_counts
df['카메라유무'] = (df['반경내카메라수'] > 0).astype(int)

feature_cols = ['반경내카메라수', '총사고건수', '총사망자수', '총중상자수']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10, max_iter=300)
df['cluster'] = kmeans.fit_predict(X_scaled)

cluster_stats = df.groupby('cluster').agg(
    평균카메라수=('반경내카메라수', 'mean'),
    평균사고건수=('총사고건수', 'mean'),
    평균사망자수=('총사망자수', 'mean'),
    평균중상자수=('총중상자수', 'mean'),
).reset_index()

label_order = ['과잉', '사망위험', '부족', '사고다발']

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
    surplus_cluster: '과잉',
    death_cluster: '사망위험',
    deficit_cluster: '부족',
    accident_cluster: '사고다발',
}
df['군집'] = df['cluster'].map(label_map)

# ── 심각도 지표 산출 ──────────────────────────────────────────
df['사망률'] = df['총사망자수'] / df['총사고건수'].replace(0, np.nan)
df['중상률'] = df['총중상자수'] / df['총사고건수'].replace(0, np.nan)
df['경상률'] = df['총경상자수'] / df['총사고건수'].replace(0, np.nan)
df['위험점수'] = df['총사고건수'] + df['총사망자수'] * 10 + df['총중상자수'] * 5
df['총피해자수'] = df['총사망자수'] + df['총중상자수'] + df['총경상자수'] + df['총부상신고자수']
df['피해자비율'] = df['총피해자수'] / df['총사고건수'].replace(0, np.nan)

print(f"군집화 완료: {len(df):,}건\n")

# ── 콘솔 출력 ─────────────────────────────────────────────────
print("=" * 70)
print("  군집별 심각도 지표 요약")
print("=" * 70)

severity = df.groupby('군집').agg(
    지점수=('총사고건수', 'count'),
    평균카메라수=('반경내카메라수', 'mean'),
    총사고=('총사고건수', 'sum'),
    총사망=('총사망자수', 'sum'),
    총중상=('총중상자수', 'sum'),
    총경상=('총경상자수', 'sum'),
    총부상신고=('총부상신고자수', 'sum'),
    평균사고건수=('총사고건수', 'mean'),
    평균사망자수=('총사망자수', 'mean'),
    평균중상자수=('총중상자수', 'mean'),
    평균경상자수=('총경상자수', 'mean'),
    평균사망률=('사망률', 'mean'),
    평균중상률=('중상률', 'mean'),
    평균위험점수=('위험점수', 'mean'),
    평균피해자비율=('피해자비율', 'mean'),
).reindex(label_order)

for label in label_order:
    row = severity.loc[label]
    emoji = {'과잉': '🔵', '사망위험': '⚫', '부족': '🔴', '사고다발': '🟡'}[label]
    print(f"\n{emoji} [{label}] ({int(row['지점수']):,}건 | 카메라 평균 {row['평균카메라수']:.1f}대)")
    print(f"   총 사고: {int(row['총사고']):,}건 | 평균 {row['평균사고건수']:.1f}건/지점")
    print(f"   총 사망: {int(row['총사망']):,}명 | 평균 {row['평균사망자수']:.3f}명/지점 | 사망률 {row['평균사망률']*100:.2f}%")
    print(f"   총 중상: {int(row['총중상']):,}명 | 평균 {row['평균중상자수']:.1f}명/지점 | 중상률 {row['평균중상률']*100:.1f}%")
    print(f"   총 경상: {int(row['총경상']):,}명 | 평균 {row['평균경상자수']:.1f}명/지점")
    print(f"   위험점수: 평균 {row['평균위험점수']:.1f} | 피해자/사고 비율: {row['평균피해자비율']:.2f}")


# ╔═══════════════════════════════════════════════════════════════╗
# ║  시각화                                                        ║
# ╚═══════════════════════════════════════════════════════════════╝
cluster_colors = {'과잉': '#3498db', '사망위험': '#2c3e50', '부족': '#e74c3c', '사고다발': '#f39c12'}

fig, axes = plt.subplots(2, 3, figsize=(24, 14))
fig.suptitle('군집별 심각도 지표 분석', fontsize=22, fontweight='bold', y=0.98)

# ── (1) 군집별 평균 사고건수·사망자·중상자 ──────────────────────
ax = axes[0, 0]
x = np.arange(len(label_order))
width = 0.25
vals_acc = [severity.loc[l, '평균사고건수'] for l in label_order]
vals_death = [severity.loc[l, '평균사망자수'] * 100 for l in label_order]  # 스케일 맞춤
vals_inj = [severity.loc[l, '평균중상자수'] for l in label_order]

bars1 = ax.bar(x - width, vals_acc, width, label='평균 사고건수', color='#e74c3c', alpha=0.85)
bars2 = ax.bar(x, vals_inj, width, label='평균 중상자수', color='#f39c12', alpha=0.85)
bars3 = ax.bar(x + width, vals_death, width, label='평균 사망자수 (×100)', color='#2c3e50', alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(label_order, fontsize=12)
ax.set_title('군집별 평균 사고·피해 규모', fontsize=14, fontweight='bold')
ax.set_ylabel('건수 / 명')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        val = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                f'{val:.1f}', ha='center', fontsize=9, fontweight='bold')

# ── (2) 군집별 사망률 & 중상률 ─────────────────────────────────
ax = axes[0, 1]
death_rates = [severity.loc[l, '평균사망률'] * 100 for l in label_order]
inj_rates = [severity.loc[l, '평균중상률'] * 100 for l in label_order]
colors = [cluster_colors[l] for l in label_order]

x = np.arange(len(label_order))
width = 0.35
bars1 = ax.bar(x - width/2, death_rates, width, label='사망률 (%)', color='#2c3e50', alpha=0.85)
bars2 = ax.bar(x + width/2, inj_rates, width, label='중상률 (%)', color='#e67e22', alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(label_order, fontsize=12)
ax.set_title('군집별 사망률 & 중상률', fontsize=14, fontweight='bold')
ax.set_ylabel('비율 (%)')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        val = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.1,
                f'{val:.2f}%', ha='center', fontsize=9, fontweight='bold')

# ── (3) 군집별 위험점수 분포 (박스플롯) ────────────────────────
ax = axes[0, 2]
bp_data = [df[df['군집'] == l]['위험점수'] for l in label_order]
bp = ax.boxplot(bp_data, labels=label_order, patch_artist=True, showfliers=False)
for patch, label in zip(bp['boxes'], label_order):
    patch.set_facecolor(cluster_colors[label])
    patch.set_alpha(0.6)
ax.set_title('군집별 위험점수 분포\n(사고×1 + 사망×10 + 중상×5)', fontsize=14, fontweight='bold')
ax.set_ylabel('위험점수')
ax.grid(axis='y', alpha=0.3)

# 중앙값 표시
for i, l in enumerate(label_order):
    median = df[df['군집'] == l]['위험점수'].median()
    ax.text(i + 1, median + 2, f'중앙값\n{median:.0f}', ha='center', fontsize=9, fontweight='bold')

# ── (4) 군집별 총 피해 규모 (누적 바) ──────────────────────────
ax = axes[1, 0]
damage = pd.DataFrame({
    '사망': [severity.loc[l, '총사망'] for l in label_order],
    '중상': [severity.loc[l, '총중상'] for l in label_order],
    '경상': [severity.loc[l, '총경상'] for l in label_order],
}, index=label_order)

damage.plot(kind='barh', stacked=True, ax=ax,
            color=['#2c3e50', '#e67e22', '#f1c40f'], edgecolor='white')
ax.set_title('군집별 총 피해자 수 (누적)', fontsize=14, fontweight='bold')
ax.set_xlabel('피해자 수 (명)')
ax.legend(fontsize=10)
ax.grid(axis='x', alpha=0.3)

for i, l in enumerate(label_order):
    total = damage.loc[l].sum()
    ax.text(total + 50, i, f'{int(total):,}명', va='center', fontsize=10, fontweight='bold')

# ── (5) 카메라 유무 × 심각도 비교 ─────────────────────────────
ax = axes[1, 1]
cam_severity = df.groupby('카메라유무').agg(
    평균사고=('총사고건수', 'mean'),
    평균사망=('총사망자수', 'mean'),
    평균중상=('총중상자수', 'mean'),
    평균위험점수=('위험점수', 'mean'),
    사망률=('사망률', 'mean'),
    중상률=('중상률', 'mean'),
)

metrics = ['평균사고', '평균중상', '평균위험점수']
labels_m = ['평균 사고건수', '평균 중상자수', '평균 위험점수']

# 정규화해서 비교
vals_no = [cam_severity.loc[0, m] for m in metrics]
vals_yes = [cam_severity.loc[1, m] for m in metrics]

# 위험점수 스케일 조정
vals_no_norm = [vals_no[0], vals_no[1], vals_no[2] / 3]
vals_yes_norm = [vals_yes[0], vals_yes[1], vals_yes[2] / 3]

x = np.arange(len(metrics))
width = 0.35
n_no = (df['카메라유무'] == 0).sum()
n_yes = (df['카메라유무'] == 1).sum()
bars1 = ax.bar(x - width/2, vals_no_norm, width,
               label=f'카메라 없음 ({n_no:,}건)', color='#e74c3c', alpha=0.85)
bars2 = ax.bar(x + width/2, vals_yes_norm, width,
               label=f'카메라 있음 ({n_yes:,}건)', color='#3498db', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(['평균 사고건수', '평균 중상자수', '평균 위험점수(÷3)'], fontsize=10)
ax.set_title('카메라 유무별 심각도 비교', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

for bars, vals in [(bars1, vals_no), (bars2, vals_yes)]:
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}', ha='center', fontsize=9, fontweight='bold')

# ── (6) 결론 텍스트 ───────────────────────────────────────────
ax = axes[1, 2]
ax.axis('off')

위험_row = severity.loc['사고다발']
부족_row = severity.loc['부족']
과잉_row = severity.loc['과잉']

conclusion = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     군집별 심각도 핵심 요약
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 사고 규모
  과잉: 평균 {과잉_row['평균사고건수']:.1f}건/지점
  부족: 평균 {부족_row['평균사고건수']:.1f}건/지점
  사고다발: 평균 {위험_row['평균사고건수']:.1f}건/지점
  → 사고다발 구역은 과잉의 {위험_row['평균사고건수']/과잉_row['평균사고건수']:.1f}배

📊 사망률
  과잉: {과잉_row['평균사망률']*100:.3f}%
  사망위험: {severity.loc['사망위험','평균사망률']*100:.3f}%
  부족: {부족_row['평균사망률']*100:.3f}%
  사고다발: {위험_row['평균사망률']*100:.3f}%

📊 중상률
  과잉: {과잉_row['평균중상률']*100:.1f}%
  부족: {부족_row['평균중상률']*100:.1f}%
  사고다발: {위험_row['평균중상률']*100:.1f}%

📊 위험점수 (사고+사망×10+중상×5)
  과잉: 평균 {과잉_row['평균위험점수']:.1f}
  부족: 평균 {부족_row['평균위험점수']:.1f}
  사고다발: 평균 {위험_row['평균위험점수']:.1f}

📊 총 피해자
  과잉: 사망 {int(과잉_row['총사망'])}명, 중상 {int(과잉_row['총중상']):,}명
  부족: 사망 {int(부족_row['총사망'])}명, 중상 {int(부족_row['총중상']):,}명
  사고다발: 사망 {int(위험_row['총사망'])}명, 중상 {int(위험_row['총중상']):,}명
"""
ax.text(0.02, 0.98, conclusion, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', fontfamily='Malgun Gothic',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#fdf2e9',
                  edgecolor='#e67e22', alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, '07_severity_by_cluster.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n→ 07_severity_by_cluster.png 저장")

# ── CSV 저장 ──────────────────────────────────────────────────
severity_out = severity.reset_index()
severity_out.columns = ['군집'] + list(severity_out.columns[1:])
severity_out.to_csv(os.path.join(OUTPUT_DIR, 'severity_by_cluster.csv'),
                    index=False, encoding='utf-8-sig')
print("→ severity_by_cluster.csv 저장")

print("\n✅ 심각도 분석 완료!")
