import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from kmodes.kprototypes import KPrototypes  # 🌟 K-Prototypes 도입
from pyproj import Transformer
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
os.makedirs('./pic', exist_ok=True)

# ==============================================================
# ⚙️ 1. 정책 시뮬레이션 파라미터 설정 (AI 위장 제거)
# ==============================================================
# C1(단순접촉) 구역에서 회수하여 C3(과속치명)로 이전할 예산의 비율 시나리오
SIMULATION_RECOVERY_RATE = 0.90

# ==============================================================
# 📥 2. 데이터 로드 및 전처리
# ==============================================================
def load_csv(file_path):
    try: return pd.read_csv(file_path, encoding='cp949')
    except: return pd.read_csv(file_path, encoding='utf-8')

df_camera = load_csv('경찰청_무인교통단속카메라_20260406.csv').dropna(subset=['위도', '경도'])
df_risk = load_csv('RiskArea.csv')

df_risk['EPDO'] = (df_risk['총사망자수']*12 + df_risk['총중상자수']*9 +
                   df_risk['총경상자수']*3 + df_risk['총부상신고자수']*1)

encoded_tags = df_risk['사고분석유형명'].str.get_dummies(sep=' / ')
df_risk = pd.concat([df_risk, encoded_tags], axis=1)

# 고속 공간 매핑
transformer = Transformer.from_crs("epsg:4326", "epsg:5179", always_xy=True)
df_camera['utmk_x'], df_camera['utmk_y'] = transformer.transform(df_camera['경도'].values, df_camera['위도'].values)
tree_camera = cKDTree(df_camera[['utmk_x', 'utmk_y']].dropna().values)
tree_risk = cKDTree(df_risk[['중심점utmkx좌표', '중심점utmky좌표']].dropna().values)
df_risk['Camera_Count'] = [len(c) for c in tree_risk.query_ball_tree(tree_camera, r=50)]

# ==============================================================
# 🧠 3. K-Prototypes 군집화 (수학적 모순 해결)
# ==============================================================
cat_cols = ['과속', '신호위반', '안전거리미확보', '중앙선침범', 'U턴중']
num_cols = ['EPDO', '츙사고건수']

# K-Prototypes를 위해 연속형 변수만 스케일링 (범주형은 그대로 유지)
scaler = StandardScaler()
df_scaled_num = pd.DataFrame(scaler.fit_transform(df_risk[num_cols]), columns=num_cols)
df_model = pd.concat([df_risk[cat_cols], df_scaled_num], axis=1)

# 범주형 컬럼의 인덱스 추출
categorical_indices = [df_model.columns.get_loc(col) for col in cat_cols]
X_matrix = df_model.values

print("🚀 K-Prototypes 최적 K 탐색 중... (데이터 혼합 연산으로 약 1~2분 소요될 수 있습니다)")
costs = []
k_range = range(2, 7)
for k in k_range:
    kp = KPrototypes(n_clusters=k, init='Cao', random_state=42, n_jobs=-1)
    kp.fit(X_matrix, categorical=categorical_indices)
    costs.append(kp.cost_)

# 최종 K=4 모델 학습
print("🎯 K=4 최종 모델 학습 중...")
kp_final = KPrototypes(n_clusters=4, init='Cao', random_state=42, n_jobs=-1)
df_risk['Cluster'] = kp_final.fit_predict(X_matrix, categorical=categorical_indices)

# 🚨 [완벽 수정본] 사고 원인(DNA) 중심의 동적 라벨링
c_names = {}
centroids = df_risk.groupby('Cluster')[num_cols + cat_cols].mean()

# 1. '과속' 비율이 가장 높은 그룹 -> C3
c3_idx = centroids['과속'].idxmax()
c_names[c3_idx] = 'C3 (Fatal Speeding)'
remaining = set(range(4)) - {c3_idx}

# 2. '신호위반' 비율이 가장 높은 그룹 -> C0
c0_idx = centroids.loc[list(remaining), '신호위반'].idxmax()
c_names[c0_idx] = 'C0 (Signal Violation)'
remaining -= {c0_idx}

# 3. '중앙선침범' 및 'U턴중' 비율이 높은 그룹 -> C2
c2_idx = centroids.loc[list(remaining), ['중앙선침범', 'U턴중']].mean(axis=1).idxmax()
c_names[c2_idx] = 'C2 (Lawless Intersection)'
remaining -= {c2_idx}

# 4. 남은 그룹 (안전거리미확보가 높고 빈도가 높은 단순 접촉/정체구역) -> C1
c1_idx = list(remaining)[0]
c_names[c1_idx] = 'C1 (Minor Crash)'

df_risk['Cluster_Name'] = df_risk['Cluster'].map(c_names)

# ==============================================================
# 📊 4. 6패널 방어 대시보드 시각화
# ==============================================================
fig, axes = plt.subplots(2, 3, figsize=(24, 14))
fig.suptitle("AI-Driven Traffic Safety Budget Optimization (K-Prototypes)", fontsize=24, fontweight='bold', y=1.02)

# [패널 1]
sns.scatterplot(data=df_risk, x='츙사고건수', y='EPDO', hue='Camera_Count', palette='Reds', size='Camera_Count', sizes=(10, 150), ax=axes[0,0], alpha=0.6)
axes[0,0].set_title("1. Camera Budget Mismatch", fontweight='bold')
axes[0,0].set_xlabel("Accident Frequency")
axes[0,0].set_ylabel("Severity (EPDO)")

# [패널 2]
dense_filter = df_risk['Camera_Count'] >= 2
if dense_filter.sum() == 0: dense_filter = df_risk['Camera_Count'] >= 1
if dense_filter.sum() > 0:
    cause_eng_map = {'과속':'Speeding', '신호위반':'Signal Violation', '안전거리미확보':'Tailgating', '중앙선침범':'Centerline Cross', 'U턴중':'Illegal U-turn'}
    dense_cameras = df_risk[dense_filter][cat_cols].mean().sort_values(ascending=False)
    dense_cameras.index = dense_cameras.index.map(lambda x: cause_eng_map.get(x, x))
    sns.barplot(x=dense_cameras.values, y=dense_cameras.index, palette='viridis', ax=axes[0,1])
axes[0,1].set_title("2. Accident Causes in Dense Camera Zones", fontweight='bold')

# [패널 3] K-Prototypes Cost (Elbow Method)
axes[0,2].plot(k_range, costs, marker='s', color='steelblue', linewidth=2, linestyle='-')
axes[0,2].set_title("3. Optimal K Validation (K-Prototypes Cost)", fontweight='bold')
axes[0,2].set_xlabel("Number of Clusters (K)")
axes[0,2].set_ylabel("Cost (Mixed Distance Measure)", color='steelblue')
axes[0,2].axvline(x=4, color='red', linestyle='--', label='Selected K=4')
axes[0,2].legend()

# [패널 4]
sns.boxplot(data=df_risk, x='Cluster_Name', y='츙사고건수', palette='Set2', ax=axes[1,0])
axes[1,0].set_title("4. Balloon Effect (Accident Freq by Cluster)", fontweight='bold')
axes[1,0].set_ylabel("Accident Frequency")
axes[1,0].tick_params(axis='x', rotation=15)

# [패널 5]
sns.barplot(data=df_risk, x='Cluster_Name', y='Camera_Count', palette='Set1', ax=axes[1,1])
axes[1,1].set_title("5. Policy Blind Spots (Avg Cameras)", fontweight='bold')
axes[1,1].set_ylabel("Avg Camera Count")
axes[1,1].tick_params(axis='x', rotation=15)

# [패널 6] 🚨 명확한 "정책 시뮬레이션" 명시
budget_current = df_risk.groupby('Cluster_Name')['Camera_Count'].sum()
budget_ai = pd.Series({
    'C0 (Signal Violation)': budget_current.get('C0 (Signal Violation)', 0),
    'C1 (Minor Crash)': budget_current.get('C1 (Minor Crash)', 0) * (1 - SIMULATION_RECOVERY_RATE),
    'C2 (Lawless Intersection)': 0,
    'C3 (Fatal Speeding)': budget_current.get('C3 (Fatal Speeding)', 0) + (budget_current.get('C1 (Minor Crash)', 0) * SIMULATION_RECOVERY_RATE)
})

df_budget = pd.DataFrame({'Current Budget': budget_current, f'Simulation ({int(SIMULATION_RECOVERY_RATE*100)}% Shift)': budget_ai}).fillna(0)
df_budget.index.name = 'Cluster_Name'
df_budget = df_budget.reset_index()

df_budget_melt = df_budget.melt(id_vars='Cluster_Name', var_name='Type', value_name='Total Cameras')
sns.barplot(data=df_budget_melt, x='Cluster_Name', y='Total Cameras', hue='Type', palette=['#95a5a6', '#e74c3c'], ax=axes[1,2])
axes[1,2].set_title("6. Budget Redistribution Simulation", fontweight='bold')
axes[1,2].set_ylabel("Total Cameras")
axes[1,2].tick_params(axis='x', rotation=15)

plt.tight_layout()
output_path = './pic/traffic_safety_kprototypes_dashboard.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Dashboard saved to: {output_path}")

# ==============================================================
# 🎯 5. [부록] 원그래프 (Radar Chart) 개별 생성
# ==============================================================
fig_radar = plt.figure(figsize=(9, 9))
ax_radar = fig_radar.add_subplot(1, 1, 1, polar=True)

eng_features = ['Speeding', 'Signal Violation', 'Tailgating', 'Centerline Cross', 'U-turn', 'EPDO (Severity)', 'Accident Freq']
features_order = cat_cols + num_cols
cluster_means_raw = df_risk.groupby('Cluster_Name')[features_order].mean()

cluster_means_normalized = cluster_means_raw / cluster_means_raw.max()
cluster_means_normalized.columns = eng_features

angles = np.linspace(0, 2 * np.pi, len(eng_features), endpoint=False).tolist()
angles += angles[:1]

colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
for idx, (cluster_name, row) in enumerate(cluster_means_normalized.iterrows()):
    values = row.tolist()
    values += values[:1]
    ax_radar.plot(angles, values, color=colors[idx], linewidth=2.5, label=cluster_name)
    ax_radar.fill(angles, values, color=colors[idx], alpha=0.08)

ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(eng_features, fontsize=11, fontweight='bold')
ax_radar.set_title("K-Prototypes Accident DNA Profiles", fontsize=16, fontweight='bold', pad=20)
ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

radar_output_path = './pic/kprototypes_radar_chart.png'
plt.savefig(radar_output_path, dpi=300, bbox_inches='tight')
print(f"✅ Radar chart saved to: {radar_output_path}")