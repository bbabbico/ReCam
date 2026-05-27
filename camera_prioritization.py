# -*- coding: utf-8 -*-
"""
카메라 재배치 우선순위 도출 및 1만 개 부족 군집 세분화
"""
import os, sys, warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = 'output_v2'

# 1. 데이터 로드 (v2 통합본 결과)
print("데이터 로딩 중...")
df = pd.read_csv(os.path.join(OUTPUT_DIR, 'cluster_result_v2.csv'))

# 위험점수 계산 (사고건수 + 사망자*10 + 중상자*5)
df['위험점수'] = df['총사고건수'] + df['총사망자수'] * 10 + df['총중상자수'] * 5
df['총피해자수'] = df['총사망자수'] + df['총중상자수'] + df['총경상자수']

# 2. 우선순위 타겟 분류
# 과잉 구역 (카메라 제공자)
과잉 = df[df['군집라벨'].str.contains('과잉', na=False)]
과잉카메라수 = 과잉['반경내카메라수'].sum()

# 재배치 대상: 반경 내 카메라가 0대인 곳만
no_cam = df[df['반경내카메라수'] == 0]

# 1순위: 사망위험 + 사고다발 (위험도 최상)
p1 = no_cam[no_cam['군집라벨'].str.contains('사망위험|사고다발', na=False)]

# 일반 부족 군집
부족_all = no_cam[no_cam['군집라벨'].str.contains('부족', na=False)].sort_values('위험점수', ascending=False)
top_10_pct_idx = int(len(부족_all) * 0.1)

# 2순위: 일반 부족 중 상위 10%
p2 = 부족_all.iloc[:top_10_pct_idx]
# 3순위: 일반 부족 중 하위 90%
p3 = 부족_all.iloc[top_10_pct_idx:]

print(f"\n[재배치 자원 파악]")
print(f"과잉 구역: {len(과잉):,}곳 (잉여 카메라 파이프라인: {과잉카메라수:,}대 활용 가능)")

print(f"\n[재배치 우선순위 분류 (카메라 미설치 12,205곳 기준)]")
print(f"1순위 (사망위험+사고다발): {len(p1):,}곳")
print(f"2순위 (일반 부족 상위 10%): {len(p2):,}곳")
print(f"3순위 (일반 부족 하위 90%): {len(p3):,}곳")


# 3. 데이터 통합 및 CSV 출력
p1['우선순위'] = '1순위 (사망/사고다발)'
p2['우선순위'] = '2순위 (일반부족 상위10%)'
p3['우선순위'] = '3순위 (일반부족 나머지)'

final_targets = pd.concat([p1, p2, p3])
final_targets = final_targets.sort_values(['우선순위', '위험점수'], ascending=[True, False])

export_cols = ['우선순위', '시도명', '사고위험지역명', '군집라벨', '위험점수', '총사고건수', '총사망자수', '총중상자수', '사고분석유형명']
final_targets[export_cols].to_csv(os.path.join(OUTPUT_DIR, 'final_camera_reallocation_targets.csv'), index=False, encoding='utf-8-sig')
print(f"→ final_camera_reallocation_targets.csv 저장 완료")


# 4. 시각화 (스토리라인 차트)
fig = plt.figure(figsize=(20, 14))
fig.suptitle('단속 카메라 재배치 3단계 전략 및 우선순위 세분화', fontsize=22, fontweight='bold', y=0.98)

# (1) 재배치 전략 흐름도 (도넛+바)
ax1 = fig.add_subplot(2, 2, 1)

sizes = [len(p1), len(p2), len(p3)]
labels = [f'1순위 타겟\n(사망/사고다발)\n{len(p1):,}곳', 
          f'2순위 타겟\n(부족 상위 10%)\n{len(p2):,}곳', 
          f'3순위 후순위\n(부족 90%)\n{len(p3):,}곳']
colors = ['#e74c3c', '#f39c12', '#bdc3c7']
explode = (0.05, 0.05, 0)

wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors, 
                                   autopct='%1.1f%%', startangle=140, pctdistance=0.8)
plt.setp(autotexts, size=11, weight="bold", color="white")
plt.setp(texts, size=12, weight="bold")

# 가운데 뚫기
centre_circle = plt.Circle((0,0),0.55,fc='white')
ax1.add_patch(centre_circle)
ax1.text(0, 0, f"카메라 없는\n위험지역\n총 {len(final_targets):,}곳", ha='center', va='center', fontsize=14, fontweight='bold')
ax1.set_title('카메라 미설치 위험지역 타겟 세분화', fontsize=15, fontweight='bold')

# (2) 우선순위별 평균 위험도 비교
ax2 = fig.add_subplot(2, 2, 2)
priorities = [p1, p2, p3]
names = ['1순위 (사망+사고다발)', '2순위 (부족 상위 10%)', '3순위 (부족 하위 90%)']

avg_risk = [p['위험점수'].mean() for p in priorities]
avg_acc = [p['총사고건수'].mean() for p in priorities]
avg_inj = [p['총중상자수'].mean() for p in priorities]

x = np.arange(len(names))
width = 0.25

bars1 = ax2.bar(x - width, avg_risk, width, label='평균 위험점수', color='#2c3e50')
bars2 = ax2.bar(x, avg_acc, width, label='평균 사고건수', color='#e74c3c')
bars3 = ax2.bar(x + width, avg_inj, width, label='평균 중상자수', color='#f39c12')

ax2.set_xticks(x)
ax2.set_xticklabels(names, fontsize=12, fontweight='bold')
ax2.set_title('우선순위 그룹별 평균 피해 규모 비교', fontsize=15, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')


# (3) 시도별 1순위, 2순위 분포
ax3 = fig.add_subplot(2, 1, 2)

sido_p1 = p1['시도명'].value_counts()
sido_p2 = p2['시도명'].value_counts()

sido_df = pd.DataFrame({'1순위 타겟': sido_p1, '2순위 타겟': sido_p2}).fillna(0)
sido_df['총합'] = sido_df['1순위 타겟'] + sido_df['2순위 타겟']
sido_df = sido_df.sort_values('총합', ascending=False).drop('총합', axis=1)

sido_df.plot(kind='bar', stacked=True, ax=ax3, color=['#e74c3c', '#f39c12'], edgecolor='white')

ax3.set_title('시도별 카메라 우선 설치 대상지 (1·2순위) 분포', fontsize=15, fontweight='bold')
ax3.set_xlabel('')
ax3.set_ylabel('대상 지점 수')
ax3.tick_params(axis='x', rotation=0)
ax3.legend(fontsize=11)
ax3.grid(axis='y', alpha=0.3)

for i, (idx, row) in enumerate(sido_df.iterrows()):
    total = row['1순위 타겟'] + row['2순위 타겟']
    if total > 0:
        ax3.text(i, total + 5, f'{int(total)}', ha='center', fontsize=10, fontweight='bold')


plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, '08_reallocation_priority.png'), dpi=150, bbox_inches='tight')
plt.close()

print("→ 08_reallocation_priority.png 시각화 완료")
print("\n✅ 우선순위 도출 및 3번 솔루션 적용 완료!")
