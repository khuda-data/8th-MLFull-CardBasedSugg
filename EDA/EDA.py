import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1) 경로 설정 (여기만 너 PC 경로로)
# =========================
FILE = r"C:\Users\admin\Desktop\심화 프로젝트\suwon_feature_table_2022_2025_with_other.csv"
OUT_DIR = r"C:\Users\admin\Desktop\심화 프로젝트\eda_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# 2) 한글 폰트(윈도우)
# =========================
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

COL_KO = {
    "year_month": "기준연월",
    "admi_cty_no": "행정동코드",
    "age": "연령대코드",
    "total_amt": "총매출금액",
    "log_total_amt": "총매출금액_로그",
    "essential_share": "필수소비비중",
    "optional_share": "선택소비비중",
    "luxury_share": "사치소비비중",
    "high-risk_share": "유흥위험소비비중",
    "night_ratio": "야간소비비중",
    "weekend_ratio": "주말소비비중",
    "buz_hhi": "업종집중도_HHI",
    "buz_entropy": "업종다양성_엔트로피",
    "daily_cv": "일별소비변동계수_CV",
    "other_share": "기타소비비중"
}

# =========================
# 3) 로드 + 기본 정리
# =========================
df = pd.read_csv(FILE)

# year_month 처리(YYYY-MM)
df["year_month"] = pd.to_datetime(df["year_month"] + "-01", errors="coerce")
df = df.dropna(subset=["year_month"]).copy()

df["admi_cty_no"] = df["admi_cty_no"].astype(str)
df["age"] = df["age"].astype(int)

# 한글 컬럼 버전(복사본)도 만들어두면 보고서/시각화 편함
df_ko = df.rename(columns=COL_KO).copy()

# =========================
# 4) 데이터 품질 점검 (저장: quality_report.txt)
# =========================
report_lines = []
report_lines.append(f"행 수: {len(df):,}")
report_lines.append(f"기간: {df['year_month'].min().date()} ~ {df['year_month'].max().date()}")
report_lines.append(f"행정동 수: {df['admi_cty_no'].nunique():,}")
report_lines.append(f"연령대코드 수: {df['age'].nunique():,} / {sorted(df['age'].unique())}")

# 결측치
missing = df.isna().mean().sort_values(ascending=False)
missing.to_csv(os.path.join(OUT_DIR, "missing_ratio.csv"), encoding="utf-8-sig")
report_lines.append("\n[결측치 비율 상위 10]")
report_lines.extend((missing.head(10).round(4).astype(str)).to_list())

# 중복 체크(완전 동일 키)
dup = df.duplicated(subset=["year_month", "admi_cty_no", "age"]).sum()
report_lines.append(f"\n키 중복(연월×동×연령) 개수: {dup}")

# 비중합 체크 (필수+선택+사치+유흥위험 == 1 근처인지)
share_cols_5 = ["essential_share", "optional_share", "luxury_share", "high-risk_share", "other_share"]
df["share_sum_5"] = df[share_cols_5].sum(axis=1)

report_lines.append(
    f"\n(5분류) 비중합 평균: {df['share_sum_5'].mean():.4f}, 최소: {df['share_sum_5'].min():.4f}, 최대: {df['share_sum_5'].max():.4f}"
)
bad_sum_5 = ((df["share_sum_5"] < 0.97) | (df["share_sum_5"] > 1.03)).sum()
report_lines.append(f"(5분류) 비중합이 [0.97, 1.03] 밖인 행 수: {bad_sum_5}")

with open(os.path.join(OUT_DIR, "quality_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

# =========================
# 5) 핵심 지표 요약표 (연령대별 분위수/평균)
# =========================
KEY = [
    "essential_share","optional_share","luxury_share","high-risk_share",
    "other_share","night_ratio","weekend_ratio","buz_hhi","buz_entropy","daily_cv",
    "total_amt"
]

def quantile_summary(g):
    out = {}
    for c in KEY:
        s = g[c]
        out[f"{c}_mean"] = s.mean()
        out[f"{c}_median"] = s.median()
        out[f"{c}_p10"] = s.quantile(0.10)
        out[f"{c}_p90"] = s.quantile(0.90)
        out[f"{c}_p95"] = s.quantile(0.95)
    return pd.Series(out)

age_summary = df.groupby("age").apply(quantile_summary).reset_index()
age_summary.to_csv(os.path.join(OUT_DIR, "age_summary_quantiles.csv"), index=False, encoding="utf-8-sig")

# =========================
# 6) 분포 확인 그래프 (연령대별 박스플롯 4종)
# =========================
def boxplot_by_age(col, title):
    data = [df.loc[df["age"] == a, col].values for a in sorted(df["age"].unique())]
    plt.figure()
    plt.boxplot(data, labels=[str(a) for a in sorted(df["age"].unique())])
    plt.title(title)
    plt.xlabel("연령대코드")
    plt.ylabel(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"box_{col}.png"))
    plt.close()

boxplot_by_age("high-risk_share", "유흥위험소비비중")
boxplot_by_age("luxury_share", "사치소비비중")
boxplot_by_age("night_ratio", "야간소비비중")
boxplot_by_age("daily_cv", "일별소비변동계수(CV)")
boxplot_by_age("other_share", "기타소비비중") # 선택

# =========================
# 7) 시간 추이(전체 평균 + 연령별 평균 저장)
# =========================
monthly_overall = df.groupby("year_month")[KEY].mean().reset_index()
monthly_overall.to_csv(os.path.join(OUT_DIR, "monthly_overall_mean.csv"), index=False, encoding="utf-8-sig")

def plot_monthly_trend(col, title):
    tmp = df.groupby("year_month")[col].mean()
    plt.figure()
    plt.plot(tmp.index, tmp.values)
    plt.title(f"월별 평균 추이: {title}")
    plt.xlabel("연월")
    plt.ylabel(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"trend_overall_{col}.png"))
    plt.close()

plot_monthly_trend("high-risk_share", "유흥위험소비비중")
plot_monthly_trend("night_ratio", "야간소비비중")
plot_monthly_trend("buz_hhi", "업종집중도(HHI)")
plot_monthly_trend("daily_cv", "일별소비변동계수(CV)")

# 연령별 월평균 (표로 저장)
monthly_by_age = df.groupby(["age","year_month"])[KEY].mean().reset_index()
monthly_by_age.to_csv(os.path.join(OUT_DIR, "monthly_by_age_mean.csv"), index=False, encoding="utf-8-sig")

# =========================
# 8) 공간(행정동) 분석: 기간 평균 기준 TOP10
# =========================
dong_avg = df.groupby(["admi_cty_no","age"])[KEY].mean().reset_index()

def top10_table(col):
    top = dong_avg.sort_values(col, ascending=False).head(10)
    bot = dong_avg.sort_values(col, ascending=True).head(10)
    top.to_csv(os.path.join(OUT_DIR, f"top10_{col}.csv"), index=False, encoding="utf-8-sig")
    bot.to_csv(os.path.join(OUT_DIR, f"bottom10_{col}.csv"), index=False, encoding="utf-8-sig")

top10_table("high-risk_share")
top10_table("luxury_share")
top10_table("night_ratio")
top10_table("daily_cv")

# =========================
# 9) 지표 간 관계(상관계수 표)
# =========================
corr_cols = [
    "essential_share","optional_share","luxury_share","high-risk_share",
    "night_ratio","weekend_ratio","buz_hhi","buz_entropy","daily_cv","log_total_amt"
]
corr = df[corr_cols].corr()
corr.to_csv(os.path.join(OUT_DIR, "correlation_table.csv"), encoding="utf-8-sig")

print("EDA outputs saved to:", OUT_DIR)
