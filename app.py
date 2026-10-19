import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# --------------------------------------------------
# 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/it-teacher-go/"
    "data-share-/main/kobis_movies.csv"
)

RANDOM_STATE = 42
N_CLUSTERS = 3

FEATURES = [
    "스크린 수",
    "누적 관객",
    "10위권 일수",
    "롱런 지수",
]

FEATURE_COLUMNS = {
    "스크린 수": "first_scrn",
    "누적 관객": "total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run_index",
}

CLUSTER_NAMES = ["㉮", "㉯", "㉰"]
CLUSTER_COLORS = {
    "㉮": "#E45756",
    "㉯": "#4C78A8",
    "㉰": "#54A24B",
}


# --------------------------------------------------
# 제목
# --------------------------------------------------
st.title("🎬 영화 유형 나누기")
st.caption(
    "영화의 흥행 규모와 장기 상영 특성을 바탕으로 "
    "K-평균 알고리즘이 영화 유형을 세 그룹으로 분류합니다."
)


# --------------------------------------------------
# 데이터 불러오기 및 전처리
# --------------------------------------------------
@st.cache_data
def load_data(url):
    df = pd.read_csv(url, encoding="utf-8")

    required_columns = [
        "movieCd",
        "movieNm",
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "CSV에 필요한 열이 없습니다: "
            + ", ".join(missing_columns)
        )

    # 쉼표가 포함된 숫자도 변환
    numeric_columns = [
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col]
            .astype("string")
            .str.replace(",", "", regex=False)
            .str.strip(),
            errors="coerce",
        )

    # 전체 편수: 원본 CSV의 데이터 행 수
    total_movies = len(df)

    # 필요한 원본 값이 모두 있고, 첫 주 관객이 0보다 큰 영화만 유지
    valid_columns = [
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]

    valid = (
        df[valid_columns].notna().all(axis=1)
        & (df["first_week_audi"] > 0)
        & (df["first_scrn"] > 0)
        & (df["total_audi"] > 0)
        & (df["days_in_top10"] >= 0)
    )

    data = df.loc[valid].copy()

    # 파생 속성 계산
    data["long_run_index"] = (
        data["total_audi"] / data["first_week_audi"]
    ).clip(upper=20)

    # 모델에 사용할 네 가지 속성
    # 로그는 상용로그(log10)
    data["스크린 수"] = data["first_scrn"].map(
        lambda x: __import__("math").log10(x)
    )
    data["누적 관객"] = data["total_audi"].map(
        lambda x: __import__("math").log10(x)
    )
    data["10위권 일수"] = data["days_in_top10"]
    data["롱런 지수"] = data["long_run_index"]

    # 화면의 통계표에는 원래 단위 값을 사용
    data["스크린 수(원래 단위)"] = data["first_scrn"]
    data["누적 관객(원래 단위)"] = data["total_audi"]

    return data, total_movies


try:
    df, total_movies = load_data(DATA_URL)
except Exception as e:
    st.error(f"데이터를 불러오거나 처리하지 못했습니다: {e}")
    st.stop()

if df.empty:
    st.error("분석할 수 있는 영화 데이터가 없습니다.")
    st.stop()

st.write(
    f"**전체 영화 {total_movies:,}편 · "
    f"분석 대상 {len(df):,}편**"
)

st.caption(
    "분석 대상은 필수 값이 존재하고, 로그 변환할 값이 양수이며, "
    "첫 주 관객이 0보다 큰 영화입니다."
)


# --------------------------------------------------
# 군집화에 사용할 속성 선택
# --------------------------------------------------
st.subheader("1. 군집화 속성 선택")

selected_features = st.multiselect(
    "군집화에 사용할 속성을 선택하세요.",
    options=FEATURES,
    default=FEATURES,
    help="두 개 이상을 선택하는 것을 권장합니다. 기본값은 네 가지 속성 전체입니다.",
)

if not selected_features:
    st.warning("군집화를 위해 속성을 하나 이상 선택하세요.")
    st.stop()

# 선택한 속성으로 표준화 후 K-means 실행
X = df[selected_features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if len(df) < N_CLUSTERS:
    st.error("군집을 세 개로 나누려면 분석 대상 영화가 최소 3편 필요합니다.")
    st.stop()

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=10,
)

df["cluster_id"] = kmeans.fit_predict(X_scaled)

# 누적 관객 평균이 큰 군집부터 ㉮, ㉯, ㉰ 부여
cluster_order = (
    df.groupby("cluster_id")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

cluster_name_map = {
    cluster_id: CLUSTER_NAMES[rank]
    for rank, cluster_id in enumerate(cluster_order)
}

df["군집"] = df["cluster_id"].map(cluster_name_map)

st.success(
    f"군집화 완료: {len(df):,}편을 세 그룹으로 분류했습니다."
)


# --------------------------------------------------
# 2D 산점도
# --------------------------------------------------
st.subheader("2. 2차원 산점도")

if len(selected_features) >= 2:
    col_x, col_y = st.columns(2)

    with col_x:
        x_feature = st.selectbox(
            "가로축(X)",
            options=selected_features,
            index=0,
            key="2d_x",
        )

    with col_y:
        y_feature = st.selectbox(
            "세로축(Y)",
            options=selected_features,
            index=1,
            key="2d_y",
        )

    plot_df = df.copy()
    plot_df["군집"] = pd.Categorical(
        plot_df["군집"],
        categories=CLUSTER_NAMES,
        ordered=True,
    )

    fig_2d = px.scatter(
        plot_df,
        x=x_feature,
        y=y_feature,
        color="군집",
        color_discrete_map=CLUSTER_COLORS,
        hover_name="movieNm",
        hover_data={
            "군집": True,
            "first_scrn": ":,.0f",
            "total_audi": ":,.0f",
            "days_in_top10": ":,.0f",
            "long_run_index": ":.2f",
        },
        labels={
            "movieNm": "영화 제목",
            "군집": "영화 유형",
            "first_scrn": "스크린 수",
            "total_audi": "누적 관객",
            "days_in_top10": "10위권 일수",
            "long_run_index": "롱런 지수",
        },
        title="영화 유형별 2차원 분포",
    )

    fig_2d.update_traces(marker={"size": 7, "opacity": 0.75})
    fig_2d.update_layout(
        height=580,
        legend_title_text="영화 유형",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(fig_2d, use_container_width=True)
else:
    st.info("2차원 산점도를 그리려면 군집화 속성을 두 개 이상 선택하세요.")


# --------------------------------------------------
# 3D 산점도
# --------------------------------------------------
st.subheader("3. 3차원 산점도")

if len(selected_features) >= 3:
    col_x3, col_y3, col_z3 = st.columns(3)

    with col_x3:
        x3 = st.selectbox(
            "X축",
            options=selected_features,
            index=0,
            key="3d_x",
        )

    with col_y3:
        y3 = st.selectbox(
            "Y축",
            options=selected_features,
            index=1,
            key="3d_y",
        )

    with col_z3:
        z3 = st.selectbox(
            "Z축",
            options=selected_features,
            index=2,
            key="3d_z",
        )

    fig_3d = px.scatter_3d(
        df,
        x=x3,
        y=y3,
        z=z3,
        color="군집",
        color_discrete_map=CLUSTER_COLORS,
        hover_name="movieNm",
        hover_data={
            "군집": True,
            "first_scrn": ":,.0f",
            "total_audi": ":,.0f",
            "days_in_top10": ":,.0f",
            "long_run_index": ":.2f",
        },
        labels={
            "군집": "영화 유형",
            "first_scrn": "스크린 수",
            "total_audi": "누적 관객",
            "days_in_top10": "10위권 일수",
            "long_run_index": "롱런 지수",
        },
        title="영화 유형별 3차원 분포",
    )

    fig_3d.update_traces(
        marker={"size": 3, "opacity": 0.8}
    )
    fig_3d.update_layout(
        height=680,
        legend_title_text="영화 유형",
        margin=dict(l=0, r=0, t=60, b=0),
    )

    st.plotly_chart(fig_3d, use_container_width=True)
else:
    st.info(
        "3차원 산점도는 군집화 속성을 세 개 이상 선택했을 때 표시됩니다."
    )


# --------------------------------------------------
# 군집별 통계
# --------------------------------------------------
st.subheader("4. 군집별 통계")

# 네 속성의 평균을 원래 단위로 계산
summary_rows = []

for cluster_name in CLUSTER_NAMES:
    group = df[df["군집"] == cluster_name]

    if group.empty:
        continue

    summary_rows.append(
        {
            "군집": cluster_name,
            "영화 수": len(group),
            "스크린 수 평균": group["first_scrn"].mean(),
            "누적 관객 평균": group["total_audi"].mean(),
            "10위권 일수 평균": group["days_in_top10"].mean(),
            "롱런 지수 평균": group["long_run_index"].mean(),
        }
    )

summary = pd.DataFrame(summary_rows)

# 표시 형식은 원래 단위를 유지
display_summary = summary.copy()
display_summary["영화 수"] = display_summary["영화 수"].map(
    lambda x: f"{x:,}"
)
display_summary["스크린 수 평균"] = display_summary[
    "스크린 수 평균"
].map(lambda x: f"{x:,.1f}")
display_summary["누적 관객 평균"] = display_summary[
    "누적 관객 평균"
].map(lambda x: f"{x:,.0f}")
display_summary["10위권 일수 평균"] = display_summary[
    "10위권 일수 평균"
].map(lambda x: f"{x:,.1f}")
display_summary["롱런 지수 평균"] = display_summary[
    "롱런 지수 평균"
].map(lambda x: f"{x:.2f}")

st.dataframe(
    display_summary,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "군집 이름은 누적 관객 평균이 높은 순서로 지정했습니다. "
    "스크린 수와 누적 관객 평균은 로그 변환 전의 원래 단위입니다."
)


# --------------------------------------------------
# 군집별 누적 관객 상위 5편
# --------------------------------------------------
st.subheader("5. 군집별 누적 관객 상위 영화")

for cluster_name in CLUSTER_NAMES:
    group = (
        df[df["군집"] == cluster_name]
        .sort_values("total_audi", ascending=False)
        .head(5)
    )

    with st.expander(f"{cluster_name} 유형 — 누적 관객 상위 영화", expanded=True):
        if group.empty:
            st.write("해당 군집에 영화가 없습니다.")
        else:
            top_movies = group[["movieNm", "total_audi"]].copy()
            top_movies.columns = ["영화 제목", "누적 관객"]
            top_movies["누적 관객"] = top_movies["누적 관객"].map(
                lambda x: f"{x:,.0f}명"
            )
            top_movies.insert(
                0,
                "순위",
                range(1, len(top_movies) + 1),
            )

            st.dataframe(
                top_movies,
                use_container_width=True,
                hide_index=True,
            )


# --------------------------------------------------
# 속성 안내
# --------------------------------------------------
with st.expander("속성 계산 및 군집화 방법"):
    st.markdown(
        """
        - **스크린 수:** 원본 `first_scrn`에 상용로그(log10)를 적용합니다.
        - **누적 관객:** 원본 `total_audi`에 상용로그(log10)를 적용합니다.
        - **10위권 일수:** 원본 `days_in_top10`을 그대로 사용합니다.
        - **롱런 지수:** `total_audi / first_week_audi`로 계산하며, 20을 초과하면 20으로 제한합니다.
        - **결측치 처리:** 필요한 원본 값이 누락된 영화와 첫 주 관객이 0인 영화는 제외합니다. 로그 계산을 위해 스크린 수와 누적 관객이 양수인 영화만 사용합니다.
        - **표준화:** 선택한 속성만 `StandardScaler`로 표준화합니다.
        - **군집화:** K-평균, 군집 수 3, `random_state=42`, `n_init=10`을 사용합니다.
        - **재현성:** 데이터와 라이브러리 환경이 동일하면 동일한 군집 결과를 얻을 수 있도록 난수를 고정했습니다.
        """
    )
