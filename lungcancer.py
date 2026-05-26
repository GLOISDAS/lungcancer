import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -------------------------
# 페이지 설정
# -------------------------

st.set_page_config(
    page_title="폐암 환자 군집 분석",
    page_icon="🫁",
    layout="wide"
)

st.title("🫁 폐암 환자 군집 분석")
st.markdown("### K-Means 군집화")

# -------------------------
# 데이터 불러오기
# -------------------------

@st.cache_data
def load_data():

    # 캐글 데이터 다운로드
    path = kagglehub.dataset_download(
        "yusufdede/lung-cancer-dataset"
    )

    # CSV 자동 찾기
    files = os.listdir(path)

    csv_file = [
        f for f in files
        if f.endswith(".csv")
    ][0]

    csv_path = os.path.join(
        path,
        csv_file
    )

    # 데이터 읽기
    df = pd.read_csv(csv_path)

    return df

try:

    df = load_data()

    st.success("✅ 데이터 불러오기 완료!")

    # -------------------------
    # 데이터 보기
    # -------------------------

    st.subheader("📄 원본 데이터")

    st.dataframe(df)

    # -------------------------
    # 컬럼 확인
    # -------------------------

    st.subheader("📌 컬럼 목록")

    st.write(df.columns)

    # -------------------------
    # 사용할 변수
    # -------------------------

    features = [
        'Age',
        'Smokes',
        'AreaQ',
        'Alkhol'
    ]

    X = df[features]

    # -------------------------
    # 스케일링
    # -------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # -------------------------
    # 사이드바
    # -------------------------

    st.sidebar.header("⚙️ 설정")

    k = st.sidebar.slider(
        "군집 개수",
        2,
        6,
        3
    )

    # -------------------------
    # 모델 생성
    # -------------------------

    model = KMeans(
        n_clusters=k,
        random_state=42
    )

    model.fit(X_scaled)

    df["cluster"] = model.labels_

    # -------------------------
    # 실루엣 점수
    # -------------------------

    score = silhouette_score(
        X_scaled,
        df["cluster"]
    )

    st.subheader("🎯 모델 성능")

    st.metric(
        "실루엣 점수",
        round(score, 3)
    )

    # -------------------------
    # 군집별 평균
    # -------------------------

    st.subheader("📊 군집별 평균")

    cluster_mean = df.groupby(
        "cluster"
    )[features].mean()

    st.dataframe(cluster_mean)

    # -------------------------
    # 산점도
    # -------------------------

    st.subheader("🎨 군집 시각화")

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    scatter = ax.scatter(
        df["Age"],
        df["Smokes"],
        c=df["cluster"],
        cmap="viridis",
        s=80
    )

    ax.set_xlabel("나이")
    ax.set_ylabel("흡연량")
    ax.set_title("폐암 환자 군집")

    st.pyplot(fig)

    # -------------------------
    # 히트맵
    # -------------------------

    st.subheader("🔥 상관관계")

    fig2, ax2 = plt.subplots(
        figsize=(8, 5)
    )

    sns.heatmap(
        df[features].corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax2
    )

    st.pyplot(fig2)

    # -------------------------
    # 새 환자 예측
    # -------------------------

    st.subheader("🧪 새 환자 군집 예측")

    age = st.slider(
        "나이",
        1,
        100,
        40
    )

    smokes = st.slider(
        "흡연량",
        0,
        50,
        5
    )

    areaq = st.slider(
        "공기질 점수",
        0,
        10,
        5
    )

    alkhol = st.slider(
        "음주량",
        0,
        10,
        3
    )

    if st.button("🔍 예측하기"):

        new_data = pd.DataFrame(
            [[
                age,
                smokes,
                areaq,
                alkhol
            ]],
            columns=features
        )

        new_scaled = scaler.transform(
            new_data
        )

        pred = model.predict(
            new_scaled
        )

        st.success(
            f"✅ 예측 군집: {pred[0]}"
        )

except Exception as e:

    st.error(f"오류 발생: {e}")