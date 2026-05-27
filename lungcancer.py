import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import kagglehub
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -----------------------------------
# 한글 폰트 설정
# -----------------------------------

try:
    font_path = "C:/Windows/Fonts/malgun.ttf"

    font_name = fm.FontProperties(
        fname=font_path
    ).get_name()

    plt.rc('font', family=font_name)

    sns.set(font=font_name)

except:
    pass

plt.rcParams['axes.unicode_minus'] = False

# -----------------------------------
# 페이지 설정
# -----------------------------------

st.set_page_config(
    page_title="폐암 환자 군집 분석",
    page_icon="🫁",
    layout="wide"
)

# -----------------------------------
# 제목
# -----------------------------------

st.title("🫁 폐암 환자 군집 분석")
st.markdown("### K-Means 군집화를 활용한 데이터 분석")

# -----------------------------------
# 데이터 불러오기
# -----------------------------------

@st.cache_data
def load_data():

    # 캐글 데이터 다운로드
    path = kagglehub.dataset_download(
        "yusufdede/lung-cancer-dataset"
    )

    # CSV 자동 탐색
    files = os.listdir(path)

    csv_file = [
        f for f in files
        if f.endswith(".csv")
    ][0]

    # CSV 경로
    csv_path = os.path.join(
        path,
        csv_file
    )

    # 데이터 읽기
    df = pd.read_csv(csv_path)

    return df

try:

    df = load_data()

    st.success("✅ Kaggle 데이터 불러오기 완료!")

    # -----------------------------------
    # 원본 데이터
    # -----------------------------------

    st.subheader("📄 원본 데이터")

    st.dataframe(df)

    # -----------------------------------
    # 컬럼 정보
    # -----------------------------------

    st.subheader("📌 컬럼 목록")

    st.write(df.columns)

    # -----------------------------------
    # 사용할 변수
    # -----------------------------------

    features = [
        'Age',
        'Smokes',
        'AreaQ',
        'Alkhol'
    ]

    X = df[features]

    # -----------------------------------
    # 데이터 스케일링
    # -----------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # -----------------------------------
    # 사이드바
    # -----------------------------------

    st.sidebar.header("⚙️ 설정")

    k = st.sidebar.slider(
        "군집 개수 선택",
        2,
        6,
        3
    )

    # -----------------------------------
    # KMeans 모델
    # -----------------------------------

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    df["cluster"] = model.labels_

    # -----------------------------------
    # 실루엣 점수
    # -----------------------------------

    score = silhouette_score(
        X_scaled,
        df["cluster"]
    )

    st.subheader("🎯 모델 성능")

    st.metric(
        "실루엣 점수",
        round(score, 3)
    )

    # -----------------------------------
    # 군집별 평균
    # -----------------------------------

    st.subheader("📊 군집별 평균")

    cluster_mean = df.groupby(
        "cluster"
    )[features].mean()

    st.dataframe(cluster_mean)

    # -----------------------------------
    # 산점도
    # -----------------------------------

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
    ax.set_title("폐암 환자 군집 분석")

    st.pyplot(fig)

    # -----------------------------------
    # 히트맵
    # -----------------------------------

    st.subheader("🔥 상관관계 히트맵")

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

    # -----------------------------------
    # 새 환자 예측
    # -----------------------------------

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

        # 새 환자 위치 시각화

        fig3, ax3 = plt.subplots(
            figsize=(8, 6)
        )

        ax3.scatter(
            df["Age"],
            df["Smokes"],
            c=df["cluster"],
            cmap="viridis",
            alpha=0.5,
            s=80
        )

        ax3.scatter(
            age,
            smokes,
            color="red",
            marker="X",
            s=300
        )

        ax3.set_xlabel("나이")
        ax3.set_ylabel("흡연량")
        ax3.set_title("새 환자 위치")

        st.pyplot(fig3)

except Exception as e:

    st.error(f"오류 발생: {e}")
