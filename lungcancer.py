import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -------------------------------
# 페이지 설정
# -------------------------------

st.set_page_config(
    page_title="폐암 환자 군집 분석",
    page_icon="🫁",
    layout="wide"
)

# -------------------------------
# 제목
# -------------------------------

st.title("🫁 폐암 환자 군집 분석")
st.markdown("### K-Means 군집화를 활용한 데이터 분석")

# -------------------------------
# 캐글 데이터 불러오기
# -------------------------------

@st.cache_data
def load_data():

    # 데이터 다운로드
    path = kagglehub.dataset_download(
        "yusufdede/lung-cancer-dataset"
    )

    # CSV 경로 설정
    csv_path = os.path.join(
        path,
        "lung_cancer_examples.csv"
    )

    # 데이터 읽기
    df = pd.read_csv(csv_path)

    return df

try:

    df = load_data()

    st.success("✅ 캐글 데이터 불러오기 완료!")

    # -------------------------------
    # 데이터 확인
    # -------------------------------

    st.subheader("📄 원본 데이터")

    st.dataframe(df)

    # -------------------------------
    # 컬럼 확인
    # -------------------------------

    st.subheader("📌 컬럼 정보")

    st.write(df.columns)

    # -------------------------------
    # 문자 데이터 숫자로 변환
    # -------------------------------

    binary_cols = [
        'GENDER',
        'LUNG_CANCER',
        'SMOKING',
        'YELLOW_FINGERS',
        'ANXIETY',
        'PEER_PRESSURE',
        'CHRONIC DISEASE',
        'FATIGUE ',
        'ALLERGY ',
        'WHEEZING',
        'ALCOHOL CONSUMING',
        'COUGHING',
        'SHORTNESS OF BREATH',
        'SWALLOWING DIFFICULTY',
        'CHEST PAIN'
    ]

    for col in binary_cols:

        df[col] = df[col].replace({
            'M': 1,
            'F': 0,
            'YES': 1,
            'NO': 0
        })

    # -------------------------------
    # 사용할 변수 선택
    # -------------------------------

    features = [
        'AGE',
        'SMOKING',
        'ALCOHOL CONSUMING',
        'COUGHING',
        'CHEST PAIN'
    ]

    X = df[features]

    # -------------------------------
    # 데이터 스케일링
    # -------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # -------------------------------
    # 사이드바 설정
    # -------------------------------

    st.sidebar.header("⚙️ 설정")

    k = st.sidebar.slider(
        "군집 개수 선택",
        2,
        6,
        3
    )

    # -------------------------------
    # KMeans 모델
    # -------------------------------

    model = KMeans(
        n_clusters=k,
        random_state=42
    )

    model.fit(X_scaled)

    df["cluster"] = model.labels_

    # -------------------------------
    # 실루엣 점수
    # -------------------------------

    score = silhouette_score(
        X_scaled,
        df["cluster"]
    )

    st.subheader("🎯 모델 성능")

    st.metric(
        "실루엣 점수",
        round(score, 3)
    )

    # -------------------------------
    # 군집별 평균
    # -------------------------------

    st.subheader("📊 군집별 평균")

    cluster_mean = df.groupby("cluster")[features].mean()

    st.dataframe(cluster_mean)

    # -------------------------------
    # 군집 시각화
    # -------------------------------

    st.subheader("🎨 군집 시각화")

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        df["AGE"],
        df["ALCOHOL CONSUMING"],
        c=df["cluster"],
        cmap="viridis",
        s=80
    )

    ax.set_xlabel("나이")
    ax.set_ylabel("음주 여부")
    ax.set_title("폐암 환자 군집 분석")

    st.pyplot(fig)

    # -------------------------------
    # 히트맵
    # -------------------------------

    st.subheader("🔥 상관관계 히트맵")

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        df[features].corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax2
    )

    st.pyplot(fig2)

    # -------------------------------
    # 새로운 환자 예측
    # -------------------------------

    st.subheader("🧪 새로운 환자 군집 예측")

    age = st.slider(
        "나이",
        1,
        100,
        40
    )

    smoking = st.selectbox(
        "흡연 여부",
        [0, 1]
    )

    alcohol = st.selectbox(
        "음주 여부",
        [0, 1]
    )

    coughing = st.selectbox(
        "기침 여부",
        [0, 1]
    )

    chest_pain = st.selectbox(
        "가슴 통증 여부",
        [0, 1]
    )

    if st.button("🔍 예측하기"):

        new_data = pd.DataFrame(
            [[
                age,
                smoking,
                alcohol,
                coughing,
                chest_pain
            ]],
            columns=features
        )

        new_scaled = scaler.transform(new_data)

        pred = model.predict(new_scaled)

        st.success(
            f"✅ 이 환자는 {pred[0]}번 군집입니다!"
        )

except Exception as e:

    st.error(f"오류 발생: {e}")
