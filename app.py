import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import ta

# 페이지 설정
st.set_page_config(
    page_title="📈 AI 주식 예측기",
    page_icon="📈",
    layout="wide"
)

# 제목
st.title("📈 AI 주식 가격 예측기")
st.markdown("---")

# 사이드바 - 주식 선택
st.sidebar.header("🔍 주식 선택")

# 인기 주식 목록
popular_stocks = {
    "Apple": "AAPL",
    "Microsoft": "MSFT", 
    "Google": "GOOGL",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
    "Samsung": "005930.KS",
    "NAVER": "035420.KS",
    "Kakao": "035720.KS"
}

# 주식 선택 방법
selection_method = st.sidebar.radio(
    "선택 방법:",
    ["인기 주식에서 선택", "직접 입력"]
)

if selection_method == "인기 주식에서 선택":
    selected_stock_name = st.sidebar.selectbox(
        "주식을 선택하세요:",
        list(popular_stocks.keys())
    )
    ticker = popular_stocks[selected_stock_name]
else:
    ticker = st.sidebar.text_input(
        "주식 심볼을 입력하세요:",
        value="AAPL",
        help="예: AAPL, MSFT, 005930.KS"
    )
    selected_stock_name = ticker

# 기간 선택
period = st.sidebar.selectbox(
    "분석 기간:",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=3
)

# 데이터 로드 함수
@st.cache_data
def load_stock_data(ticker, period):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        return data, stock.info
    except Exception as e:
        st.error(f"데이터를 불러올 수 없습니다: {e}")
        return None, None

# 메인 컨텐츠
if st.sidebar.button("📊 분석 시작", type="primary"):
    with st.spinner("데이터를 불러오는 중..."):
        data, info = load_stock_data(ticker, period)
    
    if data is not None and not data.empty:
        # 기본 정보 표시
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100
        
        with col1:
            st.metric(
                label="현재 가격",
                value=f"${current_price:.2f}",
                delta=f"{price_change:.2f} ({price_change_pct:+.2f}%)"
            )
        
        with col2:
            st.metric(
                label="최고가",
                value=f"${data['High'].max():.2f}"
            )
        
        with col3:
            st.metric(
                label="최저가", 
                value=f"${data['Low'].min():.2f}"
            )
        
        with col4:
            st.metric(
                label="평균 거래량",
                value=f"{data['Volume'].mean():,.0f}"
            )
        
        # 가격 차트
        st.subheader("📈 주가 차트")
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="주가"
        ))
        
        fig.update_layout(
            title=f"{selected_stock_name} 주가 차트",
            yaxis_title="가격 ($)",
            xaxis_title="날짜",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # AI 예측 섹션
        st.subheader("🤖 AI 가격 예측")
        
        # 기술적 지표 계산
        data['MA_5'] = data['Close'].rolling(window=5).mean()
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
        data['MACD'] = ta.trend.MACD(data['Close']).macd()
        
        # 피처 준비
        features = ['Open', 'High', 'Low', 'Volume', 'MA_5', 'MA_20', 'RSI', 'MACD']
        data_clean = data[features + ['Close']].dropna()
        
        if len(data_clean) > 30:  # 충분한 데이터가 있을 때만 예측
            X = data_clean[features]
            y = data_clean['Close']
            
            # 훈련/테스트 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 모델 훈련
            with st.spinner("AI 모델을 훈련하는 중..."):
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
            
            # 예측
            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            
            # 다음 날 예측
            last_features = X.iloc[-1:].values
            next_day_prediction = model.predict(last_features)[0]
            
            # 결과 표시
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="다음 거래일 예상 가격",
                    value=f"${next_day_prediction:.2f}",
                    delta=f"{next_day_prediction - current_price:.2f}"
                )
            
            with col2:
                st.metric(
                    label="모델 정확도 (MAE)",
                    value=f"${mae:.2f}"
                )
            
            # 예측 신뢰도
            confidence = max(0, 100 - (mae / current_price * 100))
            st.progress(confidence / 100)
            st.caption(f"예측 신뢰도: {confidence:.1f}%")
            
        else:
            st.warning("AI 예측을 위해서는 더 많은 데이터가 필요합니다.")
        
        # 기술적 지표
        st.subheader("📊 기술적 지표")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'RSI' in data.columns:
                rsi_value = data['RSI'].iloc[-1]
                st.metric("RSI", f"{rsi_value:.1f}")
                if rsi_value > 70:
                    st.warning("⚠️ 과매수 구간")
                elif rsi_value < 30:
                    st.info("💡 과매도 구간")
        
        with col2:
            if 'MA_20' in data.columns:
                ma20 = data['MA_20'].iloc[-1]
                st.metric("20일 이동평균", f"${ma20:.2f}")
                if current_price > ma20:
                    st.success("📈 상승 추세")
                else:
                    st.error("📉 하락 추세")
        
    else:
        st.error("❌ 주식 데이터를 불러올 수 없습니다. 심볼을 확인해주세요.")

else:
    # 초기 화면
    st.info("👈 사이드바에서 주식을 선택하고 '분석 시작' 버튼을 클릭하세요!")
    
    st.markdown("""
    ### 🚀 주요 기능:
    - **실시간 주식 데이터** 조회
    - **AI 기반 가격 예측**
    - **기술적 지표** 분석 (RSI, 이동평균 등)
    - **인터랙티브 차트** 시각화
    
    ### 📈 지원 주식:
    - 미국 주식 (Apple, Microsoft, Google 등)
    - 한국 주식 (삼성전자, 네이버, 카카오 등)
    - 직접 심볼 입력으로 전세계 주식 조회 가능
    """)

# 푸터
st.markdown("---")
st.markdown("💡 **주의**: 이 예측은 참고용이며, 투자 결정시 신중하게 판단하세요.")
