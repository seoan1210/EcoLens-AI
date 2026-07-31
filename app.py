import os
import json
import base64
from typing import List, Optional
from PIL import Image
import streamlit as st
from pydantic import BaseModel, Field
from groq import Groq
from tavily import TavilyClient

# ==============================================================================
# 1. Pydantic 데이터 모델 정의
# ==============================================================================

class SourceItem(BaseModel):
    title: str = Field(description="출처 기관명 또는 문서 제목")
    url: str = Field(description="원문 URL")
    published_or_checked: str = Field(description="확인일자 또는 작성일자")

class RecyclingGuide(BaseModel):
    item_name: str = Field(description="판별된 정확한 품목명")
    material: str = Field(description="추정 재질 (예: PET, PP, 종이팩, 유리, 복합재질 등)")
    category: str = Field(description="대분류 (예: 플라스틱, 비닐류, 일반쓰레기, 대형폐기물 등)")
    confidence: str = Field(description="판별 신뢰도 ('high', 'medium', 'low')")
    steps: List[str] = Field(description="실행형 3단계 배출 절차", min_items=1, max_items=4)
    cautions: List[str] = Field(description="배출 시 주의사항 및 예외 조건")
    local_note: Optional[str] = Field(default=None, description="지역별 전용 수거 기준 관련 참고사항")
    sources: List[SourceItem] = Field(default=[], description="검색된 근거 정보 출처 목록")
    eco_tip: str = Field(description="친환경 실천 팁 및 기대 효과")
    needs_confirmation: bool = Field(default=False, description="지자체/상세 재질 추가 확인 필요 여부")


# ==============================================================================
# 2. Groq & Tavily RAG 서비스 클래스
# ==============================================================================

class RecyclingService:
    def __init__(self, groq_api_key: str, tavily_api_key: str):
        self.groq_client = Groq(api_key=groq_api_key)
        self.tavily_client = TavilyClient(api_key=tavily_api_key)

    def search_local_rules(self, query: str) -> list[dict]:
        """Tavily를 통해 공공/지자체 출처 위주로 최신 분리배출 규정을 검색합니다."""
        try:
            response = self.tavily_client.search(
                query=query,
                search_depth="basic",
                max_results=3,
                include_domains=["go.kr", "or.kr", "seoul.go.kr", "keco.or.kr"]
            )
            results = response.get("results", [])
            if not results:
                response = self.tavily_client.search(query=query, search_depth="basic", max_results=3)
                results = response.get("results", [])
            return results
        except Exception as e:
            st.warning(f"Tavily 검색 연결 오류 (기본 규정으로 대체 진행): {e}")
            return []

    def encode_image_to_base64(self, image: Image.Image) -> str:
        """PIL Images를 Base64 문자열로 변환합니다."""
        import io
        buffered = io.BytesIO()
        image.convert("RGB").save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def analyze_and_guide(
        self,
        item_text: Optional[str] = None,
        image: Optional[Image.Image] = None,
        location: str = "전국 공통"
    ) -> RecyclingGuide:
        
        system_instruction = f"""
        당신은 대한민국 환경부 및 지자체 분리배출 기준에 정통한 친환경 AI 도우미 'EcoLens AI'입니다.
        사용자의 입력(이미지 및 텍스트)을 분석하여 올바른 분리배출 가이드를 반드시 주어진 JSON 스키마 형식에 맞춰 제공하세요.
        
        [지침]
        1. 이미지나 텍스트가 불명확하면 confidence를 'low'로 설정하고 needs_confirmation을 true로 설정하세요.
        2. 검색 컨텍스트에 나와있지 않은 지자체 특화 규정은 절대 지어내지 말고, 불확실하면 지자체 확인 필요라고 안내하세요.
        3. steps는 사용자가 바로 행동할 수 있는 3단계 절차로 작성하세요. (예: 비운다 -> 헹군다 -> 분리하여 배출한다)
        
        [반환할 JSON 구조]
        {json.dumps(RecyclingGuide.model_json_schema(), ensure_ascii=False, indent=2)}
        """

        # 1. Tavily 검색 질의 생성 및 실행
        search_query = f"{location} {item_text or '쓰레기'} 분리배출 기준 2026"
        search_results = self.search_local_rules(search_query)
        context_str = "\n".join([f"- 제목: {r['title']}\n  URL: {r['url']}\n  내용: {r['content']}" for r in search_results])

        user_content = [
            {
                "type": "text",
                "text": f"""[사용자 입력]
지역: {location}
물품 설명: {item_text or '사진 참고'}

[검색된 최신 지자체/공공 정보]
{context_str if context_str else "검색 결과 없음. 기본 전국 공통 기준 적용."}

위 정보를 바탕으로 스키마 형태의 JSON 응답만 출력하세요."""
            }
        ]

        # 2. 이미지 유무에 따른 모델 및 메시지 구성 (비전 이미지 지원)
        if image:
            base64_image = self.encode_image_to_base64(image)
            user_content.insert(0, {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
            model_name = "llama-3.2-11b-vision-preview" # Groq 이미지 지원 비전 모델
        else:
            model_name = "llama-3.3-70b-versatile"     # Groq 텍스트 고성능 모델

        # 3. Groq API 호출
        response = self.groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )

        json_response = response.choices[0].message.content
        return RecyclingGuide.model_validate_json(json_response)


# ==============================================================================
# 3. Streamlit 웹 UI
# ==============================================================================

st.set_page_config(
    page_title="EcoLens AI — AI 분리배출 도우미 (Groq)",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #F7F5F0; }
    .stApp { max-width: 1200px; margin: 0 auto; }
    .guide-card {
        background: rgba(255, 253, 248, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(45, 90, 39, 0.15);
        box-shadow: 0 12px 32px rgba(26,28,25,0.06);
        margin-bottom: 20px;
    }
    .step-number {
        font-weight: bold;
        color: #2D5A27;
        font-size: 1.2rem;
    }
    .badge {
        background-color: #2D5A27;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# API 키 불러오기
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY") or os.environ.get("TAVILY_API_KEY")

st.title("🌱 EcoLens AI (Powered by Groq)")
st.caption("“찍고, 묻고, 제대로 버리다.” — AI 기반 맞춤형 분리배출 도우미")

if not GROQ_API_KEY or not TAVILY_API_KEY:
    st.error("⚠️ API 키가 설정되지 않았습니다. `.streamlit/secrets.toml` 또는 환경 변수에 `GROQ_API_KEY`와 `TAVILY_API_KEY`를 추가해주세요.")
    st.info("""
    **`secrets.toml` 설정 예시:**
    ```toml
    GROQ_API_KEY = "gsk_your_groq_api_key"
    TAVILY_API_KEY = "tvly-your_tavily_api_key"
    ```
    """)
    st.stop()

service = RecyclingService(groq_api_key=GROQ_API_KEY, tavily_api_key=TAVILY_API_KEY)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 🔍 품목 분석 입력")
    
    input_type = st.radio("입력 방식을 선택하세요", ["📷 사진 촬영/업로드", "✍️ 물건명 텍스트 입력"], horizontal=True)
    
    uploaded_image = None
    item_text = None
    
    if "📷 사진" in input_type:
        img_file = st.file_uploader("물품 사진을 올려주세요 (JPEG, PNG, WebP)", type=["jpg", "jpeg", "png", "webp"])
        if img_file:
            uploaded_image = Image.open(img_file)
            st.image(uploaded_image, caption="업로드된 이미지", use_container_width=True)
    else:
        item_text = st.text_input("버리려는 물건의 이름을 입력하세요", placeholder="예: 투명 페트병, 배달용기, 햇반 용기")

    location = st.selectbox(
        "배출 지역을 선택하세요 (선택)",
        ["전국 공통", "서울특별시 강남구", "서울특별시 마포구", "경기도 수원시", "부산광역시 해운대구", "대구광역시 수성구", "인천광역시 연수구"],
        index=0
    )

    analyze_btn = st.button("✨ 분리배출 방법 분석하기", type="primary", use_container_width=True)

with col2:
    st.markdown("### 📋 분석 결과 및 배출 가이드")
    
    if analyze_btn:
        if not uploaded_image and not item_text:
            st.warning("사진을 업로드하거나 물건 이름을 입력해주세요.")
        else:
            with st.spinner("Groq AI가 초고속으로 물체를 분석하고 최신 지역 규정을 반영하는 중입니다..."):
                try:
                    guide = service.analyze_and_guide(
                        item_text=item_text,
                        image=uploaded_image,
                        location=location
                    )

                    st.markdown(f"""
                    <div class="guide-card">
                        <h2>{guide.item_name} <span class="badge">{guide.category}</span></h2>
                        <p><strong>추정 재질:</strong> {guide.material} | <strong>신뢰도:</strong> {guide.confidence.upper()}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if guide.needs_confirmation:
                        st.info("💡 **확인 필요:** 복합재질이거나 사진이 모호합니다. 용기 바닥의 재질 표기(PET, PP, OTHER 등)를 한 번 더 확인해주세요.")

                    st.markdown("#### 🛠️ 3단계 배출 프로토콜")
                    for idx, step in enumerate(guide.steps, 1):
                        st.markdown(f"**<span class='step-number'>0{idx}</span>** {step}", unsafe_allow_html=True)

                    if guide.cautions:
                        st.markdown("#### ⚠️ 주의사항")
                        for caution in guide.cautions:
                            st.warning(f"• {caution}")

                    if guide.local_note:
                        st.markdown("#### 📍 지역 수거 참고사항")
                        st.info(f"• {guide.local_note}")

                    st.markdown("#### 🌱 친환경 실천 팁")
                    st.success(f"🍃 {guide.eco_tip}")

                    if guide.sources:
                        with st.expander("🔗 최신 근거 및 공공 출처 보기"):
                            for src in guide.sources:
                                st.write(f"- [{src.title}]({src.url}) (확인일: {src.published_or_checked})")

                except Exception as e:
                    st.error(f"분석 중 오류가 발생했습니다: {e}")
    else:
        st.info("👈 좌측에서 사진이나 텍스트를 입력하고 분석 버튼을 눌러주세요.")
