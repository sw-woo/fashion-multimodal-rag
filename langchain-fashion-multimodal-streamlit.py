# 베포시 chroma db에서 sqlite3를 사용하는데 오류가 나서 추가하였습니다.
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import os
import base64
import streamlit as st
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from io import BytesIO

# .env 파일에서 환경 변수 로드
load_dotenv()

# 환경 변수에서 OpenAI API 키 가져오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# API 키가 설정되지 않았을 경우 에러 발생
if not OPENAI_API_KEY:
    raise ValueError(".env 파일에 OPENAI_API_KEY가 설정되지 않았습니다.")

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 데이터셋을 설정하는 함수
def setup_dataset(num_samples=50):
    # 패션 관련 데이터셋 불러오기 (비스트리밍 모드)
    dataset = load_dataset("detection-datasets/fashionpedia", split=f"train[:{num_samples}]")
    return dataset

# 데이터셋에서 이미지를 가져오는 함수
def get_images_from_dataset(dataset):
    images = []
    for i, sample in enumerate(dataset):
        image = sample['image']
        # 이미지 크기를 조정하여 메모리 사용량 줄이기
        image = image.resize((128, 128))
        images.append(image)
    return images

# Chroma 데이터베이스를 설정하는 함수
def setup_chroma_db():
    # Chroma 클라이언트 초기화 (인메모리 모드)
    chroma_client = chromadb.Client()
    # OpenCLIP 임베딩 함수 설정
    clip = OpenCLIPEmbeddingFunction()
    # 이미지 데이터베이스 생성 또는 가져오기
    image_vdb = chroma_client.get_or_create_collection(
        name="Streamlit", embedding_function=clip)
    return image_vdb

# 이미지를 데이터베이스에 추가하는 함수
def add_images_to_db(image_vdb, images):
    ids = []
    if len(image_vdb.get()['ids']) > 0:
        print("이미지가 이미 데이터베이스에 추가되어 있습니다.")
        return
    for i, image in enumerate(images):
        img_id = str(i)
        ids.append(img_id)
    if ids:
        image_vdb.add(ids=ids, images=images)
        print("새로운 이미지를 데이터베이스에 추가했습니다.")
    else:
        print("추가할 새로운 이미지가 없습니다.")

# 데이터베이스에서 쿼리를 실행하는 함수
def query_db(image_vdb, query, results=2):
    # 주어진 쿼리를 실행하고, 상위 결과 반환
    return image_vdb.query(
        query_texts=[query],
        n_results=results,
        include=['images', 'ids', 'distances'])

# 텍스트를 지정된 언어로 번역하는 함수
def translate(text, target_lang):
    # OpenAI의 ChatGPT 모델을 사용하여 번역
    translation_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    # 번역에 사용할 프롬프트 생성
    translation_prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a translator. Translate the following text to {target_lang}."),
        ("user", "{text}")
    ])
    # 번역 체인 설정
    translation_chain = translation_prompt | translation_model | StrOutputParser()
    # 번역 결과 반환
    return translation_chain.invoke({"text": text})

# 시각적 정보를 처리하는 체인을 설정하는 함수
def setup_vision_chain():
    # GPT-4 모델을 사용하여 시각적 정보를 처리
    gpt4 = ChatOpenAI(model="gpt-4", temperature=0.0)
    parser = StrOutputParser()
    image_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful fashion and styling assistant. Answer the user's question using the given image context with direct references to parts of the images provided. Maintain a more conversational tone, don't make too many lists. Use markdown formatting for highlights, emphasis, and structure."),
        ("user", [
            {"type": "text", "text": "What are some ideas for styling {user_query}"},
            {"type": "image_url", "image_url": "data:image/png;base64,{image_data_1}"},
            {"type": "image_url", "image_url": "data:image/png;base64,{image_data_2}"},
        ]),
    ])
    # 프롬프트, 모델, 파서 체인을 반환
    return image_prompt | gpt4 | parser

# 프롬프트 입력을 포맷하는 함수
def format_prompt_inputs(data, user_query):
    print(f"result data: {data}")
    inputs = {}

    # 사용자 쿼리를 딕셔너리에 추가
    inputs['user_query'] = user_query

    image_data_1 = data['images'][0][0]
    image_data_2 = data['images'][0][1]

    # 이미지 데이터를 Base64로 인코딩
    buffered = BytesIO()
    image_data_1.save(buffered, format="PNG")
    inputs['image_data_1'] = base64.b64encode(buffered.getvalue()).decode('utf-8')

    buffered = BytesIO()
    image_data_2.save(buffered, format="PNG")
    inputs['image_data_2'] = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return inputs

# Streamlit 앱을 실행하는 메인 함수
def main():
    st.set_page_config(page_title="FashionRAG", layout="wide")
    st.title("FashionRAG: 패션 및 스타일링 어시스턴트")

    with st.spinner("데이터셋 설정 및 이미지 로딩 중..."):
        dataset = setup_dataset(num_samples=50)
        images = get_images_from_dataset(dataset)
    st.success("데이터셋 설정 및 이미지 로딩이 완료되었습니다.")

    with st.spinner("벡터 데이터베이스 설정 및 이미지 추가 중..."):
        image_vdb = setup_chroma_db()
        add_images_to_db(image_vdb, images)
    st.success("벡터 데이터베이스 설정 및 이미지 추가가 완료되었습니다.")

    vision_chain = setup_vision_chain()

    st.header("스타일링 조언을 받아보세요")

    query_ko = st.text_input("스타일링에 대한 질문을 입력하세요:")

    if query_ko:
        with st.spinner("번역 및 쿼리 진행 중..."):
            query_en = translate(query_ko, "English")
            results = query_db(image_vdb, query_en, results=2)

            # 쿼리 결과 확인
            if not results or not results.get('images') or not results['images'][0]:
                st.error("쿼리 결과가 없습니다. 다른 질문을 시도해 보세요.")
                return

            prompt_input = format_prompt_inputs(results, query_en)
            response_en = vision_chain.invoke(prompt_input)
            response_ko = translate(response_en, "Korean")

        st.subheader("검색된 이미지:")
        for idx, image in enumerate(results['images'][0]):
            st.image(image, caption=f"ID: {results['ids'][0][idx]}", width=300)

        st.subheader("FashionRAG의 응답:")
        st.markdown(response_ko)

if __name__ == "__main__":
    main()
