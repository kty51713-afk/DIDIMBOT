from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from vector import vectorstore

load_dotenv()

import langchain
langchain.debug = True  # 모든 데이터 흐름을 터미널에 상세히 출력합니다.

llm = ChatOpenAI(model="gpt-4o-mini",temperature=0,max_completion_tokens=800)
llm2 = ChatOpenAI(model="gpt-4o-mini",temperature=0.3,max_completion_tokens=800)
parser = StrOutputParser()

# 참고 데이터 포맷팅 함수
def format_docs(docs):
    return "\n\n".join(
        f"[{d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )

# 유사도 검색을 통해 참고 데이터 가져오기(0.7 이상의 유사도 점수를 가진 문서만 반환)
def retrieve_context(user_query: str, k: int = 3, threshold: float = 0.7) -> str:
    docs = vectorstore.similarity_search_with_score(user_query, k=k)
    relevant_docs = [doc for doc, score in docs if score >= threshold]
    return format_docs(relevant_docs)

# 욕설 필터링 함수  
def bad_word_filter(data) -> str:
    with open("./data/욕설필터.txt", "r", encoding="utf-8") as f:
        bad_words = [line.strip() for line in f]
    for word in bad_words:
        if word in data['query']:
            return "비속어가 포함되어 있습니다. 다시 입력해주세요."
    return data['query']

# 응답 선택 함수
def route_request(user_query, history):
    return route_request_chain.invoke({
      "query": user_query,
      "history": history  
    })   
    
route_request_prompt = ChatPromptTemplate.from_messages([
    ("system", """
     넌 사용자의 말의 의도를 파악하는 시스템이야 출력 형식과 조건을 따르도록 해
     
     [사용자의 말]
     {query}
     
     [대화기록]
     {history}
     
     [출력형식]
     "직업추천" 혹은 "대화" 둘 중 하나로 무조건 출력하기
     
     [조건]
     대화기록을 참고하여 사용자의 말의 의도가 '직업추천'관련인지 파악할 것
     "직업추천"으로 출력하기 위해선 사용자의 말에 몸이 불편한 내용이나 특정장애를 가졌다는 내용이 포함되어야 한다
     그 외에는 무조건 "대화"로 출력할 것
     
     """)
])

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", """너는 참고자료를 참고해서 정보를 알려주는 AI야
     사용자가 어떤 장애가 있는지 [참고자료]를 참고하여 파악해줘
     사용자의 질문을 바탕으로 다음 형식, 조건으로 문장을 정리해줘
     이전 대화에서 파악된 사용자의 능력(보유역량)과 제약사항을 절대 잊지 마.
     
     [참고자료]
     {context}
     
     [대화 기록]
     {history}
     
     [형식]
     장애유형: ('참고자료 내용'과 '사용자 질의내용'을 바탕으로 사용자의 장애 유형을 추론하여 작성)
     장애등급: (1~4등급 중 선택)
     관심사: (사용자가 언급한 관심사나 선호하는 직업 분야)
     거주지역/희망지역: (사용자가 언급한 지역/사용자가 언급 안하면 기본 지역은 서울시로 할 것)
     보유역량: (사용자가 언급한 기술이나 장점)
     제한사항: (업무 시 고려해야 할 신체적/환경적 제약)
    
     [조건]
     - 한 줄로 출력할 것 (구분자 '|' 사용)
     - 정보가 없으면 '미파악'으로 기재할 것
     - 모든 정보가 '미파악'인 경우 "불가능"이라고 답변할 것
     - 장애유형이 '미파악'인 경우 "단순질문"이라고 답변할 것
     - 사용자의 질문이 '직업추천해줘'같은 말이 없으면 '단순질문'으로 취급할
     - 대화기록을 꼭 함께 참고하여 형식들을 작성할 것
     - 이전 대화 기록에 사용자가 처한 상황의 정보를 형식에 모두 기입하라
     - 사용자가 새로운 정보를 주면 그 부분만 업데이트하라. 모든 정보가 '미파악'이더라도 질문이 고용/취업과 관련 있다면 최대한 형식을 유지하라
     """),
    ("human", "{query}")
])

other_prompt = ChatPromptTemplate.from_messages([
    ("system", """너는 장애인 고용 및 직업상담 보조 시스템이야.
    사용자의 질문에 아래 형식에 맞게 친절하게 답변해줘.
    
    [형식]
    1. 질문에 대하여 너가 아는데로 친절히 답변해줘
    2. 가능한 경우엔 참고 정보도 참고해서 답변하도록 해
    
    [대화 기록]
    {history}
    
    [검색된 참고 정보]
    {context}
    
    [조건]
    1. 질문자의 질문에 친절히 답변만 한 뒤에 너가 답할 수 있는 장애인 취업관련 질문 3가지를 예시로 들어줘
    2. 사는지역, 불편을 겪는 부위, 관심사항을 알려주면 더 정확한 취업정보를 알려줄 수 있다고 안내할 것
    3. 참고자료에 사용자와 봇 사이의 대화이력에 관련 내용이 있으면 그 내용을 바탕으로 작성할 것
    
    """),                          
    ("human", "{query}")
])

recommend_prompt = ChatPromptTemplate.from_template("""
    너는 장애인 고용 및 직업상담 보조 시스템이야.
        장애유형과 제한사항, 보유역량을 우선적으로 고려해서 직업을 추천해


    [사용자 질문]
    {query}
    
    [대화 기록]
    {history}

    [검색된 참고 정보]
    {context}

   아래 규칙을 반드시 지켜라.
    1. 답변은 지금까지 쌓인 대화내용과 검색된 참고 정보에 근거해서 작성하라.
    2. 검색된 참고 정보에 근거가 있는 내용만 답하라.
    3. 참고 정보에 없는 직업, 기관, 장소를 추측해서 만들지 마라.
    4. 질문과 직접 관련 없는 정보는 제외하라.
    5. 불확실한 내용은 단정하지 마라.
    6. 미파악정보들은 "가까운 컨설팅 기관에서 추가 상담 필요"라고 적어라.
    7. 대화기록을 꼭 함께 참고하여 답변하되, 컨설팅 기관은 중복되지 않도록 작성하라.
    8. 미파악된 정보들을 알려주고, 그 정보들을 알려주면 더 정확한 답변을 줄 수 있다고 친절하게 안내하라.
    9. '장애유형' 혹은 '제한사항'이 '정신','자폐','지능' 관련일 경우, 업무강도 2이하의 근무들만 추천하라
    10. 정신, 자폐, 지능과 관련 없는 장애의 경우에는 업무강도는 너가 판단해서 추천하라

    다음 형식으로 답변하라.
    1. 현재 상황 요약
    2. 추천 직업 2~3개(업무 강도는 1~10까지 있다)
    3. 컨설팅 기관 또는 지원 장소 2~3개(컨설팅기관은 주소지와 가까운 곳, 연락처 필수 포함, 구분이 잘 가도록 한 줄씩 줄바꿈작성)
    4. 해당 직업과 기관들을 추천하는 이유
    5. 업무강도에 대한 추가 설명: 직업 옆에 숫자(1~10)는 업무 강도를 나타내며, 
    숫자가 높을수록 업무 강도가 높다는 것을 추가 설명으로 사용자에게 알려주기.

[조건]
- 답변은 한국어로, 짧고 명확하게 작성하라.
- 장애유형이 확인되지 않았을 땐, 추천직업을 보여주지 않는다
""")

    # 수정된 체인 구성
rewriter = rewrite_chain = (
    rewrite_prompt 
    | llm 
    | parser
    # x는 이미 문자열이므로 바로 출력하면 됩니다.
    | RunnableLambda(lambda x: print(f"\n✨ [LLM 변환 결과]:\n{x}\n") or x) 
)
route_request_chain = route_request_prompt | llm | parser
other_chain = other_prompt | llm | parser
recommend_chain = recommend_prompt | llm2 | parser

# 최종 답변 생성 함수
def get_chat_response(user_query: str, history: list = None) -> str:
                                    # history는 이전 대화기록, 없어도 에러발생 안나도록 None으로 초기화
    context = retrieve_context(user_query, k=3)

    history_text = ""
    
    if history: #history가 존재할 때만 대화 기록을 포맷팅하여 history_text에 추가
        history_text = "\n[대화 기록]\n" + "\n".join([
            f"{'유저' if c['type'] == 'userChat' else 'AI'}: {c['text']}" for c in history[-3:]
        ])
    
    # 참고 데이터와 대화 기록을 합쳐서 최종 컨텍스트 생성, rewrite_chain과 recommend_chain에 전달
    full_context = f"{context}\n{history_text}"
    
    # 사용자 질문에서 욕설이 포함되어 있는지 필터링, 욕설이 포함된 경우 해당 메시지 반환
    user_query = bad_word_filter({"query": user_query})
    if "비속어" in user_query:
        return user_query
    
    inten = route_request(user_query, history_text)
    rewrite=""
    
    if inten=="직업추천":
        rewrite = rewrite_chain.invoke({
        "query": user_query,
        "context": context,
        "history": history_text
    }).strip()
        
    elif inten=="대화":
        return other_chain.invoke({
        "query": user_query,
        "context": full_context,
        "history": history_text
    }).strip()
        
    return recommend_chain.invoke({
        "query": rewrite,
        "context": full_context,
        "history": history_text
    }).strip()

   
    # 내가 시력이 매우 나쁜데 아예 안보이는 건 아니야. 나는 무슨 일을 할 수 있을까