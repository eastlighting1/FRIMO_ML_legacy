# FRIMO_ML

FRIMO_ML은 FRIMO 프로젝트에서 머신러닝을 이용하는 부분들을 담은 레포지토리입니다.

언어 : [영어](https://github.com/Friend-for-Modern-people/FRIMO_ML/blob/main/README.md) | 한국어

<br> <br>

## <b> 레포지토리 구조 </b>

<br>

* <b> Documentation </b>
    > 프로그래밍 언어로 구성되지 않는 파일들이 위치합니다.
    * lib-requirement.txt
        * 모델들에서 요구하는 라이브러리 설정을 정리한 파일입니다.
    * README_ko.md
        * README 한국어 버전입니다.
* <b> Model </b>
    > FRIMO의 모델들이 위치합니다.
    * Chatting
        * AI와 대화하는 기능을 구현합니다.
        * 사용 모델 : [KoGPT2](https://github.com/SKT-AI/KoGPT2)
        * 세부 폴더 : [해당 깃허브](https://github.com/haven-jeon/KoGPT2-chatbot) 참고
    * Style-Change
        * AI의 말투[^1]를 변경해주는 기능을 구현합니다.
        * 사용 모델 : [KoBART](https://github.com/SKT-AI/KoBART)
        * 세부 폴더 : [해당 문서]() 참고
    * Summarization
        * 대화의 내용을 요약하는 기능을 구현합니다.
        * 사용 모델 : [KoBART](https://github.com/SKT-AI/KoBART)
        * 세부 폴더 : [해당 깃허브](https://github.com/seujung/KoBART-summarization) 참고
    * Emotion Recognition
        * 대화에 담긴 감정을 파악하는 기능을 구현합니다.
        * 사용 모델 : [KoBERT]()
        * 세부 폴더 : [해당 문서]() 참고 

* <b> Execution </b>
    > 모델 구동을 위한 파일들이 위치합니다.
    * ai_reply_making.bat
        * 채팅을 구현하기 위한 배치 파일입니다.
        * Input : <b>String</b> - User Chat
        * Output : <b>String</b> - AI Chat
    * diary_update.bat
        * 채팅 내용을 종합하여 일기로 만들기 위한 배치 파일입니다.
        * Input : <b>String</b> - User's daily conversation data
        * Output : <b>String</b> - Summarized Data
    * key_emo_making.bat
        * 일기의 내용을 바탕으로 키워드와 감정을 추출하기 위한 배치 파일입니다.
        * Input Data : <b>String</b> - Summarized Data
        * Output Data : 
            * <b>String</b> - Keyword
            * <b>Integer</b> - sentiment_pk

* <b> README </b>
    > FRIMO_ML에 대한 설명이 담긴 마크다운 파일입니다.

<br> <br>

## <b> 프로젝트 구조 </b>

<br>

1. 사용자가 채팅을 입력한다.

2. 입력된 채팅을 기반으로 AI 답변을 생성한다.

3. AI 답변의 말투를 변경하여 사용자에게 보여준다.

4. 유저의 채팅과 AI의 답변을 처리한다 :
    * 유저의 채팅을 CSV 파일로 만든 후, 이 csv 파일을 이용하여 채팅 모델을 재학습함
    * 유저의 채팅을 txt 파일로 만든 후, 이 csv 파일을 Firebase로 넘김

5. Firebase로부터 유저의 채팅을 tsv 파일로 받아서 요약한다.

6. 요약된 내용을 기반으로, 키워드와 감정을 추출한다.

7. 요약, 감정, 키워드를 데이터베이스로 보낸다.

<br> <br>

## <b> 모델 구현 사항 </b>

<br>

* Chatting
    * [X] <b> 모델 구현 </b>
    * [X] <b> requirements 처리 </b>
    * [ ] <b> re-train 구조 만들기 </b>
    * [ ] <b> epoch 충분히 돌리기 </b>
* Style-Change
    * [X] <b> 모델 구현 </b>
    * [X] <b> requirements 처리 </b>
    * [X] <b> 데이터셋[^2] 적용 </b>
    * [ ] <b> epoch 충분히 돌리기 </b>
* Summarization
    * [X] <b> 모델 구현 </b>
    * [X] <b> requirements 처리 </b>
    * [ ] <b> 데이터셋[^3] 적용 </b>
    * [ ] <b> epoch 충분히 돌리기 </b>
* Keyword Extraction
    * [X] <b> 관련 모델 발견 </b>
    * [ ] <b> tokenizer 발견 </b>
* Emotion Recognition
    * [X] <b> 모델 구현 </b>
    * [X] <b> requirements 처리 </b>
    * [X] <b> 데이터셋[^4] 적용 </b>
    * [ ] <b> 모델 최적화 </b>
    * [ ] <b> epoch 충분히 돌리기 </b>

<br> <br>

## <b> 참고 자료 및 각주 </b>

<br>

[1] https://heegyukim.medium.com/korean-smilestyle-dataset%EC%9C%BC%EB%A1%9C-%EB%AC%B8%EC%B2%B4-%EC%8A%A4%ED%83%80%EC%9D%BC%EC%9D%84-%EB%B0%94%EA%BE%B8%EB%8A%94-%EB%AA%A8%EB%8D%B8-%EB%A7%8C%EB%93%A4%EC%96%B4%EB%B3%B4%EA%B8%B0-d15d32a2c303

[^1]: 말투 변경 예시 : <br>
<b>기존</b> : 저는 지금 사막에 와 있어요. <br>
<b>변경</b> : 나 지금 그.. 사막인데..ㅠㅠ

[^2]: https://github.com/smilegate-ai/korean_smile_style_dataset

[^3]: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=117

[^4]: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86

<br> <br>

## <b> 기여자 </b>

<br>

| 이름 | 학번 | 학교 | 기여한 부분 | 깃허브 링크 |
| :---: | :---: | :---: | :---: | :---: |
|김동현 | 201935217 | 가천대학교 | 전체적인 모델 개발 | [깃허브](https://github.com/eastlighting1) |

