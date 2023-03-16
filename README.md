# FRIMO_ML

FRIMO_ML is a repository for the parts of the FRIMO project that use machine learning.

<br> <br>

## <b> Repository Structure </b>

<br>

* <b> Documentation </b>
    > This is where files that are not written in a programming language are located.
    * lib-requirement.txt
        * This file organizes the library settings required by the models.
    * README_ko.md
        * Korean Version of README
* <b> Model </b>
    > FRIMO's models are located here.
    * Chatting
        * Implement the ability to talk to AI.
        * Implemented Model : [KoGPT2](https://github.com/SKT-AI/KoGPT2)
        * Folder Description : See [this github](https://github.com/haven-jeon/KoGPT2-chatbot).
    * Style-Change
        * Implement the ability to change the AI's speech style[^1].
        * Implemented Model : [KoBART](https://github.com/SKT-AI/KoBART)
        * Folder Description : See [this doc]().
    * Summarization
        * Implement the ability to summarize the content of a conversation.
        * Implemented Model : [KoBART](https://github.com/SKT-AI/KoBART)
        * Folder Description : See [this github](https://github.com/seujung/KoBART-summarization).
    * Emotion Recognition
        * Implement the ability to understand the emotion behind a conversation.
        * Implemented Model : [KoBERT]()
        * Folder Description : See [this doc]().

* <b> Execution </b>
    > This is where the files to run the model are located.
    * ai_reply_making.bat
        * A batch file for implementing chat.
        * Input : <b>String</b> - User Chat
        * Output : <b>String</b> - AI Chat
    * diary_update.bat
        * A batch file for converting chat transcripts into diaries.
        * Input : <b>String</b> - User's daily conversation data
        * Output : <b>String</b> - Summarized Data
    * key_emo_making.bat
        * A batch file for extracting keywords and sentiments based on the contents of a diary.
        * Input Data : <b>String</b> - Summarized Data
        * Output Data : 
            * <b>String</b> - Keyword
            * <b>Integer</b> - sentiment_pk

* <b> README </b>
    > A markdown file containing a description of FRIMO_ML.

<br> <br>

## <b> Project Structure </b>

<br>

1. Users enter a chat.

2. Generate an AI answer based on the entered chat.

3. Change the style of the AI answer and display it to the user.

4. Process the user's chat and the AI answer :
    *Create a CSV file of the user's chats and use it to retrain the chat model
    * Create a txt file of the user's chats and pass the csv file to Firebase.

5. Get the user's chats from Firebase as a tsv file and summarize them.

6. Extract keywords and sentiments based on the summarized content.

7. Send the summary, sentiment, and keywords to the database.

<br> <br>

## <b> Model Implementation </b>

<br>

* Chatting
    * [X] <b> Implement Prototype </b>
    * [X] <b> Handle Requirements </b>
    * [ ] <b> Create a Structure to Re-train </b>
    * [ ] <b> Train repeatedly for enough reps </b>
* Style-Change
    * [X] <b> Implement Prototype </b>
    * [X] <b> Handle Requirements </b>
    * [X] <b> Apply the Corpus[^2] </b>
    * [ ] <b> Train repeatedly for enough reps </b>
* Summarization
    * [X] <b> Implement Prototype </b>
    * [X] <b> Handle Requirements </b>
    * [ ] <b> Apply the Corpus[^3] </b>
    * [ ] <b> Train repeatedly for enough reps </b>
* Keyword Extraction
    * [X] <b> Find Related Model</b>
    * [ ] <b> Find proper tokenizer </b>
* Emotion Recognition
    * [X] <b> Implement Prototype </b>
    * [X] <b> Handle Requirements </b>
    * [X] <b> Apply the Corpus[^4] </b>
    * [ ] <b> Optimize the model </b>
    * [ ] <b> Train repeatedly for enough reps </b>

<br> <br>

## <b> References and Footnotes </b>

<br>

[1] https://heegyukim.medium.com/korean-smilestyle-dataset%EC%9C%BC%EB%A1%9C-%EB%AC%B8%EC%B2%B4-%EC%8A%A4%ED%83%80%EC%9D%BC%EC%9D%84-%EB%B0%94%EA%BE%B8%EB%8A%94-%EB%AA%A8%EB%8D%B8-%EB%A7%8C%EB%93%A4%EC%96%B4%EB%B3%B4%EA%B8%B0-d15d32a2c303

[^1]: a Example for Style-Change : <br>
<b>Existing</b> : 저는 지금 사막에 와 있어요. <br>
<b>Changed </b> : 나 지금 그.. 사막인데..ㅠㅠ

[^2]: https://github.com/smilegate-ai/korean_smile_style_dataset

[^3]: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=117

[^4]: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86

<br> <br>

## <b> Contributors </b>

<br>

| Name | Student Number | University | Contributed Parts | Github Link |
| :---: | :---: | :---: | :---: | :---: |
|Kim Donghyeon | 201935217 | Gachon Univ. | Develop Overall Model | [Github](https://github.com/eastlighting1) |

