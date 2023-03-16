## Folder Structure

* 감성대화말뭉치
    * JSON file
        * Structure :
            ```JSON
            {"profile": {
                "persona-id": "Pro_05349", 
                "persona": {
                    "persona-id": "A02_G02_C01", 
                    "human": ["A02", "G02"], 
                    "computer": ["C01"]}, 
                "emotion": {
                    "emotion-id": "S06_D02_E18", 
                    "type": "E18", 
                    "situation": ["S06", "D02"]}}, 
             "talk": {
                "id": {
                    "profile-id": "Pro_05349", 
                    "talk-id": "Pro_05349_00062"},        
                "content": {
                    "HS01": "직장에 다니고 있지만 시간만 버리는 거 같아. 진지하게 진로에 대한 고민이 생겨.", 
                    "SS01": "진로에 대해서 고민하고 계시는군요. 어떤 점이 고민인가요?", 
                    "HS02": "직장 상사한테 자주 지적을 받아. 그럴 때마다 이 업무는 나랑 맞지 않는 거 같이 느껴져.", 
                    "SS02": "업무가 나와 맞지 않아 시간을 버리는 것 같이 느껴지셨군요.", 
                    "HS03": "", 
                    "SS03": ""}}}

            ```
        * Training   : 51628 rows
        * Validation : 6640 rows
    * train.py
