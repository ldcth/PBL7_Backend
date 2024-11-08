import requests


def testModel():

    API_URL = "https://api-inference.huggingface.co/models/PB7-DUT-2023/finetuned_Bloomz_1b1_v1"
    headers = {"Authorization": "Bearer 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"}

    response = requests.post(API_URL, headers=headers, json={
        "inputs": """###QUESTION: Công nghiệp văn hóa Việt Nam bao gồm ngành nào sau đây?
        A. Du lịch văn hóa
        B. Công nghệ thông tin.
        C. Sinh học.
        D. Y khoa.
        ### ANSWER:""",
        "parameters": {
            "max_length": 100,
            "wait_for_model": True,
            "use_cache": True
        }
    })

    output = response.json()[0]['generated_text']

    print(output)


def model_api(question, answers, version=1):
    # API_URL = "https://x9p9u1p794crk2ph.us-east-1.aws.endpoints.huggingface.cloud"
    API_URL = ""
    if (version == 1):
        # API_URL = "https://btjecpsw9w4zs50z.us-east-1.aws.endpoints.huggingface.cloud"
        API_URL = "https://c3fstdbsvvfdw5kj.us-east-1.aws.endpoints.huggingface.cloud"

    if (version == 2):
        API_URL = "https://x9p9u1p794crk2ph.us-east-1.aws.endpoints.huggingface.cloud"

    if (version == 3):
        API_URL = "https://aic51kvyskq769ar.us-east-1.aws.endpoints.huggingface.cloud"

    headers = {
        "Accept": "application/json",
        "Authorization": "Bearer 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",
        "Content-Type": "application/json"
    }

    formatted_question = """### QUESTION: {}\n{}\n### ANSWER:""".format(
        question, '\n'.join(answers))
    # print("formatted_question:", formatted_question)
    response = requests.post(API_URL, headers=headers, json={
        "inputs": formatted_question,
        "parameters": {
            "max_length": 100,
            "wait_for_model": True,
            "use_cache": True
        }
    })
    print(response.json())
    res = response.json()
    if type(res) == list:
        output = res[0]['generated_text']
        start_index = output.find("### ANSWER:") + len("### ANSWER:")
        end_index = output.find("EXPLAIN:", start_index)

        explain_index = output.find("EXPLAIN:")

        answer_part = output[start_index:end_index]
        explain_part = output[explain_index+len("EXPLAIN:"):].strip()

        # Tách đoạn văn thành các câu bằng dấu chấm
        sentences = explain_part.split('.')
        
        # Loại bỏ các khoảng trắng thừa từ các câu
        sentences = [sentence.strip() for sentence in sentences]
        
        # Kiểm tra và loại bỏ câu cuối nếu nó không có dấu chấm
        if sentences[-1]:
            if not explain_part.endswith('.'):
                sentences = sentences[:-1]
        
        # Ghép lại các câu còn lại thành đoạn văn
        new_text = '. '.join(sentences)
        new_text += "."

        return True, answer_part, new_text
    else:
        return False, "", ""
