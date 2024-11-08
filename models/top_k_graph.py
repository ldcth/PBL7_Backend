from datasets import load_dataset
from huggingface_hub import notebook_login
import pandas as pd
import numpy as np
import re
import os
from openai import OpenAI
import ast
from collections import defaultdict

def load_data():
    notebook_login("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")
    dataset_edges = load_dataset(
        "BaoHuynh2002/final_keywords_edges_data")
    df_dataset_edges = dataset_edges['train'].to_pandas()
    dataset_nodes = load_dataset(
        "BaoHuynh2002/final_nodes_data")
    df_dataset_nodes = dataset_nodes['train'].to_pandas()
    return df_dataset_edges, df_dataset_nodes

def load_data_qwen():
    notebook_login("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")
    dataset_edges = load_dataset(
        "ductho263/final_keywords_edges_data")
    df_dataset_edges = dataset_edges['train'].to_pandas()
    dataset_nodes = load_dataset(
        "ductho263/final_nodes_data")
    df_dataset_nodes = dataset_nodes['train'].to_pandas()
    return df_dataset_edges, df_dataset_nodes
# Hàm làm sạch cho cột 'nội dung liên kết' và 'keyword'
def formatting_func_mcqa_gen(example):
    # Remove sentences containing specific phrases from the explanation
    answers_list = ast.literal_eval(example['answers'])
    i = ord(example['correct_answer']) - 65
    text = "{} {} ".format(
        example['question'],
        ' '.join(answers_list))
    return text

def formatting_func_mcqa_doc_id(example):
    # Directly use 'answers' as it is already a list
    answers_list = example['answers']
    # Combine the question and answers
    text = "{} {}".format(
        example['question'],
        ' '.join(answers_list)
    )
    return text


def clean_text_columns(df, column_name):
    """
    Làm sạch cột văn bản bằng cách xóa các ký tự đặc biệt và chuyển thành chữ thường.
    """
    def clean_content(content):
        # Xóa các ký tự đặc biệt bằng regex và chuyển thành chữ thường
        content = re.sub(r'[^a-zA-Z0-9aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆ fFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTu UùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ\s]', '', content)
        # Xóa dấu ngoặc kép và các ký tự dư thừa
        content = content.replace('"', '').strip().lower()
        return content

    # Áp dụng hàm làm sạch cho cột đã chỉ định
    df[column_name] = df[column_name].apply(clean_content)
def get_extracted_keywords(question):
    key = "xai-E39CyX81ARslfGwYTBrwwca9xR74jMCzIEKvvlP7Ci78pQp93LUSLXq549jtdHkwqIjNz9POLpQIN388"
    requirement = """Bạn là một chuyên gia phân tích dữ liệu lịch sử. Dưới đây là một câu hỏi trắc nghiệm lịch sử, và nhiệm vụ của bạn là trích xuất các thực thể từ câu hỏi này để tạo một **Knowledge Graph (KG)**.

    - Các thực thể (entities) quan trọng bao gồm:
    1. **Sự kiện**: Các sự kiện lịch sử quan trọng (trận đánh, ký kết hiệp ước, cách mạng, v.v.).
    2. **Nhân vật lịch sử**: Các cá nhân có vai trò quan trọng trong lịch sử (vua, tướng lĩnh, nhà khoa học, nghệ sĩ, v.v.).
    3. **Địa điểm**: Các địa điểm quan trọng (thành phố, quốc gia, vùng lãnh thổ, chiến trường, v.v.).
    4. **Tổ chức**: Các tổ chức, quốc gia, quân đội, chính phủ, liên minh, v.v.
    5. **Tư tưởng/Phong trào**: Các tư tưởng hoặc phong trào chính trị, triết học hoặc tôn giáo (Cộng sản, Phát xít, Khai sáng, v.v.).
    6. **Tài liệu**: Các tài liệu quan trọng (hiệp ước, tuyên bố, văn bản pháp lý, v.v.).
    7. **Thời kỳ**: Thời gian hoặc giai đoạn lịch sử.

    Dựa trên câu hỏi trắc nghiệm dưới đây, hãy trích xuất, và định dạng thành **bảng** như sau:

    1. **Bảng 1: Thông tin các nodes**
    Format:
    - **id**: mã định danh duy nhất cho thực thể (Q1,Q2,Q3: tương ứng với thực thể ở câu hỏi; A1,A2,..: tương ứng với thực thể ở đáp án)
    - **từ khóa (keyword)**: tên của thực thể.
    """

    request = f"""**Câu hỏi trắc nghiệm:**
    "{question}"

    Hãy trích xuất các thực thể trong câu hỏi trắc nghiệm, tối đa 3 thực thể ở trong câu hỏi và 2 thực thể ở mỗi đáp án trắc nghiệm.
    Nếu nội dung câu hỏi trắc nghiệm không về nội dung học thuật môn lịch sử và các nội dung vô nghĩa thì mặc định trả về 'null'"""

    XAI_API_KEY = key
    client = OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1",
    )

    completion = client.chat.completions.create(
        model="grok-beta",
        messages=[
            {"role": "system", "content": requirement},
            {"role": "user", "content": request},
        ],
    )

    return completion.choices[0].message.content

from collections import defaultdict

def calculate_relevance_score_Q(keyword, keywords_Q):
    # Tìm từ dài nhất trong keywords_A mà chứa keyword
    matching_keywords = [k for k in keywords_Q if keyword in k or k in keyword]
    
    if matching_keywords:
        # Lấy từ dài nhất trong các từ khớp
        longest_match = max(matching_keywords, key=len)
        # Tính tỉ lệ độ dài
        ratio = min(len(keyword), len(longest_match)) / max(len(keyword), len(longest_match)) * 100
        
        if ratio == 100:  # Khớp hoàn toàn
            return 4
        elif ratio > 50:  # Khớp hơn 50%
            return 2
        elif ratio > 30:  # Khớp một phần nhỏ
            return 1
        else:
            return 0
    return 0

def calculate_relevance_score_A(keyword, keywords_A):
    # Tìm từ dài nhất trong keywords_A mà chứa keyword
    matching_keywords = [k for k in keywords_A if keyword in k or k in keyword]
    
    if matching_keywords:
        # Lấy từ dài nhất trong các từ khớp
        longest_match = max(matching_keywords, key=len)
        # Tính tỉ lệ độ dài
        ratio = min(len(keyword), len(longest_match)) / max(len(keyword), len(longest_match)) * 100
        
        if ratio == 100:  # Khớp hoàn toàn
            return 4
        elif ratio > 50:  # Khớp hơn 50%
            return 2
        elif ratio > 30:  # Khớp một phần nhỏ
            return 1
        else:
            return 0
    return 0
def find_most_relevant_doc(df_nodes, df_edges, df_question, top_n=5):
    # Chuyển đổi keyword thành danh sách để dễ dàng kiểm tra sự xuất hiện
    keywords_Q = df_question[df_question['id'].str.startswith('Q')]['keyword'].tolist()
    keywords_A = df_question[df_question['id'].str.startswith('A')]['keyword'].tolist()
    print(keywords_Q)
    print(keywords_A)
    # Count number of 'null' values in both lists
    null_count_Q = keywords_Q.count('null')
    null_count_A = keywords_A.count('null')
    
    # Check if null count is less than 50% of list length for both lists
    is_valid_Q = null_count_Q < len(keywords_Q) * 0.5
    is_valid_A = null_count_A < len(keywords_A) * 0.5
    
    if is_valid_Q and is_valid_A and len(keywords_Q) != 0 and len(keywords_A) != 0 and len(keywords_Q) > 1 and len(keywords_A) > 3:
        print("ok check")
        # Sử dụng defaultdict để lưu điểm cho mỗi id bài
        relevance_score = defaultdict(int)
        # Đếm số lần xuất hiện của keywords trong df_nodes
        for index, row in df_nodes.iterrows():
            doc_base = row['id_doc'].rsplit('_', 1)[0] + '_'
            
            relevance_score[doc_base] += calculate_relevance_score_Q(row['keyword'], keywords_Q)
            #relevance_score[doc_base] += calculate_relevance_score_A(row['keyword'], keywords_A)
# Đếm số lần xuất hiện của keywords trong df_edges và kiểm tra liên kết giữa Q và A
        
        if relevance_score[doc_base] > 0:
            for index, row in df_edges.iterrows():
                doc_base = row['id_doc'].rsplit('_', 1)[0] + '_'
                if row['keyword_head'] in keywords_Q and row['keyword_tail'] in keywords_A:
                    relevance_score[doc_base] += 10
                elif row['keyword_head'] in keywords_Q or row['keyword_tail'] in keywords_Q:
                    relevance_score[doc_base] += 5
                elif row['keyword_head'] in keywords_A and row['keyword_tail'] in keywords_Q:
                    relevance_score[doc_base] += 10
                elif row['keyword_head'] in keywords_A or row['keyword_tail'] in keywords_A:
                    relevance_score[doc_base] += 2

        # Sắp xếp và lọc các tài liệu có điểm > 0
        sorted_docs = sorted(relevance_score.items(), key=lambda x: x[1], reverse=True)
        print(sorted_docs)
        filtered_docs = [doc for doc in sorted_docs if doc[1] > 0]
        print(filtered_docs)
        top_relevant_docs = [doc[0].rstrip('_') for doc in filtered_docs[:top_n]]
        print(top_relevant_docs)
    else:
        print('null')
        top_relevant_docs = []
    if top_relevant_docs == ['10_10']:
        top_relevant_docs = []
    return top_relevant_docs

def get_top_k_graph(question,refType):
    # question = "Bộ máy hành chính giúp việc cho vua ở các quốc gia cổ đại phương Đônggồm A.nông dân công xã và quý tộc. B.các tầng lớp trong xã hội. C.toàn quý tộc. D.toàn tăng lữ."
    df_edges, df_nodes  = load_data() if refType == "L" else load_data_qwen()
    # question_text = formatting_func_mcqa_doc_id(question)
    question_text = question
    result = get_extracted_keywords(question_text)
    # Chuỗi chứa dữ liệu bảng
    data = result

    # Sử dụng biểu thức chính quy để trích xuất các hàng chứa dữ liệu
    rows = re.findall(r"\|\s*(\w+)\s*\|\s*(.+?)\s*\|", data)

    # Chuyển dữ liệu thành DataFrame
    df_question = pd.DataFrame(rows, columns=["id", "keyword"])

    # Loại bỏ dòng tiêu đề nếu có
    df_question = df_question[df_question['id'] != "id"]

    # Gọi hàm làm sạch cho cột 'keyword'
    clean_text_columns(df_question, 'keyword')

    # df_question_dict = df_question.set_index('id')['keyword'].to_dict()

    # Tìm tài liệu liên quan nhất
    most_relevant = find_most_relevant_doc(df_nodes, df_edges, df_question)

    return most_relevant

    
