from bs4 import BeautifulSoup
import requests
import time
import re



def Chitiet_HTML(base_url):
    # Lấy dữ liệu HTML
    response = requests.get(base_url)

    # Parse content
    content = response.content
    parsed_content = BeautifulSoup(content, 'html.parser')

    # Tìm tất cả các thẻ <p>
    paragraphs = parsed_content.find_all('p')

    # List để lưu trữ thông tin của các câu hỏi và đáp án
    questions_and_answers = []

    # Biến để lưu trữ câu hỏi hiện tại
    current_question = None

    # Lặp qua tất cả các thẻ <p>
    for paragraph in paragraphs:
        # Lấy nội dung của thẻ <p> và loại bỏ các khoảng trắng thừa
        text = paragraph.get_text(strip=True)
        if text.startswith("Câu "):  # Kiểm tra nếu đây là một câu hỏi mới
            # Tạo một dict mới để lưu trữ câu hỏi và các đáp án
            current_question = {"question": text,
                                "answers": [], "explanation": "NaN"}
            questions_and_answers.append(current_question)
        elif current_question is not None:
            # Kiểm tra nếu đây là lời giải
            if text.startswith("Lời giải:") or text.startswith("Giải thích:"):
                next_paragraphs = paragraph.find_all_next(
                    "p")  # Tìm tất cả các phần tử <p> tiếp theo
                explanation = ""
                for next_paragraph in next_paragraphs:
                    # Nếu gặp phần tử bắt đầu bằng "Đáp án"
                    if next_paragraph.get_text(strip=True).startswith("Đáp án"):
                        break  # Thoát khỏi vòng lặp
                    # Cộng các giá trị text của các phần tử <p> vào explanation
                    explanation += next_paragraph.get_text(strip=True) + " "
                # Gán giá trị explanation đã cộng được vào current_question
                current_question["explanation"] = explanation.strip()
            elif text.startswith("Giải thích:"):
                current_question["explanation"] = text.split(":")[1].strip()
            # Kiểm tra nếu đây là đáp án cần chọn
            elif text.startswith("Đáp án cần chọn là:") or text.startswith("Đáp án:") or text.startswith("Đáp án đúng là:"):
                current_question["correct_answer"] = text.split(":")[1].strip()
                if text.startswith("Đáp án đúng là:"):
                    # Tìm thẻ div hiện tại
                    current_div = paragraph.find_parent(
                        'div', class_="toggle-content")
                    if current_div:
                        # Tìm tất cả các thẻ <p> trong thẻ div hiện tại
                        next_paragraphs = current_div.find_all("p")
                        explanation = ""
                        for next_paragraph in next_paragraphs:
                            if not next_paragraph.get_text(strip=True).startswith("Đáp án"):
                                explanation += next_paragraph.get_text(
                                    strip=True) + " "
                        # Gán giá trị explanation sau khi đã lấy được toàn bộ nội dung
                        current_question["explanation"] = explanation.strip()
            elif text.startswith("A.") or text.startswith("B.") or text.startswith("C.") or text.startswith("D."):
                # Thêm các đáp án vào danh sách của câu hỏi hiện tại
                current_question["answers"].append(text)

    # In thông tin của các câu hỏi và đáp án
    for i, qa in enumerate(questions_and_answers, start=1):
        print(f"{qa['question']}")
        for answer in qa['answers']:
            print(answer)
        print(f"Đáp án cần chọn là: {qa['correct_answer']}")
        print(f"Lời giải: {qa['explanation']}")
        print()


def Tracnghiem_Lichsu_url(base_url):
    url = []
    base_website = "https://vietjack.com"

    # Tính thời gian chạy function
    start_time = time.time()

    # Lấy dữ liệu HTML
    response = requests.get(base_url)

    # Parse content
    content = response.content
    parsed_content = BeautifulSoup(content, 'html.parser')

    # Tìm tất cả các thẻ <a> có thẻ con <b> chứa ký tự lowercase liên quan đến từ "lịch sử"
    links = parsed_content.find_all('a', href=True)
    for link in links:
        # Tìm thẻ <b> trong <a> mà không đi sâu vào các thẻ con khác
        b_tag = link.find('b', recursive=False)
        if b_tag and b_tag.text.lower().startswith('trắc nghiệm') and 'color:green' in b_tag.get('style', ''):
            # Lấy giá trị của thuộc tính href trong thẻ <a>
            href = link['href']
            url.append(base_website+href[2:])

    end_time = time.time()
    running_time = end_time - start_time
    print("Running time:", running_time, "seconds")

    return url


def Test_Chitiet_HTML(base_url):
    # Tính thời gian chạy function
    start_time = time.time()

    # Lấy dữ liệu HTML
    response = requests.get(base_url)

    # Parse content
    content = response.content
    parsed_content = BeautifulSoup(content, 'html.parser')

    # Tìm tất cả các thẻ <p>
    paragraphs = parsed_content.find_all('p', class_=None)

    # List để lưu trữ thông tin của các câu hỏi và đáp án
    questions_and_answers = []

    # Biến để lưu trữ câu hỏi hiện tại
    current_question = None

    # Lặp qua tất cả các thẻ <p>
    try:
       # Lặp qua tất cả các thẻ <p>
        for paragraph in paragraphs:
            # Lấy nội dung của thẻ <p> và loại bỏ các khoảng trắng thừa
            text = paragraph.get_text(strip=True)
            if text.startswith("Câu "):  # Kiểm tra nếu đây là một câu hỏi mới
                # Tạo một dict mới để lưu trữ câu hỏi và các đáp án
                current_question = {"question": re.sub(r'Câu \d+[:.]', '', text),
                                    "answers": [], "explanation": "NaN"}
                questions_and_answers.append(current_question)
            elif current_question is not None:
                # Kiểm tra nếu đây là đáp án cần chọn
                if text.startswith("Đáp án cần chọn là:") or text.startswith("Đáp án:") or text.startswith("Đáp án đúng là:"):
                    current_question["correct_answer"] = text.split(":")[
                        1].strip()
                elif text.startswith("A.") or text.startswith("B.") or text.startswith("C.") or text.startswith("D."):
                    # Thêm các đáp án vào danh sách của câu hỏi hiện tại
                    current_question["answers"].append(text)
                # Kiểm tra nếu đây là lời giải
                elif text.startswith("Lời giải:") or text.startswith("Giải thích:"):
                    next_paragraphs = paragraph.find_all_next(
                        "p", class_=None)  # Tìm tất cả các phần tử <p> tiếp theo
                    explanation = text.split(":")[1].strip()
                    # Danh sách các class có thể xuất hiện trong phần footer
                    footer_classes = [
                        'footer', 'footer-content', 'footer-wrapper']
                    for next_paragraph in next_paragraphs:
                        parent_classes = [cls.strip() for cls in next_paragraph.parent.get(
                            'class', [])]  # Lấy danh sách class của thẻ cha
                        if any(cls in parent_classes for cls in footer_classes):
                            break  # Nếu thẻ cha có một trong các class thuộc footer_classes, thoát khỏi vòng lặp
                        # Nếu gặp phần tử bắt đầu bằng "Đáp án"
                        if next_paragraph.get_text(strip=True).startswith("Đáp án") or next_paragraph.get_text(strip=True).startswith("Xem thêm") or next_paragraph.get_text(strip=True).startswith("Câu"):
                            break  # Thoát khỏi vòng lặp
                        # Cộng các giá trị text của các phần tử <p> vào explanation
                        explanation += next_paragraph.get_text(
                            strip=True) + " "
                    # Gán giá trị explanation đã cộng được vào current_question
                    current_question["explanation"] = explanation.strip()
                else:
                    explanation = text
                    next_paragraphs = paragraph.find_all_next(
                        "p", class_=None)  # Tìm tất cả các phần tử <p> tiếp theo
                    # Danh sách các class có thể xuất hiện trong phần footer
                    footer_classes = [
                        'footer', 'footer-content', 'footer-wrapper']
                    for next_paragraph in next_paragraphs:
                        parent_classes = [cls.strip() for cls in next_paragraph.parent.get(
                            'class', [])]  # Lấy danh sách class của thẻ cha
                        if any(cls in parent_classes for cls in footer_classes):
                            break  # Nếu thẻ cha có một trong các class thuộc footer_classes, thoát khỏi vòng lặp
                        # Nếu gặp phần tử bắt đầu bằng "Đáp án"
                        if next_paragraph.get_text(strip=True).startswith("Đáp án") or next_paragraph.get_text(strip=True).startswith("Xem thêm") or next_paragraph.get_text(strip=True).startswith("Câu"):
                            break  # Thoát khỏi vòng lặp
                        # Cộng các giá trị text của các phần tử <p> vào explanation
                        explanation += next_paragraph.get_text(
                            strip=True) + " "
                    # Gán giá trị explanation đã cộng được vào current_question
                    current_question["explanation"] = explanation.strip()

    except:
        return questions_and_answers
    return questions_and_answers


def handle():
    urls_10 = Tracnghiem_Lichsu_url(
        "https://vietjack.com/bai-tap-trac-nghiem-lich-su-10/index.jsp")
    urls_11 = Tracnghiem_Lichsu_url(
        "https://vietjack.com/bai-tap-trac-nghiem-lich-su-11/index.jsp")
    urls_12 = Tracnghiem_Lichsu_url(
        "https://vietjack.com/bai-tap-trac-nghiem-lich-su-12/index.jsp")
    urls = urls_10 + urls_11 + urls_12
    final_data = []
    for url in urls:
        data = Test_Chitiet_HTML(url)
        final_data = final_data + data
    return final_data
