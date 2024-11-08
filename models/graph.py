from datasets import load_dataset
from huggingface_hub import notebook_login
import pandas as pd
import numpy as np
from pyvis.network import Network
import networkx as nx
import os
from datetime import datetime

def load_data():
    notebook_login("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")
    # dataset_doc = load_dataset(
    #     "BaoHuynh2002/final_keywords_edges_data",  split="train")
    dataset_doc = load_dataset(
        "BaoHuynh2002/final_keywords_edges_data")
    df_dataset_doc = dataset_doc['train'].to_pandas()
    return df_dataset_doc

def load_data_qwen():
    notebook_login("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")
    # dataset_doc = load_dataset(
    #     "BaoHuynh2002/final_keywords_edges_data",  split="train")
    dataset_doc = load_dataset(
        "ductho263/final_keywords_edges_data")
    df_dataset_doc = dataset_doc['train'].to_pandas()
    return df_dataset_doc
def create_graph_by_lessons(dataset, grade, lesson):
    # Load appropriate dataset based on parameter
    data = load_data() if dataset == "L" else load_data_qwen()
    
    # Tạo đồ thị NetworkX
    G = nx.Graph()
    
    # Kiểm tra nếu `lesson = "all"` thì lấy toàn bộ bài học của lớp
    if lesson == "all":
        filtered_data = data[data['id_doc'].str.startswith(f"{grade}_")]
        
        # Tạo node cha đại diện cho lớp
        class_node = f"Lớp {grade}"
        G.add_node(class_node, label=f"Lớp {grade}", color="green", size=25)

        # Duyệt qua dữ liệu của từng bài học trong lớp
        for _, row in filtered_data.iterrows():
            node_1 = row['final_head_node']
            node_2 = row['final_tail_node']
            relationship = row['nội dung liên kết']
            lesson_id = row['id_doc'].split('_')[1]  # Lấy id bài học

            # Tạo node đại diện cho bài học
            lesson_node = f"{grade}_bài_{lesson_id}"
            G.add_node(lesson_node, label=f"Bài {lesson_id}", color="purple", size=20)
            G.add_edge(class_node, lesson_node, title=f"Bao gồm Bài {lesson_id}", weight=2)

            # Thêm các node vào đồ thị với các thuộc tính tùy chỉnh
            G.add_node(node_1, title=row['keyword_head'], color="lightblue", size=15, label=row['keyword_head'])
            G.add_node(node_2, title=row['keyword_tail'], color="lightblue", size=15, label=row['keyword_tail'])
            G.add_edge(node_1, node_2, title=relationship, weight=4)

            # Thêm edges từ node bài học đến các node của bài
            G.add_edge(lesson_node, node_1, title=f"Trong bài {lesson_id}", weight=2)
            G.add_edge(lesson_node, node_2, title=f"Trong bài {lesson_id}", weight=2)
    else:
        # Xử lý input của bài học, hỗ trợ nhập dạng '10-15' hoặc '10 13 15'
        lessons = set()
            # Tối đa số bài học theo lớp
        max_lessons = {
            "10": 40,
            "11": 24,
            "12": 25
        }
                # Kiểm tra xem lớp có trong danh sách không và lấy số bài tối đa
        if grade in max_lessons:
            max_lesson = max_lessons[grade]
        else:
            print("Lớp không tồn tại.")
            return None  # Trả về None nếu lớp không tồn tại trong danh sách
        # Tách từng phần của đầu vào bởi dấu phẩy
        parts = lesson.split(',')
        
        for part in parts:
            # Kiểm tra xem phần này có dạng "a-b" không
            if '-' in part:
                start, end = map(int, part.split('-'))
                if start > end or end > max_lesson:
                    print(f"Lỗi: Bài {part} vượt quá số bài tối đa của lớp {grade} ({max_lesson} bài).")
                    return None  # Trả về None nếu số bắt đầu lớn hơn số kết thúc hoặc vượt quá giới hạn
                lessons.update(range(start, end + 1))
            # Kiểm tra xem phần này có phải là một số lẻ không
            elif part.isdigit():
                lesson_num = int(part)
                if lesson_num > max_lesson:
                    print(f"Lỗi: Bài {lesson_num} vượt quá số bài tối đa của lớp {grade} ({max_lesson} bài).")
                    return None  # Trả về None nếu số bài vượt quá giới hạn
                lessons.add(lesson_num)
            else:
                print("Lỗi: Đầu vào không hợp lệ.")
                return None  # Trả về None nếu đầu vào không hợp lệ
            
        # Duyệt qua từng bài học trong danh sách
        for lesson in lessons:
            # Tạo node cha đại diện cho bài
            lesson_node = f"{grade}_bài_{lesson}"
            G.add_node(lesson_node, label=f"Bài {lesson}", color="purple", size=20)

            # Lọc dữ liệu theo `id_doc` của từng bài
            lesson_filter = f"{grade}_{lesson}_"
            filtered_data = data[data['id_doc'].str.startswith(lesson_filter)]

            # Thêm nodes và edges vào đồ thị
            for _, row in filtered_data.iterrows():
                node_1 = row['final_head_node']
                node_2 = row['final_tail_node']
                relationship = row['nội dung liên kết']

                # Thêm các node vào đồ thị với các thuộc tính tùy chỉnh
                G.add_node(node_1, title=row['keyword_head'], color="lightblue", size=15, label=row['keyword_head'])
                G.add_node(node_2, title=row['keyword_tail'], color="lightblue", size=15, label=row['keyword_tail'])
                G.add_edge(node_1, node_2, title=relationship, weight=4)

                # Thêm edges từ node cha đến các node của bài
                G.add_edge(lesson_node, node_1, title=f"Trong bài {lesson}", weight=1)
                G.add_edge(lesson_node, node_2, title=f"Trong bài {lesson}", weight=1)

   
    # Tạo PyVis network từ đồ thị
    net = Network(height="750px", width="100%", directed=True)  # directed=True để kích hoạt mũi tên
    net.from_nx(G)

    # Bật mũi tên trên các cạnh
    for edge in net.edges:
        edge['arrows'] = 'to'
        edge['width'] = 1 
        edge['arrowStrikethrough'] = False
        edge['arrowSize'] = 0.3

    # Tùy chỉnh PyVis
    net.show_buttons()
    net.toggle_physics(False)

    # Lưu đồ thị dưới dạng file HTML
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join('static', f"graph_{dataset}_{grade}_{lesson}_{current_time}.html")
    if os.path.exists(output_file):
        os.remove(output_file)  # Delete existing file if it exists
    net.save_graph(output_file)
    
    return output_file


def create_full_graph(data):
    purple = "#9370DB"
    light_blue = "#ADD8E6"
    green = "#00FFCC"
    # Tạo đồ thị NetworkX
    G = nx.Graph()

    # Duyệt qua từng lớp (dựa trên giá trị `id_doc` cột đầu)
    for grade in sorted(data['id_doc'].str.split('_').str[0].unique()):
        class_node = f"Lớp {grade}"
        G.add_node(class_node, label=class_node, color=green, size=25)

        # Lọc dữ liệu theo lớp hiện tại
        grade_data = data[data['id_doc'].str.startswith(grade)]

        # Duyệt qua từng bài trong lớp
        for lesson in sorted(grade_data['id_doc'].str.split('_').str[1].unique()):
            lesson_node = f"{class_node} - Bài {lesson}"
            G.add_node(lesson_node, label=f"Bài {lesson}", color=purple, size=20)
            
            # Liên kết node lớp với node bài
            G.add_edge(class_node, lesson_node, title=f"Bao gồm Bài {lesson}", weight=2)

            # Lọc dữ liệu theo bài hiện tại
            lesson_filter = f"{grade}_{lesson}_"
            lesson_data = grade_data[grade_data['id_doc'].str.startswith(lesson_filter)]

            # Thêm các node nội dung và liên kết chúng
            for _, row in lesson_data.iterrows():
                node_1 = row['final_head_node']
                node_2 = row['final_tail_node']
                relationship = row['nội dung liên kết']

                # Thêm node nội dung với màu tím
                G.add_node(node_1, title=row['keyword_head'], color=light_blue, size=15, label=row['keyword_head'])
                G.add_node(node_2, title=row['keyword_tail'], color=light_blue, size=15, label=row['keyword_tail'])
                
                # Thêm edge cho nội dung bài học
                G.add_edge(node_1, node_2, title=relationship, weight=4)

                # Liên kết các node nội dung với node bài hiện tại
                G.add_edge(lesson_node, node_1, title=f"Thuộc Bài {lesson}", weight=2)
                G.add_edge(lesson_node, node_2, title=f"Thuộc Bài {lesson}", weight=2)

    
    # Tạo PyVis network từ đồ thị
    net = Network(height="750px", width="100%", directed=True)  # directed=True để kích hoạt mũi tên
    net.from_nx(G)

    # Bật mũi tên trên các cạnh
    for edge in net.edges:
        edge['arrows'] = 'to'
        
    # Tùy chỉnh PyVis
    net.show_buttons()
    net.toggle_physics(False)

    # Lưu đồ thị dưới dạng file HTML
    output_file = "full_graph.html"
    if os.path.exists(output_file):
        os.remove(output_file)  # Delete existing file if it exists
    net.save_graph(output_file)
    
    return output_file