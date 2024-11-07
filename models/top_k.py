import json
from database.document import Document
from datasets import load_dataset
from huggingface_hub import notebook_login
import pandas as pd
import numpy as np
from pyvi.ViTokenizer import tokenize
from sentence_transformers import SentenceTransformer
import unicodedata
import regex as re
from pyvi import ViTokenizer, ViPosTagger
from tqdm import tqdm
from tqdm.notebook import tqdm
from rank_bm25 import BM25Okapi

def load_data():
    notebook_login("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")
    dataset_doc = load_dataset(
        "PB7-DUT-2023/Document_History_Embedding",  split="train")
    # df_dataset_doc = dataset_doc['train'].to_pandas()
    data_array = [dict(item) for item in dataset_doc]
    for data in data_array:
        embedding = json.loads(data['embedding'])
        document = Document(
            data['text_preprocessed_vietnamese'], embedding, data['id'])
        document.save_to_db()

def get_top_k(question, topk =10):
    notebook_login("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")
    dataset_doc = load_dataset("PB7-DUT-2023/Document_History_Embedding")
    df_dataset_doc = dataset_doc['train'].to_pandas()
    df_dataset_doc_processed = pd.DataFrame(columns=['id', 'text_preprocessed_vietnamese', 'embedding','id_book'])

    for idx,row in df_dataset_doc.iterrows():
        second_underscore_index = row.id.find("_", row.id.find("_") + 1)
        if second_underscore_index > -1:
            modified_item = row.id[:second_underscore_index]
            df_dataset_doc_processed.loc[idx]={"id": row.id, "text_preprocessed_vietnamese": row.text_preprocessed_vietnamese, "embedding": row.embedding, "id_book": modified_item}
    
    df_doc_embeddings = df_dataset_doc_processed.embedding
    df_doc_embeddings = [np.array([float(e) for e in ele[1:-1].split(',')]) for ele in df_doc_embeddings ]
    df_doc_embeddings /= np.linalg.norm(df_doc_embeddings, axis=1)[:, np.newaxis]

    model = SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')


    bang_nguyen_am= [['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
                    ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
                    ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
                    ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
                    ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
                    ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
                    ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
                    ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
                    ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
                    ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
                    ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
                    ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']]
    
    nguyen_am_to_ids = {}

    for i in range(len(bang_nguyen_am)):
        for j in range(len(bang_nguyen_am[i]) - 1):
            nguyen_am_to_ids[bang_nguyen_am[i][j]] = (i, j)
    
    def chuan_hoa_unicode(text):
        text = unicodedata.normalize('NFC', text)
        return text
    
    def chuan_hoa_dau_tu_tieng_viet(word):
        if not is_valid_vietnam_word(word):
            return word

        chars = list(word)
        dau_cau = 0
        nguyen_am_index = []
        qu_or_gi = False
        for index, char in enumerate(chars):
            x, y = nguyen_am_to_ids.get(char, (-1, -1))
            if x == -1:
                continue
            elif x == 9:  # check qu
                if index != 0 and chars[index - 1] == 'q':
                    chars[index] = 'u'
                    qu_or_gi = True
            elif x == 5:  # check gi
                if index != 0 and chars[index - 1] == 'g':
                    chars[index] = 'i'
                    qu_or_gi = True
            if y != 0:
                dau_cau = y
                chars[index] = bang_nguyen_am[x][0]
            if not qu_or_gi or index != 1:
                nguyen_am_index.append(index)
        if len(nguyen_am_index) < 2:
            if qu_or_gi:
                if len(chars) == 2:
                    x, y = nguyen_am_to_ids.get(chars[1])
                    chars[1] = bang_nguyen_am[x][dau_cau]
                else:
                    x, y = nguyen_am_to_ids.get(chars[2], (-1, -1))
                    if x != -1:
                        chars[2] = bang_nguyen_am[x][dau_cau]
                    else:
                        chars[1] = bang_nguyen_am[5][dau_cau] if chars[1] == 'i' else bang_nguyen_am[9][dau_cau]
                return ''.join(chars)
            return word

        for index in nguyen_am_index:
            x, y = nguyen_am_to_ids[chars[index]]
            if x == 4 or x == 8:  # ê, ơ
                chars[index] = bang_nguyen_am[x][dau_cau]
                # for index2 in nguyen_am_index:
                #     if index2 != index:
                #         x, y = nguyen_am_to_ids[chars[index]]
                #         chars[index2] = bang_nguyen_am[x][0]
                return ''.join(chars)

        if len(nguyen_am_index) == 2:
            if nguyen_am_index[-1] == len(chars) - 1:
                x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
                chars[nguyen_am_index[0]] = bang_nguyen_am[x][dau_cau]
                # x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
                # chars[nguyen_am_index[1]] = bang_nguyen_am[x][0]
            else:
                # x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
                # chars[nguyen_am_index[0]] = bang_nguyen_am[x][0]
                x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
                chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
        else:
            # x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            # chars[nguyen_am_index[0]] = bang_nguyen_am[x][0]
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
            # x, y = nguyen_am_to_ids[chars[nguyen_am_index[2]]]
            # chars[nguyen_am_index[2]] = bang_nguyen_am[x][0]
        return ''.join(chars)
    
    def is_valid_vietnam_word(word):
        chars = list(word)
        nguyen_am_index = -1
        for index, char in enumerate(chars):
            x, y = nguyen_am_to_ids.get(char, (-1, -1))
            if x != -1:
                if nguyen_am_index == -1:
                    nguyen_am_index = index
                else:
                    if index - nguyen_am_index != 1:
                        return False
                    nguyen_am_index = index
        return True


    def chuan_hoa_dau_cau_tieng_viet(sentence):
        """
            Chuyển câu tiếng việt về chuẩn gõ dấu kiểu cũ.
            :param sentence:
            :return:
            """
        sentence = sentence.lower()
        words = sentence.split()
        for index, word in enumerate(words):
            cw = re.sub(r'(^\p{P}*)([p{L}.]*\p{L}+)(\p{P}*$)', r'\1/\2/\3', word).split('/')
            if len(cw) == 3:
                cw[1] = chuan_hoa_dau_tu_tieng_viet(cw[1])
            words[index] = ''.join(cw)
        return ' '.join(words)
    
    def tach_tu_tieng_viet(text):
        text = ViTokenizer.tokenize(text)
        return text

    # Đưa về chữ viết thường
    def chuyen_chu_thuong(text):
        return text.lower()

    # Xóa đi các dấu cách thừa, các từ không cần thiết cho việc phân loại vẳn bản
    def chuan_hoa_cau(text):
        text = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]',' ',text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tien_xu_li(text):
        text = chuan_hoa_unicode(text)
        text = chuan_hoa_dau_cau_tieng_viet(text)
        # text = tach_tu_tieng_viet(text)
        text = chuyen_chu_thuong(text)
        text = chuan_hoa_cau(text)

        return text
    
    def embed_sentence(i = 0, row= "") :
        tokenized_sentence = tokenize(row)
        return model.encode(tokenized_sentence)

    def split_text(text):
        words = text.lower().split()
        words = [word for word in words if len(word.strip()) > 0]
        return words
    
    data = df_dataset_doc_processed['text_preprocessed_vietnamese'].tolist()

    tokenized_corpus = [split_text(e) for e in tqdm(data)]
    bm25 = BM25Okapi(tokenized_corpus)

    def get_doc_relevant_id(list_key, list_value):
        sums = {}
        counts = {}
        for key, value in zip(list_key,list_value):
            if key in sums:
                sums[key] += value
                counts[key] += 1
            else:
                sums[key] = value
                counts[key] = 1

        # Tính giá trị trung bình cho từng phần tử
        averages = {key: sums[key] / counts[key] for key in sums}

        # Tìm phần tử có giá trị trung bình cao nhất
        max_key = max(averages, key=averages.get)
        max_value = averages[max_key]

        doc_relevant = max_key[:max_key.find("_", max_key.find("_") + 1)]
        return doc_relevant
    
    def get_topk(question, topk =10):
        question = tien_xu_li(question)
        tokenized_query = split_text(question)
        question_emb = [embed_sentence(0,question)]
        question_emb /= np.linalg.norm(question_emb, axis=1)[:, np.newaxis]
        ## get BM25 and semantic scores
        bm25_scores = bm25.get_scores(tokenized_query)
        semantic_scores = question_emb @ df_doc_embeddings.T
        semantic_scores = semantic_scores[0]
        ## update chunks' scores.
        max_bm25_score = max(bm25_scores)
        min_bm25_score = min(bm25_scores)
        def normalize(x):
            return (x - min_bm25_score + 0.1) / (max_bm25_score - min_bm25_score + 0.1)

        normalize_bm25 = []
        for i in range(len(bm25_scores)):
            normalize_bm25.append(normalize(bm25_scores[i]))

        df_dataset_doc_processed.loc[:, 'bm25_score'] = bm25_scores
        df_dataset_doc_processed.loc[:, 'bm25_normed_score'] = normalize_bm25
        df_dataset_doc_processed.loc[:, 'semantic_score'] = semantic_scores
        ## compute combined score (BM25 + semantic)
        for i, book_row in df_dataset_doc_processed.iterrows():
            df_dataset_doc_processed.loc[i, "combined_score"] = book_row["bm25_normed_score"] * 0.5 + book_row["semantic_score"] * 0.5
        ## sort passages by the combined score
        df_sorted = df_dataset_doc_processed.sort_values(by=['combined_score'], ascending=False)
        list_combined_score = df_sorted[:topk]['combined_score'].to_list()
        list_books_id = df_sorted[:topk]['id'].to_list()

        doc_id = get_doc_relevant_id(list_books_id,list_combined_score)
        return doc_id
    
    doc_id = get_topk(question,topk)
    return doc_id
