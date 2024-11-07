from flask import Flask, request, jsonify
import schedule
import time
import threading
from bson import ObjectId
from datetime import datetime, timedelta
from database.settings import Settings
from database.links import Links
from database.answers import Answers
from database.record import Records
from database.user import Users
from database.finetune import Finetune
from database.conversation import Conversations
from database.content import Contents
from utils.convert import custom_json_respone
from services.crawl import (
    Tracnghiem_Lichsu_url,
    Chitiet_HTML,
    handle,
    Test_Chitiet_HTML,
)
from flask_cors import CORS
from models.tranformer import testModel, model_api
from models.finetuned import finetune_model
from models.top_k import get_top_k
from models.top_k_graph import get_top_k_graph
from models.graph import create_graph_by_lessons
import math
from flask_bcrypt import Bcrypt
from flask_jwt_extended import (
    JWTManager,
    jwt_required,
    create_access_token,
    get_jwt_identity,
)
import os

# app = Flask(__name__, static_folder='public', static_url_path='')
app = Flask(__name__, static_folder="static")

app.config["SECRET_KEY"] = "PBL7"
bcrypt = Bcrypt(app)
jwt = JWTManager(app)
CORS(app)


def crawl_data():
    urls_10 = Tracnghiem_Lichsu_url(
        "https://vietjack.com/bai-tap-trac-nghiem-lich-su-10/index.jsp"
    )
    urls_11 = Tracnghiem_Lichsu_url(
        "https://vietjack.com/bai-tap-trac-nghiem-lich-su-11/index.jsp"
    )
    urls_12 = Tracnghiem_Lichsu_url(
        "https://vietjack.com/bai-tap-trac-nghiem-lich-su-12/index.jsp"
    )
    urls = urls_10 + urls_11 + urls_12
    current_urls = Links().get_all_links()
    new_links_to_add = []
    for link in urls:
        if (
            link not in [existing_link["name"] for existing_link in current_urls]
            and len(new_links_to_add) < 1000
        ):
            new_links_to_add.append(link)

    total_questions = 0
    for url in new_links_to_add:
        link = Links(url)
        link.save_to_db()
        # results.append(link.__dict__)
        answers = Test_Chitiet_HTML(url)
        questions = []
        for data in answers:
            if len(data) == 4:
                has_nan = False
                for key, value in data.items():
                    if isinstance(value, float) and math.isnan(value):
                        has_nan = True
                        break
                if not has_nan:
                    answer = Answers(
                        data["question"],
                        data["answers"],
                        data["explanation"],
                        data["correct_answer"],
                        ObjectId(link._id),
                    )
                    answer.save_to_db()
                    questions.append(ObjectId(answer._id))
        link.questions = questions
        link.save_to_db()
        total_questions += len(link.questions)

    record = Records(total_questions, new_links_to_add)
    record.save_to_db()


def my_job():
    setting = Settings()
    setting.find_one()
    current_date = datetime.now()
    time_str = current_date.strftime("%H:%M")
    print(current_date.day, current_date.weekday(), time_str)
    if setting.crawl:
        if setting.type == "month":
            if current_date.day == setting.date and time_str == setting.time:
                print("crawl")
                crawl_data()
        else:
            current_date = datetime.now()
            if current_date.weekday() == setting.date and time_str == setting.time:
                print("crawl")
                crawl_data()


# schedule.every(10).seconds.do(my_job)


def schedule_thread():
    while True:
        schedule.run_pending()
        time.sleep(1)


# schedule_t = threading.Thread(target=schedule_thread)
# schedule_t.start()
# schedule_t.join()


@app.route("/")
def home():
    try:
        q = "Việc Liên Xô chế tạo thành công bom nguyên tử có ý nghĩa gì ? A. Khẳng định vai trò to lớn của Liên Xô đối cách mạng thế giới. B. Đưa thế giới bước vào thời đại chiến tranh hạt nhân. C. Thế độc quyền vũ khí nguyên tử của Mĩ bị phá vỡ. D. Khiến Liên Xô trở thành nước đầu tiên sở hữu vũ khí nguyên tử."
        top_k = get_top_k(q, topk=15)
        print(top_k)
        return jsonify({"message": top_k})
    except Exception as e:
        return jsonify({"message": f"Lỗi khi kết nối đến MongoDB: {e}"}), 500


@app.get("/api/load-data-from-huggingface")
def get_data_hugging_face():
    try:
        # load_data()
        return jsonify({"message": "Tải xuống thành công"})
    except Exception as e:
        return jsonify({"message": f"Lỗi khi kết nối đến MongoDB: {e}"}), 500


@app.get("/api/test")
def test():
    # handle()
    # urls_10 = Tracnghiem_Lichsu_url(
    #     "https://vietjack.com/bai-tap-trac-nghiem-lich-su-10/index.jsp")
    # data = Test_Chitiet_HTML(urls_10[1])
    # print(urls_10)

    return {}, 200


@app.post("/api/auth/create")
def create_account():
    data = request.json
    user = Users()
    res = user.find_by_email(data["email"])
    if res:
        return jsonify({"message": "This email is already used!"}), 400
    else:
        user.name = data["name"]
        user.email = data["email"]
        hashed_password = bcrypt.generate_password_hash(data["password"]).decode(
            "utf-8"
        )
        user.password = hashed_password
        user.role = 1
        user.save_to_db()
        return jsonify(user.__dict__), 200


@app.post("/api/auth/login")
def login():
    data = request.json
    user = Users()
    check = user.find_by_email(data["email"])
    if check:
        is_valid = bcrypt.check_password_hash(user.password, data["password"])
        if is_valid:
            access_token = create_access_token(
                identity=user.__dict__, expires_delta=timedelta(days=30)
            )
            return jsonify({"user": user.__dict__, "access_token": access_token}), 200
        else:
            return jsonify({"message": "Invalid email or password"}), 401
    else:
        return jsonify({"message": "Invalid email or password"}), 401


@app.post("/api/crawl")
def crawl():
    urls_10 = Tracnghiem_Lichsu_url(
        "https://vietjack.com/bai-tap-trac-nghiem-lich-su-10/index.jsp"
    )
    urls_11 = Tracnghiem_Lichsu_url(
        "https://vietjack.com/bai-tap-trac-nghiem-lich-su-11/index.jsp"
    )
    urls_12 = Tracnghiem_Lichsu_url(
        "https://vietjack.com/bai-tap-trac-nghiem-lich-su-12/index.jsp"
    )
    urls = urls_10 + urls_11 + urls_12
    current_urls = Links().get_all_links()
    new_links_to_add = []
    for link in urls:
        if (
            link not in [existing_link["name"] for existing_link in current_urls]
            and len(new_links_to_add) < 1000
        ):
            new_links_to_add.append(link)

    total_questions = 0
    for url in new_links_to_add:
        link = Links(url)
        link.save_to_db()
        # results.append(link.__dict__)
        answers = Test_Chitiet_HTML(url)
        questions = []
        for data in answers:
            if len(data) == 4:
                has_nan = False
                for key, value in data.items():
                    if isinstance(value, float) and math.isnan(value):
                        has_nan = True
                        break
                if not has_nan:
                    answer = Answers(
                        data["question"],
                        data["answers"],
                        data["explanation"],
                        data["correct_answer"],
                        ObjectId(link._id),
                    )
                    answer.save_to_db()
                    questions.append(ObjectId(answer._id))
        link.questions = questions
        link.save_to_db()
        total_questions += len(link.questions)

    record = Records(total_questions, new_links_to_add)
    record.save_to_db()

    return jsonify(record.__dict__), 200


@app.get("/api/settings")
def get_setting():
    setting = Settings()
    res = setting.find_one()
    return jsonify(res), 200


@app.post("/api/finetune")
@jwt_required()
def finetune():
    url = finetune_model()
    finetuned = Finetune(100, url)
    finetuned.save_to_db()
    return jsonify(finetuned.__dict__), 200


@app.get("/api/finetune")
@jwt_required()
def get_finetune():
    finetuned = Finetune()
    data = finetuned.get_all()
    return jsonify(data), 200


@app.post("/api/settings")
def create_setting():
    # data = request.json
    # name = data['name']
    # time = data['time']
    # id = data["id"]
    # setting = Settings(True, 'week', 6, '15:15:00')
    # setting.save_to_db()
    return jsonify({"message": "Setting created successfully"}), 200


@app.put("/api/settings")
def update_setting():
    setting = Settings()
    res = setting.find_one()
    data = request.json
    print(setting._id)
    setting.crawl = data["crawl"]
    setting.time = data["time"]
    setting.type = data["type"]
    setting.date = data["date"]
    setting.save_to_db()
    return jsonify({"message": "Setting update successfully"}), 200


@app.get("/api/links")
def get_links():
    page = int(request.args.get("page", 0))
    # Số lượng bản ghi trên mỗi trang
    per_page = int(request.args.get("per_page", 0))
    res = Links()
    result = res.get_all_links(page, per_page)
    return jsonify(result), 200


@app.delete("/api/links/<id>")
def delete_link(id):
    link = Links()
    ans = Answers()
    try:
        link.find_by_id(id)
        for answer_id in link.questions:
            data = str(answer_id)
            ans.delete_by_id(data)
        link.delete_by_id(id)

        return jsonify({"message": "Delete success"}), 200
    except:
        return jsonify({"message": "Delete fail"}), 400


@app.get("/api/links/<id>")
def get_link_detail(id):
    try:
        res = Links()
        result = res.find_by_id(id)
        if result:
            return jsonify(res.__dict__), 200
        else:
            return jsonify({"message": "Not found"}), 404
    except:
        return jsonify({}), 400


@app.get("/api/links/<id>/answers")
def get_link_answers(id):
    link = Links()
    link.find_by_id(id)
    answers = Test_Chitiet_HTML(link.name)
    return jsonify(answers), 200


@app.post("/api/links/<id>/crawl")
def crawl_answers_by_link(id):
    link = Links()
    link.find_by_id(id)
    answers = Test_Chitiet_HTML(link.name)
    questions = []
    for data in answers:
        if len(data) == 4:
            has_nan = False
            for key, value in data.items():
                if isinstance(value, float) and math.isnan(value):
                    has_nan = True
                    break
            if not has_nan:
                answer = Answers(
                    data["question"],
                    data["answers"],
                    data["explanation"],
                    data["correct_answer"],
                    ObjectId(id),
                )
                answer.save_to_db()
                questions.append(ObjectId(answer._id))

    link.questions = questions
    link.save_to_db()
    return jsonify(answers), 200


@app.get("/api/crawl")
def get_crawl_his():
    record = Records()
    data = record.get_all()
    return jsonify(data), 200


@app.get("/api/answers")
def get_answers():
    params = request.args
    name = params.get("name")
    setting = Settings()
    result = setting.find_by_id(name)
    return jsonify({"message": "success"}), 200


@app.delete("/api/delete")
def test_delete():
    link = Links()
    answers = Answers()
    link.delete_by_month()
    answers.delete_by_month()
    return jsonify({"message": "success"}), 200


@app.get("/api/dashboard")
def get_infor_dashboard():
    link = Links()
    answer = Answers()
    user = Users()
    content = Contents()

    total_content = content.get_total()
    total_content_feedback = content.get_total_feedback()
    type_feedback = content.get_type_feedback()

    total_users = user.get_total()

    links_data = link.get_lated_data()
    total_links = link.get_total()

    total_answers = answer.get_total()
    total_pending = link.get_pending()

    links_months = link.get_data_months()
    links_grown = 0
    if datetime.utcnow().month > 1:
        cur_month = links_months[datetime.utcnow().month - 1]
        last_month = links_months[datetime.utcnow().month - 2]
        links_grown = (cur_month - last_month) / last_month

    answers_months = answer.get_data_months()
    answers_grown = 0
    if datetime.utcnow().month > 1:
        cur_month = answers_months[datetime.utcnow().month - 1]
        last_month = answers_months[datetime.utcnow().month - 2]
        answers_grown = (cur_month - last_month) / last_month

    return jsonify(
        {
            "contents": {
                "feedback": total_content_feedback,
                "no_feedback": total_content - total_content_feedback,
                "type_feedback": type_feedback,
            },
            "users": {"total": total_users, "growth": True, "value": "0.0%"},
            "links": {
                "total": total_links,
                "growth": links_grown > 0,
                "value": str(abs(round(links_grown * 100, 2))) + "%",
                "per_months": links_months,
                "data": links_data,
            },
            "answers": {
                "total": total_answers,
                "growth": answers_grown > 0,
                "value": str(abs(round(answers_grown * 100, 2))) + "%",
                "per_months": answers_months,
            },
            "pending": {
                "total": total_pending,
            },
        }
    ), 200


@app.post("/api/model/test")
def test_model():
    # API_URL = "https://api-inference.huggingface.co/models/PB7-DUT-2023/finetuned_Bloomz_1b1_v1"
    # headers = {"Authorization": "Bearer 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"}
    testModel()
    return {}, 200


# @app.post('/api/customer/question')
# def generate_question_customer():
#     data = request.json
#     try:
#         success, answer_part, explain_part = model_api(
#             data["question"], data["answers"], data["version"])
#         top_k = get_top_k(data["question"], 15)
#         if success == True:
#             return jsonify({
#                 "correct_answer": answer_part,
#                 "explanation": explain_part,
#                 "top_k": top_k
#             })
#         else:
#             [grade, lesson] = top_k.split('_')
#             return jsonify({
#                 "message": f"Waiting for loading model, but you can search the answer in history book grade {grade} lesson {lesson}"
#             }), 400
#     except ValueError as e:
#         return jsonify({
#             "message": e,
#         }), 400
def format_question_answers(question, answers):
    # If answers is already a list, use it directly
    if isinstance(answers, list):
        answer_list = answers
    else:
        # Convert string answers to list if needed
        try:
            # Use ast.literal_eval for safer string to list conversion
            import ast

            answer_list = ast.literal_eval(answers)
        except:
            # Fallback to simple string splitting if ast.literal_eval fails
            answer_list = answers.strip("[]").replace("'", "").split(", ")

    # Combine question and answers with a single space
    result = "{} {}".format(question, " ".join(answer_list))

    return result


@app.post("/api/customer/question")
def generate_question_customer():
    data = request.json
    try:
        formatted_question = format_question_answers(data["question"], data["answers"])
        # success, answer_part, explain_part = model_api(
        #     data["question"], data["answers"], data["version"])
        top_k = get_top_k_graph(formatted_question)
        # if success == True:
        #     return jsonify({
        #         "correct_answer": answer_part,
        #         "explanation": explain_part,
        #         "top_k": top_k
        #     })
        # else:
        #     # [grade, lesson] = top_k.split('_')
        #     return jsonify({
        #         # "message": f"Waiting for loading model, but you can search the answer in history book grade {grade} lesson {lesson}"
        #         "message": "Waiting for loading model, but you can search the answer {top_k} "
        #     }), 400
        print(top_k)
        return jsonify({"top_k": top_k}), 200
    except ValueError as e:
        return jsonify(
            {
                "message": e,
            }
        ), 400


@app.post("/api/customer/graph")
def generate_graph():
    data = request.json
    graph = ""
    try:
        graph = create_graph_by_lessons(data["dataset"], data["grade"], data["lesson"])
        if graph:
            print(graph)
            return jsonify(
                {
                    "graph": graph,
                    # "message": f"Graph showed!!"
                }
            ), 200
    except ValueError as e:
        return jsonify(
            {
                "message": e,
            }
        ), 400


@app.post("/api/user/question")
@jwt_required()
def generate_question_user():
    current_user = get_jwt_identity()
    data = request.json
    try:
        success, answer_part, explain_part = model_api(
            data["question"], data["answers"], data["version"]
        )
        top_k = get_top_k(data["question"], 15)
        print(top_k)
        if success == True:
            conversation = Conversations(current_user["_id"], [], data["question"])

            if data["conversationId"] != "":
                print("here")
                conversation.find_by_id(data["conversationId"])
            else:
                conversation.save_to_db()

            contentId = conversation.contents
            ask = Contents(conversation._id, "ask", data["question"], data["answers"])
            ask.save_to_db()
            contentId.append(ask._id)
            respone = Contents(
                conversation._id,
                "answer",
                data["question"],
                data["answers"],
                explain_part,
                answer_part,
                version=data["version"],
                top_k=top_k,
            )
            respone.save_to_db()
            contentId.append(respone._id)
            conversation.contents = contentId
            conversation.save_to_db()

            return jsonify(
                {
                    "correct_answer": answer_part,
                    "explanation": explain_part,
                    "conversation_id": conversation._id,
                    "answer_id": respone._id,
                    "top_k": top_k,
                }
            )
        else:
            [grade, lesson] = top_k.split("_")
            return jsonify(
                {
                    "message": f"Waiting for loading model, but you can search the answer in history book grade {grade} lesson {lesson}"
                }
            ), 400
    except ValueError as e:
        return jsonify(
            {
                "message": e,
            }
        ), 400


@app.get("/api/user/conversation")
@jwt_required()
def get_user_conversation():
    current_user = get_jwt_identity()
    conversation = Conversations()
    print(current_user["_id"])
    res = conversation.find_by_user_id(current_user["_id"])

    return jsonify(res), 200


@app.get("/api/user/conversation/<id>")
@jwt_required()
def get_conversation_content(id):
    conversation = Conversations()
    conversation.find_by_id(id)
    data = []
    for text in conversation.contents:
        content = Contents()
        content.find_by_id(text)
        data.append(content.__dict__)

    return jsonify(data), 200


@app.get("/api/admin/users")
@jwt_required()
def admin_get_list_user():
    user = Users()
    current_user = get_jwt_identity()
    if current_user["role"] == 2:
        res = user.get_all_user()
        return jsonify(res), 200
    else:
        return jsonify({"message": "You don't have permission to do this action."}), 403


@app.post("/api/user/content/<id>/rate")
@jwt_required()
def rate_content(id):
    content = Contents()
    content.find_by_id(id)
    data = request.json

    content.bad_response = True
    content.feedback = data["feedback"]

    content.save_to_db()
    return jsonify({"message": "Thank you for your feedback"}), 200


if __name__ == "__main__":
    with open("pid_backend", "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    schedule.every(1).minutes.do(my_job)

    schedule_t = threading.Thread(target=schedule_thread)
    schedule_t.start()

    app.run(debug=False, port=3001, host="0.0.0.0")
