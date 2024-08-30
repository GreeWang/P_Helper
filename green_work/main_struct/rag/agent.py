from green_work.main_struct.rag.create_kb import create_kb
from green_work.main_struct.rag.upload_files import upload_files
from green_work.main_struct.rag.question_request import send_request_without_history, send_request_with_history
from green_work.main_struct.rag.prompt import SUMMARIZER, SUMMARIZE_REQUIREMENT, QUESTIONER, QUESTIONER_REQUIREMENT, ANSWERER, ANSWERER_REQUIREMENT
from green_work.main_struct.rag.del_kb import delete_kb
from green_work.main_struct.rag.get_files_status import get_files_status

def summarize(localhost, folder_path):
    localhost = localhost
    folder_path = folder_path
    user_id = "zzp"
    kb_name = "test_kb"
    kb_id = create_kb(user_id, kb_name, localhost)
    upload_files(user_id, kb_id, folder_path, 'soft', localhost)
    get_files_status(user_id, kb_id, localhost)
    question1, answer1 = send_request_without_history(user_id, [kb_id], SUMMARIZE_REQUIREMENT, SUMMARIZER, localhost)
    question2, answer2 = send_request_with_history(user_id, [kb_id], QUESTIONER_REQUIREMENT, [[question1, answer1]], QUESTIONER, localhost)
    question3, answer3 = send_request_with_history(user_id, [kb_id], ANSWERER_REQUIREMENT, [[question1, answer1], [question2, answer2]], ANSWERER, localhost)
    delete_kb(user_id, [kb_id], localhost)
    return answer3

#summarize("0.0.0.0", '/home/dongpeijie/workspace/marker1/QAnything/test')