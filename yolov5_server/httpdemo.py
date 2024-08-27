from flask import Flask, request, Response
import logging
from yolo.yolov5_bmcv import create_task_yolo, get_result_yolo
from multiprocessing import Process, Queue
import json
from flask import jsonify
import argparse


class TASKS:
    def __init__(self) -> None:
        self.task_processes = {}
        self.result_queues = {}
        self.max_size = 0
        self.index = 0
    
    def set_queue_size(self, queue_size):
        self.max_size = queue_size

    def create_task(self, task_id, bmodel, dev_id, input_path, conf_thresh, nms_thresh):
        result_queue = Queue(maxsize=self.max_size)
        self.result_queues[task_id] = result_queue
        task_process = Process(target=create_task_yolo, args=(bmodel, dev_id, input_path, conf_thresh, nms_thresh, result_queue), daemon=True)
        self.task_processes[task_id] = task_process
        task_process.start()
    
    def get_result(self, task_id):
        return get_result_yolo(self.result_queues[task_id])


# 创建Flask应用对象
app = Flask(__name__)    
tasks = TASKS()


@app.route('/create', methods=['POST'])
def handle_create_request():
        
    # 获取客户端发送的JSON数据
    request_data = request.get_json()
    
    # 检查JSON数据中是否包含rtsp地址
    if 'video_pt' in request_data and 'task_id' in request_data:
        video_pt = request_data['video_pt']
        task_id = int(request_data['task_id'])
        if task_id in tasks.task_processes.keys():
            response_data = {'error': 'Task %d is already started'%(task_id)}
            return response_data, 400
        tasks.create_task(task_id, "model/yolov5s_v6.1_3output_int8_1b.bmodel", 0, video_pt, 0.5, 0.5)
        response_data = {'message': 'Task %d started'%(task_id)}
        return response_data, 200
    else:
        # 如果JSON数据中没有rtsp_url，则返回错误响应
        response_data = {'error': 'Missing video_pt or task_id'}
        return response_data, 400

@app.route('/result', methods=['POST'])
def handle_result_request():
    
    # 获取客户端发送的JSON数据
    request_data = request.get_json()

    if 'task_id' in request_data:
        task_id = int(request_data['task_id'])

        if task_id not in tasks.task_processes.keys():
            response_data = {'error': 'The task id %d does not exist'%(task_id)}
            return response_data, 400
        # 处理结果请求
        result = tasks.get_result(task_id)
        
        json_data = json.dumps(result)
        def generate_chunks():
            chunk_size = 4096
            for i in range(0, len(json_data), chunk_size):
                yield json_data[i:i+chunk_size]
        response = Response(generate_chunks(), content_type='application/json')
        return response, 200
    else:
        response_data = {'error': 'Missing task_id'}
        return response_data, 400
        

if __name__ == '__main__':
    # 设置日志记录到文件
    logging.basicConfig(filename='app.log', level=logging.INFO)

    tasks.set_queue_size(1)

    # 启动Flask应用，接收远程请求
    app.run(host='0.0.0.0', port=5000)
