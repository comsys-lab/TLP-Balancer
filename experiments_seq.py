#!/usr/bin/python3

# Before run this code, 
# Execute 'nvcc -Xcompiler -fPIC -shared -o tlp_balancer.so src/ResNet50_ViT.cu' first.

import time
import sys
from multiprocessing import Queue, Value
from threading import Thread
from numpy import random
import ctypes
from ctypes import *

def inference(count, user_request, random_model_list, total, stopping, producer_status, finisher, execution_start_end_time_list, avg_inference_time):
    dll = ctypes.CDLL('./tlp_balancer.so', mode=ctypes.RTLD_GLOBAL)

    Res_ViT_host2gpu = dll.Res_ViT_host2gpu
    Res_ViT_host2gpu.argtypes = [POINTER(POINTER(POINTER(c_float)))] * 11

    Res_ViT_inference = dll.Res_ViT_inference
    Res_ViT_inference.argtypes = [POINTER(POINTER(c_float))] * 11 + [c_int] * 3

    Vgg_host2gpu = dll.Vgg_host2gpu
    Vgg_host2gpu.argtypes = [POINTER(POINTER(c_float))] * 54

    Vgg_inference = dll.Vgg_inference
    Vgg_inference.argtypes = [POINTER(c_float)] * 54 + [c_int]

    D_ResVit = [0] * 11
    for i in range(len(D_ResVit)):
        D_ResVit[i] = POINTER(POINTER(c_float))()

    print("[Inference] Res_ViT_host2gpu Start")
    Res_ViT_host2gpu( 
        D_ResVit[0], D_ResVit[1], D_ResVit[2], 
        D_ResVit[3], D_ResVit[4], D_ResVit[5],
        D_ResVit[6], D_ResVit[7], D_ResVit[8], D_ResVit[9], D_ResVit[10]
    )
    print("[Inference] Res_ViT_host2gpu Complete")

    # warm up
    for i in range(5):
        print("[Inference] Res_ViT Warmup", i+1, "Start")
        Res_ViT_inference(
            D_ResVit[0], D_ResVit[1], D_ResVit[2], 
            D_ResVit[3], D_ResVit[4], D_ResVit[5],
            D_ResVit[6], D_ResVit[7], D_ResVit[8], D_ResVit[9], D_ResVit[10],
            1, 1, 1
        )
    print("[Inference] Res_ViT Warmup Complete")

    D_Vgg = [0] * 54
    for i in range(len(D_Vgg)):
        D_Vgg[i] = POINTER(c_float)()

    print("[Inference] Vgg_host2gpu Start")
    Vgg_host2gpu(
        D_Vgg[0], D_Vgg[1], D_Vgg[2], D_Vgg[3], D_Vgg[4], D_Vgg[5], D_Vgg[6], D_Vgg[7], D_Vgg[8], D_Vgg[9], 
        D_Vgg[10], D_Vgg[11], D_Vgg[12], D_Vgg[13], D_Vgg[14], D_Vgg[15], D_Vgg[16], D_Vgg[17], D_Vgg[18], D_Vgg[19], 
        D_Vgg[20], D_Vgg[21], D_Vgg[22], D_Vgg[23], D_Vgg[24], D_Vgg[25], D_Vgg[26], D_Vgg[27], D_Vgg[28], D_Vgg[29], 
        D_Vgg[30], D_Vgg[31], D_Vgg[32], D_Vgg[33], D_Vgg[34], D_Vgg[35], D_Vgg[36], D_Vgg[37], D_Vgg[38], D_Vgg[39], 
        D_Vgg[40], D_Vgg[41], D_Vgg[42], D_Vgg[43], D_Vgg[44], D_Vgg[45], D_Vgg[46], D_Vgg[47], D_Vgg[48], D_Vgg[49],
        D_Vgg[50], D_Vgg[51], D_Vgg[52], D_Vgg[53]
    )
    print("[Inference] Vgg_host2gpu Complete")

    # warm up
    for i in range(5):
        print("[Inference] Vgg Warmup", i+1, "Start")
        Vgg_inference(
            D_Vgg[0], D_Vgg[1], D_Vgg[2], D_Vgg[3], D_Vgg[4], D_Vgg[5], D_Vgg[6], D_Vgg[7], D_Vgg[8], D_Vgg[9], 
            D_Vgg[10], D_Vgg[11], D_Vgg[12], D_Vgg[13], D_Vgg[14], D_Vgg[15], D_Vgg[16], D_Vgg[17], D_Vgg[18], D_Vgg[19], 
            D_Vgg[20], D_Vgg[21], D_Vgg[22], D_Vgg[23], D_Vgg[24], D_Vgg[25], D_Vgg[26], D_Vgg[27], D_Vgg[28], D_Vgg[29], 
            D_Vgg[30], D_Vgg[31], D_Vgg[32], D_Vgg[33], D_Vgg[34], D_Vgg[35], D_Vgg[36], D_Vgg[37], D_Vgg[38], D_Vgg[39], 
            D_Vgg[40], D_Vgg[41], D_Vgg[42], D_Vgg[43], D_Vgg[44], D_Vgg[45], D_Vgg[46], D_Vgg[47], D_Vgg[48], D_Vgg[49],
            D_Vgg[50], D_Vgg[51], D_Vgg[52], D_Vgg[53],
            1
        )
    print("[Inference] Vgg Warmup Complete")

    with open("warp_allocation_results.txt", "w"):
        pass

    with stopping.get_lock():
        stopping.value += 1
    print("[Inference] Ready!")

    while(stopping.value < 1):
        continue

    print("[Inference] Start!")

    total_res_num = 0
    total_vit_num = 0
    total_vgg_num = 0

    # inference 시작 시간 측정
    while True:
        # user_request가 5개 미만일 때 producer의 상태에 따라 continue 또는 break
        if user_request.qsize() < 5:
            if(producer_status.value == 0):
                continue
            elif (producer_status.value == 1):
                break

        # 
        file = open("results/" + "0_" + str(avg_inference_time) + ".txt","a")

        # 처리된 request가 total개가 되면 inference 종료
        request_count = count.qsize()
        if request_count > (total - 1):
            file.close()
            finisher.value = 1 
            break

        # queue에서 1개 빼기
        request_time = user_request.get()

        # model 확인하기
        model = random_model_list[0]
        del random_model_list[0]

        # inference 시작
        queuing_delay = 0
        inference_time = 0

        # Queuing delay 계산
        start_time = time.time()
        queuing_delay = max(start_time - request_time, 0)

        if model == 'Resnet':
            print("[Inference] Resnet Start")
            Res_ViT_inference(
                D_ResVit[0], D_ResVit[1], D_ResVit[2], 
                D_ResVit[3], D_ResVit[4], D_ResVit[5],
                D_ResVit[6], D_ResVit[7], D_ResVit[8], D_ResVit[9], D_ResVit[10],
                1, 0, 0
            )
        elif model == 'ViT':
            print("[Inference] ViT Start")
            Res_ViT_inference(
                D_ResVit[0], D_ResVit[1], D_ResVit[2], 
                D_ResVit[3], D_ResVit[4], D_ResVit[5],
                D_ResVit[6], D_ResVit[7], D_ResVit[8], D_ResVit[9], D_ResVit[10],
                0, 1, 0
            )
        elif model == 'Vgg':
            print("[Inference] Vgg Start")
            Vgg_inference(
                D_Vgg[0], D_Vgg[1], D_Vgg[2], D_Vgg[3], D_Vgg[4], D_Vgg[5], D_Vgg[6], D_Vgg[7], D_Vgg[8], D_Vgg[9], 
                D_Vgg[10], D_Vgg[11], D_Vgg[12], D_Vgg[13], D_Vgg[14], D_Vgg[15], D_Vgg[16], D_Vgg[17], D_Vgg[18], D_Vgg[19], 
                D_Vgg[20], D_Vgg[21], D_Vgg[22], D_Vgg[23], D_Vgg[24], D_Vgg[25], D_Vgg[26], D_Vgg[27], D_Vgg[28], D_Vgg[29], 
                D_Vgg[30], D_Vgg[31], D_Vgg[32], D_Vgg[33], D_Vgg[34], D_Vgg[35], D_Vgg[36], D_Vgg[37], D_Vgg[38], D_Vgg[39], 
                D_Vgg[40], D_Vgg[41], D_Vgg[42], D_Vgg[43], D_Vgg[44], D_Vgg[45], D_Vgg[46], D_Vgg[47], D_Vgg[48], D_Vgg[49],
                D_Vgg[50], D_Vgg[51], D_Vgg[52], D_Vgg[53],
                1
            )

        # inference time 계산
        end_time = time.time()
        inference_time = end_time - start_time

        # queueing delay 읽어와서 파일에 쓰기
        file.write(model + f"\t{(queuing_delay):.6f} \t {(inference_time):.6f}\n")
        count.put(0)

    # total execution의 종료 시간 측정
    execution_end_time = time.time()
    execution_start_end_time_list.append(execution_end_time)

def generate_request(user_request, queue_size, arrival_time, stopping, producer_status, finisher, execution_start_end_time_list, not_processed_list):
    not_processed = 0

    time.sleep(2)  

    print("[Request generator] Ready!")

    # 다른 process 기다리기
    while(stopping.value < 1):
        continue

    print("[Request generator] Start!")

    _lambda = arrival_time
    random.seed(0)
    poisson_list = random.exponential(_lambda, 2000)
    
    request_count = 0

    execution_start_time = time.time()
    execution_start_end_time_list.append(execution_start_time)

    for timeslot in poisson_list:
        time.sleep (timeslot)
        requested_time = time.time() # request 시간
        if finisher.value == 1:
            break

        if user_request.qsize() < queue_size:  # user_request에 공간이 있고 끝나지 않았으면 user_request에 requested_time 추가
            request_count += 1
            print("[Request Generator] Request generated",
                "(" + str(user_request.qsize()) + "/" + str(queue_size) + ") " + str(request_count))
            user_request.put(requested_time)                                
        elif finisher.value == 1:                                       # 끝나면 break
            break
        else:                                                           # user_request에 requested_time을 추가하지 못하면 not_processed 증가
            not_processed += 1
            print("[Request Generator] Request not queued up", not_processed)
            continue 
    
    # producer_status를 1로 바꿔서 producer가 동작을 마쳤음을 표시
    with producer_status.get_lock():
        producer_status.value = 1

    not_processed_list.append(not_processed)
    print('{0} requests not added to request queue'.format(not_processed))

stopping = Value('i', 0)            # for simultaneous starting of inferences between processes
producer_status = Value('i', 0)     # producer가 request 생성을 종료한 상태
finisher = Value('i', 0)            # request가 total개 처리되어 producer가 request를 만들지 않아도 되는 상태

if __name__ == '__main__':

    try:
        total = int(sys.argv[1])
        avg_inference_time = float(sys.argv[2])
    except:
        print("Error")
        print("Usage: ./experiments_seq.py total_num_requests delay")
        exit(1)

    user_request = Queue()  # user의 request를 저장하는 queue (request time 저장)
    count = Queue()         # 처리한 request의 개수를 저장하기 위한 queue

    execution_start_end_time_list = []
    not_processed_list = []
    random_model_list = []
    execution_start_end_time_list = []

    with open("random_model.txt") as f:
        for line in f:
            random_model_list.append(line.strip()) 

    arrival_time = avg_inference_time * (100/95)
     
    rho = avg_inference_time/arrival_time
    queue_size = round((rho*rho)/(1-rho))
    print("queue size:", queue_size)

    children = []

    children.append(Thread(target = inference,        args=(count, user_request, random_model_list, total, stopping, producer_status, finisher, execution_start_end_time_list, avg_inference_time)))
    children.append(Thread(target = generate_request, args=(user_request, queue_size, arrival_time, stopping, producer_status, finisher, execution_start_end_time_list, not_processed_list)))

    for child in children:
        child.start()

    for child in children:
        child.join()

    total_execution_time = execution_start_end_time_list[1] - execution_start_end_time_list[0]

    with open("results/" + "0_" + str(avg_inference_time) + ".txt","a") as file:
        file.write("------------------------------------------------" + '\n')
        file.write("mode: Sequential execution" + '\n')
        file.write(f"Total execution time: {(total_execution_time):.6f}" + '\n')
        file.write('Total processed requests: {0}'.format(count.qsize()) + '\n')
        file.write('Number of requests not queued: ' + str(not_processed_list[0]) + '\n')
        file.write("------------------------------------------------" + '\n')
