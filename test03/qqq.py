import cv2
import time

# MPII에서 각 파트 번호, 선으로 연결될 POSE_PAIRS
BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                "Background": 15 }

POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]
    
# 각 파일 path
protoFile = "C:/Users/iseo/Downloads/openpose-master/openpose-master/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "C:/Users/iseo/Downloads/openpose-master/openpose-master/models/pose/mpi/pose_iter_160000.caffemodel"
 
# network 불러오기
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# 캠 연ㄴ결
cap = cv2.VideoCapture(0)

# 디버깅 시작 시간 기록
start_time = time.time()

# 10초 디버깅 시작 후 횟수 측정 플래그
counting_started = False
tilted_count = 0
#-----------------------------------

# 초기화
tilt_threshold = 50
tilted_count = 0
prev_tilted = False
prev_x = None

while True:
    # 웹캠에서 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 크기 얻기
    imageHeight, imageWidth, _ = frame.shape
 
    # 전처리
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)
 
    # 네트워크에 입력
    net.setInput(inpBlob)

    # 결과 받아오기
    output = net.forward()

    # 결과 처리
    # 키포인트 검출시 이미지에 그려줌
    points = []
    for i in range(0,15):
        # 해당 신체부위 신뢰도
        probMap = output[0, i, :, :]
     
        # global 최대값 찾기
        _, prob, _, point = cv2.minMaxLoc(probMap)

        # 원래 이미지에 맞게 점 위치 변경
        x = int(imageWidth * point[0]) // output.shape[3]
        y = int(imageHeight * point[1]) // output.shape[2]

        # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로    
        if prob > 0.1 :    
            cv2.circle(frame, (x, y), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
            cv2.putText(frame, "{}".format(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            points.append((x, y))
        else :
            points.append(None)

    # 각 POSE_PAIRS별로 선 (머리 - 목, 목 - 왼쪽어깨.......)
    for pair in POSE_PAIRS:
        partA = pair[0]             # Head
        partA = BODY_PARTS[partA]   # 0
        partB = pair[1]             # Neck
        partB = BODY_PARTS[partB]   # 1
        
        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 0), 2)
# -----------------------------------------------------------------------------
    if not counting_started and time.time() - start_time > 10:
        counting_started = True
        
    # 현재 0번 신체부위의 x 좌표값
    current_x = points[0][0] if points[0] else None

    # 변화량이 30 이상이고, 이전 x 좌표값이 None이 아니고 현재 x 좌표값도 None이 아니면 카운트 증가
    if prev_x is not None and current_x is not None and abs(current_x - prev_x) >= 20:
        tilted_count += 1

    prev_x = current_x  # 현재 x 좌표값을 이전 x 좌표값으로 업데이트

    # 카운트를 화면에 표시
    cv2.putText(frame, f'moving count: {tilted_count}', (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#------------
# 10초가 지난 후부터 횟수 세기 시작
    #if not counting_started and time.time() - start_time > 10:
        #counting_started = True
    
    #if counting_started:
        # 0번 포인트의 좌표가 30 이상 움직일 때마다 카운트 증가
        #if points[0] and points[0][0] >= 30:
            #tilted_count += 1
            #print("moving count:", tilted_count)
#---------------------
    # 영상 출력
    cv2.imshow('ppogeulpops', frame)

    # 중지
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()