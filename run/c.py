import json, sys, time, urllib.request, numpy as np, cv2
cp=json.load(open("calibration/front_camera_pose_arm_wide.json")); cam=json.load(open("calibration/camera_front.json"))
K=np.array(cam["K"]); D=np.array(cam["dist"]).ravel(); R=cv2.Rodrigues(np.array(cp["rvec"]))[0]; C=np.array(cp["camera_pos_robot_m"])
def at_z(u,v,z):
    d=cv2.undistortPoints(np.array([[[u,v]]],np.float64),K,D).reshape(2); r=R.T@np.array([d[0],d[1],1.]); return C+(z-C[2])/r[2]*r
A=lambda p,b=None,t=60: json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8891"+p,data=None if b is None else json.dumps(b).encode(),method="GET" if b is None else "POST"),timeout=t).read())
def shots(tag):
    for c in ("table","wrist"):
        im=cv2.imdecode(np.frombuffer(urllib.request.urlopen(f"http://127.0.0.1:8893/frame.jpg?c={c}",timeout=5).read(),np.uint8),1); cv2.imwrite(f"run/camscan/c_{tag}_{c}.jpg",im)
    fr=cv2.imread(f"run/camscan/c_{tag}_table.jpg")[140:340,80:400]; fr=cv2.resize(fr,None,fx=2,fy=2)
    for x in range(80,401,20):
        X=(x-80)*2; cv2.line(fr,(X,0),(X,fr.shape[0]),(190,190,190),1)
        if x%40==0: cv2.putText(fr,str(x),(X+2,12),0,.4,(0,0,255),1)
    for y in range(140,341,20):
        Y=(y-140)*2; cv2.line(fr,(0,Y),(fr.shape[1],Y),(190,190,190),1); cv2.putText(fr,str(y),(2,Y-2),0,.4,(0,0,255),1)
    w=cv2.resize(cv2.imread(f"run/camscan/c_{tag}_wrist.jpg"),(fr.shape[1],int(fr.shape[1]*0.75))); cv2.imwrite(f"run/camscan/c_{tag}.jpg",np.vstack([fr,w]))
