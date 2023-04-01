import numpy as np
import cv2 as cv
import time
import mediapipe as mp
import bpy
import threading
import math
from mathutils import Vector 
from mathutils import Matrix

from bpy import context



_cap = cv.VideoCapture(0)
_cap.set(cv.CAP_PROP_FRAME_WIDTH, 512)
_cap.set(cv.CAP_PROP_FRAME_HEIGHT, 512)
_cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

mpPose = mp.solutions.pose
pose=mpPose.Pose()
mpDraw= mp.solutions.drawing_utils

facemark = cv.face.createFacemarkLBF()




#Puts a specified bone in its origional location
def putInOrig(bone):
    armature_obj = bpy.data.objects["rig"]
    bpy.context.view_layer.objects.active = armature_obj
    edit_bone= armature_obj.data.bones[bone]
    boney= armature_obj.pose.bones[edit_bone.name]
    boney.bone.select = True
    
    boney.location.x = 0
    boney.location.y = 0
    boney.location.z = 0
    

#gets the global location of a bone given the bone's armature name
def getZ(bone):
       
    glo_po=bone.matrix @ bone.location
    
    
    X = glo_po[0]
    
    Y = glo_po[1]
    
    Z = glo_po[2] 
    
    return X,Y,Z
    

#gets the global location of a bone given the bone's string name
def getXYZ(bone):
    armature_obj = bpy.data.objects["rig"]
    bpy.context.view_layer.objects.active = armature_obj
    

    
    edit_bone= armature_obj.data.bones[bone]
    boney= armature_obj.pose.bones[edit_bone.name]
    boney.bone.select = True
    
    
    
    glo_po=boney.matrix @ boney.location
    
    

    
    X = glo_po[0]
    
    Y = glo_po[1]
    
    Z = glo_po[2] 
    
    return X,Y,Z


#gets the distance between two points

def getDist(firstpoint, secondpoint):
    
    x=(secondpoint[0]-firstpoint[0])**2
    
    y=(secondpoint[1]-firstpoint[1])**2
    
    z=(secondpoint[2]-firstpoint[2])**2
    
    d= math.sqrt((x+y+z))
    
    return d
    

#gets the change vector between two points
def getVect(v1, v2):
    
    x1 = v1[0]
    y1 = v1[1]
    z1 = v1[2]
    
    x2 = v2[0]
    y2 = v2[1]
    z2 = v2[2]
    
    vx = x2-x1
    vy = y2-y1
    vz = z2-z1
    

    
    v=(vx,vz,vy)
    
    return v


#returns a vector of distance d in the direction of v 
def slopeVec(d,v):
    

    d=d**2
    
    nv= v[0]*v[0] + v[1]*v[1] + v[2] * v[2]
    a= math.sqrt(d/(nv))
    
    x= (v[0] *a, v[1]*a, v[2]*a)
    
    
    
    return x

#moves the bone to its new location
def movePoint(bone, obone, point):
    

    
    armature_obj = bpy.data.objects["rig"]
    bpy.context.view_layer.objects.active = armature_obj
    edit_bone= armature_obj.data.bones[bone]
    boney= armature_obj.pose.bones[edit_bone.name]
    boney.bone.select = True
    
    OboneLoc= getXYZ(obone)

    
    
    location= (point[0]+OboneLoc[0],point[1]+OboneLoc[1],point[2]+OboneLoc[2])
    

    
 

    
    locate= np.array([location[0],location[1],location[2],1])
    
    loc=getCords(boney,locate)
    
    
    boney.location.x=loc[0]
    boney.location.y=loc[1]
    boney.location.z=loc[2]

    
# gets the slope vecotr and moves the data point
def runData(bone,obone,d,vector):


        
    v= slopeVec(d,vector)
      
    movePoint(bone,obone, v)
        

        
# gets the local coordinates of a bone to move it to a global position
def getCords(boney1,loc):

    A = np.array([[boney1.matrix[0][0],boney1.matrix[0][1],boney1.matrix[0][2],boney1.matrix[0][3]],[boney1.matrix[1][0],boney1.matrix[1][1],boney1.matrix[1][2],boney1.matrix[1][3]],[boney1.matrix[2][0],boney1.matrix[2][1],boney1.matrix[2][2],boney1.matrix[2][3]],[boney1.matrix[3][0],boney1.matrix[3][1],boney1.matrix[3][2],boney1.matrix[3][3]]])
    B = loc
    
    s=np.linalg.solve(A,B)
    
    
    return s
    

#updates the rig 
def establish(Joint, frame_c):
    

    Hold=[]
    

    
    for i in range (len(Joint)):
        
        JointPoint=(getZ(Joint[i]))
        
        
        Hold.append(JointPoint)
        
    
    #updating the rig
    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    
    #setting the bones back to the position they were in before
    for i in range (len(Hold)):
        Holdx=(Hold[i][0])
        Holdy=(Hold[i][1])
        Holdz=(Hold[i][2])
        destination=np.array([Holdx,Holdy,Holdz,1])
        newLoc=getCords(Joint[i],destination)
        Joint[i].location.x=newLoc[0]
        Joint[i].location.y=newLoc[1]
        Joint[i].location.z=newLoc[2]


    
    bpy.context.scene.frame_set(frame_c)
    



#uses openCV to get the keydata points and return a vector of the change in points between each data point
def getAllVect(frame):
        
    
    
    Hold = []
    


    imgRGB= cv.cvtColor(frame, cv.COLOR_BGR2RGB)
   
    results = pose.process(imgRGB)
   
    mpDraw.draw_landmarks(frame, results.pose_landmarks)
    
    for id, lm in enumerate(results.pose_landmarks.landmark):
        
        x, y, z=(lm.x), (lm.y), (lm.z)
        
        x= (round(x*100))/100
        
        y= (round(y*100))/100
        
        z= (round(z*100))/100
        
        h,w, c = frame.shape
        

        
        x, y, z=(lm.x*w), (lm.y*h), (lm.z)
        
        

            
        
        temp = (lm.x,2*(1-lm.y),lm.z)
        
        
        
        
        
        Hold.append(temp)
        
#getting the vector of each bone
        
    RArm = getVect(Hold[12],Hold[14])
    
    RFArm = getVect(Hold[14],Hold[16])
    
    LArm = getVect(Hold[11],Hold[13])
    
    LFArm = getVect(Hold[13],Hold[15])
    
    RLeg = getVect(Hold[24],Hold[26])
    
    RShin = getVect(Hold[26],Hold[28])
    
    RFoot= getVect(Hold[28],Hold[32])
    
    LLeg = getVect(Hold[23],Hold[25])
    
    LShin = getVect(Hold[25],Hold[27])
    
    LFoot= getVect(Hold[27],Hold[31])

    
    
    Vect=[RArm,RFArm,LArm,LFArm,RLeg,RShin,RFoot,LLeg,LShin,
    LFoot]
    
    return Vect, Hold[9] , Hold[10], Hold[0]
            
        

#function that rotates the head   
def moveHead(head, LHP,RHP,THP):
    
    d = math.sqrt((LHP[0]-RHP[0])**2 + (LHP[1]-RHP[1])**2)

    
    horizontalChange = (LHP[0]/d-RHP[0]/d, LHP[1]/d-RHP[1]/d)
    
    midPoint = ((LHP[0]+RHP[0])/2, (LHP[1]+RHP[1])/2)
    
    TopDistance = math.sqrt((THP[0]-midPoint[0])**2 + (THP[1]-midPoint[1])**2)

    
    TopPoint= (THP[0]/TopDistance, THP[1]/TopDistance)
    
    verticalChange = TopPoint[0]-midPoint[0]/TopDistance, TopPoint[1]- midPoint[1]/TopDistance

    
    head.rotation_euler.y =   math.pi*verticalChange[0]
    head.rotation_euler.z = math.pi* horizontalChange[1]/2

    
     
#calls all the other functions and runs the program
def avatarMe():

    
    frame_count = 0
    update_frequency = 10
    
    #setting the names of all the joints
    
    LeftHand = "hand_tweak.L"
 
    
    LeftElbow = "forearm_tweak.L"

    
    LeftShoulder = "upper_arm_tweak.L"

    
    RightHand = "hand_tweak.R"

    RightElbow = "forearm_tweak.R"

    
    RightShoulder = "upper_arm_tweak.R"
    
    RightHip = "thigh_tweak.R"
    
    RightKnee = "shin_tweak.R"
    
    RightAnkle = "foot_tweak.R"
    
    RightToe =  "toe_ik.R"

    
    LeftHip = "thigh_tweak.L"
    
    LeftKnee = "shin_tweak.L"
    
    LeftAnkle = "foot_tweak.L"
    
    LeftToe =  "toe_ik.L"

    
    hbone = "head"
    
    

    
    #loading the bones armature so operations can be ran on them

    armature_obj = bpy.data.objects["rig"]
    bpy.context.view_layer.objects.active = armature_obj

    
    edit_hbone= armature_obj.data.bones[hbone]
    Head= armature_obj.pose.bones[edit_hbone.name]
    Head.bone.select = True

    
    edit_bone1= armature_obj.data.bones[LeftHand]
    boney1= armature_obj.pose.bones[edit_bone1.name]
    boney1.bone.select = True
    
    edit_bone2= armature_obj.data.bones[LeftElbow]
    boney2= armature_obj.pose.bones[edit_bone2.name]
    boney2.bone.select = True
    
    edit_bone3= armature_obj.data.bones[LeftShoulder]
    boney3= armature_obj.pose.bones[edit_bone3.name]
    boney3.bone.select = True
    
    edit_bone4= armature_obj.data.bones[RightHand]
    boney4= armature_obj.pose.bones[edit_bone4.name]
    boney4.bone.select = True
    
    edit_bone5= armature_obj.data.bones[RightElbow]
    boney5= armature_obj.pose.bones[edit_bone5.name]
    boney5.bone.select = True
    
    
    edit_bone6= armature_obj.data.bones[RightShoulder]
    boney6= armature_obj.pose.bones[edit_bone6.name]
    boney6.bone.select = True
    
    edit_bone7= armature_obj.data.bones[RightHip]
    boney7= armature_obj.pose.bones[edit_bone7.name]
    boney7.bone.select = True
    
    edit_bone8= armature_obj.data.bones[RightKnee]
    boney8= armature_obj.pose.bones[edit_bone8.name]
    boney8.bone.select = True
    
    edit_bone9= armature_obj.data.bones[RightAnkle]
    boney9= armature_obj.pose.bones[edit_bone9.name]
    boney9.bone.select = True
    
    
    edit_bone10= armature_obj.data.bones[RightToe]
    boney10= armature_obj.pose.bones[edit_bone10.name]
    boney10.bone.select = True
    
    
    edit_bone11= armature_obj.data.bones[LeftHip]
    boney11= armature_obj.pose.bones[edit_bone11.name]
    boney11.bone.select = True
    
    edit_bone12= armature_obj.data.bones[LeftKnee]
    boney12= armature_obj.pose.bones[edit_bone12.name]
    boney12.bone.select = True
    
    edit_bone13= armature_obj.data.bones[LeftAnkle]
    boney13= armature_obj.pose.bones[edit_bone13.name]
    boney13.bone.select = True
    
    
    edit_bone14= armature_obj.data.bones[LeftToe]
    boney14= armature_obj.pose.bones[edit_bone14.name]
    boney14.bone.select = True
    

 
    #getting the distance between joints (The distance of the bone)
    
    LELSD = getDist(getXYZ(LeftElbow),getXYZ(LeftShoulder))
        
        
        
    LHLED = getDist(getXYZ(LeftElbow),getXYZ(LeftHand))
        
    
        
    RERSD = getDist(getXYZ(RightElbow),getXYZ(RightShoulder))
        
    RHRED = getDist(getXYZ(RightElbow),getXYZ(RightHand))
    
    
    RHRKD = getDist(getXYZ(RightHip),getXYZ(RightKnee))
        
    RKRAD = getDist(getXYZ(RightKnee),getXYZ(RightAnkle))
        
    RARTD = getDist(getXYZ(RightAnkle),getXYZ(RightToe))
    
    
        
    LHLKD = getDist(getXYZ(LeftHip),getXYZ(LeftKnee))
        
    LKLAD = getDist(getXYZ(LeftKnee),getXYZ(LeftAnkle))
        
    LALTD = getDist(getXYZ(LeftAnkle),getXYZ(LeftToe))
    
  


 
    while True:
        
        
        
        _, frame = _cap.read()
        
        vec,LHP, RHP, THP = getAllVect(frame)
        
        cv.imshow("img", frame)
        
        moveHead(Head, LHP, RHP, THP)
        
        
        #Function moves the bones to the location given by the opencv KeyPoint Detection    
    
        runData(LeftElbow, LeftShoulder,LELSD,vec[2])

        
        runData(LeftHand,LeftElbow,LHLED,vec[3])
      
        
        
        runData(RightElbow, RightShoulder,RERSD,vec[0])
        
        
        runData(RightHand, RightElbow,RHRED,vec[1])
       
        runData(RightKnee, RightHip,RHRKD, vec[4])
        
        
        runData(RightAnkle, RightKnee, RKRAD, vec[5])
        
        runData(RightToe, RightAnkle, RARTD, vec[6])
      
        runData(LeftKnee, LeftHip,LHLKD, vec[7])
        
        
        
        runData(LeftAnkle, LeftKnee, LKLAD, vec[8])


        
        runData(LeftToe, LeftAnkle, LALTD, vec[9])
        
        
        #A list off all the bones
       
        JointA=[boney1,boney2,boney3,boney4,boney5,boney6,
        boney7,boney8,boney9,boney10,boney11,boney12,boney13,boney14]

        #updates the rig
        
        establish(JointA,frame_count)
        

        frame_count += 1
        
        if(cv.waitKey(1) == ord('q')):
            break
        
    #puts the bones back in their origional location
    putInOrig(LeftHand)
    putInOrig(LeftElbow)
    putInOrig(LeftShoulder)
    putInOrig(RightHand)
    putInOrig(RightElbow)
    putInOrig(RightShoulder)
    
    putInOrig(RightHip)
    putInOrig(RightKnee)
    putInOrig(RightAnkle)
    putInOrig(RightToe)
    
    
    putInOrig(LeftHip)
    putInOrig(LeftKnee)
    putInOrig(LeftAnkle)
    putInOrig(LeftToe)
    Head.rotation_euler=  (0,0,0)
    

if __name__ == "__main__" :
    avatarMe()
    