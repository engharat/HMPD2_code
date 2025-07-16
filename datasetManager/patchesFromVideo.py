#this file is devoted to get the patches from the selected video exploiting the metadata saved in the db

import pandas as pd
import pandas as pd
import logging
import os
from neo4j import GraphDatabase
from PIL import Image
import cv2

class graphDatabaseManager:

    def __init__(self, uri, user, pwd, db):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__db = db
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
        except Exception as e:
            print("Failed to create the driver:", e)

    def close(self):
        if self.__driver is not None:
            self.__driver.close()

    def query(self, query, parameters=None, db=None):
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try:
            session = self.__driver.session(database=db) if db is not None else self.__driver.session()
            response = list(session.run(query, parameters))
        except Exception as e:
            print("Query failed:", e)
        finally:
            if session is not None:
                session.close()
        return response


    def getAnnoted(self, vedeoNameNOE4J):
        nodes = self.query("""
                            MATCH (a:Annotation)<-[:MARK]-(u:User {userid:"Teresa"})
                            WITH a
                            MATCH (v:Video {video_id: $vedeoNameNOE4J})<-[:BELONG_TO_VIDEO]-(f:Frame)<-[:BELONG_TO_FRAME]-(p:Patch)<-[:REGARD]-(a)-[:POINT_AT]->(c:Class)
                            RETURN p.patchid as patchid, p.bb as bb, c.class  as class, f.frame_n as fn
                            ORDER BY fn 
        """, db=self.__db)
        return [node.get('patchid') for node in nodes], [node.get('bb') for node in nodes], [node.get('fn') for node in nodes]

#22-03-10-14-09
#th_150_h100_var6_mk5_merge1_maxRegion300_startFrame10 --> #22-03-10-14-30

def saveBBofFramePair(bb, frame_R, frame_P, frame_A, id, datasetDir):

    imgFilePath_R = f'{datasetDir}/{id}_R.bmp'
    roi_R = frame_R[bb[0]:bb[2] , bb[1]:bb[3]]
    cv2.imwrite(imgFilePath_R, roi_R)

    imgFilePath_A = f'{datasetDir}/{id}_A.bmp'
    roi_A = frame_A[bb[0]:bb[2], bb[1]:bb[3]]
    cv2.imwrite(imgFilePath_A, roi_A)

    imgFilePath_P = f'{datasetDir}/{id}_P.bmp'
    roi_P = frame_P[bb[0]:bb[2], bb[1]:bb[3]]
    cv2.imwrite(imgFilePath_P, roi_P)

    return id




videoPathPC = "/Users/beppe2hd/Data/acquisizioni_20221012/22-03-10-14-09"
vedeoNameNOE4J = "22-03-10-14-09"
destinationPath = "/Users/beppe2hd/Data/Microplastiche/patches"

querymanager = graphDatabaseManager('bolt://192.168.54.202:7687', 'neo4j', 'password', 'neo4j')
patchids, bb, fn = querymanager.getAnnoted(vedeoNameNOE4J)


df = pd.DataFrame(list(zip(patchids, bb, fn)),
               columns =['patchids', 'bb', 'fn'])

videofilePath_R = f"{videoPathPC}/R.avi"
videofilePath_P = f"{videoPathPC}/P.avi"
videofilePath_A = f"{videoPathPC}/A.avi"

cap_R = cv2.VideoCapture(videofilePath_R)
cap_P = cv2.VideoCapture(videofilePath_P)
cap_A = cv2.VideoCapture(videofilePath_A)

# The query retrieve patchesid, bounding boxe and frame number (fn)

num_of_frame_R = int(cap_R.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
for x in range(1, num_of_frame_R):
    ret, frame_R = cap_R.read()
    ret, frame_P = cap_P.read()
    ret, frame_A = cap_A.read()
    if x in fn:  # if the current frame is in the list of frame number retrieved from the db
        u = df[df["fn"]==x]  ## the df eleemntf of that frame are selected
        for item in u.iterrows(): # for each of them the component or R A ane P are stored with the right id (that correspond to the class)
            currentpsatchid = item[1].patchids
            currentBB = item[1].bb
            saveBBofFramePair(currentBB, frame_R, frame_P, frame_A, currentpsatchid, destinationPath)



