from astropy.coordinates import EarthLocation, AltAz
from astropy.time import Time,TimezoneInfo
from astropy.coordinates import SkyCoord
#from astropy.coordinates import ICRS, Galactic, FK4, FK5
from astroquery.simbad import Simbad
import astropy.units as u
import cv2
#import matplotlib.pyplot as plt
import numpy as np
import datetime

from PIL import Image, ImageDraw, ImageFont, ImageTk
import tkinter as tk
import pandas as pd
import datetime
import os
import math

#目的：複数の画像の星の対応を正確に取りたい
#手法：各画像の星と星のカタログで対応を取る

#v0.1
################タスク
#画像から星を抽出するしきい値を画面から可変とする
#カタログから星を抽出するしきい値を画面から可変とする
#マウスの位置を拡大した画像を表示する：ポイントしやすいように
################済み
#v0.1 カタログと画像の星の対応をクリックでつけ直せるようにする

_squarelength = 200
_framelength = 50

#def draw_stars(img,stars,color_s):
def draw_stars(img,stars,color_s,maker):
	for s_point in stars:
#		cv2.drawMarker(img, s_point, color_s, markerType=cv2.MARKER_STAR, markerSize=5, thickness=1, line_type=cv2.LINE_8)
		cv2.drawMarker(img, s_point, color_s, markerType=maker, markerSize=10, thickness=1, line_type=cv2.LINE_8)
	
class MainApplication(tk.Frame):
	def __init__(self, master):
		super().__init__(master)
		self.master = master
		self.master.title("MOSAIC")
		self.master.geometry('1300x700')

		# 画像を読み込み
		self.read_img()
#画像から星を抽出
#		self.stars_img=star_detect(self.img)
		self.star_detect()
		self.map_catalog()
		color_img=(0,0,255)
		draw_stars(self.img,self.stars_img,color_img,cv2.MARKER_SQUARE)
#		self.img_stars=self.img.copy()
		#self.img_stars：抽出した星が書き込まれている
		self.img_stars=self.img.copy()	
		self.draw_stars_d()

#		img_star_catalog = np.full(self.img.shape, 0, dtype=np.uint8)
#		color_img=(255,255,255)
##		draw_stars(img_star_catalog,self.stars_catalog,color_img)
#		for s_point in self.stars_catalog:
##			cv2.drawMarker(img_star_catalog, s_point, color_img, markerType=cv2.MARKER_DIAMOND, markerSize=5, thickness=1, line_type=cv2.LINE_8)
#			cv2.circle(img_star_catalog, s_point, 2, color_img, thickness=1, lineType=cv2.LINE_AA, shift=0)
#		cv2.imwrite('20231021032633.jpg', img_star_catalog)
		
		self.kp_catalog=[]
		self.kp_catalog_original=[]
		self.kp_img=[]
		self.flag=0
		self.mtx_old=np.array([[1,0,0],[0,1,0]])
		self.line=[]

		self.create_widget()
		self.center_angle()

#	def star_detect(image):
	def star_detect(self):
		#グレースケール画像にする
		img_gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
		#明るさに閾値を設ける(ここでは適当に200)
		threshold=150
		ret, new = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
		#画像は黒背景に白点になっているため、白点の輪郭を検出
		contours, hierarchy = cv2.findContours(new, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		#各輪郭について、重心を求めて配列に入れる
		stars = []
		for cnt in contours:
		    M = cv2.moments(cnt)
		    if M['m00'] != 0:
		        cx = int(M['m10']/M['m00'])
		        cy = int(M['m01']/M['m00'])
		        stars.append([cx,cy])
		    else:
		        stars.append(cnt[0][0])
		self.stars_img=np.array(stars,dtype='int32') 
	#	return stars

	def center_angle(self):
#中心角度の表示
		self.x=(self.left+self.right)/2
		self.y=(self.top+self.bottom)/2
		self.label4["text"] = str([self.x,self.y]) 

	def create_widget(self):
		h,w=self.img.shape[:2]
		self.canvas1 = tk.Canvas(self.master, width=w, height=h)
#		self.canvas1.pack()
		self.canvas1.place(x=0, y=0)
		_slength=200
		self.canvas2 = tk.Canvas(self.master, width=_slength, height=_slength)
#		self.canvas2.pack()
		self.canvas2.place(x=1000, y=10)
		self.canvas2.create_line(_slength/2, 0,_slength/2, _slength,tag="line1")
		self.canvas2.create_line(0, _slength/2,_slength, _slength/2,tag="line2")

		self.label1 = tk.Label(self.master, bg="white", width=10, height=3)
		self.label1.place(x=100, y=600)
		self.label2 = tk.Label(self.master, bg="green", width=10, height=3)
		self.label2.place(x=400, y=600)
		self.label3 = tk.Label(self.master, bg="red", width=10, height=3)
		self.label3.place(x=550, y=600)
		self.label4 = tk.Label(self.master, bg="white", width=10, height=3)
		self.label4.place(x=200, y=600)

		self.button = tk.Button(self.master, text="Adjust",command=self.remake_img)
		self.button.place(x=650, y=600)

		self.button = tk.Button(self.master, text="Auto stars",command=self.auto)
		self.button.place(x=750, y=600)

		self.disp_img()

		# canvas1にマウスが乗った場合、離れた場合のイベントをセット。
		self.canvas1.bind('<Motion>', self.mouse_motion)
		self.canvas1.bind("<ButtonPress-1>", self.point_get)
#		self.button.bind("<ButtonPress-1>", self.point_get)
#		self.canvas1.bind('<KeyPress>',self.key_event)
#		self.master.bind('<KeyPress>',self.key_event)

#		font = tk.font.Font(family='Arial', size=16, weight='bold')
#		image_title = tk.Label(text='=>', bg = "white", font=font)
		image_title = tk.Label(text='=>', bg = "white")
		image_title.place(x=500, y=610, anchor=tk.NW)

	def key_event(self,event):
		self.key=event.keysym
#		print(self.key)
		if self.key == "j":
			self.stars_catalog[:,0] -=-1
			self.left -= 1
			self.right -= 1
		elif self.key == "l":
			self.stars_catalog[:,0] +=-1
			self.left += 1
			self.right += 1
		elif self.key == "i":
			self.stars_catalog[:,1] +=-1
			self.top += 1
			self.bottom += 1
		elif self.key == "k":
			self.stars_catalog[:,1] -=-1
			self.top -= 1
			self.bottom -= 1
		self.center_angle()
		self.remake_img()
#		self.read_img()
##画像から星を抽出
#		star_detect()
#		
#		self.map_catalog()
#		self.draw_stars_d()

	def Reline(self):
#		p1=[1,1]
#		p2=[2,2]
		self.kp_img=np.array(self.kp_img)
		for i in range(self.kp_img.shape[0]):
			p1=self.kp_catalog_original[i]
			p2=self.kp_img[i]
			
			a=(p1[1]-p2[1])/(p1[0]-p2[0])
			a=-1/a
	
			mx=(p1[0]+p2[0])/2
			my=(p1[1]+p2[1])/2
	
		#切片
			b=my-a*mx
	
			self.line.append([a,b])
#		print(self.line)

	def calc_center(self):
		a0,b0=self.line[0]
		a1,b1=self.line[1]

		cx=-(b0-b1)/(a0-a1)
		cy=(a0*b1-a1*b0)/(a0-a1)
		self.center=np.array([cx,cy])
		print(self.center)

		add_vect=0
		for line in self.line:
			a,b=line
			A=np.array([0,b])
			B=np.array([1,a+b])
			ab=B-A
			ap=self.center-A

			ai_norm=np.dot(ap,ab)/np.linalg.norm(ab)
			neighbor_point=a+(ab)/np.linalg.norm(ab)*ai_norm
#			add_vect+=self.center-neighbor_point
			print(neighbor_point)
			print(neighbor_point-self.center)
			add_vect+=neighbor_point-self.center

		print(add_vect)
		self.center+=add_vect
		self.center=np.array(self.center,dtype="int32")
		print("center")
		print(self.center)

	#緑の星に最近接の赤の星を自動的に対応付ける
	#距離で制限する
	def auto(self):
		R=10
		for catalog_indx in range(self.stars_catalog.shape[0]):
			dist_list=[]	#距離のリスト
			for star_img in self.stars_img:
				dist=np.linalg.norm(self.stars_catalog[catalog_indx]-star_img)
				dist_list.append(dist)
			min_dist=min(dist_list)
			if min_dist<R:
				min_dist_indx=np.argmin(dist_list)	#最短のstarsのインデックス
				self.kp_catalog.append(self.stars_catalog[catalog_indx])
				self.kp_catalog_original.append(self.stars_catalog_original[catalog_indx]) #add
				self.kp_img.append(self.stars_img[min_dist_indx])
#				cv2.line(self.img, self.kp_catalog[-1], self.kp_img[-1], (255,255,255), thickness=1)
		self.drawline()
		self.disp_img()

	def drawline(self):
		#self.img_stars2:画像から抽出したマークまでつけた画像
		self.img=self.img_stars2.copy()
		print(len(self.kp_catalog))
		for i in range(len(self.kp_catalog)):
			cv2.line(self.img, self.kp_catalog[i], self.kp_img[i], (255,255,255), thickness=1)
		

	def map_catalog(self):
		simbad = Simbad()
		simbad.add_votable_fields('flux(V)')
		hoshi = simbad.query_criteria('Vmag<4',otype='star')
		
		LOCATION = EarthLocation(lon=139.3370196674786*u.deg, lat=36.41357867541122*u.deg, height=122*u.m)
		utcoffset = 0*u.hour
		tz = TimezoneInfo(9*u.hour) # 時間帯を決める。
		basename = os.path.splitext(os.path.basename(self.filename))[0]
#		basename = "20231021023633"
#		t_base=basename[10:]
#		t_base=basename[5:]	#trim_
		t_base=basename
		print(t_base)
		toki = datetime.datetime(int(t_base[:4]),int(t_base[4:6]),int(t_base[6:8]),int(t_base[8:10]),int(t_base[10:12]),int(t_base[12:]),tzinfo=tz)
		OBSTIME = Time(toki)
		OBSERVER = AltAz(location= LOCATION, obstime = OBSTIME)
		
		RA=hoshi['RA']
		DEC=hoshi['DEC']
		STAR_COORDINATES = SkyCoord(RA,DEC, unit=['hourangle','deg'])
		STAR_ALTAZ       = STAR_COORDINATES.transform_to(OBSERVER)
		self.seiza = STAR_ALTAZ.get_constellation()
		z = (self.seiza[:,None]==np.unique(self.seiza)).argmax(1)
		iro = np.stack([z/87,z%5/4,1-z%4/4],1)
		self.s = (5-hoshi['FLUX_V'])*1
		
		self.AZ  = STAR_ALTAZ.az.deg
		self.ALT = STAR_ALTAZ.alt.deg
		self.stars_catalog=np.array([self.AZ,self.ALT])
		self.stars_catalog=self.stars_catalog.T
		
		#AZ N 0 : E 90 : S 180 : W 270
#		center_x=191
#		width=86
#		center_y=43
#		height=50
		
		#top=center_y+height/2
		#bottom=center_y-height/2
		#left=center_x-width/2
		#right=center_x+width/2
		self.top=68
		self.bottom=18
		self.left=146
		self.right=240

		h,w = self.img.shape[:2]
#		print(h,w)
		ws=w/(self.right-self.left)
		hs=h/(self.top-self.bottom)

		self.stars_catalog[:,0]=(self.stars_catalog[:,0]-self.left)*ws
		self.stars_catalog[:,1]=h-(self.stars_catalog[:,1]-self.bottom)*hs
		
#		self.stars_catalog=np.array(self.stars_catalog,dtype='int32')
#		print(self.stars_catalog.shape)

#		st = [s for s in self.stars_catalog if self.left-10<s[0] and s[0]<self.right+10]
#		self.stars_catalog = [s for s in st if self.bottom-10<s[1] and s[1]<self.top+10]
		st = [s for s in self.stars_catalog if -100<s[0] and s[0]<w+100]
		self.stars_catalog = [s for s in st if -100<s[1] and s[1]<h+100]
		
		self.stars_catalog=np.array(self.stars_catalog,dtype='int32')
		self.stars_catalog_original=self.stars_catalog
#		print(self.stars_catalog.shape)
		
	def draw_stars_d(self):
		self.img=self.img_stars.copy()
		color_catalog=(0,255,0)
#		draw_stars(self.img,self.stars_catalog,color_catalog)
		draw_stars(self.img,self.stars_catalog,color_catalog,cv2.MARKER_STAR)
		#self.img_stars2：画像とカタログから抽出した星が描かれている
		self.img_stars2=self.img.copy()
#		cv2.imwrite('out.jpg', self.img)

	def disp_img(self):
		#表示を半分のサイズに変更：画面内に表示するため
		h,w = self.img.shape[:2]
		self.img_disp = cv2.resize(self.img, dsize=(int(w/2),int(h/2)))
		self.img_rgb = cv2.cvtColor(self.img_disp, cv2.COLOR_BGR2RGB) # imreadはBGRなのでRGBに変換
		self.img_pil = Image.fromarray(self.img_rgb) # RGBからPILフォーマットへ変換
		self.img_tk  = ImageTk.PhotoImage(self.img_pil) # ImageTkフォーマットへ変換
		#photo = ImageTk.PhotoImage(img_tk)
		# キャンバスに新しい画像を表示
		self.canvas1.create_image(0, 0, anchor=tk.NW, image=self.img_tk,tag="img")

	def read_img(self):
		self.filename="20231111040000.jpg"
#		self.filename="/home/kunitofukuda/workspace/Meteor/Optical2Wave/20231021/trim_img/20231021023633.jpg"
#		self.filename="20231021035238.jpg"
		self.img = cv2.imread(self.filename)
	#画像のマスク
		mask = cv2.imread("mask_original.png")
		mask = cv2.bitwise_not(mask)
		self.img = cv2.bitwise_and(self.img, mask)
#		cv2.imwrite('mask_result.jpg', img_AND)

	def remake_img(self):
		self.canvas1.delete("img")
#		self.read_img()	#画像を再構成
#画像から星を抽出
#		self.stars_img=star_detect(self.img)
#		self.star_detect()
#		self.map_catalog()

#		self.Reline()
#		self.calc_center()

#カタログの星を移動させる
		self.stars_adjust()
		self.draw_stars_d()
#		self.img_stars=self.img.copy()
		self.disp_img()

		self.kp_catalog=[]
		self.kp_catalog_original=[]
		self.kp_img=[]

	def camera(self):
		gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
#		print(gray.shape[::-1])
		objpoints=np.insert(self.kp_catalog,2,0.0,axis=1)
		objpoints=np.array([objpoints],dtype="float32")
#		print(objpoints)
#		self.mtx=np.insert(self.mtx,2,insert_m,axis=0)
		imgpoints=self.kp_img
		imgpoints=np.array([imgpoints],dtype="float32")
		ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
#		ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(self.kp_catalog, self.kp_img, np.array([1920,1080),None,None)
#		ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
		#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],initial_mtx,initial_dist)
		
		#ret, mtx, dist, rvecs, tvecs =cv2.fisheye.calibrate(objpoints,imgpoints,gray.shape[::-1],K,d,tvecs,calibration_flags,self.criteria)
		
		print("Err : ")
		print(ret)
		print("\n")
		print("Camera matrix : ")
		print(self.mtx)
		print("\n")
		print("dist : ")
		print(self.dist)
		print("rvecs : ")
		print(rvecs)
		print("tvecs : ")
		print(tvecs)

#		s_point=np.array([self.mtx[0,2],self.mtx[1,2]],dtype="int32")
#		print(s_point)
#		color_s=(255,0,0)
#		cv2.drawMarker(self.img, s_point, color_s, markerType=cv2.MARKER_STAR, markerSize=10, thickness=1, line_type=cv2.LINE_8)

		self.save_para()
		self.undistortion()

	def save_para(self):
		k_filename="K_calibcamera.csv"
		d_filename="D_calibcamera.csv"
		np.savetxt(k_filename, self.mtx, delimiter=',', fmt="%0.5e")
		np.savetxt(d_filename, self.dist, delimiter=',', fmt="%0.5e")

	def undistortion(self):
		# Using the derived camera parameters to undistort the image
		self.filename="20231111040000.jpg"
#		for filepath in self.images:
		img = cv2.imread(self.filename)
		h,w = img.shape[:2]
#		img = cv2.resize(img, dsize=(int(w/2),int(h/2)))
		# ROI:Region Of Interest(対象領域)
		#newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
		#newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (self.w,self.h), 0, (self.w,self.h))
		newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, img.shape[:2], 0, img.shape[:2])
    
		# Method 1 to undistort the image
		dst = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)
		
		# undistort関数と同じ結果が返されるので、今回はコメントアウト(initUndistortRectifyMap()関数)
		# Method 2 to undistort the image
		#mapx,mapy=cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
		#dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
    
		#歪み補正した画像をimg_undistortフォルダへ保存
		cv2.imwrite('undistort_' + self.filename, dst)
#		cv2.imwrite('undistort_' + str(os.path.basename(filepath)), dst)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def stars_adjust(self):
		self.kp_catalog=np.array(self.kp_catalog)
		self.kp_catalog_original=np.array(self.kp_catalog_original)
		self.kp_img=np.array(self.kp_img)
#		print(self.kp_catalog,self.kp_img)
		if self.kp_catalog.shape[0]>5:
			self.camera()
			self.mtx,inliers=cv2.estimateAffinePartial2D(self.kp_catalog_original,self.kp_img)
#			mtx,inliers=cv2.estimateAffinePartial2D(self.kp_img,self.kp_catalog)
#			mtx,inliers=cv2.estimateAffine2D(self.kp_catalog_original,self.kp_img)
		else:
			self.mtx=np.array([[1,0,0],[0,1,0]])
			inliers=np.array([[0]])
		insert_m=[0,0,1]
		self.mtx=np.insert(self.mtx,2,insert_m,axis=0)
		self.mtx_old=np.insert(self.mtx_old,2,insert_m,axis=0)
		self.mtx=np.dot(self.mtx,self.mtx_old)
#		mtx=np.delete(mtx,2,axis=0)
#		print(inliers)
#		print(sum(inliers))
#		print(self.mtx)
#回転角を示す
		degree = np.rad2deg(-np.arctan2(self.mtx[0, 1], self.mtx[0, 0]))

#		self.calc_stars_point()
#		self.rotate_point()
#		print(self.mtx_old)
		self.mxt_old=self.mtx
		self.mtx_old=self.mtx_old[:2,:]
#		print(self.mtx_old)

	def calc_stars_point(self):
		stars_catalog_calc=np.insert(self.stars_catalog_original, 2, 1, axis=1)

		stars_catalog_re=[]
		for s_point in stars_catalog_calc:
			s_point=np.dot(self.mtx,s_point)
			s_point=np.array(s_point,dtype='int32')
			stars_catalog_re.append(s_point)
		stars_catalog_re=np.array(stars_catalog_re)
		stars_catalog_re=stars_catalog_re[:,:2]
		self.stars_catalog=stars_catalog_re

	def rotate_point(self):
		angle=math.radians(1)
		sin_angle = math.sin(angle)
		cos_angle = math.cos(angle)
#		center=[348,1483]
		center=self.center
		
		# 回転した座標を格納するリスト
		rotated_points = []
		
#		for i in range(0, len(point), 2):
#		for point in self.stars_catalog_original:
		for point in self.stars_catalog:
			x=point[0]-center[0]
			y=point[1]-center[1]
			
			# 回転後の座標を計算
			rotated_x = x * cos_angle - y * sin_angle + center[0]
			rotated_y = y * cos_angle + x * sin_angle + center[1]
			
			rotated_points.append([rotated_x, rotated_y])

		self.stars_catalog_re=np.array(rotated_points,dtype="int32")
#		self.stars_catalog_re=self.stars_catalog_re.reshape(self.stars_catalog_original.shape)
		self.stars_catalog_re=self.stars_catalog_re.reshape(self.stars_catalog.shape)
		self.stars_catalog=self.stars_catalog_re
#		print(self.stars_catalog[0])	
#		print(self.stars_catalog[1])	

	def point_get(self,event):
		sd=self.nearst(event)
#		print(sd,self.flag)
		if sd!=0:
			if self.flag==0:
				self.kp_catalog.append(self.stars_catalog[sd])
				self.kp_catalog_original.append(self.stars_catalog_original[sd]) #add
				self.label2["text"]=str(self.stars_catalog[sd])
				self.flag=1
			else:
				self.kp_img.append(self.stars_img[sd])
				self.label3["text"]=str(self.stars_img[sd])
				if len(self.kp_img)>2:
					self.check_kp()

				self.drawline()
				self.disp_img()
				self.flag=0
	
	def check_kp(self):
	#既にその星の登録があるならそれを解除して登録する。
		kc=np.array(self.kp_catalog[-1])
		kco=np.array(self.kp_catalog[-1])
		ki=np.array(self.kp_img[-1])
		tmp1 = []
		tmp2 = []
		tmp3 = []
		for i in range(len(self.kp_catalog[:-1])):
			c = np.array(self.kp_catalog[i])
			co = np.array(self.kp_catalog_original[i])
			i = np.array(self.kp_img[i])
			if not (np.array_equal(c,kc) or np.array_equal(i,ki)):
				tmp1.append(c)
				tmp2.append(i)
				tmp3.append(co)
		
		self.kp_catalog=tmp1
		self.kp_catalog_original=tmp3
		self.kp_img=tmp2
		self.kp_catalog.append(kc)
		self.kp_catalog_original.append(kco)
		self.kp_img.append(ki)
#		print(self.kp_catalog)
#		print(self.kp_catalog_original)
#		print(self.kp_img)

#flagはカタログ0か画像1かの違い
	def nearst(self,event):
		R=50
		mause_point=np.array([event.x*2,event.y*2])	#画像表示との対応を取るため2倍
		if self.flag==0:
			stars=self.stars_catalog
		else:
			stars=self.stars_img
		dist_list=[]	#距離のリスト
		for star in stars:
			dist=np.linalg.norm(star-mause_point)
			dist_list.append(dist)
		min_dist=min(dist_list)
		if min_dist<R:
			sd=np.argmin(dist_list)	#最短のstarsのインデックス
		else:
			sd=0
		return sd

	def mouse_motion(self, event):
		# マウス最近傍の星の座標を得る
		x = event.x*2
		y = event.y*2
		self.label1["text"] = str([x,y]) 
		self.frame_rect(x, y)
		self.canvas_set(x, y)

	def frame_rect(self, x, y):
		# 過去に枠線が描画されている場合はそれを削除し、メイン画像内マウス位置に枠線を描画
		x=x/2
		y=y/2
		self.frame_refresh()
		self.crop_frame = (x-_framelength/2, y-_framelength/2, x+_framelength/2, y+_framelength/2)	
		self.rectframe = self.canvas1.create_rectangle(self.crop_frame, outline='#AAA', width=2, tag='rect')

	def frame_refresh(self):
		try:
			self.canvas1.delete('rect')
		except:
			pass

	def canvas_set(self, x, y):
		# 枠線内をクロップし、ズームする
		zoom_mag = _squarelength / _framelength
#		croped = self.resize_image.crop(self.crop_frame)
		croped = self.img[int(self.crop_frame[0]):int(self.crop_frame[1]),int(self.crop_frame[2]):int(self.crop_frame[3]),:]
		zoom_image = cv2.resize(croped,dsize=(200,200))
#		self.img_disp = cv2.resize(self.img, dsize=(int(w/2),int(h/2)))

		# ズームした画像を右側canvasに当てはめる
		self.image_refresh()
		self.sub_image1 = ImageTk.PhotoImage(zoom_image)

		self.sub_cv1 = self.canvas2.create_image(0, 0, image=self.sub_image1, anchor=tk.NW, tag='cv1')

		self.canvas1.delete("line1")  # すでに"rect1"タグの図形があれば削除
		self.canvas1.delete("line2")  # すでに"rect1"タグの図形があれば削除
		self.canvas2.create_line(_squarelength/2, 0,_squarelength/2, _squarelength,tag="line1")
		self.canvas2.create_line(0, _squarelength/2,_squarelength, _squarelength/2,tag="line2")

def main():
	root = tk.Tk()
	app = MainApplication(master = root)
# tkinterのメインループを開始
	app.mainloop()

if __name__ == '__main__':
	main()
