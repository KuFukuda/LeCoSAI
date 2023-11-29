#LeCoSAI
#Lens correction software using astronomical images
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

#from PIL import Image, ImageDraw, ImageFont, ImageTk
from PIL import Image, ImageTk
import tkinter as tk
import tkinter.filedialog
#import pandas as pd
import os
import math
import csv

#目的：レンズのゆがみを星の位置関係から補正する
#手法：本来の星の位置のカタログと撮影された星を対応させ、カメラのパラメータを取得し、補正する。

_squarelength = 200
_framelength = 50

#def draw_stars(img,stars,color_s):
def draw_stars(img,stars,color_s,maker,msizes):
	for s_point,msize in zip(stars,msizes):
#		print(msize)
#		cv2.drawMarker(img, s_point, color_s, markerType=cv2.MARKER_STAR, markerSize=5, thickness=1, line_type=cv2.LINE_8)
#		cv2.drawMarker(img, s_point, color_s, markerType=maker, markerSize=10, thickness=1, line_type=cv2.LINE_8)
		cv2.circle(img, s_point, msize, color_s, thickness=-1, lineType=cv2.LINE_AA)
#		cv2.drawMarker(img, s_point, color_s, markerType=maker, markerSize=msize, thickness=1, line_type=cv2.LINE_8)
#		cv2.drawMarker(img, s_point, color_s, markerType=maker, thickness=msize, line_type=cv2.LINE_8)
	
class MainApplication(tk.Frame):
	def __init__(self, master):
		super().__init__(master)
		self.master = master
		self.master.title("LeCoSAI")
		self.master.geometry('1300x700')

		self.init_para()
		self.filename_mask=None
#		#kp_index : [catalog_index,img_index]
#		self.img_flag=0
#		self.kp_index=[]
#		self.flag=0
#		self.specific_flag=0
#		self.line=[]
#		self.filename_mask=None

		self.create_widget()
#		self.center_angle()

	def init_para(self):
		#kp_index : [catalog_index,img_index]
		self.img_flag=0
		self.kp_index=[]
		self.flag=0
		self.specific_flag=0
		self.line=[]

	def star_detect(self):
		#グレースケール画像にする
		img_gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
		#明るさに閾値を設ける(ここでは適当に200)
#		self.threshold_view["text"] =str(self.var_scale.get())
		#self.threshold=120
		self.threshold=self.var_scale.get()
		ret, new = cv2.threshold(img_gray, self.threshold, 255, cv2.THRESH_BINARY)
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
		print(self.stars_img.shape)

		self.stars_number["text"] = str(self.stars_img.shape[0]) 
	#	return stars

#	def center_angle(self):
##中心角度の表示
#		self.x=(self.left+self.right)/2
#		self.y=(self.top+self.bottom)/2
#		self.label4["text"] = str([self.x,self.y]) 

	def create_widget(self):
#		h,w=self.img.shape[:2]
		h=1080
		w=1920
		self.canvas1 = tk.Canvas(self.master, width=w, height=h)
		self.canvas1.place(x=0, y=0)
		_slength=200
		self.canvas2 = tk.Canvas(self.master, width=_slength, height=_slength)
		self.canvas2.place(x=1000, y=10)
		self.canvas2.create_line(_slength/2, 0,_slength/2, _slength,tag="line1")
		self.canvas2.create_line(0, _slength/2,_slength, _slength/2,tag="line2")

		#マウスの位置
		self.mause_posi = tk.Label(self.master, width=10)
		self.mause_posi.place(x=100, y=600)
#		self.label2 = tk.Label(self.master, bg="green", width=10, height=3)
#		self.label2.place(x=400, y=600)
#		self.label3 = tk.Label(self.master, bg="red", width=10, height=3)
#		self.label3.place(x=550, y=600)

		#星に関する情報
		self.Location = tk.Label(text='Star',font=("normal","13","bold"))
		self.Location.place(x=200, y=550, anchor=tk.NW)
		#画像から抽出された星の数
		self.label3 = tk.Label(text='Image')
		self.label3.place(x=250, y=570, anchor=tk.NW)
		self.stars_number = tk.Label(self.master, width=5)
		self.stars_number.place(x=250, y=600)
		#カタログから抽出された星の数
		self.label4 = tk.Label(text='Catalog')
		self.label4.place(x=250, y=640, anchor=tk.NW)
		self.catalog_number = tk.Label(self.master, width=5)
		self.catalog_number.place(x=250, y=660)
		#対応付けられた星の数
		self.label2 = tk.Label(text='Map')
		self.label2.place(x=350, y=600, anchor=tk.NW)
		self.check_stars = tk.Label(self.master, width=5)
		self.check_stars.place(x=350, y=630)

		#画像から抽出する星の閾値を設定するスライドバー
		x_posi=430
		self.label_scale1 = tk.Label(text='Image')
		self.label_scale1.place(x=x_posi, y=570, anchor=tk.NW)
		self.var_scale = tk.IntVar()
		self.star_threshold = tk.Scale(self.master, 
										from_=100, to=250, variable=self.var_scale, 
										command=self.threshold_view,orient=tk.HORIZONTAL)
		self.star_threshold.place(x=x_posi+70, y=550)
		#カタログから抽出する星の閾値を設定するスライドバー
		self.label_scale2 = tk.Label(text='Catalog')
		self.label_scale2.place(x=x_posi, y=620, anchor=tk.NW)
		self.var_scale_catalog = tk.IntVar()
		self.catalog_threshold = tk.Scale(self.master, 
										from_=3, to=9, variable=self.var_scale_catalog, 
										command=self.catalog_threshold_view,orient=tk.HORIZONTAL)
		self.catalog_threshold.place(x=x_posi+70, y=600)

		#画像の選択
		self.entry_box_img = tk.Entry(width=20,state="readonly")
		self.entry_box_img.place(x=1000, y=230)
		self.button_img=tk.Button(text="Image",command=self.openfile_img,width=7)
		self.button_img.place(x=1200,y=222)

		#マスクの選択
		self.entry_box_mask = tk.Entry(width=20,state="readonly")
		self.entry_box_mask.place(x=1000, y=270)
		self.button_mask=tk.Button(text="Mask",command=self.openfile_mask,width=7)
		self.button_mask.place(x=1200,y=262)

		#日時入力
		x_posi=1000
		y_posi=310
		self.date_y = tk.Label(text='Date:')
		self.date_y.place(x=x_posi, y=y_posi, anchor=tk.NW)

		self.entry_box_y = tk.Entry(width=4)
		x_posi+=40
		self.entry_box_y.place(x=x_posi, y=y_posi)
		self.date_y = tk.Label(text='/')
		x_posi+=40
		self.date_y.place(x=x_posi, y=y_posi, anchor=tk.NW)

		self.entry_box_mo = tk.Entry(width=3)
		x_posi+=10
		self.entry_box_mo.place(x=x_posi, y=y_posi)
		self.date_mo = tk.Label(text='/')
		x_posi+=35
		self.date_mo.place(x=x_posi, y=y_posi, anchor=tk.NW)

		self.entry_box_day = tk.Entry(width=3)
		x_posi+=10
		self.entry_box_day.place(x=x_posi, y=y_posi)

		self.entry_box_h = tk.Entry(width=3)
		x_posi+=40
		self.entry_box_h.place(x=x_posi, y=y_posi)
		self.date_h = tk.Label(text=':')
		x_posi+=26
		self.date_h.place(x=x_posi, y=y_posi, anchor=tk.NW)

		self.entry_box_min = tk.Entry(width=3)
		x_posi+=10
		self.entry_box_min.place(x=x_posi, y=y_posi)
		self.date_min = tk.Label(text=':')
		x_posi+=26
		self.date_min.place(x=x_posi, y=y_posi, anchor=tk.NW)

		self.entry_box_sec = tk.Entry(width=3)
		x_posi+=10
		self.entry_box_sec.place(x=x_posi, y=y_posi)

		#場所の入力
		y_posi=350
		self.Location = tk.Label(text='Location',font=("normal","13","bold"))
		self.Location.place(x=1000, y=y_posi, anchor=tk.NW)

		self.lon = tk.Label(text='Lon[deg]:')
		self.lon.place(x=1000, y=y_posi+30, anchor=tk.NW)
		self.entry_box_lon = tk.Entry(width=10)
		self.entry_box_lon.place(x=1080, y=y_posi+30)

		self.lat = tk.Label(text='Lat[deg]:')
		self.lat.place(x=1000, y=y_posi+60, anchor=tk.NW)
		self.entry_box_lat = tk.Entry(width=10)
		self.entry_box_lat.place(x=1080, y=y_posi+60)

		self.H = tk.Label(text='Hight[m]:')	#高度[m]
		self.H.place(x=1000, y=y_posi+90, anchor=tk.NW)
		self.entry_box_H = tk.Entry(width=10)
		self.entry_box_H.place(x=1080, y=y_posi+90)

		#方位角と仰角の設定
		self.angle = tk.Label(text='Angle[deg]:')
		self.angle.place(x=1000,y=y_posi+135,anchor=tk.NW)
		x_posi=1080

		self.entry_box_top = tk.Entry(width=5)
		self.entry_box_top.place(x=x_posi+50, y=y_posi+125)

		self.entry_box_left = tk.Entry(width=5)
		self.entry_box_left.place(x=x_posi, y=y_posi+135)

		self.entry_box_right = tk.Entry(width=5)
		self.entry_box_right.place(x=x_posi+100, y=y_posi+135)

		self.entry_box_bottom = tk.Entry(width=5)
		self.entry_box_bottom.place(x=x_posi+50, y=y_posi+145)

		self.button_posi=tk.Button(text="Input",command=self.DateLocInput,width=7)
		self.button_posi.place(x=1050,y=y_posi+200)

		self.button_saveloc=tk.Button(text="Save",command=self.LocationSave,width=7)
		self.button_saveloc.place(x=1150,y=y_posi+200)

		#場所情報の読み込み
		self.entry_box_loc = tk.Entry(width=20,state="readonly")
		self.entry_box_loc.place(x=1000, y=y_posi+280)
		self.button_loc=tk.Button(text="Load",command=self.openfile_loc,width=7)
		self.button_loc.place(x=1200,y=y_posi+272)


		#レンズ歪みの計算
		y_posi=550
		x_posi=650
		self.Calib = tk.Label(text='Calibration',font=("normal","13","bold"))
		self.Calib.place(x=x_posi, y=y_posi, anchor=tk.NW)

		self.button = tk.Button(self.master, text="Calc",command=self.remake_img,width=7)
		self.button.place(x=x_posi, y=y_posi+30)

		self.button_mstar=tk.Button(text="Move",command=self.stars_move,width=7)
		self.button_mstar.place(x=x_posi+100,y=y_posi+30)

		self.button = tk.Button(self.master, text="Auto",command=self.auto,width=7)
		self.button.place(x=x_posi+200, y=y_posi+30)


		self.entry_box_calib = tk.Entry(width=20,state="readonly") 
		self.entry_box_calib.place(x=x_posi, y=y_posi+70)
		self.button_calib=tk.Button(text="Load",command=self.openfile,width=7)
		self.button_calib.place(x=x_posi+200,y=y_posi+70)

		self.button_save=tk.Button(text="Save",command=self.CalibSave,width=7)
		self.button_save.place(x=x_posi,y=y_posi+110)

		self.button_save=tk.Button(text="Reset",command=self.Reset,width=7)
		self.button_save.place(x=x_posi+100,y=y_posi+110)

#		self.disp_img()

		# canvas1にマウスが乗った場合、離れた場合のイベントをセット。
		self.canvas1.bind('<Motion>', self.mouse_motion)
		self.canvas1.bind("<ButtonPress-1>", self.point_get)
#		self.canvas1.bind('<Leave>', self.mouse_leave)
#		self.button.bind("<ButtonPress-1>", self.point_get)
#		self.canvas1.bind('<KeyPress>',self.key_event)
		self.master.bind('<KeyPress>',self.key_event)

#		font = tk.font.Font(family='Arial', size=16, weight='bold')
#		image_title = tk.Label(text='=>', bg = "white", font=font)
#		image_title = tk.Label(text='=>')
#		image_title.place(x=500, y=610, anchor=tk.NW)

	def catalog_threshold_view(self,event=None):
		print("catalog_threshold_view")
		self.read_img()
#画像から星を抽出
		self.star_detect()
		color_s=(0,0,255)
		for s_point in self.stars_img:
			cv2.drawMarker(self.img, s_point, color_s, markerType=cv2.MARKER_SQUARE, thickness=1, line_type=cv2.LINE_8)
		self.img_stars=self.img.copy()	
		if self.specific_flag==1:
			self.map_catalog()
			self.draw_stars_d()
		self.canvas1.delete("img")
		self.disp_img()

	def threshold_view(self,event=None):
		self.read_img()
#画像から星を抽出
		self.star_detect()
		color_s=(0,0,255)
		for s_point in self.stars_img:
			cv2.drawMarker(self.img, s_point, color_s, markerType=cv2.MARKER_SQUARE, thickness=1, line_type=cv2.LINE_8)
		self.img_stars=self.img.copy()	
		if self.specific_flag==1:
#			self.map_catalog()
			self.draw_stars_d()
		self.canvas1.delete("img")
		self.disp_img()

	def CalibSave(self):
		filename = tk.filedialog.asksaveasfilename(
		filetypes = [("CSV", ".csv")], # ファイルフィルタ
		initialdir = "./", # 自分自身のディレクトリ
		)
		print(filename)
		specific_data=[self.plon,self.plat,self.pH]
		np.savetxt(filename, specific_data, delimiter=',', fmt="%0.5e")
		print("SaveClib")
		self.save_para()

	def openfile_loc(self):
		self.entry_box_loc.configure(state="normal") #Entry_boxを書き込み可に設定
		idir="./" #初期フォルダを指定
		filetype=[("CSV",".csv")]
		filename = tk.filedialog.askopenfilename(filetypes=filetype,initialdir=idir)
		self.entry_box_loc.insert(tk.END,filename) #選択ファイルを表示
		self.entry_box_loc.configure(state="readonly") #読み込み専用に戻す
		print(filename)

		with open(filename, encoding='utf8', newline='') as f:
			csvreader = csv.reader(f)
			content = [row for row in csvreader]
		self.plon=np.float32(content[0])
		self.plat=np.float32(content[1])
		self.pH=np.float32(content[2])

		self.entry_box_lon.insert(tk.END,str(self.plon[0]))
		self.entry_box_lat.insert(tk.END,str(self.plat[0]))
		self.entry_box_H.insert(tk.END,str(self.pH[0]))

		self.DateLocInput()

	def LocationSave(self):
		filename = tk.filedialog.asksaveasfilename(
		filetypes = [("CSV", ".csv")], # ファイルフィルタ
		initialdir = "./", # 自分自身のディレクトリ
		)
		print(filename)
#		filename="Location.csv"
#		specific_data=[self.plon,self.plat,self.pH,self.top,self.left,self.right,self.bottom]
		specific_data=[self.plon,self.plat,self.pH]
		np.savetxt(filename, specific_data, delimiter=',', fmt="%0.5e")
		print("SaveLocation")

#Location
	def DateLocInput(self):
		y=self.entry_box_y.get()
		mo=self.entry_box_mo.get()
		day=self.entry_box_day.get()
		ho=self.entry_box_h.get()
		minu=self.entry_box_min.get()
		sec=self.entry_box_sec.get()

		tz = TimezoneInfo(9*u.hour) # 時間帯を決める。Tokyoで9時間
		toki = datetime.datetime(int(y),int(mo),int(day),int(ho),int(minu),int(sec),tzinfo=tz)
		self.obstime = Time(toki)
		print("DateInput")

#	def PosiInput(self):
		self.plon=self.entry_box_lon.get()
		self.plon=np.float32(self.plon)
		self.plat=self.entry_box_lat.get()
		self.plat=np.float32(self.plat)
		self.pH=self.entry_box_H.get()
		self.pH=np.float32(self.pH)
		self.location = EarthLocation(lon=self.plon*u.deg, lat=self.plat*u.deg, height=self.pH*u.m)
		print("PosiInput")
		self.specific_flag=1

		self.canvas1.delete("img")
		self.map_catalog()
		self.draw_stars_d()
		self.disp_img()

	def Reset(self):
		self.init_para()
		self.canvas1.delete("img")
		self.stars_catalog=self.stars_catalog_original.copy()
		self.draw_stars_d()
#		self.make_kp_list()
		self.drawline()
		self.disp_img()

	def stars_move(self):
		#星を移動させる
		self.calc_catalog_point()
		self.draw_stars_d()
		self.make_kp_list()
		self.drawline()
		self.disp_img()
#		print(move)

	def remake_img(self):
		self.canvas1.delete("img")
#		self.Reline()
#		self.calc_center()

#カタログの星を移動させる
		self.stars_adjust()
		self.draw_stars_d()
		self.make_kp_list()
		self.drawline()
		self.disp_img()

	def openfile_mask(self):
		self.entry_box_mask.configure(state="normal") #Entry_boxを書き込み可に設定
		idir="./" #初期フォルダを指定
#		filetype=[("JPG",".jpg")]
#		self.filename_mask = tk.filedialog.askopenfilename(filetypes=filetype,initialdir=idir)
		self.filename_mask = tk.filedialog.askopenfilename(initialdir=idir)
		self.entry_box_mask.insert(tk.END,self.filename_mask) #選択ファイルを表示
		self.entry_box_mask.configure(state="readonly") #読み込み専用に戻す
		print(self.filename_mask)

		# 画像を読み込み
#		self.img_flag=1
	#画像のマスク
#		mask = cv2.imread("mask_original.png")
#		mask = cv2.bitwise_not(mask)
#		self.img = cv2.bitwise_and(self.img, mask)
		self.read_img()
#画像から星を抽出
		self.star_detect()
		color_s=(0,0,255)
		for s_point in self.stars_img:
			cv2.drawMarker(self.img, s_point, color_s, markerType=cv2.MARKER_SQUARE, thickness=1, line_type=cv2.LINE_8)
#		draw_stars(self.img,self.stars_img,color_img,cv2.MARKER_SQUARE,msizes)
#		self.img_stars=self.img.copy()
		#self.img_stars：抽出した星が書き込まれている
		self.img_stars=self.img.copy()	
		if self.specific_flag==1:
			self.map_catalog()
			self.draw_stars_d()
		self.canvas1.delete("img")
		self.disp_img()

	def openfile_img(self):
		self.init_para()
		self.entry_box_img.configure(state="normal") #Entry_boxを書き込み可に設定
		idir="./" #初期フォルダを指定
		filetype=[("JPG",".jpg")]
		self.filename = tk.filedialog.askopenfilename(filetypes=filetype,initialdir=idir)
		self.entry_box_img.insert(tk.END,self.filename) #選択ファイルを表示
		self.entry_box_img.configure(state="readonly") #読み込み専用に戻す
		print(self.filename)

		# 画像を読み込み
		self.img_flag=1
		self.read_img()
#画像から星を抽出
#		self.stars_img=star_detect(self.img)
		self.star_detect()
		color_s=(0,0,255)
		for s_point in self.stars_img:
			cv2.drawMarker(self.img, s_point, color_s, markerType=cv2.MARKER_SQUARE, thickness=1, line_type=cv2.LINE_8)
#		draw_stars(self.img,self.stars_img,color_img,cv2.MARKER_SQUARE,msizes)
#		self.img_stars=self.img.copy()
		#self.img_stars：抽出した星が書き込まれている
		self.img_stars=self.img.copy()	
		if self.specific_flag==1:
			self.map_catalog()
			self.draw_stars_d()
		self.canvas1.delete("img")
		self.disp_img()

	def openfile(self):
		#パラメータの読み込み mtx(K) dist(d) tvecs rvecs
		self.entry_box_calib.configure(state="normal") #Entry_boxを書き込み可に設定
		idir="./" #初期フォルダを指定
		filetype=[("CSV",".csv")]
		filename = tk.filedialog.askopenfilename(filetypes=filetype,initialdir=idir)
		self.entry_box_calib.insert(tk.END,filename) #選択ファイルを表示
		self.entry_box_calib.configure(state="readonly") #読み込み専用に戻す
		print(filename)
		with open(filename, encoding='utf8', newline='') as f:
			csvreader = csv.reader(f)
			content = [row for row in csvreader]
#		print(content)
		self.mtx=np.float32(content[:3]).reshape(3,3)
#		print(self.read_mtx)
		self.dist=np.float32(content[3])
#		print(self.read_dist)
		self.rvecs=np.float32(content[4])
#		print(self.read_rvecs)
		self.tvecs=np.float32(content[5])
#		print(self.read_tvecs)

	def key_event(self,event):
		self.key=event.keysym
#		print(self.key)
		if self.key == "j":
			self.stars_catalog[:,0] +=-5
#			self.left += 1
#			self.right += 1
		elif self.key == "l":
			self.stars_catalog[:,0] -=-5
#			self.left -= 1
#			self.right -= 1
		elif self.key == "i":
			self.stars_catalog[:,1] +=-5
#			self.top += 1
#			self.bottom += 1
		elif self.key == "k":
			self.stars_catalog[:,1] -=-5
#			self.top -= 1
#			self.bottom -= 1
#		self.center_angle()
#		self.remake_img()
#		self.read_img()
##画像から星を抽出
#		star_detect()
#		
#		self.map_catalog()
#		self.draw_stars_d()
		if self.specific_flag==1:
			self.canvas1.delete("img")
	#カタログの星を移動させる
	#		self.stars_adjust()
			self.draw_stars_d()
			self.make_kp_list()
			self.drawline()
			self.disp_img()

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

	def calc_catalog_point(self):
#		stars_catalog=self.stars_catalog.copy()
		stars_catalog=self.stars_catalog_original.copy()
		stars_catalog=np.insert(stars_catalog, 2, 0.0, axis=1)
		stars_catalog=np.array(stars_catalog,dtype="float32")
#		stars_catalog[:,0]=(stars_catalog[:,0]-self.mtx[0,2])/self.mtx[0,0]
#		stars_catalog[:,1]=(stars_catalog[:,1]-self.mtx[1,2])/self.mtx[1,1]

		# project 3D points to image plane
#		self.rvecs = np.zeros((3, 1))
#		self.tvecs = np.zeros((3, 1))
#		print(self.rvecs)
		self.calc_catalog, jac = cv2.projectPoints(stars_catalog, self.rvecs, self.tvecs, self.mtx, self.dist)
#		self.calc_catalog, jac = cv2.projectPoints(stars_catalog, self.rvecs, self.tvecs, self.mtx, self.dist)
#		imgpts, jac = cv2.projectPoints(self.stars_catalog, rvecs, self.tvecs, self.mtx, self.dist)
#		self.calc_catalog=np.int32(self.calc_catalog).reshape(self.calc_catalog.shape[0],self.calc_catalog.shape[2])
		self.calc_catalog=np.int32(self.calc_catalog).reshape(-1,2)
#		self.calc_catalog=self.calc_catalog.reshape(self.calc_catalog.shape[0],self.calc_catalog.shape[2])
#		print(self.calc_catalog)
#		print(self.calc_catalog.shape)
#		print(self.calc_catalog[0])
		self.stars_catalog=self.calc_catalog

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
#				self.kp_catalog.append(self.stars_catalog[catalog_indx])
#				self.kp_catalog_original.append(self.stars_catalog_original[catalog_indx]) #add
#				self.kp_img.append(self.stars_img[min_dist_indx])
				self.kp_index.append([catalog_indx,min_dist_indx])
				if len(self.kp_index)>2:
#					self.check_kp()
					self.check_kp_index()
#				cv2.line(self.img, self.kp_catalog[-1], self.kp_img[-1], (255,255,255), thickness=1)
					self.check_stars["text"]=str(len(self.kp_index))
		self.canvas1.delete("img")
#		self.Reline()
#		self.calc_center()

#		self.remake_img()
#カタログの星を移動させる
#		self.stars_adjust()
#		self.draw_stars_d()
		self.make_kp_list()
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
		hoshi = simbad.query_criteria('Vmag<5',otype='star')
#		hoshi = simbad.query_criteria('Vmag<5',otype='*')
		
		OBSERVER = AltAz(location= self.location, obstime = self.obstime)
		
		RA=hoshi['RA']
		DEC=hoshi['DEC']
		STAR_COORDINATES = SkyCoord(RA,DEC, unit=['hourangle','deg'])
		STAR_ALTAZ       = STAR_COORDINATES.transform_to(OBSERVER)
		self.seiza = STAR_ALTAZ.get_constellation()
#		self.flux_v=(6-hoshi[hoshi['FLUX_V']<threshold])+2
		self.flux_v = (6-hoshi['FLUX_V'])+2
		self.flux_v=np.int32(self.flux_v)
		print(self.flux_v.max())
		print(self.flux_v.min())
#		print(self.flux_v.shape)
		
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
		#ATOMCAM2 水平 102°、垂直 54.9°
		self.left=int(self.entry_box_left.get())
		self.right=int(self.entry_box_right.get())
		self.top=int(self.entry_box_top.get())
		self.bottom=int(self.entry_box_bottom.get())
#		self.left=125
#		self.right=270
#		self.top=68
#		self.bottom=12

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
		print(self.stars_catalog.shape)
#		st = [s for s in self.stars_catalog if -200<s[0] and s[0]<w+200]
#		self.stars_catalog = [s for s in st if -200<s[1] and s[1]<h+200]
		stars_catalog=[]
		flux_v=[]
		threshold=self.var_scale_catalog.get()
		for s,f in zip(self.stars_catalog,self.flux_v):
			if -200<s[0] and s[0]<w+200 and -200<s[1] and s[1]<h+200 and threshold<=f:
				stars_catalog.append(s)
				flux_v.append(f)
#		self.stars_catalog = [s for s in st if -200<s[1] and s[1]<h+200]
		
		self.flux_v=np.int32(flux_v)
#		self.stars_catalog=np.array(self.stars_catalog,dtype='int32')
		self.stars_catalog=np.int32(stars_catalog)
		self.stars_catalog_original=self.stars_catalog
		print(self.flux_v.shape)
		print(self.stars_catalog.shape)
#		print(self.stars_catalog.shape)
##################################
		self.catalog_number["text"]=str(self.flux_v.shape[0])
#		self.check_stars = tk.Label(self.master, width=5)
#		self.check_stars.place(x=350, y=610)
		
	def draw_stars_d(self):
		self.img=self.img_stars.copy()
		color_catalog=(0,255,0)
#		draw_stars(self.img,self.stars_catalog,color_catalog)
#		draw_stars(self.img,self.stars_catalog,color_catalog,cv2.MARKER_STAR)
		draw_stars(self.img,self.stars_catalog,color_catalog,cv2.MARKER_DIAMOND,self.flux_v)
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
#		self.filename="20231111042000.jpg"	#test_img ok
#		self.filename="20231111040000.jpg"	#real ok
#		self.filename="20231122050000.jpg"	#real ok
#		self.filename="20231123010820.jpg"
#		self.filename="20231021035238.jpg"
		self.img = cv2.imread(self.filename)
#	#画像のマスク
		if self.filename_mask!=None:
#		mask = cv2.imread("mask_original.png")
			mask = cv2.imread(self.filename_mask)
			mask = cv2.bitwise_not(mask)
			self.img = cv2.bitwise_and(self.img, mask)
##		cv2.imwrite('mask_result.jpg', img_AND)

	def camera(self):
		#init
		fx=1.14731e3
		fy=1.14814e3
		cx=9.68865e2	#pixel
		cy=5.32106e2	#pixel
		init_mtx = np.array([[fx, 0.0,cx], # カメラ行列
						[0.0,fy, cy],
						[0.0,0.0,1.0]])
		init_dist = np.array([-3.84386e-01,2.00727e-01,7.27513e-04,3.32499e-04,-6.59210e-02])	#正式

		gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
#		print(gray.shape[::-1])
#		self.objps=np.insert(self.kp_catalog,2,0.0,axis=1)
		self.objps=np.insert(self.kp_catalog_original,2,0.0,axis=1)
		self.objps=np.array([self.objps],dtype="float32")
#		self.objps=np.float32(self.objps).reshape(-1,3)
#		print(self.objps)
#		self.mtx=np.insert(self.mtx,2,insert_m,axis=0)
		imgpoints=self.kp_img
		imgpoints=np.array([imgpoints],dtype="float32")
		ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objps, imgpoints, gray.shape[::-1],init_mtx,init_dist,flags=cv2.CALIB_USE_INTRINSIC_GUESS+cv2.CALIB_THIN_PRISM_MODEL+cv2.CALIB_RATIONAL_MODEL )
#		ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objps, imgpoints, gray.shape[::-1],init_mtx,init_dist,flags=1)
#		ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objps, imgpoints, gray.shape[::-1],None,None)
#		ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(self.objps, imgpoints, gray.shape[::-1],None,None)
		self.rvecs=np.float32(self.rvecs).reshape((3,1))
		self.tvecs=np.float32(self.tvecs).reshape((3,1))

		#回転は、z軸だけをひろう
		#並進は、x軸、y軸だけひろう。また、fx,fyの割合として扱う。
#		self.rvecs=np.zeros((3,1))
#		self.tvecs=np.zeros((3,1))
#		self.rvecs=rvecs
#		self.rvecs[2]=rvecs[2]
#		self.tvecs=tvecs
#		self.tvecs[0]=tvecs[0]
#		self.tvecs[1]=tvecs[1]
#		self.tvecs[0]=tvecs[0]/self.mtx[0,0]
#		self.tvecs[1]=tvecs[1]/self.mtx[1,1]

#		ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(self.kp_catalog, self.kp_img, np.array([1920,1080),None,None)
#		ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
		#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],initial_mtx,initial_dist)
		
		#ret, mtx, dist, rvecs, tvecs =cv2.fisheye.calibrate(objpoints,imgpoints,gray.shape[::-1],K,d,tvecs,calibration_flags,self.criteria)

#ベタ打ち
#		fx=1.14731e3
#		fy=1.14814e3
#		cx=9.68865e2	#pixel
#		cy=5.32106e2	#pixel
#		self.mtx = np.array([[fx, 0.0,cx], # カメラ行列
#						[0.0,fy, cy],
#						[0.0,0.0,1.0]])
#		self.dist = np.array([-3.84386e-01,2.00727e-01,7.27513e-04,3.32499e-04,-6.59210e-02])
#		self.rvecs = np.zeros((3, 1))
#		self.tvecs = np.zeros((3, 1))
		
		print("Err : ")
		print(ret)
		print("\n")
		print("Camera matrix : ")
		print(self.mtx)
		print("\n")
		print("dist : ")
		print(self.dist)
		print("rvecs : ")
		print(self.rvecs)
#		print(rvecs)
		print("tvecs : ")
		print(self.tvecs)
#		print(tvecs)

#		s_point=np.array([self.mtx[0,2],self.mtx[1,2]],dtype="int32")
#		print(s_point)
#		color_s=(255,0,0)
#		cv2.drawMarker(self.img, s_point, color_s, markerType=cv2.MARKER_STAR, markerSize=10, thickness=1, line_type=cv2.LINE_8)

#		self.save_para()
#		self.undistortion()

	def save_para(self):
		save_dist=np.float32(self.dist).reshape(1,-1)
		save_rvecs=np.float32(self.rvecs).reshape(1,-1)
		save_tvecs=np.float32(self.tvecs).reshape(1,-1)
		filename_calib="calibcamera.csv"
		np.savetxt(filename_calib, self.mtx, delimiter=',', fmt="%0.5e")
		with open(filename_calib, "a") as f:
			np.savetxt(f, save_dist, delimiter=',', fmt="%0.5e")
			np.savetxt(f, save_rvecs, delimiter=',', fmt="%0.5e")
			np.savetxt(f, save_tvecs, delimiter=',', fmt="%0.5e")
#		k_filename="K_calibcamera.csv"
#		d_filename="D_calibcamera.csv"
#		np.savetxt(k_filename, self.mtx, delimiter=',', fmt="%0.5e")
#		np.savetxt(d_filename, self.dist, delimiter=',', fmt="%0.5e")

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
		
		#歪み補正した画像をimg_undistortフォルダへ保存
		cv2.imwrite('undistort_' + self.filename, dst)
#		cv2.imwrite('undistort_' + str(os.path.basename(filepath)), dst)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def stars_adjust(self):
		if len(self.kp_index)!=0:
			self.kp_catalog=np.array(self.kp_catalog)
			self.kp_catalog_original=np.array(self.kp_catalog_original)
			self.kp_img=np.array(self.kp_img)
#		print(self.kp_catalog,self.kp_img)
			if self.kp_catalog.shape[0]>20:
				#カメラの歪みを考慮したカタログ星の位置を示す
				self.camera()
				self.calc_catalog_point()

#			self.mtx,inliers=cv2.estimateAffinePartial2D(self.kp_catalog_original,self.kp_img)
#			mtx,inliers=cv2.estimateAffinePartial2D(self.kp_img,self.kp_catalog)
#			mtx,inliers=cv2.estimateAffine2D(self.kp_catalog_original,self.kp_img)
#		else:
#			self.mtx=np.array([[1,0,0],[0,1,0]])
#			inliers=np.array([[0]])
#		insert_m=[0,0,1]
#		self.mtx=np.insert(self.mtx,2,insert_m,axis=0)
#		self.mtx_old=np.insert(self.mtx_old,2,insert_m,axis=0)
#		self.mtx=np.dot(self.mtx,self.mtx_old)
##		mtx=np.delete(mtx,2,axis=0)
##		print(inliers)
##		print(sum(inliers))
##		print(self.mtx)
##回転角を示す
##		degree = np.rad2deg(-np.arctan2(self.mtx[0, 1], self.mtx[0, 0]))
#
##		self.calc_stars_point()
##		self.rotate_point()
#		self.mxt_old=self.mtx
#		self.mtx_old=self.mtx_old[:2,:]

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

	def point_get(self,event):
		sd=self.nearst(event)
		if sd!=0:
			if self.flag==0:
				self.kp_index_catalog=sd
				self.flag=1
			else:
				self.kp_index.append([self.kp_index_catalog,sd])
				if len(self.kp_index)>2:
					self.check_kp_index()
				
				self.check_stars["text"]=str(len(self.kp_index))
				self.make_kp_list()
				self.drawline()
				self.disp_img()
				self.flag=0
#		print(self.kp_index)

	#位置のリストを作成する
	def make_kp_list(self):
		self.kp_catalog=[]
		self.kp_catalog_original=[]
		self.kp_img=[]
		for index in self.kp_index:
			self.kp_catalog.append(self.stars_catalog[index[0]])
			self.kp_catalog_original.append(self.stars_catalog_original[index[0]]) #add
			self.kp_img.append(self.stars_img[index[1]])

	def check_kp_index(self):
	#既にその星の登録があるならそれを解除して登録する。
		new_index=np.array(self.kp_index[-1])
		tmp1 = []
		for i in range(len(self.kp_index[:-1])):
			c = np.array(self.kp_index[i])
			#catalogまたはimgのindexが一致していたら被り
#			print(new_index,c)
#			print(np.equal(c,new_index))
#			print(sum(np.equal(c,new_index)))
#			if not (np.array_equal(c,new_index)):
			if not (sum(np.equal(c,new_index))):
				tmp1.append(c)
		
		self.kp_index=tmp1
		self.kp_index.append(new_index)
	
#	def check_kp(self):
#	#既にその星の登録があるならそれを解除して登録する。
#		kc=np.array(self.kp_catalog[-1])
#		kco=np.array(self.kp_catalog[-1])
#		ki=np.array(self.kp_img[-1])
#		tmp1 = []
#		tmp2 = []
#		tmp3 = []
#		for i in range(len(self.kp_catalog[:-1])):
#			c = np.array(self.kp_catalog[i])
#			co = np.array(self.kp_catalog_original[i])
#			i = np.array(self.kp_img[i])
#			if not (np.array_equal(c,kc) or np.array_equal(i,ki)):
#				tmp1.append(c)
#				tmp2.append(i)
#				tmp3.append(co)
#		
#		self.kp_catalog=tmp1
#		self.kp_catalog_original=tmp3
#		self.kp_img=tmp2
#		self.kp_catalog.append(kc)
#		self.kp_catalog_original.append(kco)
#		self.kp_img.append(ki)
##		print(self.kp_catalog)
##		print(self.kp_catalog_original)
##		print(self.kp_img)

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
		if self.img_flag!=0 and 0<=x and x<=self.img.shape[1] and 0<=y and y<=self.img.shape[0]:
			self.mause_posi["text"] = str([x,y]) 
			self.frame_rect(x, y)
			self.canvas_set(x, y)

	def frame_rect(self, x, y):
		# 過去に枠線が描画されている場合はそれを削除し、メイン画像内マウス位置に枠線を描画
		x=x/2
		y=y/2
#		self.frame_refresh()
#		self.crop_frame = (x-_framelength/2, y-_framelength/2, x+_framelength/2, y+_framelength/2)	
		self.crop_frame = [x-_framelength/2, y-_framelength/2, x+_framelength/2, y+_framelength/2]
		if self.crop_frame[0]<0:
			self.crop_frame[0]=0
		if self.crop_frame[1]<0:
			self.crop_frame[1]=0
#		self.rectframe = self.canvas1.create_rectangle(self.crop_frame, outline='#AAA', width=2, tag='rect')

	def frame_refresh(self):
		try:
			self.canvas1.delete('rect')
		except:
			pass

	def image_refresh(self):
		try:
			self.canvas2.delete('cv1')
		except:
			pass

	def canvas_set(self, x, y):
		# 枠線内をクロップし、ズームする
#		zoom_mag = _squarelength / _framelength
		self.croped = self.img[2*int(self.crop_frame[1]):2*int(self.crop_frame[3]),2*int(self.crop_frame[0]):2*int(self.crop_frame[2]),:]
#		self.croped = self.img[int(self.crop_frame[1]):int(self.crop_frame[3]),int(self.crop_frame[0]):int(self.crop_frame[2]),:]

		# ズームした画像を右側canvasに当てはめる
		self.image_refresh()

#		print(self.croped.shape)
		#画像の端の場合の対処
		h,w=self.croped.shape[:2]
		if w<100 or h<100:
			size=(2*_framelength,2*_framelength,3)
			croped=np.zeros(size,np.uint8)
			if self.crop_frame[0]==0:
				xs=2*_framelength-w
				xe=2*_framelength
			else:
				xs=0
				xe=w
			if self.crop_frame[1]==0:
				ys=2*_framelength-h
				ye=2*_framelength
			else:
				ys=0
				ye=h

#			print(w,h)
#			print(xs,xe,ys,ye)
			croped[ys:ye,xs:xe,:]=self.croped
			self.croped=croped

		self.img_dispa = cv2.resize(self.croped, dsize=(200,200))
		self.img_rgba = cv2.cvtColor(self.img_dispa, cv2.COLOR_BGR2RGB) # imreadはBGRなのでRGBに変換
		self.img_pila = Image.fromarray(self.img_rgba) # RGBからPILフォーマットへ変換
		self.img_tka  = ImageTk.PhotoImage(self.img_pila) # ImageTkフォーマットへ変換

#		self.sub_cv1 = self.canvas2.create_image(0, 0, image=self.sub_image1, anchor=tk.NW, tag='cv1')
		self.sub_cv1 = self.canvas2.create_image(0, 0, image=self.img_tka, anchor=tk.NW, tag='cv1')

		self.canvas2.create_line(_squarelength/2, 0,_squarelength/2, _squarelength,fill="#ff0000",tag="line1")
		self.canvas2.create_line(0, _squarelength/2,_squarelength, _squarelength/2,fill="#ff0000",tag="line2")

def main():
	root = tk.Tk()
	app = MainApplication(master = root)
# tkinterのメインループを開始
	app.mainloop()

if __name__ == '__main__':
	main()
