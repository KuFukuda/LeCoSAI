import cv2
import numpy as np
from astroquery.simbad import Simbad
from astropy.coordinates import EarthLocation, AltAz
import astropy.units as u
from astropy.time import Time,TimezoneInfo
import os
import datetime
from astropy.coordinates import SkyCoord

def map_catalog(filename,img):
	simbad = Simbad()
	simbad.add_votable_fields('flux(V)')
	hoshi = simbad.query_criteria('Vmag<5',otype='star')
	
	LOCATION = EarthLocation(lon=139.3370196674786*u.deg, lat=36.41357867541122*u.deg, height=122*u.m)
	utcoffset = 0*u.hour
	tz = TimezoneInfo(9*u.hour) # 時間帯を決める。
	basename = os.path.splitext(os.path.basename(filename))[0]
	t_base=basename
	print(t_base)
	toki = datetime.datetime(int(t_base[:4]),int(t_base[4:6]),int(t_base[6:8]),int(t_base[8:10]),int(t_base[10:12]),int(t_base[12:]),tzinfo=tz)
	OBSTIME = Time(toki)
	OBSERVER = AltAz(location= LOCATION, obstime = OBSTIME)
	
	RA=hoshi['RA']
	DEC=hoshi['DEC']
	STAR_COORDINATES = SkyCoord(RA,DEC, unit=['hourangle','deg'])
	STAR_ALTAZ       = STAR_COORDINATES.transform_to(OBSERVER)
	seiza = STAR_ALTAZ.get_constellation()
	z = (seiza[:,None]==np.unique(seiza)).argmax(1)
	iro = np.stack([z/87,z%5/4,1-z%4/4],1)
	s = (5-hoshi['FLUX_V'])*1
	
	AZ  = STAR_ALTAZ.az.deg
	ALT = STAR_ALTAZ.alt.deg
	stars_catalog=np.array([AZ,ALT])
	stars_catalog=stars_catalog.T
	
	#AZ N 0 : E 90 : S 180 : W 270
	#水平 102°、垂直 54.9°
	left=146
	right=248
	top=68
	bottom=12

	h,w = img.shape[:2]
	ws=w/(right-left)
	hs=h/(top-bottom)

	stars_catalog[:,0]=(stars_catalog[:,0]-left)*ws
	stars_catalog[:,1]=h-(stars_catalog[:,1]-bottom)*hs
	
	st = [s for s in stars_catalog if -100<s[0] and s[0]<w+100]
	stars_catalog = [s for s in st if -100<s[1] and s[1]<h+100]
	
	stars_catalog=np.int32(stars_catalog)
#	stars_catalog_original=stars_catalog
	return stars_catalog

#カメラのパラメータ定義
#fx=2.51989762e+04
#fy=3.33388930e+04
#cx=8.91984110e+02
#cy=5.06379701e+02

#正式
fx=1.14731e3
fy=1.14814e3
cx=9.68865e2	#pixel
cy=5.32106e2	#pixel
mtx = np.array([[fx, 0.0,cx], # カメラ行列
				[0.0,fy, cy],
				[0.0,0.0,1.0]])
dist = np.array([-3.84386e-01,2.00727e-01,7.27513e-04,3.32499e-04,-6.59210e-02])	#正式
#dist = np.array([-1.71425457e+02, 2.17435144e+04, 9.53123407e-02, 4.22363938e-01,-5.48430241e+03])
rvecs = np.array([0.0,0.0,0.39])	#z軸の値をいじると画面上回転する
#tvecs = np.array([-0.1493574, -0.17211831, 0.0])
tvecs = np.array([0.006, 0.185, 0.0])
#mvecs = np.zeros((3, 1))
#print(mvecs.shape)
#mvecs[0]=0.1
#print(mvecs)
#rvecs = np.zeros((3, 1))
#tvecs = np.zeros((3, 1))

#星をカタログから読み込み
filename="20231111042000.jpg"
img = cv2.imread(filename)
stars_catalog=map_catalog(filename,img)
stars_catalog=np.float32(stars_catalog)
#print(stars_catalog)
#stars_catalog=np.insert(stars_catalog, 2, 0.0, axis=1)
stars_catalog_m=np.zeros((stars_catalog.shape[0],3))

stars_catalog_m[:,0]=(stars_catalog[:,0]-mtx[0,2])/mtx[0,0]
stars_catalog_m[:,1]=(stars_catalog[:,1]-mtx[1,2])/mtx[1,1]

#print(stars_catalog)
#星の位置を修正
calc_catalog, jac = cv2.projectPoints(stars_catalog_m, rvecs, tvecs, mtx, dist)
calc_catalog=np.int32(calc_catalog).reshape(-1,2)

#キャンバス作製
size=(1080,1920,3)
black_img=np.zeros(size,np.uint8)

#カタログの配置そのまま 
color_s=(0,255,0)
maker=cv2.MARKER_SQUARE
stars_catalog=np.int32(stars_catalog[:,:2])
#print(stars_catalog)
for s_point in stars_catalog:
	cv2.drawMarker(black_img, s_point, color_s, markerType=maker, markerSize=10, thickness=1, line_type=cv2.LINE_8)

#修正した星の配置
print(calc_catalog)
color_s=(255,255,255)
maker=cv2.MARKER_TRIANGLE_UP 
for s_point in calc_catalog:
#	cv2.drawMarker(black_img, s_point, color_s, markerType=maker, markerSize=10, thickness=1, line_type=cv2.LINE_8)
	cv2.circle(black_img, s_point, 5, color_s, thickness=-1)

cv2.imwrite('out.jpg', black_img)
