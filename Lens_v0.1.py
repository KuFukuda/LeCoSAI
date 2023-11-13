import cv2
import numpy as np
import os
import glob
import argparse

class CameraCalibUndist:
	def __init__(self):
		# Defining the dimensions of checkerboard
		self.CHECKERBOARD = (7,10)
		
		# コマンドライン引数
		args = self.get_args()
		    
		self.filepath = args.file
		self.k_filename = args.k_filename
		self.d_filename = args.d_filename
		
		self.images=glob.glob(self.filepath+"*.png")
#		print(self.images)

		# cv2.TERM_CRITERIA_EPS:指定された精度(epsilon)に到達したら繰り返し計算を終了する
		# cv2.TERM_CRITERIA_MAX_ITER:指定された繰り返し回数(max_iter)に到達したら繰り返し計算を終了する
		# cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER : 上記のどちらかの条件が満たされた時に繰り返し計算を終了する
		self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	def get_args(self):
		parser = argparse.ArgumentParser()
		
		parser.add_argument("--file", type=str, default="")
		parser.add_argument("--k_filename", type=str, default="K_fisheye.csv")
		parser.add_argument("--d_filename", type=str, default="d_fisheye.csv")
		
		args = parser.parse_args()
		
		return args

	def calibcamera(self):
		# Creating vector to store vectors of 3D points for each checkerboard image
		objpoints = []
		# Creating vector to store vectors of 2D points for each checkerboard image
		imgpoints = [] 
		
		# Defining the world coordinates for 3D points
		objp = np.zeros((1, np.prod(self.CHECKERBOARD), 3), np.float32)
		objp[0, :, :2] = np.indices(self.CHECKERBOARD).T.reshape(-1, 2)
		#objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
		
		# Extracting path of individual image stored in a given directory
#		images = glob.glob('./old/*.png')
#		print(images)

		#images+=glob.glob('./old2/*.png')
		findchess_flag=cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE
		for filepath in self.images:
			print(filepath)
			img = cv2.imread(filepath)
			#img = cv2.resize(img, (1280, 720))
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		# Find the chess board corners
		# If desired number of corners are found in the image then ret = true
		#ret, corners = cv2.findChessboardCorners(gray, self.CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
			ret, corners = cv2.findChessboardCorners(gray, self.CHECKERBOARD, findchess_flag)
		
			"""
			If desired number of corner are detected,
			we refine the pixel coordinates and display 
			them on the images of checker board
			"""
			if ret == True:
				objpoints.append(objp)
				# refining pixel coordinates for given 2d points.
				corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),self.criteria)
				
				imgpoints.append(corners2)
				
				# Draw the corners
				img = cv2.drawChessboardCorners(img, self.CHECKERBOARD, corners2,ret)
			else:
				print("No corner\n")
				
				# img_drawChessboardCornersフォルダにチェスボードのコーナー検出画像を保存
			cv2.imwrite('./img_drawChessboardCorners/' + str(os.path.basename(filepath)), img)
			cv2.waitKey(0)
				
		cv2.destroyAllWindows()
			
			#self.h,self.w = img.shape[:2]
		
		"""
		Performing camera calibration by 
		passing the value of known 3D points (objpoints)
		and corresponding pixel coordinates of the 
		detected corners (imgpoints)
		"""
		
		print(objpoints)
		print(imgpoints)
		ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
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
		#print("rvecs : \n")
		#print(rvecs)
		#print("tvecs : \n")
		#print(tvecs)
		
		self.save_para()

	def save_para(self):
		k_filename="../K_calibcamera.csv"
		d_filename="../D_calibcamera.csv"
		np.savetxt(k_filename, self.mtx, delimiter=',', fmt="%0.5e")
		np.savetxt(d_filename, self.dist, delimiter=',', fmt="%0.5e")

	def undistortion(self):
		# Using the derived camera parameters to undistort the image
		for filepath in self.images:
        
			img = cv2.imread(filepath)
			#self.h,self.w = img.shape[:2]
			#Refining the camera matrix using parameters obtained by calibration
			#ROI:Region Of Interest(対象領域)
			#newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
			#newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (self.w,self.h), 0, (self.w,self.h))
			newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, img.shape[:2], 0, img.shape[:2])
        
			# Method 1 to undistort the image
			dst = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)
        
			#undistort関数と同じ結果が返されるので、今回はコメントアウト(initUndistortRectifyMap()関数)
			#Method 2 to undistort the image
			#mapx,mapy=cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
			#dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
        
            # 歪み補正した画像をimg_undistortフォルダへ保存
			cv2.imwrite('./img_undistort/undistort_' + str(os.path.basename(filepath)), dst)
			cv2.waitKey(0)
		cv2.destroyAllWindows()

def main():
	hoge=CameraCalibUndist()
	hoge.calibcamera()
	hoge.undistortion()

if __name__=='__main__':
	main()
