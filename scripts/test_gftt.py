import cv2

# 步骤2：读取图片
img = cv2.imread('your_image_path.jpg')  # 替换为你的图片路径

# 步骤3：创建GFTT检测器
gftt_detector = cv2.GFTTDetector_create()

# 步骤4：使用GFTT检测特征点
keypoints = gftt_detector.detect(img, None)

# 步骤5：绘制特征点
img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(255, 0, 0))

# 步骤6：显示图片
cv2.imshow('GFTT Features', img_with_keypoints)

# 步骤7：等待键盘输入
cv2.waitKey(0)

# 步骤8：释放资源
cv2.destroyAllWindows()