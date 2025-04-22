#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np

def image_callback(msg):
    # 获取图像的高度、宽度和编码
    height = msg.height
    width = msg.width
    encoding = msg.encoding

    # 将字节数据转换为NumPy数组
    if encoding == "rgb8":
        # RGB图像，每个像素3个字节
        step = 3 * width
    elif encoding == "bgr8":
        # BGR图像，每个像素3个字节
        step = 3 * width
    elif encoding == "mono8":
        # 单通道灰度图像，每个像素1个字节
        step = 1 * width
    else:
        rospy.logwarn("Unsupported encoding: %s", encoding)
        return

    # 将字节数据转换为一维NumPy数组
    image_array = np.frombuffer(msg.data, dtype=np.uint8)

    # 重新塑形数组以匹配图像的维度和通道
    image_array = image_array.reshape(height, width, step // width)

    # 将BGR图像转换为RGB图像（如果需要）
    # if encoding == "bgr8":
        # image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    out_image = np.zeros((image_array.shape[0],image_array.shape[1],image_array.shape[2]))
    out_image[:,:,0] = image_array[:,:,2]
    out_image[:,:,1] = image_array[:,:,1]
    out_image[:,:,2] = image_array[:,:,0]
    # print(out_image.shape)

    # 保存图像
    cv2.imwrite("temp/image.png", out_image)

def talker():
    # 初始化ROS节点
    rospy.init_node('image_getter', anonymous=True)

    # 创建一个Publisher，发布到'chatter'话题
    sub = rospy.Subscriber('/camera/color/image_raw', Image, image_callback)

    # 创建一个rate对象，用于控制循环频率
    rate = rospy.Rate(1)  # 10hz

    while not rospy.is_shutdown():
        # 创建一个消息
        hello_str = "hello world %s" % rospy.get_time()
        # 打印消息
        # rospy.loginfo(hello_str)
        # 发布消息
        # pub.publish(hello_str)

        # 等待一段时间
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
