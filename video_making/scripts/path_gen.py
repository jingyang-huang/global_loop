#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
import sys, select, termios, tty

class OdomToMarker:
    def __init__(self):
        rospy.init_node('odom_to_marker', anonymous=True)
        
        self.path1 = Path()
        self.path1.header.frame_id = "world"
        
        self.marker_pub = rospy.Publisher('/path_marker', Marker, queue_size=10)
        self.path1_pub = rospy.Publisher('/ros_reloc_node/pose_graph_path', Path, queue_size=10)
        self.odom_sub = rospy.Subscriber('/ros_reloc_node/odometry_rect', Odometry, self.odom_callback)
        
        self.marker = Marker()
        self.marker.header.frame_id = "world"
        self.marker.ns = "trajectory"
        self.marker.type = Marker.POINTS # LINE_STRIP = path # POINTS = CUBE
        self.marker.action = Marker.ADD
        self.marker.scale.x = 0.5
        self.marker.scale.y = 0.5
        self.marker.scale.z = 0.5
        self.marker.color.r = 0.95
        self.marker.color.g = 0.5
        self.marker.color.b = 0.0
        self.marker.color.a = 1.0
        self.marker.lifetime = rospy.Duration(0)  # 设置lifetime为0

        self.rate = rospy.Rate(100)  # 20 Hz
        self.id = 0
        self.counter = 0  # 初始化计数器
        self.pushback_enabled = False
        
        self.get_key_thread = rospy.Timer(rospy.Duration(0.01), self.get_key)  # Adjusted to 20 Hz
        
    def odom_callback(self, msg):
        self.counter += 1  # 每次回调时递增计数器
        
        # 仅在计数器达到特定值时执行操作，例如每10次回调执行一次
        if self.counter % 10 == 0:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose = msg.pose.pose

            self.path1.poses.append(pose)
            self.path1.header.stamp = rospy.Time.now()
            self.path1_pub.publish(self.path1)
            
            if self.pushback_enabled:
                point = Marker()
                point.header.frame_id = "world"
                point.header.stamp = rospy.Time.now()
                point.ns = "trajectory"
                point.id = self.id
                self.id += 1
                point.type = Marker.POINTS
                point.action = Marker.ADD
                point.pose.position.x = msg.pose.pose.position.x
                point.pose.position.y = msg.pose.pose.position.y
                point.pose.position.z = msg.pose.pose.position.z
                point.scale.x = 0.2
                point.scale.y = 0.2
                point.scale.z = 0.2
                point.color.r = 0.0
                point.color.g = 1.0
                point.color.b = 0.0
                point.color.a = 1.0
                point.lifetime = rospy.Duration(0)  # 设置lifetime为0

                self.marker.points.append(point.pose.position)
                self.marker_pub.publish(self.marker)

    def get_key(self, event):
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, termios.tcgetattr(sys.stdin))

        if key == 's':
            print('start cha\n')
            self.pushback_enabled = True
        elif key == 'd':
            print('stop cha\n')
            self.pushback_enabled = False

    def signal_handler(self, sig, frame):
        print("Ctrl+C pressed, shutting down...")
        rospy.signal_shutdown("Ctrl+C pressed")

if __name__ == '__main__':
    print("odom_to_marker running, Press 's' to enable pushback, 'd' to disable pushback")
    try:
        OdomToMarker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass