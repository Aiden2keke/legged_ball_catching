import lcm
from dog_in_frame_info import dog_in_frame_info
import time
# 配置为使用特定网络接口（替换为发送端实际IP）
lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
# lc = lcm.LCM("udpm://192.168.31.230:7667?ttl=1")

while True:
    msg = dog_in_frame_info()
    
    # 设置示例数据
    msg.dog_coord = [1.5, 2.3, 0.8]
    msg.dog_orientation = [0.1, -0.2, 1.57]
    
    # 发布到频道 "DOG_CHANNEL"
    lc.publish("dog_in_frame_info", msg.encode())
    
    print("Sent dog info")
    time.sleep(1)  # 每秒发送一次