import tkinter as tk
from PIL import Image, ImageTk
import os
import rospy
from std_msgs.msg import Int32, String

class CustomUI:
    def __init__(self, root):
        self.root = root
        self.root.title("VLM UI 界面")
        self.root.geometry("1500x1000")  # 减小窗口高度

        font_style = ("Arial", 30, "bold")
        font_style2 = ("Arial", 30, "bold")
        font_style3 = ("Arial", 20, "bold")
        font_style4 = ("Arial", 20, "bold")

        # 调整根窗口的布局
        self.root.grid_rowconfigure(3, weight=0)  # 确保VLM2下面的行不拉伸
        self.root.grid_rowconfigure(4, weight=0)  # 控制最后一行不分配额外空间
        self.root.grid_columnconfigure(0, weight=1)  # 确保列正常分配

        # 初始化指示标签
        self.instruction_label = tk.Label(root, text="指示:", font=font_style, anchor="w", wraplength=15000)
        self.instruction_label.grid(row=0, column=0, padx=10, pady=5, sticky="nw", columnspan=2)

        self.vlm1_text = tk.Text(root, height=20, width=60, font=font_style4)
        self.vlm1_text.grid(row=1, column=0, padx=20, pady=5, sticky="nw", columnspan=2)
        self.vlm1_text.config(state=tk.DISABLED)

        self.vlm3_text = tk.Text(root, height=12, width=60, font=font_style3)
        self.vlm3_text.grid(row=2, column=0, padx=20, pady=5, sticky="nw", columnspan=2)
        self.vlm3_text.config(state=tk.DISABLED)

        # 图片0 和 图片1 (在一行)
        self.image_frames = []
        for i in range(2):
            frame = tk.Label(root)
            frame.grid(row=1, column=2 + i, padx=0, pady=0, sticky="nsew")
            self.image_frames.append(frame)

        # 图片2 和 图片3 (在一行)
        for i in range(2):
            frame = tk.Label(root)
            frame.grid(row=2, column=2 + i, padx=0, pady=0, sticky="nsew")
            self.image_frames.append(frame)

        # VLM2 放在图片0, 1, 2, 3的下面
        self.vlm2_label = tk.Label(root, text="VLM2:", font=font_style2, anchor="center", fg="black")
        self.vlm2_label.grid(row=3, column=2, padx=0, pady=5, sticky="n", columnspan=2)

        self.update_images()

        # 初始化 ROS 节点并订阅消息
        rospy.init_node('vlm_ui_subscriber', anonymous=True)

        # 订阅 'user_check_topic'，用来更新 VLM2 的状态
        rospy.Subscriber('user_check_topic', Int32, self.ros_callback_vlm2)

        # 订阅 'prom_topic'，用来更新 VLM1 的文本
        rospy.Subscriber('prom_topic', String, self.ros_callback_vlm1)

        # 订阅 'action2_topic'，用来更新 VLM3 的文本
        rospy.Subscriber('action2_topic', String, self.ros_callback_vlm3)

        # 订阅 'instre_topic'，用来更新指示内容
        rospy.Subscriber('instre_topic', String, self.ros_callback_instruction)

    def ros_callback_vlm2(self, data):
        # 如果接收到的消息是10，设置 VLM2 为 True（绿色）；如果是20，设置为 False（红色）
        if data.data == 10:
            self.vlm2_value = True
        elif data.data == 20:
            self.vlm2_value = False

        # 更新 VLM2 的状态
        self.update_vlm2_status()

        # 在 5 秒后重置 VLM2 为黑色
        self.root.after(5000, self.reset_vlm2_status)

    def update_vlm2_status(self):
        # 根据 vlm2_value 的值设置颜色和状态
        status_text = "True" if self.vlm2_value else "False"
        color = "green" if self.vlm2_value else "red"
        self.vlm2_label.config(text=f"VLM2: {status_text}", fg=color)

    def reset_vlm2_status(self):
        # 将 VLM2 重置为黑色
        self.vlm2_label.config(text="VLM2:", fg="black")

            
            
            

    def ros_callback_vlm1(self, data):
        # 清空 VLM1 的文本框内容
        self.vlm1_text.config(state=tk.NORMAL)
        #self.vlm1_text.delete(1.0, tk.END)  # 清空文本框
        self.vlm1_text.insert(tk.END, "**********************************\n")
        # 插入接收到的新的文本内容
        start_index = self.vlm1_text.index(tk.END) 
        print("!",start_index)
        self.vlm1_text.insert(tk.END, data.data)
        self.vlm1_text.see(start_index)
        self.vlm1_text.config(state=tk.DISABLED)

    def ros_callback_vlm3(self, data):
        #i=0
        # 清空 VLM3 的文本框内容
        self.vlm3_text.config(state=tk.NORMAL)
        #self.vlm3_text.delete(1.0, tk.END)  # 清空文本框
        #self.vlm3_text.insert(tk.END, '----------------------\n')
        # 插入接收到的新的文本内容
        self.vlm3_text.insert(tk.END, data.data+"\n")
        self.vlm3_text.see(tk.END)
        self.vlm3_text.config(state=tk.DISABLED)
        


    def ros_callback_instruction(self, data):
        # 更新指示标签的内容，添加收到的指示内容
        instruction_text = f"指示: {data.data}"
        self.instruction_label.config(text=instruction_text, wraplength=150000, anchor="w")

    def update_vlm2_status(self):
        status_text = "T" if self.vlm2_value else "F"
        color = "green" if self.vlm2_value else "red"
        self.vlm2_label.config(text=f"VLM2: {status_text}", fg=color)

    def update_images(self):
        image_paths = ["/home/cwh/Desktop/expTF/0.jpg", "/home/cwh/Desktop/expTF/1.jpg",
                       "/home/cwh/Desktop/expTF/2.jpg", "/home/cwh/Desktop/expTF/3.jpg"]

        for i in range(4):
            if os.path.exists(image_paths[i]):
                img = Image.open(image_paths[i])
                img = img.resize((350, 350), Image.ANTIALIAS)
                photo = ImageTk.PhotoImage(img)
                self.image_frames[i].config(image=photo)
                self.image_frames[i].image = photo
            else:
                self.image_frames[i].config(image='')

        self.root.after(5000, self.update_images)


if __name__ == "__main__":
    root = tk.Tk()
    app = CustomUI(root)
    root.mainloop()

