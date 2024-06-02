import customtkinter as ctk
import tkinter as tk
import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk
import threading
import os
import pandas as pd
import time

class YOLOApp:
    def __init__(self, root):
        # ルートウィンドウの初期設定
        self.root = root
        self.root.title("CYOLOv8 Real-Time Detection / トヨタ自動車株式会社 高杉 旭 問い合わせ先：akira_takasugi@mail.toyota.co.jp")
        self.window_width = 1200
        self.window_height = 635

        # ウィンドウサイズを設定し、スクリーン中央に配置
        self.set_geometry()

        # フォントの定義
        self.title_font = ctk.CTkFont(family="Brush Script MT", size=40, weight="bold")
        self.default_font = ctk.CTkFont(family="Meiryo", size=12)

        # カメラウィンドウフレームの作成（サイズを1.5倍に拡大）
        self.camera_frame = ctk.CTkFrame(root, width=900, height=540)
        self.camera_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # カメラのフィードを表示するラベルの作成
        self.camera_label = ctk.CTkLabel(self.camera_frame, text="", font=self.default_font)
        self.camera_label.pack()

        # モデルとカメラ情報を表示するラベルの作成
        self.info_label = ctk.CTkLabel(self.camera_frame, text="", font=self.default_font, anchor="w")
        self.info_label.pack(pady=(5, 0))  # 上下のパディングを狭くする

        # 解説文を表示するラベルの作成（横幅をカメラウィンドウの1.5倍に設定）
        self.description_label = ctk.CTkLabel(self.camera_frame, text="", font=self.default_font, anchor="w", wraplength=900)
        self.description_label.pack(pady=(5, 0))  # 上下のパディングを狭くする

        # モデルリストフレームの作成
        self.model_frame = ctk.CTkFrame(root)
        self.model_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # タイトルラベルの作成
        self.title_label = ctk.CTkLabel(self.model_frame, text="CYOLOv8", font=self.title_font)
        self.title_label.pack(pady=10)

        # モデルディレクトリからモデルファイルを取得
        self.model_dir = os.path.join(os.path.dirname(__file__), 'model')
        self.models = [f for f in os.listdir(self.model_dir) if f.endswith('.pt')]

        # モデル選択コンボボックスの作成
        default_model = 'yolov8n.pt' if 'yolov8n.pt' in self.models else (self.models[0] if self.models else "")
        self.model_var = tk.StringVar(value=default_model)
        self.model_combobox = ctk.CTkComboBox(self.model_frame, variable=self.model_var, values=self.models, font=self.default_font)
        self.model_combobox.pack(pady=10)

        # モデル切り替えボタンの作成
        self.switch_button = ctk.CTkButton(self.model_frame, text="モデルを切り替える", command=self.switch_model, font=self.default_font)
        self.switch_button.pack(pady=10)

        # カメラ選択コンボボックスの作成
        self.camera_var = tk.StringVar(value="0")
        self.camera_combobox = ctk.CTkComboBox(self.model_frame, variable=self.camera_var, values=["0", "1", "2", "3"], font=self.default_font)
        self.camera_combobox.pack(pady=10)

        # カメラ切り替えボタンの作成
        self.switch_camera_button = ctk.CTkButton(self.model_frame, text="カメラを切り替える", command=self.switch_camera, font=self.default_font)
        self.switch_camera_button.pack(pady=10)

        # カメラ左右反転スイッチの作成
        self.flip_var = tk.BooleanVar(value=False)
        self.flip_switch = ctk.CTkSwitch(self.model_frame, text="カメラ左右反転", variable=self.flip_var, font=self.default_font)
        self.flip_switch.pack(pady=10)

        # conf変更用のジョグダイヤル（スライダー）の作成
        self.conf_var = tk.DoubleVar(value=0.5)
        self.conf_slider = ctk.CTkSlider(self.model_frame, from_=0.0, to=1.0, variable=self.conf_var, number_of_steps=100, command=self.update_conf_label)
        self.conf_slider.pack(pady=(10, 0))

        # スライダーの値表示ラベルの作成
        self.conf_label = ctk.CTkLabel(self.model_frame, text=f"Conf: {self.conf_var.get():.2f}", font=self.default_font)
        self.conf_label.pack(pady=(0, 10))

        # CSVファイルからモデルの解説文を読み込む
        self.model_descriptions = self.load_model_descriptions(os.path.join(os.path.dirname(__file__), 'model_descriptions.csv'))

        # コンボボックスの選択変更時に解説文を更新
        self.model_combobox.bind("<<ComboboxSelected>>", self.update_description_label)

        # 初期モデルの読み込み
        if self.models:
            self.model = YOLO(os.path.join(self.model_dir, default_model))

        # カメラストリームのスレッドを開始
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.thread = threading.Thread(target=self.update_frame)
        self.thread.start()

        self.update_info_label()  # 初期表示のために追加
        self.update_description_label()  # 初期表示のために追加

    def set_geometry(self):
        # ウィンドウサイズを設定し、スクリーン中央に配置
        self.root.geometry(f'{self.window_width}x{self.window_height}')
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.window_width // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.window_height // 2)
        self.root.geometry(f'{self.window_width}x{self.window_height}+{x}+{y}')

    def load_model_descriptions(self, file_path):
        # モデルの解説文をCSVファイルから読み込む
        try:
            df = pd.read_csv(file_path)
            return dict(zip(df['model'], df['description']))
        except Exception as e:
            print(f"Error loading model descriptions: {e}")
            return {}

    def switch_model(self):
        # モデルを切り替える
        selected_model = self.model_var.get()
        model_path = os.path.join(self.model_dir, selected_model)
        self.model = YOLO(model_path)
        self.update_info_label()
        self.update_description_label()
        print(f"Switched to model: {selected_model}")

    def switch_camera(self):
        # カメラを切り替える
        camera_index = int(self.camera_var.get())
        self.cap.release()
        self.cap = cv2.VideoCapture(camera_index)
        self.update_info_label()
        print(f"Switched to camera: {camera_index}")

    def update_conf_label(self, value):
        # スライダーの値を更新
        self.conf_label.configure(text=f"Conf: {float(value):.2f}")
        self.update_info_label()

    def update_info_label(self):
        # モデル、カメラ、信頼度の情報を更新
        selected_model = self.model_var.get()
        selected_camera = self.camera_var.get()
        conf_value = self.conf_var.get()
        self.info_label.configure(text=f"Selected Model: {selected_model} / Selected Camera: {selected_camera} / Conf: {conf_value:.2f}")

    def update_description_label(self, event=None):
        # モデルの解説文を更新
        selected_model = self.model_var.get()
        description = self.model_descriptions.get(selected_model, "No description available.")
        self.description_label.configure(text=description)

    def update_frame(self):
        # フレームを更新し続けるスレッド
        while self.running:
            start_time = time.time()  # 現在の時刻を取得
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (900, 540))  # サイズを1.5倍に設定

                if self.flip_var.get():
                    frame = cv2.flip(frame, 1)

                conf_value = self.conf_var.get()
                results = self.model(frame, conf=conf_value)
                annotated_frame = results[0].plot()
                img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)

                del frame, annotated_frame, img

                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)
            
            self.root.update()
            elapsed_time = time.time() - start_time
            time.sleep(max(0, 0.033 - elapsed_time))  # 1フレームあたり約30fpsになるように調整

    def on_closing(self):
        # ウィンドウを閉じるときの処理
        self.running = False
        self.thread.join()
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    # アプリケーションのメインルーチン
    ctk.set_appearance_mode("dark")
    root = ctk.CTk()
    app = YOLOApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
