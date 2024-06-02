import customtkinter as ctk
import tkinter as tk
import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk
import threading
import os
import pandas as pd
import time
import csv
from collections import deque  # データを保持するためのデックをインポート
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class YOLOApp:
    def __init__(self, root):
        # ルートウィンドウの初期設定
        self.root = root
        self.root.title("CYOLOv8 Real-Time Detection / トヨタ自動車株式会社 高杉 旭 問い合わせ先：akira_takasugi@mail.toyota.co.jp")
        self.window_width = 1600
        self.window_height = 635

        # ウィンドウサイズを設定し、スクリーン中央に配置
        self.set_geometry()

        # フォントの定義
        self.title_font = ctk.CTkFont(family="Brush Script MT", size=40, weight="bold")
        self.default_font = ctk.CTkFont(family="Meiryo", size=12)

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

        # カメラウィンドウフレームの作成（サイズを600×480に設定）
        self.camera_frame = ctk.CTkFrame(root, width=600, height=480)
        self.camera_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # カメラのフィードを表示するラベルの作成
        self.camera_label = ctk.CTkLabel(self.camera_frame, text="", font=self.default_font)
        self.camera_label.pack()

        # モデルとカメラ情報を表示するラベルの作成
        self.info_label = ctk.CTkLabel(self.camera_frame, text="", font=self.default_font, anchor="w", width=600, padx=10)
        self.info_label.pack(pady=(5, 0), fill="x")  # 上下のパディングを狭くする

        # 解説文を表示するラベルの作成
        self.description_label = ctk.CTkLabel(self.camera_frame, text="", font=self.default_font, anchor="w", wraplength=600, width=600, padx=10)
        self.description_label.pack(pady=(5, 0), fill="x")  # 上下のパディングを狭くする

        # 散布図グラフフレームの作成
        self.scatter_frame = ctk.CTkFrame(root, width=600, height=480)
        self.scatter_frame.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

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

        # ログファイルの設定
        self.log_file = os.path.join(os.path.dirname(__file__), 'detection_log.csv')
        self.detections = deque(maxlen=10000)  # 最新10000件を保持するデックを初期化
        self.reset_log_file()  # 起動時にログファイルをリセット

        # 散布図の初期表示と定期更新
        self.update_scatter_plot()
        self.update_scatter_plot_periodically()

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

    def reset_log_file(self):
        # ログファイルをリセット
        with open(self.log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time", "ClassID", "ClassName", "X", "Y", "Width", "Height"])
        self.detections.clear()

    def log_detection(self, detection):
        # 検出結果をデックに追加し、CSVファイルに記録
        self.detections.append(detection)
        with open(self.log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time", "ClassID", "ClassName", "X", "Y", "Width", "Height"])  # ヘッダーを書き込む
            writer.writerows(self.detections)

    def update_scatter_plot(self):
        # 散布図を更新
        if not os.path.exists(self.log_file):
            return

        df = pd.read_csv(self.log_file)
        if df.empty:
            return

        fig, ax = plt.subplots()

        # 背景色を黒に設定
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        try:
            for class_name, group in df.groupby(df.columns[2]):  # ClassName列を3番目の列として指定
                ax.scatter(group[df.columns[3]], group[df.columns[4]], label=class_name)  # Xは4番目の列、Yは5番目の列

            ax.set_xlabel("X", color='white')
            ax.set_ylabel("Y", color='white')
            ax.legend(title="ClassName", facecolor='black', edgecolor='white', title_fontsize='large', labelcolor='white')

            # 軸の色を白に設定
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')

            # 軸の線の色を白に設定
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')

            ax.invert_yaxis()  # Y軸を上下反転
        except KeyError:
            print("ClassName, X, または Y 列が見つかりません。CSVファイルの内容を確認してください。")
            return

        for widget in self.scatter_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=self.scatter_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def update_scatter_plot_periodically(self):
        # 5秒ごとに散布図を更新
        self.update_scatter_plot()
        self.root.after(5000, self.update_scatter_plot_periodically)

    def update_frame(self):
        # フレームを更新し続けるスレッド
        while self.running:
            start_time = time.time()  # 現在の時刻を取得
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (600, 480))  # サイズを600×480に設定

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

            self.root.update_idletasks()  # ウィンドウの更新を滑らかにするために追加
            elapsed_time = time.time() - start_time
            time.sleep(max(0, 0.033 - elapsed_time))  # 1フレームあたり約30fpsになるように調整

            # 検出結果をCSVにログとして保存
            for result in results[0].boxes.data:
                class_id = int(result[5])
                class_name = results[0].names[class_id]
                x, y, w, h = map(int, result[:4])
                timestamp = time.strftime("%Y%m%d%H%M%S")
                detection = [timestamp, class_id, class_name, x, y, w, h]
                self.log_detection(detection)

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
