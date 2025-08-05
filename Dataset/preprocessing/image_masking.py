'''
커리큘럼 러닝용 이미지 생산 시 활용하는 도구(GUI)

원천데이터 내 레이블링하기 불확실한 영역이 존재하는 경우 사용.

이미지 내 레이블링 시 논란의 여지가 없는 영역에 바운딩박스(Bounding Box)를 그린 후 저장하면, 하위 폴더에 새로운 이미지 저장.
새로 생성되는 이미지는 바운딩 박스 영역을 제외한 나머지 영역의 픽셀값을 (0, 0, 0)으로 변경.
'''
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


class ImageBBoxApp(tk.Tk):
    """
    이미지 디렉토리 선택, 다중 바운딩 박스 그리기 및 삭제, 영역 외 마스킹 후 저장 기능을 제공하는 GUI 애플리케이션
    """
    def __init__(self):
        super().__init__()
        self.title("Image BBox Processor")
        self.dir_path = None            # 선택된 디렉토리 경로
        self.image_paths = []           # 처리할 이미지 파일 목록
        self.index = 0                  # 현재 이미지 인덱스
        self.orig_image = None          # 원본 PIL 이미지
        self.display_image = None       # 리사이즈된 PIL 이미지
        self.scale = 1.0                # 리사이즈 비율
        self.boxes = []                 # [{'coords':(x0,y0,x1,y1), 'rect_id':id}, ...]
        self.current_rect = None        # 드래그 중일 때 캔버스 사각형 ID

        # 버튼 프레임
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=5)

        self.dir_btn = tk.Button(btn_frame, text="Image Dir", command=self.select_dir)
        self.dir_btn.pack(side=tk.LEFT, padx=5)

        self.prev_btn = tk.Button(btn_frame, text="Previous", command=self.prev_image, state=tk.DISABLED)
        self.prev_btn.pack(side=tk.LEFT, padx=5)

        self.next_btn = tk.Button(btn_frame, text="Next", command=self.next_image, state=tk.DISABLED)
        self.next_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = tk.Button(btn_frame, text="Save", command=self.save_image, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        # 이미지 캔버스
        self.canvas = tk.Canvas(self, width=1280, height=720)
        self.canvas.pack()

        # 바운딩 박스 그리기 및 삭제 바인딩
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.bind("<Button-3>", self.on_right_click)  # 우클릭 시 삭제

    def select_dir(self):
        path = filedialog.askdirectory()
        if not path:
            return
        self.dir_path = path
        processed_dir = os.path.join(path, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        files = os.listdir(path)
        self.image_paths = [os.path.join(path, f)
                            for f in files
                            if f.lower().endswith('.jpg')
                            and not os.path.exists(os.path.join(processed_dir,
                                                                 os.path.splitext(f)[0] + '_processed.jpg'))]
        if not self.image_paths:
            messagebox.showinfo("정보", "처리할 jpg 파일이 없습니다.")
            return

        self.index = 0
        self.prev_btn.config(state=tk.NORMAL)
        self.next_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)
        self.load_image()

    def load_image(self):
        img_path = self.image_paths[self.index]
        self.orig_image = Image.open(img_path).convert('RGB')
        ow, oh = self.orig_image.size
        dw = 1280
        self.scale = dw / ow
        dh = int(oh * self.scale)
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        self.display_image = self.orig_image.resize((dw, dh), resample)
        self.photo = ImageTk.PhotoImage(self.display_image)

        self.canvas.config(width=dw, height=dh)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # 박스 초기화
        self.boxes.clear()
        self.current_rect = None

    def on_button_press(self, event):
        # 사각형 드래그 시작
        self.start_x = event.x
        self.start_y = event.y
        # 드래그 중 사각형 초기화
        if self.current_rect:
            self.canvas.delete(self.current_rect)
            self.current_rect = None

    def on_move_press(self, event):
        # 드래그하며 사각형 갱신
        if self.current_rect:
            self.canvas.delete(self.current_rect)
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline='red', width=2)

    def on_button_release(self, event):
        # 드래그 완료, 박스 저장
        if not self.current_rect:
            return
        x0, y0 = min(self.start_x, event.x), min(self.start_y, event.y)
        x1, y1 = max(self.start_x, event.x), max(self.start_y, event.y)
        self.boxes.append({'coords': (x0, y0, x1, y1), 'rect_id': self.current_rect})
        self.current_rect = None

    def on_right_click(self, event):
        # 우클릭한 영역 안의 박스 삭제
        for box in reversed(self.boxes):
            x0, y0, x1, y1 = box['coords']
            if x0 <= event.x <= x1 and y0 <= event.y <= y1:
                self.canvas.delete(box['rect_id'])
                self.boxes.remove(box)
                break

    def save_image(self):
        if not self.boxes:
            messagebox.showwarning("경고", "적어도 하나의 바운딩 박스를 그려주세요.")
            return
        ow, oh = self.orig_image.size
        # 검정 배경 이미지 생성
        new_img = Image.new('RGB', (ow, oh), (0, 0, 0))
        # 각 박스 영역 복사
        for box in self.boxes:
            x0, y0, x1, y1 = box['coords']
            x0o = int(x0 / self.scale)
            y0o = int(y0 / self.scale)
            x1o = int(x1 / self.scale)
            y1o = int(y1 / self.scale)
            region = self.orig_image.crop((x0o, y0o, x1o, y1o))
            new_img.paste(region, (x0o, y0o))

        # 저장
        processed_dir = os.path.join(self.dir_path, 'processed')
        base = os.path.splitext(os.path.basename(self.image_paths[self.index]))[0]
        save_path = os.path.join(processed_dir, f"{base}_processed.jpg")
        new_img.save(save_path)
        messagebox.showinfo("정보", f"저장 완료:\n{save_path}")

        # 목록에서 제거 후 다음
        self.image_paths.pop(self.index)
        if not self.image_paths:
            messagebox.showinfo("정보", "모든 이미지 작업이 완료되었습니다.")
            self.canvas.delete('all')
            self.prev_btn.config(state=tk.DISABLED)
            self.next_btn.config(state=tk.DISABLED)
            self.save_btn.config(state=tk.DISABLED)
            return
        if self.index >= len(self.image_paths):
            self.index = len(self.image_paths) - 1
        self.load_image()

    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.load_image()

    def next_image(self):
        if self.index < len(self.image_paths) - 1:
            self.index += 1
            self.load_image()


if __name__ == '__main__':
    app = ImageBBoxApp()
    app.mainloop()
