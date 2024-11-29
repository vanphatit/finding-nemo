import math
import time
import random
from simpleai.search import SearchProblem, astar, breadth_first, depth_first, greedy
import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import time

# Chi phí di chuyển giữa các ô
cost_regular = 1.0
COSTS = {
    "up": cost_regular,
    "down": cost_regular,
    "left": cost_regular,
    "right": cost_regular,
}

# Tạo mê cung bằng thuật toán DFS Backtracking
def generate_maze(width, height):
    maze = [["#" for _ in range(width)] for _ in range(height)]
    start_x, start_y = 1, 1
    maze[start_y][start_x] = " "
    stack = [(start_x, start_y)]
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    while stack:
        current_x, current_y = stack[-1]
        random.shuffle(directions)
        moved = False
        for dx, dy in directions:
            nx, ny = current_x + dx * 2, current_y + dy * 2
            if 1 <= nx < width - 1 and 1 <= ny < height - 1 and maze[ny][nx] == "#":
                maze[current_y + dy][current_x + dx] = " "
                maze[ny][nx] = " "
                stack.append((nx, ny))
                moved = True
                break
        if not moved:
            stack.pop()
    return maze

# Khởi tạo bản đồ
M, N = 25, 45
generated_maze = generate_maze(N, M)
MAP = ["".join(row) for row in generated_maze]
MAP = [list(row) for row in MAP]

class MazeSolver(SearchProblem):
    def __init__(self, board):
        self.board = board
        self.goal = (0, 0)
        self.explored = []

        for y in range(len(self.board)):
            for x in range(len(self.board[y])):
                if self.board[y][x].lower() == "o":
                    self.initial = (x, y)
                elif self.board[y][x].lower() == "x":
                    self.goal = (x, y)

        super(MazeSolver, self).__init__(initial_state=self.initial)

    def actions(self, state):
        actions = []
        for action in COSTS.keys():
            newx, newy = self.result(state, action)
            if self.board[newy][newx] != "#":
                actions.append(action)
        return actions

    def result(self, state, action):
        x, y = state
        if action == "up":
            y -= 1
        elif action == "down":
            y += 1
        elif action == "left":
            x -= 1
        elif action == "right":
            x += 1
        new_state = (x, y)
        if self.board[y][x] != "#" and new_state not in self.explored:
            self.explored.append(new_state)
        return new_state

    def is_goal(self, state):
        return state == self.goal

    def cost(self, state, action, state2):
        return COSTS[action]

    def heuristic(self, state):
        x, y = state
        gx, gy = self.goal
        return math.sqrt((x - gx) ** 2 + (y - gy) ** 2)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.dem = 0
        self.selected_algorithm = tk.StringVar(value="astar")
        self.title('Pinky tìm đường trong mê cung')
        self.geometry(f"{N*21+270}x{M*21+60}")
        self.cvs_me_cung = tk.Canvas(self, width=N*21, height=M*21, relief=tk.SUNKEN, border=1)
        self.image_tk = ImageTk.PhotoImage(Image.open("img_wallpaper.png"))
        self.update_map()
        self.cvs_me_cung.bind("<Button-1>", self.xu_ly_mouse)

        pinky_img = Image.open("pinky.png").resize((21, 21))
        self.pinky_tk = ImageTk.PhotoImage(pinky_img)

        # Frame menu chức năng
        menu_frame = tk.LabelFrame(self, text="Chức năng", font=("Arial", 13, "bold"), padx=10, pady=10, bg="#F7F7F7")
        btn_start = tk.Button(menu_frame, text='🟢 Bắt đầu', width=18, bg="#4CAF50", fg="white", font=("Arial", 11, "bold"),
                            command=self.btn_start_click)
        btn_start.pack(pady=6)
        btn_reset = tk.Button(menu_frame, text='🔄 Đặt lại', width=18, bg="#F44336", fg="white", font=("Arial", 11, "bold"),
                            command=self.btn_reset_click)
        btn_reset.pack(pady=6)
        btn_generate = tk.Button(menu_frame, text='🌟 Tạo bản đồ', width=18, bg="#2196F3", fg="white", font=("Arial", 11, "bold"),
                            command=self.generate_new_map)
        btn_generate.pack(pady=6)
        btn_empty_map = tk.Button(menu_frame, text='❌ Không bản đồ', width=18, bg="#FFC107", fg="black", font=("Arial", 11, "bold"),
                            command=self.generate_empty_map)
        btn_empty_map.pack(pady=6)

        # Label chọn thuật toán
        lbl_algorithm = tk.Label(menu_frame, text="Thuật toán:", font=("Arial", 12, "bold"), bg="#F7F7F7")
        lbl_algorithm.pack(pady=10)
        algorithm_menu = tk.OptionMenu(menu_frame, self.selected_algorithm, "astar", "bfs", "dfs", "greedy")
        algorithm_menu.config(width=12, font=("Arial", 12), bg="#EEEEEE")
        algorithm_menu.pack(pady=0)

        # Thêm hình ảnh dưới mục chọn thuật toán
        maze_image = Image.open("img_wallpaper.png").resize((200, 200))  # Thay bằng đường dẫn tới tệp ảnh của bạn
        self.maze_tk = ImageTk.PhotoImage(maze_image)  # Chuyển ảnh thành định dạng Tkinter
        maze_image_label = tk.Label(menu_frame, image=self.maze_tk)  # Gắn ảnh vào Label
        maze_image_label.pack(pady=10)  # Hiển thị ảnh

        # Label để hiển thị thông tin trạng thái
        self.info_label = tk.Label(self, text="Số ô đã thăm: 0 | Số ô đường đi: 0",
                                font=("Arial", 12), anchor="center", justify="center", bg="#EEEEEE", relief="sunken", padx=5)
        self.info_label.grid(row=1, column=0, columnspan=2, sticky="we", padx=5, pady=10)

        self.cvs_me_cung.grid(row=0, column=0, padx=10, pady=0)
        menu_frame.grid(row=0, column=1, padx=0, pady=0, sticky=tk.N)
        self.resizable(False, False)

    def update_map(self):
        global MAP
        dark_blue = np.zeros((21, 21, 3), np.uint8) + (np.uint8(68), np.uint8(37), np.uint8(31))
        white_color = np.zeros((21, 21, 3), np.uint8) + (np.uint8(238), np.uint8(237), np.uint8(235))
        image = np.ones((M * 21, N * 21, 3), np.uint8) * 255
        for x in range(0, M):
            for y in range(0, N):
                if MAP[x][y] == '#':
                    image[x * 21:(x + 1) * 21, y * 21:(y + 1) * 21] = dark_blue
                elif MAP[x][y] == ' ':
                    image[x * 21:(x + 1) * 21, y * 21:(y + 1) * 21] = white_color
        color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_coverted)
        self.image_tk = ImageTk.PhotoImage(pil_image)
        self.cvs_me_cung.create_image(0, 0, anchor=tk.NW, image=self.image_tk)

    def generate_new_map(self):
        global MAP
        generated_maze = generate_maze(N, M)
        MAP = ["".join(row) for row in generated_maze]
        MAP = [list(row) for row in MAP]
        self.update_map()
        self.info_label.config(text="Số ô đã thăm: 0 | Số ô đường đi: 0")
        self.btn_reset_click()
    
    def generate_empty_map(self):
        """
        Tạo bản đồ không có tường (toàn bộ là đường đi).
        """
        self.btn_reset_click()
        global MAP
        MAP = [["#" if x == 0 or y == 0 or x == N-1 or y == M-1 else " " for x in range(N)] for y in range(M)]
        self.update_map()
        self.info_label.config(text="Số ô đã thăm: 0 | Số ô đường đi: 0")

    def xu_ly_mouse(self, event):
        global MAP
        px, py = event.x, event.y
        x, y = px // 21, py // 21
        if self.dem == 0 and MAP[y][x] != '#':
            MAP[y][x] = 'o'
            self.cvs_me_cung.create_oval(x * 21 + 2, y * 21 + 2, (x + 1) * 21 - 2, (y + 1) * 21 - 2,
                                        outline='#FF0000', fill='#FF0000')
            self.dem += 1
        elif self.dem == 1 and MAP[y][x] != '#':
            MAP[y][x] = 'x'
            self.cvs_me_cung.create_rectangle(x * 21 + 2, y * 21 + 2, (x + 1) * 21 - 2, (y + 1) * 21 - 2,
                                        outline='#FF0000', fill='#FF0000')
            self.dem += 1

    def btn_start_click(self):
        problem = MazeSolver(MAP)
        
        # Bắt đầu đo thời gian
        start_time = time.perf_counter()
        
        algorithm = self.selected_algorithm.get()
        if algorithm == "astar":
            result = astar(problem, graph_search=True)
        elif algorithm == "bfs":
            result = breadth_first(problem, graph_search=True)
        elif algorithm == "dfs":
            result = depth_first(problem, graph_search=True)
        elif algorithm == "greedy":
            result = greedy(problem, graph_search=True)
        
        # Kết thúc đo thời gian
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time  # Tính thời gian chạy
        
        # Lấy các trạng thái đã dò
        explored = problem.explored
        for state in explored:
            x, y = state
            self.cvs_me_cung.create_rectangle(x * 21, y * 21, (x + 1) * 21, (y + 1) * 21,
                                        outline='#FFFF00', fill='#FFFF00')  # Màu vàng nhạt
            self.cvs_me_cung.update()
            time.sleep(0.01)

        # Lấy đường đi tối ưu từ kết quả
        path = [x[1] for x in result.path()]
        for i in range(len(path)):
            x, y = path[i]
            pinky = self.cvs_me_cung.create_image(x * 21, y * 21, anchor=tk.NW, image=self.pinky_tk)
            self.cvs_me_cung.update()
            time.sleep(0.01)
            if i < len(path) - 1:
                self.cvs_me_cung.delete(pinky)
                self.cvs_me_cung.create_rectangle(x * 21, y * 21, (x + 1) * 21, (y + 1) * 21,
                                            outline='#5AB2FF', fill='#5AB2FF')
                self.cvs_me_cung.update()

        # Cập nhật thông tin số ô đã thăm và số ô đường đi
        self.info_label.config(text=f"Số ô đã thăm: {len(explored)} | Số ô đường đi: {len(path)} | Thời gian dò tìm: {elapsed_time:.10f} giây | Đã tạo xong đường đi, bấm Reset trước khi chọn ô mới")

    def btn_reset_click(self):
        global MAP
        self.update_map()  # Vẽ lại bản đồ
        self.dem = 0  # Đặt lại số điểm đã chọn về 0
        for y in range(len(MAP)):
            for x in range(len(MAP[y])):
                # Kiểm tra nếu ô hiện tại là 'o' hoặc 'x', đổi lại thành ' '
                if MAP[y][x] in ['o', 'x']:
                    MAP[y][x] = ' '
        self.info_label.config(text="Số ô đã thăm: 0 | Số ô đường đi: 0")


if __name__ == "__main__":
    app = App()
    app.mainloop()