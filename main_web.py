import math
import time
import random
from PIL import Image, ImageDraw
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from simpleai.search import SearchProblem, astar, breadth_first, depth_first, greedy


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

# Vẽ mê cung thành hình ảnh
def draw_maze_image(maze, cell_size=21):
    # Kích thước hình ảnh
    img_width = len(maze[0]) * cell_size
    img_height = len(maze) * cell_size
    img = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(img)

    # Vẽ các ô trong mê cung
    for y, row in enumerate(maze):
        for x, cell in enumerate(row):
            top_left = (x * cell_size, y * cell_size)
            bottom_right = ((x + 1) * cell_size, (y + 1) * cell_size)
            if cell == "#":
                draw.rectangle([top_left, bottom_right], fill="black")
            elif cell == "o":
                draw.ellipse([top_left, bottom_right], fill="green")
            elif cell == "x":
                draw.ellipse([top_left, bottom_right], fill="red")

    return img

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

# Kích thước mê cung
M, N = 21, 33
cell_size = 21  # Kích thước ô

# Streamlit
st.title("Pinky tìm đường trong mê cung")

# Lưu mê cung gốc khi tạo mới
if "original_maze" not in st.session_state:
    st.session_state["original_maze"] = generate_maze(N, M)

if "maze" not in st.session_state:
    st.session_state["maze"] = [row[:] for row in st.session_state["original_maze"]]

if "dem" not in st.session_state:
    st.session_state["dem"] = 0

if "points" not in st.session_state:
    st.session_state["points"] = []

# Nút tạo mê cung mới
if st.button("Tạo mê cung mới (map mới)"):
    st.session_state["original_maze"] = generate_maze(N, M)
    st.session_state["maze"] = [row[:] for row in st.session_state["original_maze"]]
    st.session_state["points"] = []  # Reset các điểm đã chọn
    st.session_state["dem"] = 0
    if "directed" in st.session_state:
        del st.session_state["directed"]  # Xóa trạng thái 'directed' nếu tồn tại

# Nút Reset Map
if st.button("Vẽ lại mê cung (map cũ)"):
    st.session_state["maze"] = [row[:] for row in st.session_state["original_maze"]]
    st.session_state["points"] = []  # Reset các điểm đã chọn
    st.session_state["dem"] = 0
    if "directed" in st.session_state:
        del st.session_state["directed"]  # Xóa trạng thái 'directed' nếu tồn tại

algorithm = st.selectbox(
    "Chọn thuật toán tìm đường",
    ("A*", "BFS", "DFS", "Greedy")
)

# Hiển thị canvas với mê cung
maze_image = draw_maze_image(st.session_state["maze"], cell_size)
canvas_width = len(st.session_state["maze"][0]) * cell_size
canvas_height = len(st.session_state["maze"]) * cell_size

canvas_result = st_canvas(
    background_image=maze_image,
    height=canvas_height,
    width=canvas_width,
    drawing_mode="point",
    point_display_radius = 0,
    display_toolbar = False,
)

if st.session_state["dem"] == 2:
    if st.button('Tìm đường'):
        if "directed" not in st.session_state:
            st.session_state["directed"] = True
            x1 = st.session_state["points"][0][0]
            y1 = st.session_state["points"][0][1]

            x2 = st.session_state["points"][1][0]
            y2 = st.session_state["points"][1][1]

            MAP = [list(row) for row in st.session_state["maze"]]
            MAP[y1][x1] = 'o'
            MAP[y2][x2] = 'x'

            # Bắt đầu đo thời gian
            start_time = time.perf_counter()

            # Áp dụng giải thuật A*
            problem = MazeSolver(MAP)

            if algorithm == "A*":
                result = astar(problem, graph_search=True)
            elif algorithm == "BFS":
                result = breadth_first(problem, graph_search=True)
            elif algorithm == "DFS":
                result = depth_first(problem, graph_search=True)
            elif algorithm == "Greedy":
                result = greedy(problem, graph_search=True)

            # Kết thúc đo thời gian
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time  # Tính thời gian chạy

            # Trích xuất đường đi và các điểm đã dò
            path = [x[1] for x in result.path()]
            explored = problem.explored  # Các điểm đã dò

            # Tính số bước
            steps_explored = len(explored)
            steps_path = len(path)

            frames = []
            frame = draw_maze_image(st.session_state["maze"], cell_size)

            # Vẽ các điểm đã dò
            for ex in explored:
                x, y = ex
                draw = ImageDraw.Draw(frame)
                top_left = (x * cell_size, y * cell_size)
                bottom_right = ((x + 1) * cell_size, (y + 1) * cell_size)
                draw.rectangle([top_left, bottom_right], fill="#FFFF00", outline="#FFFF00")  # Màu vàng
                frames.append(frame.copy())

            # Vẽ đường đi
            for p in path:
                x, y = p
                draw = ImageDraw.Draw(frame)
                top_left = (x * cell_size, y * cell_size)
                bottom_right = ((x + 1) * cell_size, (y + 1) * cell_size)
                draw.rectangle([top_left, bottom_right], fill="#5AB2FF", outline="#5AB2FF")  # Màu xanh
                frames.append(frame.copy())

            # Tạo ảnh GIF từ các khung
            frame_one = frames[0]
            frame_one.save("maze.gif", format="GIF", append_images=frames, save_all=True, duration=0.01, loop=1)

            st.image("maze.gif")
            # In kết quả
            st.write(f"Số bước đã dò: {steps_explored}")
            st.write(f"Số bước đã đi: {steps_path}")
            st.write(f"Thời gian dò tìm: {elapsed_time:.10f} giây")
            st.info("Nhấn Ctrl + R để làm mới ứng dụng")

# Lấy tọa độ các điểm từ canvas
if canvas_result.json_data is not None:
    lst_points = canvas_result.json_data["objects"]
    if len(lst_points) > 0:
        px = lst_points[-1]["left"]
        py = lst_points[-1]["top"]

        print(f"Tọa độ: ({px}, {py})")

        x = int(px) // cell_size
        y = int(py) // cell_size

        print(f"Chọn tọa độ ({x}, {y})")
        if st.session_state["maze"][y][x] == "o":
            st.error("Điểm này đã là điểm bắt đầu (o). Vui lòng chọn điểm khác!")
        elif st.session_state["maze"][y][x] == "x":
            st.error("Điểm này đã là điểm kết thúc (x). Vui lòng chọn điểm khác!")
        elif st.session_state["maze"][y][x] != "#":  # Không chọn vào tường
            if st.session_state["dem"] < 2:
                st.session_state["dem"] += 1
                if st.session_state["dem"] == 1:  # Điểm đầu
                    st.session_state["maze"][y][x] = "o"
                    st.session_state["points"].append((x, y))
                    maze_image = draw_maze_image(st.session_state["maze"], cell_size)
                elif st.session_state["dem"] == 2:  # Điểm cuối
                    st.session_state["maze"][y][x] = "x"
                    st.session_state["points"].append((x, y))
                    maze_image = draw_maze_image(st.session_state["maze"], cell_size)
        else:
            st.error("Không thể chọn vào tường")

# Hiển thị các điểm đã chọn
st.write(f"Các điểm đã chọn: {st.session_state['points']}")