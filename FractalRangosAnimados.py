import pygame as pg
import numpy as np
import psutil
import time
import threading
import numba
from numba import jit, prange

# Configuración inicial
res = width, height = 900, 500
texture_size = 256

# Generación de una textura artificial
texture_array = np.zeros((texture_size, texture_size, 3), dtype=np.uint32)
for i in range(texture_size):
    color = (i % 256, (i * 5) % 256, (i * 10) % 256)
    texture_array[i, i] = color

@jit(nopython=True, parallel=True)
def render_kernel(screen_array, texture_array, width, height, max_iter, x_min, x_max, y_min, y_max):
    """
    Renderiza el fractal con optimización JIT usando Numba.
    """
    zoom_x = (x_max - x_min) / width
    zoom_y = (y_max - y_min) / height
    for x in prange(width):
        for y in range(height):
            c = (x_min + x * zoom_x, y_min + y * zoom_y)
            z = (0.0, 0.0)
            num_iter = 0
            for i in range(max_iter):
                zx, zy = z
                z = (zx * zx - zy * zy + c[0], 2.0 * zx * zy + c[1])
                if zx * zx + zy * zy > 4.0:
                    break
                num_iter += 1
            col = int(texture_size * num_iter / max_iter)
            col = min(col, texture_size - 1)
            screen_array[x, y] = texture_array[col, col]

def medir_uso_recursos():
    print("\n--- MONITOREO DE RECURSOS COMPUTACIONALES ---")
    while True:
        cpu_usage = psutil.cpu_percent(interval=1)  # % de uso de CPU
        memory_info = psutil.virtual_memory()  # Uso de RAM
        print(f"CPU: {cpu_usage:.2f}% | RAM: {memory_info.percent:.2f}%")
        time.sleep(1)

# Estado inicial antes de ejecutar el programa
print("\n--- ESTADO INICIAL DE RECURSOS ---")
cpu_inicial = psutil.cpu_percent(interval=1)  # Uso inicial de la CPU
ram_inicial = psutil.virtual_memory().percent  # Uso inicial de la RAM

print(f"Uso Inicial de CPU: {cpu_inicial:.2f}%")
print(f"Uso Inicial de RAM: {ram_inicial:.2f}%")
print(f"Hilos utilizados por Numba: {numba.get_num_threads()}")
print(f"CPU Disponible: {psutil.cpu_count(logical=True)} núcleos")
print(f"RAM Total: {psutil.virtual_memory().total / 1e9:.2f} GB")
print("------------------------------------\n")

# Hilo separado para medir recursos sin afectar pygame
monitor_thread = threading.Thread(target=medir_uso_recursos, daemon=True)
monitor_thread.start()

class Fractal:
    def __init__(self, app):
        self.app = app
        self.screen_array = np.zeros((width, height, 3), dtype=np.uint32)
        self.x_min, self.x_max, self.y_min, self.y_max = -2, 1, -1, 1
        self.scale = 0.993

        self.target_x_min, self.target_x_max = self.x_min, self.x_max
        self.target_y_min, self.target_y_max = self.y_min, self.y_max

        self.smooth_factor = 0.1 
        self.threshold = 1e-5 

        self.max_iter, self.max_iter_limit = 30, 5500
        self.app_speed = 1 / 4000
        self.prev_time = pg.time.get_ticks()
        self.needs_redraw = True  # Renderizar solo si hay cambios
        self.needs_rescale = True

    def delta_time(self):
        time_now = pg.time.get_ticks()
        dt = (time_now - self.prev_time) * self.app_speed
        self.prev_time = time_now
        return dt

    def control(self):
        pressed_key = pg.key.get_pressed()
        dt = self.delta_time()

        rango_x = (self.x_max - self.x_min)
        rango_y = (self.y_max - self.y_min)
        move_factor = 0.3 * dt

        if pressed_key[pg.K_a]:
            desp = rango_x * move_factor
            self.x_min -= desp
            self.x_max -= desp
            self.needs_redraw = True
        if pressed_key[pg.K_d]:
            desp = rango_x * move_factor
            self.x_min += desp
            self.x_max += desp
            self.needs_redraw = True

        if pressed_key[pg.K_w]:
            desp = rango_y * move_factor
            self.y_min += desp
            self.y_max += desp
            self.needs_redraw = True
        if pressed_key[pg.K_s]:
            desp = rango_y * move_factor
            self.y_min -= desp
            self.y_max -= desp
            self.needs_redraw = True

        center_x = 0.5 * (self.x_min + self.x_max)
        center_y = 0.5 * (self.y_min + self.y_max)

        if pressed_key[pg.K_UP]:
            new_width = rango_x * self.scale
            new_height = rango_y * self.scale
            self.x_min = center_x - new_width / 2
            self.x_max = center_x + new_width / 2
            self.y_min = center_y - new_height / 2
            self.y_max = center_y + new_height / 2
            self.needs_redraw = True

        if pressed_key[pg.K_DOWN]:
            new_width = rango_x / self.scale
            new_height = rango_y / self.scale
            self.x_min = center_x - new_width / 2
            self.x_max = center_x + new_width / 2
            self.y_min = center_y - new_height / 2
            self.y_max = center_y + new_height / 2
            self.needs_redraw = True

        if pressed_key[pg.K_LEFT]:
            self.max_iter -= 1
            self.needs_redraw = True
        if pressed_key[pg.K_RIGHT]:
            self.max_iter += 1
            self.needs_redraw = True
        self.max_iter = min(max(self.max_iter, 2), self.max_iter_limit)

    def update(self):
        self.control()
        if self.needs_redraw:
            render_kernel(self.screen_array, texture_array, width, height, self.max_iter,
                          self.x_min, self.x_max, self.y_min, self.y_max)
            self.needs_redraw = False

        if self.needs_rescale:
            self.x_min += (self.target_x_min - self.x_min) * self.smooth_factor
            self.x_max += (self.target_x_max - self.x_max) * self.smooth_factor
            self.y_min += (self.target_y_min - self.y_min) * self.smooth_factor
            self.y_max += (self.target_y_max - self.y_max) * self.smooth_factor

            if (abs(self.target_x_min - self.x_min) < self.threshold and
                abs(self.target_x_max - self.x_max) < self.threshold and
                abs(self.target_y_min - self.y_min) < self.threshold and
                abs(self.target_y_max - self.y_max) < self.threshold):
                self.x_min, self.x_max = self.target_x_min, self.target_x_max
                self.y_min, self.y_max = self.target_y_min, self.target_y_max
                self.needs_rescale = False
            start_time = time.time()
            render_kernel(
                self.screen_array, texture_array,
                width, height,
                self.max_iter,
                self.x_min, self.x_max,
                self.y_min, self.y_max
            )
            end_time = time.time()
            print(f"Tiempo de renderizado: {end_time - start_time:.4f} seg")

    def draw(self):
        tmp_surface = pg.surfarray.make_surface(self.screen_array)
        flipped_surface = pg.transform.flip(tmp_surface, False, True)
        self.app.screen.blit(flipped_surface, (0, 0))
        self.draw_reference_map()

    def draw_reference_map(self):
        map_width, map_height = 125, 125
        offset_x = 10
        offset_y = height - map_height - 10

        if not hasattr(self, 'reference_surface'):
            reference_array = np.zeros((map_width, map_height, 3), dtype=np.uint32)
            render_kernel(
                reference_array, texture_array,
                map_width, map_height,
                self.max_iter, -2, 1, -1.5, 1.5
            )
            self.reference_surface = pg.surfarray.make_surface(reference_array)
            self.reference_surface = pg.transform.flip(self.reference_surface, False, True)

        self.app.screen.blit(self.reference_surface, (offset_x, offset_y))

        center_x_view = 0.5 * (self.x_min + self.x_max)
        center_y_view = 0.5 * (self.y_min + self.y_max)
        x_ratio = (center_x_view + 2) / 3
        y_ratio = (center_y_view + 1.5) / 3

        point_x = int(offset_x + x_ratio * map_width)
        point_y = int(offset_y + (1 - y_ratio) * map_height)
        pg.draw.circle(self.app.screen, (255, 0, 0), (point_x, point_y), 5)

    def run(self):
        self.update()
        self.draw()

    def set_limits(self, x_min, x_max, y_min, y_max):
        self.target_x_min, self.target_x_max = x_min, x_max
        self.target_y_min, self.target_y_max = y_min, y_max
        self.needs_redraw = True

class App:
    def __init__(self):
        self.screen = pg.display.set_mode(res, pg.SCALED)
        self.clock = pg.time.Clock()
        self.fractal = Fractal(self)
        self.font = pg.font.Font(None, 24)
        # Organización de la UI en dos columnas y dos filas:
        #   - Columna izquierda: valores mínimos
        #   - Columna derecha: valores máximos
        #   - Primera fila para x y segunda para y
        self.inputs = {
            "x_min": [60, 40, "-2.0"],
            "x_max": [180, 40, "1.0"],
            "y_min": [60, 80, "-1.0"],
            "y_max": [180, 80, "1.0"]
        }
        # Se usan rectángulos para detección de clics en los inputs
        self.text_boxes = {}
        for key, (x, y, _) in self.inputs.items():
            self.text_boxes[key] = pg.Rect(x, y, 100, 30)
        self.update_button = pg.Rect(60, 120, 100, 30)
        self.active_input = None

    def draw_current_range_labels(self):
        """Dibuja los valores actuales del fractal en la parte inferior."""
        label1 = self.font.render(f"x [ {self.fractal.x_min:.6f}; {self.fractal.x_max:.6f} ]", True, (0, 0, 0))
        label2 = self.font.render(f"y [ {self.fractal.y_min:.6f}; {self.fractal.y_max:.6f} ]", True, (0, 0, 0))
        self.screen.blit(label1, (150, height - 50))
        self.screen.blit(label2, (150, height - 30))

    def draw_color_bar(self):
        bar_width = 20
        bar_height = height - 20
        bar_x = width - bar_width - 15
        bar_y = 5

        colors = [
    (128, 128, 0),  # Olive
    (137, 173, 90), # Light Olive
    (88, 184, 112), # Light Green
    (147, 223, 190),# Light Aqua
    (98, 234, 212), # Aqua
    (68, 84, 168),  # Light Blue
    (25, 125, 250), # Light Blue
    (19, 95, 190),  # Blue
    (9, 45, 90),    # Dark Blue
    (59, 39, 78),   # Dark Purple
    (118, 78, 156), # Purple
    (177, 117, 234),# Light Purple
    (167, 67, 134), # Magenta
    (109, 33, 66),  # Dark Red
    (157, 17, 34),  # Red
    (216, 56, 112), # Pink
    (236, 156, 56), # Orange
    (255, 251, 246),# Off White
    (255, 251, 246) # Off White
]



        color_bar = pg.Surface((bar_width, bar_height))
        for y in range(bar_height):
            t = y / (bar_height - 1) * (len(colors) - 1)
            idx = int(t)
            frac = t - idx
            color1 = colors[idx]
            color2 = colors[min(idx + 1, len(colors) - 1)]
            blended_color = (
                int(color1[0] + (color2[0] - color1[0]) * frac),
                int(color1[1] + (color2[1] - color1[1]) * frac),
                int(color1[2] + (color2[2] - color1[2]) * frac)
            )
            pg.draw.line(color_bar, blended_color, (0, y), (bar_width, y))
        self.screen.blit(color_bar, (bar_x, bar_y))
        label_min = self.font.render("0", True, (0, 0, 0))
        label_max = self.font.render(str(self.fractal.max_iter), True, (0, 0, 0))
        self.screen.blit(label_max, (bar_x - bar_width - 20, bar_y + bar_height - 20))
        self.screen.blit(label_min, (bar_x - bar_width, bar_y))

    def draw_ui(self):
        # Dibujar encabezados para las columnas
        header_min = self.font.render("Mínimo", True, (0, 0, 0))
        header_max = self.font.render("Máximo", True, (0, 0, 0))
        self.screen.blit(header_min, (self.inputs["x_min"][0], self.inputs["x_min"][1] - 25))
        self.screen.blit(header_max, (self.inputs["x_max"][0], self.inputs["x_max"][1] - 25))
        
        # Dibujar etiquetas para las filas (x e y)
        label_x = self.font.render("x:", True, (0, 0, 0))
        label_y = self.font.render("y:", True, (0, 0, 0))
        self.screen.blit(label_x, (10, self.inputs["x_min"][1] + 5))
        self.screen.blit(label_y, (10, self.inputs["y_min"][1] + 5))
        
        # Dibujar cada cuadro de entrada con fondo blanco y borde negro
        for key, rect in self.text_boxes.items():
            pg.draw.rect(self.screen, (255, 255, 255), rect)
            pg.draw.rect(self.screen, (0, 0, 0), rect, 2)
            text_surface = self.font.render(self.inputs[key][2], True, (0, 0, 0))
            self.screen.blit(text_surface, (rect.x + 5, rect.y + 5))
        
        # Dibujar el botón de actualización con fondo blanco y borde negro
        pg.draw.rect(self.screen, (255, 255, 255), self.update_button)
        pg.draw.rect(self.screen, (0, 0, 0), self.update_button, 2)
        button_text = self.font.render("Actualizar", True, (0, 0, 0))
        self.screen.blit(button_text, (self.update_button.x + 5, self.update_button.y + 5))

    def handle_ui_events(self, event):
        if event.type == pg.MOUSEBUTTONDOWN:
            for key, rect in self.text_boxes.items():
                if rect.collidepoint(event.pos):
                    self.active_input = key
                    break
            else:
                self.active_input = None

            if self.update_button.collidepoint(event.pos):
                try:
                    x_min = float(self.inputs["x_min"][2])
                    x_max = float(self.inputs["x_max"][2])
                    y_min = float(self.inputs["y_min"][2])
                    y_max = float(self.inputs["y_max"][2])
                    self.fractal.set_limits(x_min, x_max, y_min, y_max)
                    self.fractal.needs_rescale = True
                except ValueError:
                    print("Por favor, ingrese valores numéricos válidos.")

        elif event.type == pg.KEYDOWN:
            if self.active_input:
                if event.key == pg.K_BACKSPACE:
                    self.inputs[self.active_input][2] = self.inputs[self.active_input][2][:-1]
                else:
                    self.inputs[self.active_input][2] += event.unicode

    def run(self):
        while True:
            self.screen.fill((0, 0, 0))
            self.fractal.run()
            self.draw_current_range_labels()  # Muestra los valores actuales en la parte inferior
            self.draw_ui()
            self.draw_color_bar()
            pg.display.flip()

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    exit()
                self.handle_ui_events(event)

            self.clock.tick(60)

if __name__ == '__main__':
    pg.font.init()
    app = App()
    app.run()
