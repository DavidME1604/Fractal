import pygame as pg
import numpy as np
from numba import jit, prange

# Configuración inicial
res = width, height = 800, 450
offset = np.array([0, 0])
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
            
            c = (x_min + x  * zoom_x , y_min + y * zoom_y )
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

class Fractal:
    def __init__(self, app):
        self.app = app
        self.screen_array = np.zeros((width, height, 3), dtype=np.uint32)
        self.x_min, self.x_max, self.y_min, self.y_max = -1.943, -1.94, -0.0012, 0.0012
        self.zoom_x = (self.x_max - self.x_min) / width 
        self.zoom_y = (self.y_max - self.y_min) / height
        self.scale = 0.993

        self.target_x_min, self.target_x_max = self.x_min, self.x_max
        self.target_y_min, self.target_y_max = self.y_min, self.y_max

        self.smooth_factor = 0.1 
        self.threshold = 1e-5 

        self.max_iter, self.max_iter_limit = 30, 5500
        self.app_speed = 1 / 4000
        self.prev_time = pg.time.get_ticks()
        self.needs_redraw = True  # Indicador para renderizar solo si hay cambios
        self.needs_rescale = False

    def delta_time(self):
        """
        Calcula el tiempo delta.
        """
        time_now = pg.time.get_ticks()
        dt = (time_now - self.prev_time) * self.app_speed
        self.prev_time = time_now
        return dt

    def control(self):
        pressed_key = pg.key.get_pressed()
        dt = self.delta_time()

        rango_x = (self.x_max - self.x_min)
        rango_y = (self.y_max - self.y_min)

        # Por ejemplo, un factor de '0.3' * dt => más rápido o más lento a gusto
        move_factor = 0.3 * dt

        # A / D => mover en X
        if pressed_key[pg.K_a]:  # Izquierda
            desp = rango_x * move_factor
            self.x_min -= desp
            self.x_max -= desp
            self.needs_redraw = True
        if pressed_key[pg.K_d]:  # Derecha
            desp = rango_x * move_factor
            self.x_min += desp
            self.x_max += desp
            self.needs_redraw = True

        # W / S => mover en Y
        if pressed_key[pg.K_w]:  # Arriba (en el plano, y_min > y_max?)
            desp = rango_y * move_factor
            self.y_min += desp
            self.y_max += desp
            self.needs_redraw = True
        if pressed_key[pg.K_s]:  # Abajo
            desp = rango_y * move_factor
            self.y_min -= desp
            self.y_max -= desp
            self.needs_redraw = True

        center_x = 0.5 * (self.x_min + self.x_max)
        center_y = 0.5 * (self.y_min + self.y_max)

        # Zoom in
        if pressed_key[pg.K_UP]:
            new_width = rango_x * self.scale
            new_height = rango_y * self.scale
            self.x_min = center_x - new_width / 2
            self.x_max = center_x + new_width / 2
            self.y_min = center_y - new_height / 2
            self.y_max = center_y + new_height / 2
            self.needs_redraw = True

        # Zoom out
        if pressed_key[pg.K_DOWN]:
            new_width = rango_x / self.scale
            new_height = rango_y / self.scale
            self.x_min = center_x - new_width / 2
            self.x_max = center_x + new_width / 2
            self.y_min = center_y - new_height / 2
            self.y_max = center_y + new_height / 2
            self.needs_redraw = True

        # Resolución del fractal
        if pressed_key[pg.K_LEFT]:
            self.max_iter -= 1
            self.needs_redraw = True
        if pressed_key[pg.K_RIGHT]:
            self.max_iter += 1
            self.needs_redraw = True
        self.max_iter = min(max(self.max_iter, 2), self.max_iter_limit)

    def update(self):
        self.control()
        if self.needs_redraw:  # Solo renderizar si es necesario
            render_kernel(self.screen_array, texture_array, width, height, self.max_iter,self.x_min, self.x_max, self.y_min, self.y_max)
            self.needs_redraw = False
        # Interpolar los valores actuales hacia los objetivos
        if self.needs_rescale:
            # Interpolación suave
            self.x_min += (self.target_x_min - self.x_min) * self.smooth_factor
            self.x_max += (self.target_x_max - self.x_max) * self.smooth_factor
            self.y_min += (self.target_y_min - self.y_min) * self.smooth_factor
            self.y_max += (self.target_y_max - self.y_max) * self.smooth_factor

            # Verificar si la animación terminó
            if (
                abs(self.target_x_min - self.x_min) < self.threshold and
                abs(self.target_x_max - self.x_max) < self.threshold and
                abs(self.target_y_min - self.y_min) < self.threshold and
                abs(self.target_y_max - self.y_max) < self.threshold
            ):
                # Finalizar la animación
                self.x_min, self.x_max = self.target_x_min, self.target_x_max
                self.y_min, self.y_max = self.target_y_min, self.target_y_max
                self.needs_rescale = False

            # Llamar al render kernel para dibujar el fractal
            render_kernel(
                self.screen_array, texture_array,
                width, height,
                self.max_iter,
                self.x_min, self.x_max,
                self.y_min, self.y_max
            )

    def draw(self):
        tmp_surface = pg.surfarray.make_surface(self.screen_array)

        # Voltear la imagen en el eje Y
        flipped_surface = pg.transform.flip(tmp_surface, False, True)

        # Blit de la imagen volteada
        self.app.screen.blit(flipped_surface, (0, 0))

    def run(self):
        self.update()
        self.draw()

    def set_limits(self, x_min, x_max, y_min, y_max):
        """Define los límites objetivo para la transición animada."""
        self.target_x_min, self.target_x_max = x_min, x_max
        self.target_y_min, self.target_y_max = y_min, y_max
        self.needs_redraw = True

class App:
    def __init__(self):
        self.screen = pg.display.set_mode(res, pg.SCALED)
        self.clock = pg.time.Clock()
        self.fractal = Fractal(self)

       
        # Crear campos de entrada simples y un botón con Pygame
        self.font = pg.font.Font(None, 24)

        # Coordenadas iniciales de los inputs
        self.inputs = {
            "x_min": [10, 10, "-2.0"],
            "x_max": [120, 10, "1.0"],
            "y_min": [10, 50, "-1.0"],
            "y_max": [120, 50, "1.0"]
        }
        self.text_boxes = {}
        for key, (x, y, value) in self.inputs.items():
            self.text_boxes[key] = pg.Rect(x, y, 100, 30)

        self.update_button = pg.Rect(10, 90, 100, 30)
        self.active_input = None

    def draw_ui(self):
        # Dibuja los campos de texto y el botón
        for key, rect in self.text_boxes.items():
            pg.draw.rect(self.screen, (255, 255, 255), rect, 2)
            text_surface = self.font.render(self.inputs[key][2], True, (255, 255, 255))
            self.screen.blit(text_surface, (rect.x + 5, rect.y + 5))

        # Dibuja el botón
        pg.draw.rect(self.screen, (0, 255, 0), self.update_button)
        button_text = self.font.render("Actualizar", True, (0, 0, 0))
        self.screen.blit(button_text, (self.update_button.x + 5, self.update_button.y + 5))

    def handle_ui_events(self, event):
        if event.type == pg.MOUSEBUTTONDOWN:
            # Verifica si se hizo clic en un campo de texto
            for key, rect in self.text_boxes.items():
                if rect.collidepoint(event.pos):
                    self.active_input = key
                    break
            else:
                self.active_input = None

            # Verifica si se hizo clic en el botón
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
            self.draw_ui()
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
