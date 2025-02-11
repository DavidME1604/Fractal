# Fractal

---
## 🔍Introducción
Este programa permite generar fractales en tiempo real, incluye una interfaz gráfica con controles para modificar la visualización del fractal.

## 🎮 Controles
- W,A,S,D : Mover la vista del fractal.
- Arriba/Abajo:hacer zoom in/out.
- Izquierda/Derecha: Ajustar cantidad de iteraciones.
## 🖥️ Requerimientos

- Pygame: permite crear la interfaz gráfica y la renderización.
- Numba: optimización mediante JIT.
- Numpy: manejo eficiente de matrices y operaciones matemáticas. 
- Psutil: monitoreo del uso de CPU y memoria.
- threading: ejecutar el monitoreo en un hilo separado.
- time: medir el tiempo de renderizado.
## 🧮Configuración inicial
Se define la resolución de la ventana con ancho de 900 pixels y de 500 pixels de alto, a demás se crea una textura artificial donde cada punto tiene un color basado en un patrón cíclico RGB.
## ⚡Funciones de renderizado
@jit(nopython=True, parallel=True)

def render_kernel(screen_array, texture_array, width, height, max_iter, x_min, x_max, y_min, y_max)

Esta función optimizada genera el fractal de Mandelbrot usando un algoritmo iterativo, con paralelismo.

1. Se asigna un valor complejo C.
2. Se itera según la ecuación de Mandelbrot z=z^2 + c.
3. Si el número excede max_iter, se asigna un color basaddo en la cantidad de iteraciones.

## 📊Monitoreo de recursos
def medir_uso_recursos():
Ejecuta un bucle en un hilo separado que imprime el uso de CPU y memoria en tiempo real

## 🎨Clase fractal
Permite manejar el renderizado y control del fractal.
### Atributos:

- screen_array: matriz en la que se almacena el fractal.
- x_min,y_min,x_max,y_max: Definen la ventana de visualización.
- scale: Factor de zoom.
- needs_redraw: Indicador de si es necesario volver a dibujar el fractal.

### Métodos

- control(): Maneja la entrada del teclado y zoom.
- update(): Actualiza la imagen.
- draw()): Dibuja el fractal en pantalla.
- draw_reference_map(): Dibuja un minimapa.
- run(): Ejecuta la actualización del fractal.
## 🖥️Clase App
### Atributos
- screen: Ventana generaada con pygame.
- clock: Control de FPS.
- fractal: Objeto de la clase Fractal.
- inputs: Diccionario con los cuadros de entrada.
### Métodos
- draw_ui(): Dibuja la interfaz.
- draw_color_bar():Muestra una barra de colores que representa la cantidad de iteraciones.
- handle_ui_events: Manejar las entradas del usuario.
- run(): Bucle principal.