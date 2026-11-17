"""
Fase 2: detectar objetos por color (OpenCV) y pasar de píxel -> coordenadas 3D.

Requiere estar en la misma carpeta que mujoco_inicio.py (reutiliza su escena).

Uso:
    python3 p2__percepcion.py
Genera: detecciones.png (imagen anotada) e imprime una tabla estimado vs real.
"""
import cv2
import mujoco
import numpy as np

from p1__escena import ALTO, ANCHO, XML

CAMARA = "cam_mesa"

# ---------------------------------------------------------------------------
# Rangos de color en HSV. En OpenCV: H va de 0 a 179, S y V de 0 a 255.
# El rojo "da la vuelta" en el círculo de matiz, por eso tiene dos rangos.
# La mesa es naranja/marrón (H~15, S~145), así que exigimos S alta (>180)
# para no confundirla con los objetos.
# ---------------------------------------------------------------------------
COLORES = {
    "cubo_rojo": [((0, 180, 40), (5, 255, 255)), ((175, 180, 40), (179, 255, 255))],
    "cubo_verde": [((40, 150, 40), (85, 255, 255))],
    "esfera_amarilla": [((22, 180, 40), (35, 255, 255))],
}
AREA_MINIMA = 80  # píxeles; descarta ruido


# ---------------------------------------------------------------------------
# 1) Percepción 2D: máscara por color -> centroide en píxeles
# ---------------------------------------------------------------------------
def detectar_por_color(rgb, rangos):
    """Devuelve (mascara, (u, v) del centroide) del blob más grande, o None."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)  # ojo: MuJoCo entrega RGB, no BGR
    mascara = np.zeros(rgb.shape[:2], np.uint8)
    for bajo, alto in rangos:
        mascara |= cv2.inRange(hsv, np.array(bajo), np.array(alto))

    # Limpieza: quita puntitos sueltos
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = [c for c in contornos if cv2.contourArea(c) >= AREA_MINIMA]
    if not contornos:
        return None

    mejor = max(contornos, key=cv2.contourArea)
    m = cv2.moments(mejor)
    u, v = m["m10"] / m["m00"], m["m01"] / m["m00"]

    # Máscara solo del blob elegido (para promediar profundidad dentro de él)
    solo = np.zeros_like(mascara)
    cv2.drawContours(solo, [mejor], -1, 255, thickness=-1)
    return solo, (u, v)


# ---------------------------------------------------------------------------
# 2) Cámara: intrínsecos y extrínsecos
# ---------------------------------------------------------------------------
def intrinsecos(model, camara):
    """Focal en píxeles (fx = fy) y centro óptico, a partir del fov vertical."""
    cam_id = model.camera(camara).id
    fovy = np.deg2rad(model.cam_fovy[cam_id])
    f = (ALTO / 2) / np.tan(fovy / 2)
    return f, ANCHO / 2, ALTO / 2


def pixel_a_mundo(u, v, z, f, cx, cy, data, camara):
    """
    Pixel (u, v) + profundidad z (metros, a lo largo del eje óptico)
    -> punto 3D en el marco del MUNDO.

    Convención de cámara en MuJoCo (distinta de OpenCV!):
        x -> derecha, y -> arriba, y la cámara MIRA hacia -z.
    """
    x_cam = (u - cx) / f * z
    y_cam = -(v - cy) / f * z  # el eje v de la imagen crece hacia abajo
    z_cam = -z
    p_cam = np.array([x_cam, y_cam, z_cam])

    R = data.cam_xmat[data.model.camera(camara).id].reshape(3, 3)  # cámara -> mundo
    t = data.cam_xpos[data.model.camera(camara).id]
    return R @ p_cam + t


# ---------------------------------------------------------------------------
# 3) Programa principal
# ---------------------------------------------------------------------------
def main():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    # Brazo recto hacia arriba, para que no tape los objetos
    data.ctrl[:] = [0.0, 0.0, 0.0, 0.0]
    for _ in range(int(2.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)

    # --- Render RGB + profundidad ---
    renderer = mujoco.Renderer(model, height=ALTO, width=ANCHO)
    renderer.update_scene(data, camera=CAMARA)
    rgb = renderer.render()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=CAMARA)
    depth = renderer.render()
    renderer.close()

    f, cx, cy = intrinsecos(model, CAMARA)
    print(f"Intrínsecos: f={f:.1f} px, centro=({cx:.0f}, {cy:.0f})\n")

    anotada = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    print(f"{'objeto':16s} {'pixel (u,v)':14s} {'estimado (m)':24s} {'real (m)':24s} error")

    for nombre, rangos in COLORES.items():
        res = detectar_por_color(rgb, rangos)
        if res is None:
            print(f"{nombre:16s} NO DETECTADO")
            continue
        mascara, (u, v) = res

        # Profundidad: mediana dentro del blob (más robusta que un solo píxel)
        z = float(np.median(depth[mascara > 0]))
        estimado = pixel_a_mundo(u, v, z, f, cx, cy, data, CAMARA)
        real = data.body(nombre).xpos
        error_mm = np.linalg.norm(estimado - real) * 1000

        print(
            f"{nombre:16s} ({u:5.1f},{v:5.1f})   "
            f"{np.round(estimado, 3)!s:24s} {np.round(real, 3)!s:24s} {error_mm:.0f} mm"
        )

        # Anotar la imagen
        cv2.circle(anotada, (int(u), int(v)), 5, (255, 255, 255), -1)
        cv2.putText(anotada, nombre, (int(u) + 8, int(v) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite("src/detecciones.png", anotada)
    print("\nGuardado: detecciones.png")


if __name__ == "__main__":
    main()