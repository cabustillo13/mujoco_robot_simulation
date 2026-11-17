"""
Fase 3: cinemática inversa (IK) para llevar el efector a las posiciones 3D
que calcula la percepción de la Fase 2.

Flujo completo:  cámara -> detección por color -> 3D -> IK -> brazo se mueve.

Uso:
    python3 p3__cinematica_inversa.py
Genera: demo__cinematica_inversa.mp4 (video del brazo visitando cada objeto) e imprime una tabla.
"""
import cv2
import mujoco
import numpy as np

from p2__percepcion import (CAMARA, COLORES, detectar_por_color, intrinsecos,
                              pixel_a_mundo)
from p1__escena import ALTO, ANCHO, XML

N_ARM = 4  # las 4 primeras articulaciones del modelo son las del brazo
FPS = 30


# ---------------------------------------------------------------------------
# 1) Percepción (lo que hicimos en la Fase 2, empaquetado en una función)
# ---------------------------------------------------------------------------
def percibir(model, data, renderer):
    """Devuelve {nombre: xyz en el mundo} para cada objeto detectado."""
    renderer.update_scene(data, camera=CAMARA)
    rgb = renderer.render()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=CAMARA)
    depth = renderer.render()
    renderer.disable_depth_rendering()

    f, cx, cy = intrinsecos(model, CAMARA)
    objetos = {}
    for nombre, rangos in COLORES.items():
        res = detectar_por_color(rgb, rangos)
        if res is None:
            continue
        mascara, (u, v) = res
        z = float(np.median(depth[mascara > 0]))
        objetos[nombre] = pixel_a_mundo(u, v, z, f, cx, cy, data, CAMARA)
    return objetos


# ---------------------------------------------------------------------------
# 2) Cinemática inversa por Jacobiano (damped least squares)
# ---------------------------------------------------------------------------
def resolver_ik(model, data, objetivo, q_ini, sitio="efector",
                iters=300, tol=1e-3, lam=0.05, paso_max=0.3):
    """
    Busca los ángulos q de las articulaciones del brazo tal que el sitio
    `efector` quede en `objetivo` (xyz, mundo).

    Idea: en cada iteración linealizamos  dx = J(q) * dq  y resolvemos
          dq = J^T (J J^T + lam^2 I)^-1 * error
    El término lam^2 I (amortiguamiento) evita explosiones cerca de singularidades.

    Se trabaja sobre una COPIA de los datos (d) para no mover el brazo real:
    esto es "planificar", no "ejecutar".
    """
    d = mujoco.MjData(model)
    d.qpos[:] = data.qpos
    d.qpos[:N_ARM] = q_ini

    sid = model.site(sitio).id
    rango = model.jnt_range[:N_ARM]
    jacp = np.zeros((3, model.nv))

    for _ in range(iters):
        mujoco.mj_kinematics(model, d)  # posiciones de cuerpos y sitios
        mujoco.mj_comPos(model, d)      # necesario antes de calcular el Jacobiano
        error = objetivo - d.site_xpos[sid]
        if np.linalg.norm(error) < tol:
            break

        mujoco.mj_jacSite(model, d, jacp, None, sid)
        J = jacp[:, :N_ARM]  # solo columnas del brazo (los objetos tienen freejoints)
        dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(3), error)

        # Limitar el paso para estabilidad y respetar límites articulares
        n = np.linalg.norm(dq)
        if n > paso_max:
            dq *= paso_max / n
        d.qpos[:N_ARM] = np.clip(d.qpos[:N_ARM] + dq, rango[:, 0], rango[:, 1])

    mujoco.mj_kinematics(model, d)
    err_final = np.linalg.norm(objetivo - d.site_xpos[sid])
    return d.qpos[:N_ARM].copy(), err_final


def q_inicial(data, objetivo):
    """Semilla razonable: yaw apuntando hacia el objetivo, resto semi-flexionado."""
    base = data.body("base").xpos
    yaw = np.arctan2(objetivo[1] - base[1], objetivo[0] - base[0])
    return np.array([yaw, 0.6, 1.0, 0.5])


# ---------------------------------------------------------------------------
# 3) Ejecución: mover el brazo con una transición suave y grabar video
# ---------------------------------------------------------------------------
class Ejecutor:
    def __init__(self, model, data, renderer, video):
        self.model, self.data, self.renderer, self.video = model, data, renderer, video
        self.pasos_por_frame = max(1, int(1 / FPS / model.opt.timestep))
        self.contador = 0

    def _grabar(self):
        self.contador += 1
        if self.contador % self.pasos_por_frame == 0:
            self.renderer.update_scene(self.data, camera=CAMARA)
            self.video.write(cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR))

    def mover_a(self, q_objetivo, duracion=1.2, espera=0.3):
        """Interpola ctrl de forma suave (smoothstep) y luego espera a que asiente."""
        q0 = self.data.ctrl[:N_ARM].copy()
        n = int(duracion / self.model.opt.timestep)
        for i in range(n):
            s = (i + 1) / n
            s = s * s * (3 - 2 * s)  # smoothstep: arranca y frena suave
            self.data.ctrl[:N_ARM] = q0 + (q_objetivo - q0) * s
            mujoco.mj_step(self.model, self.data)
            self._grabar()
        for _ in range(int(espera / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
            self._grabar()


def main():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=ALTO, width=ANCHO)

    # --- Paso 1: brazo en reposo (recto, sin tapar nada) y percibir ---
    data.ctrl[:N_ARM] = 0.0
    for _ in range(int(1.5 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    objetivos = percibir(model, data, renderer)
    print("Percepción:")
    for k, v in objetivos.items():
        print(f"  {k:16s} {np.round(v, 3)}")

    # --- Paso 2: para cada objeto, IK + movimiento ---
    video = cv2.VideoWriter("src/demo__cinematica_inversa.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                            FPS, (ANCHO, ALTO))
    ejec = Ejecutor(model, data, renderer, video)
    sid = model.site("efector").id
    ALTURA_APROX = 0.15  # el efector primero va 15 cm por encima del objeto

    print(f"\n{'objeto':16s} {'err IK':>8s} {'err final al objetivo':>22s} {'err vs posición real':>22s}")
    for nombre, p in objetivos.items():
        q_sig = q_inicial(data, p)

        # (a) punto de aproximación, encima del objeto
        q_arriba, e1 = resolver_ik(model, data, p + [0, 0, ALTURA_APROX], q_sig)
        # (b) punto de contacto; semilla = solución anterior (movimiento continuo)
        q_abajo, e2 = resolver_ik(model, data, p, q_arriba)

        if max(e1, e2) > 5e-3:
            print(f"{nombre:16s} OBJETIVO FUERA DE ALCANCE (error IK {max(e1, e2)*1000:.0f} mm)")
            continue

        ejec.mover_a(q_arriba)
        ejec.mover_a(q_abajo, duracion=0.8)

        # Medir dónde quedó el efector de verdad (tras la dinámica)
        efector = data.site_xpos[sid].copy()
        real_tope = data.body(nombre).xpos + [0, 0, 0.03]  # parte superior del objeto
        print(f"{nombre:16s} {e2*1000:6.1f}mm {np.linalg.norm(efector - p)*1000:19.1f} mm "
              f"{np.linalg.norm(efector - real_tope)*1000:19.1f} mm")

        ejec.mover_a(q_arriba, duracion=0.8)  # volver a subir antes del siguiente

    ejec.mover_a(np.zeros(N_ARM), duracion=1.2)  # volver a reposo
    video.release()
    renderer.close()
    print("\nGuardado: demo__cinematica_inversa.mp4")


if __name__ == "__main__":
    main()