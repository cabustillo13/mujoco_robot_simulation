"""
Fase 4: pick & place completo.
    cámara -> detección por color -> 3D -> IK con orientación -> agarre con pinza
    -> transporte -> soltar en una zona de entrega.

Uso:
    python3 p4__pick_and_place.py
Genera: demo__pick_and_place.mp4 e imprime una tabla con el resultado de cada objeto.
"""
import cv2
import mujoco
import numpy as np

from p2__percepcion import CAMARA
from p3__cinematica_inversa import percibir
from p1__escena import ALTO, ANCHO

# ---------------------------------------------------------------------------
# 1) Escena: brazo más largo + muñeca con roll + pinza de dos dedos
# ---------------------------------------------------------------------------
XML = """
<mujoco model="brazo_con_pinza">
  <compiler angle="radian"/>
  <!-- impratio alto = fricción más "pegajosa" y estable al agarrar -->
  <option timestep="0.002" integrator="implicitfast" impratio="10"/>
  <visual><global offwidth="640" offheight="480"/></visual>

  <!-- Contactos más rígidos (menos penetración al apretar objetos livianos) -->
  <default>
    <geom solref="0.006 1"/>
  </default>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.6 0.8 1" rgb2="0.1 0.1 0.2"
             width="256" height="256"/>
    <material name="madera" rgba="0.7 0.5 0.3 1"/>
    <material name="brazo"  rgba="0.2 0.4 0.8 1"/>
    <material name="pinza"  rgba="0.15 0.15 0.15 1"/>
  </asset>

  <worldbody>
    <light pos="0.6 0 2" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="piso" type="plane" size="3 3 0.1" rgba="0.8 0.8 0.8 1"/>

    <!-- Mesa: la cara superior queda en z = 0.4 -->
    <body name="mesa" pos="0.6 0 0.2">
      <geom type="box" size="0.4 0.4 0.2" material="madera"/>
    </body>

    <!-- Brazo. Los eslabones NO colisionan (contype=0); la pinza SÍ. -->
    <body name="base" pos="0.3 0 0.43" gravcomp="1">
      <geom type="cylinder" size="0.05 0.03" material="brazo" contype="0" conaffinity="0"/>
      <joint name="hombro_yaw" type="hinge" axis="0 0 1" range="-3.1 3.1" damping="2"/>
      <body name="link1" pos="0 0 0.03" gravcomp="1">
        <joint name="hombro_pitch" type="hinge" axis="0 1 0" range="-1.5 1.5" damping="2"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.30" size="0.025" material="brazo" contype="0" conaffinity="0"/>
        <body name="link2" pos="0 0 0.30" gravcomp="1">
          <joint name="codo" type="hinge" axis="0 1 0" range="-2.5 2.5" damping="1"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.30" size="0.02" material="brazo" contype="0" conaffinity="0"/>
          <body name="link3" pos="0 0 0.30" gravcomp="1">
            <joint name="muneca" type="hinge" axis="0 1 0" range="-2.5 2.5" damping="0.5"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.08" size="0.015" material="brazo" contype="0" conaffinity="0"/>

            <!-- Palma: roll (gira sobre el eje de la muñeca) + dos dedos prismáticos -->
            <body name="palma" pos="0 0 0.08" gravcomp="1">
              <joint name="roll" type="hinge" axis="0 0 1" range="-3.2 3.2" damping="0.3"/>
              <geom type="box" size="0.012 0.07 0.008" material="pinza"/>
              <!-- TCP (Tool Center Point): punto entre las puntas de los dedos -->
              <site name="tcp" pos="0 0 0.04" size="0.008" rgba="1 0 1 1"/>

              <body name="dedo_izq" pos="0 0.055 0" gravcomp="1">
                <joint name="dedo_izq" type="slide" axis="0 -1 0" range="0 0.045" damping="3"/>
                <geom type="box" size="0.012 0.006 0.025" pos="0 0 0.035" material="pinza"
                      friction="2 0.01 0.001" condim="4"/>
              </body>
              <body name="dedo_der" pos="0 -0.055 0" gravcomp="1">
                <joint name="dedo_der" type="slide" axis="0 1 0" range="0 0.045" damping="3"/>
                <geom type="box" size="0.012 0.006 0.025" pos="0 0 0.035" material="pinza"
                      friction="2 0.01 0.001" condim="4"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Objetos (free joint = pueden moverse) -->
    <body name="cubo_rojo" pos="0.6 0.15 0.43">
      <freejoint/>
      <geom type="box" size="0.03 0.03 0.03" rgba="1 0 0 1" mass="0.05"/>
    </body>
    <body name="cubo_verde" pos="0.75 -0.1 0.43">
      <freejoint/>
      <geom type="box" size="0.03 0.03 0.03" rgba="0 0.8 0 1" mass="0.05"/>
    </body>
    <body name="esfera_amarilla" pos="0.5 -0.2 0.43">
      <freejoint/>
      <!-- condim=6 + fricción de rodadura: sin esto la esfera gira libre entre los
           dedos y rueda sin parar sobre la mesa -->
      <geom type="sphere" size="0.03" rgba="1 0.9 0 1" mass="0.05"
            condim="6" friction="1 0.005 0.005"/>
    </body>

    <!-- Cámara de percepción (la misma de antes) y cámara solo para el video -->
    <body name="foco" pos="0.5 0 0.5"/>
    <camera name="cam_mesa"  pos="1.4 0 1.1" mode="targetbody" target="foco"/>
    <camera name="cam_video" pos="1.1 -0.9 1.0" mode="targetbody" target="foco"/>
  </worldbody>

  <actuator>
    <position name="a_yaw"    joint="hombro_yaw"   kp="80" kv="8"/>
    <position name="a_pitch"  joint="hombro_pitch" kp="80" kv="8"/>
    <position name="a_codo"   joint="codo"         kp="60" kv="6"/>
    <position name="a_muneca" joint="muneca"       kp="30" kv="3"/>
    <position name="a_roll"   joint="roll"         kp="10" kv="1"/>
    <position name="a_dedo_i" joint="dedo_izq"     kp="200" kv="10"/>
    <position name="a_dedo_d" joint="dedo_der"     kp="200" kv="10"/>
  </actuator>
</mujoco>
"""

# Índices en data.ctrl
N_ARM = 4                 # yaw, pitch, codo, muñeca (los que resuelve la IK)
I_ROLL, I_DEDO_I, I_DEDO_D = 4, 5, 6
DEDO_ABRIR, DEDO_CERRAR = 0.0, 0.045

# Geometría del mundo (conocida)
Z_MESA = 0.40
RADIO_OBJ = 0.03          # medio tamaño de los objetos (cubos de 6 cm / esfera r=3 cm)
Z_CENTRO = Z_MESA + RADIO_OBJ
Z_SEGURA = 0.60           # altura del TCP para transportar sin chocar con nada
SIGNO_ROLL = 1.0          # +1: el roll compensa el yaw para alinear la pinza con el mundo

# Zona de entrega: dónde dejar cada objeto (centro del objeto sobre la mesa)
DESTINOS = {
    "cubo_rojo": np.array([0.45, 0.27, Z_CENTRO]),
    "cubo_verde": np.array([0.55, 0.27, Z_CENTRO]),
    "esfera_amarilla": np.array([0.65, 0.27, Z_CENTRO]),
}

FPS = 30
Q_LISTO = np.array([0.0, 0.3, 1.9, np.pi - 0.3 - 1.9])  # pose "lista", pinza hacia abajo


# ---------------------------------------------------------------------------
# 2) Corrección de la percepción: de "punto de la superficie" a "centro del objeto"
# ---------------------------------------------------------------------------
def superficie_a_centro(model, data, p_superficie, radio=RADIO_OBJ):
    """
    La profundidad mide la superficie visible, no el centro. Corregimos avanzando
    `radio` a lo largo del rayo de la cámara (exacto para una esfera, buena
    aproximación para un cubo). La altura se fija con lo que sabemos: el objeto
    está apoyado en la mesa.
    """
    cam = data.cam_xpos[model.camera(CAMARA).id]
    rayo = p_superficie - cam
    rayo /= np.linalg.norm(rayo)
    centro = p_superficie + rayo * radio
    centro[2] = Z_CENTRO
    return centro


# ---------------------------------------------------------------------------
# 3) IK con restricción de orientación (pinza apuntando hacia abajo)
# ---------------------------------------------------------------------------
def resolver_ik(model, data, objetivo, q_ini, sitio="tcp",
                iters=150, tol=5e-4, lam=0.02, paso_max=0.2):
    """
    4 articulaciones (yaw + 3 de pitch) y 4 condiciones:
        3 de posición del TCP  +  1 de orientación.
    Para este brazo, "pinza apuntando hacia abajo" equivale a que la suma de los
    tres ángulos de pitch sea pi:  q_pitch + q_codo + q_muneca = pi.
    Eso es una fila más en el Jacobiano:  [0 1 1 1].
    """
    d = mujoco.MjData(model)
    d.qpos[:] = data.qpos
    d.qpos[:N_ARM] = q_ini
    sid = model.site(sitio).id
    rango = model.jnt_range[:N_ARM]
    jacp = np.zeros((3, model.nv))

    for _ in range(iters):
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)
        err_pos = objetivo - d.site_xpos[sid]
        err_ori = np.pi - d.qpos[1:4].sum()
        if np.linalg.norm(err_pos) < tol and abs(err_ori) < 1e-3:
            break

        mujoco.mj_jacSite(model, d, jacp, None, sid)
        J = np.vstack([jacp[:, :N_ARM], [0.0, 1.0, 1.0, 1.0]])  # 4x4
        err = np.r_[err_pos, err_ori]
        dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(4), err)
        n = np.linalg.norm(dq)
        if n > paso_max:
            dq *= paso_max / n
        d.qpos[:N_ARM] = np.clip(d.qpos[:N_ARM] + dq, rango[:, 0], rango[:, 1])

    mujoco.mj_kinematics(model, d)
    return d.qpos[:N_ARM].copy(), np.linalg.norm(objetivo - d.site_xpos[sid])


# ---------------------------------------------------------------------------
# 4) Robot: movimientos de alto nivel + grabación de video
# ---------------------------------------------------------------------------
class Robot:
    def __init__(self, model, data, renderer, video):
        self.model, self.data, self.renderer, self.video = model, data, renderer, video
        self.dt = model.opt.timestep
        self.pasos_frame = max(1, int(1 / FPS / self.dt))
        self.contador = 0
        self.tcp = model.site("tcp").id
        self.q = np.zeros(N_ARM)  # última configuración comandada

    # --- bajo nivel ---
    def _step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            self.contador += 1
            if self.contador % self.pasos_frame == 0:
                self.renderer.update_scene(self.data, camera="cam_video")
                self.video.write(cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR))

    def _comandar(self, q):
        self.q = q
        self.data.ctrl[:N_ARM] = q
        self.data.ctrl[I_ROLL] = SIGNO_ROLL * q[0]  # roll = yaw -> dedos alineados con el mundo

    def esperar(self, seg):
        self._step(int(seg / self.dt))

    # --- movimientos ---
    def mover_articular(self, q_obj, duracion=2.0):
        """Interpola en espacio de articulaciones (para movimientos grandes y libres)."""
        q0 = self.q.copy()
        n = int(duracion / self.dt)
        for i in range(n):
            s = (i + 1) / n
            s = s * s * (3 - 2 * s)
            self._comandar(q0 + (q_obj - q0) * s)
            self._step()
        self.esperar(0.2)

    def mover_lineal(self, destino, duracion=1.0):
        """Mueve el TCP en LÍNEA RECTA en el espacio cartesiano (IK en cada subpaso)."""
        p0 = self.data.site_xpos[self.tcp].copy()
        n = max(1, int(duracion / 0.02))
        for i in range(n):
            s = (i + 1) / n
            s = s * s * (3 - 2 * s)
            q, err = resolver_ik(self.model, self.data, p0 + (destino - p0) * s, self.q)
            if err > 5e-3:
                raise RuntimeError(f"Punto fuera de alcance: {np.round(destino, 3)} (err {err*1000:.0f} mm)")
            self._comandar(q)
            self._step(10)  # 10 pasos de 2 ms = 20 ms
        self.esperar(0.2)

    def abrir(self):
        self.data.ctrl[I_DEDO_I] = self.data.ctrl[I_DEDO_D] = DEDO_ABRIR
        self.esperar(0.5)

    def cerrar(self):
        self.data.ctrl[I_DEDO_I] = self.data.ctrl[I_DEDO_D] = DEDO_CERRAR
        self.esperar(0.8)

    def ancho_agarrado(self):
        """Ancho entre dedos (m). Un cubo de 6 cm debería dar ~0.06; si dio ~0 no agarró nada."""
        q = 0.5 * (self.data.qpos[self.model.joint("dedo_izq").qposadr[0]]
                   + self.data.qpos[self.model.joint("dedo_der").qposadr[0]])
        return 2 * (0.055 - q) - 2 * 0.006

    # --- tarea ---
    def pick_and_place(self, centro, destino):
        sobre = lambda p, z=Z_SEGURA: np.array([p[0], p[1], z])

        self.abrir()
        self.mover_lineal(sobre(centro), 1.2)       # 1. ir encima del objeto
        self.mover_lineal(centro, 0.9)              # 2. bajar en vertical
        self.cerrar()                               # 3. cerrar la pinza
        ancho = self.ancho_agarrado()
        self.mover_lineal(sobre(centro), 0.9)       # 4. subir con el objeto
        self.mover_lineal(sobre(destino), 1.4)      # 5. transportar
        self.mover_lineal(destino + [0, 0, 0.004], 0.9)  # 6. bajar hasta apoyarlo
        self.abrir()                                # 7. soltar
        self.mover_lineal(sobre(destino), 0.8)      # 8. subir
        return ancho


def main():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=ALTO, width=ANCHO)
    video = cv2.VideoWriter("src/demo__pick_and_place.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                            FPS, (ANCHO, ALTO))
    robot = Robot(model, data, renderer, video)

    # --- Percepción con el brazo en reposo (recto hacia arriba, no tapa nada) ---
    robot._comandar(np.zeros(N_ARM))
    robot.esperar(1.0)
    superficie = percibir(model, data, renderer)
    objetos = {n: superficie_a_centro(model, data, p) for n, p in superficie.items()}

    print("Percepción (centro estimado vs real):")
    for n, p in objetos.items():
        err = np.linalg.norm(p - data.body(n).xpos) * 1000
        print(f"  {n:16s} {np.round(p, 3)}  real {np.round(data.body(n).xpos, 3)}  error {err:.0f} mm")

    # --- Pick & place de cada objeto detectado ---
    robot.mover_articular(Q_LISTO, 2.0)
    resultados = []
    for nombre, centro in objetos.items():
        ancho = robot.pick_and_place(centro, DESTINOS[nombre])
        robot.esperar(0.5)
        final = data.body(nombre).xpos.copy()
        err = np.linalg.norm(final[:2] - DESTINOS[nombre][:2]) * 1000
        ok = err < 20 and abs(final[2] - Z_CENTRO) < 0.01
        resultados.append((nombre, ancho, final, err, ok))

    robot.mover_articular(np.zeros(N_ARM), 2.0)
    video.release()
    renderer.close()

    print(f"\n{'objeto':16s} {'ancho agarre':>13s} {'posición final':>26s} {'error xy':>9s}  resultado")
    for nombre, ancho, final, err, ok in resultados:
        print(f"{nombre:16s} {ancho*1000:10.0f} mm {np.round(final, 3)!s:>26s} {err:6.0f} mm  "
              f"{'OK' if ok else 'FALLÓ'}")
    print("\nGuardado: demo__pick_and_place.mp4")


if __name__ == "__main__":
    main()