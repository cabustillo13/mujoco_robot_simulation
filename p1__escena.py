"""
Primer script de MuJoCo: brazo + mesa + objetos + cámara.
Renderiza una imagen RGB y un mapa de profundidad y los guarda en disco.

Instalación:
    pip install mujoco opencv-python numpy

Uso:
    python3 p1__escena.py            # genera rgb.png y depth.png
    python3 p1__escena.py --viewer   # además abre el visor interactivo
"""
import sys

import cv2
import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# 1) Escena en XML (formato MJCF). Todo en un string para que sea autocontenido.
# ---------------------------------------------------------------------------
XML = """
<mujoco model="brazo_simple">
  <compiler angle="radian"/>  <!-- por defecto MuJoCo usa grados en el XML -->
  <option timestep="0.002" integrator="implicitfast"/>
  <visual>
    <global offwidth="640" offheight="480"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.6 0.8 1" rgb2="0.1 0.1 0.2"
             width="256" height="256"/>
    <material name="madera" rgba="0.7 0.5 0.3 1"/>
    <material name="brazo"  rgba="0.2 0.4 0.8 1"/>
  </asset>

  <worldbody>
    <light pos="0.6 0 2" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="piso" type="plane" size="3 3 0.1" rgba="0.8 0.8 0.8 1"/>

    <!-- Mesa: la cara superior queda en z = 0.4 -->
    <body name="mesa" pos="0.6 0 0.2">
      <geom type="box" size="0.4 0.4 0.2" material="madera"/>
    </body>

    <!-- Brazo de 4 articulaciones, montado en el borde de la mesa.
         contype/conaffinity="0" = sin colisiones (por ahora, para simplificar).
         gravcomp="1" = compensación de gravedad: los actuadores de posición
         no se "caen" por el peso de los eslabones. -->
    <body name="base" pos="0.3 0 0.43" gravcomp="1">
      <geom type="cylinder" size="0.05 0.03" material="brazo" contype="0" conaffinity="0"/>
      <joint name="hombro_yaw" type="hinge" axis="0 0 1" range="-3.1 3.1" damping="2"/>
      <body name="link1" pos="0 0 0.0" gravcomp="1">
        <joint name="hombro_pitch" type="hinge" axis="0 1 0" range="-1.5 1.5" damping="2"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.25" size="0.025" material="brazo" contype="0" conaffinity="0"/>
        <body name="link2" pos="0 0 0.25" gravcomp="1">
          <joint name="codo" type="hinge" axis="0 1 0" range="-2.5 2.5" damping="1"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.25" size="0.02" material="brazo" contype="0" conaffinity="0"/>
          <body name="link3" pos="0 0 0.25" gravcomp="1">
            <joint name="muneca" type="hinge" axis="0 1 0" range="-2.5 2.5" damping="0.5"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.10" size="0.015" material="brazo" contype="0" conaffinity="0"/>
            <site name="efector" pos="0 0 0.10" size="0.01" rgba="1 0 0 1"/>
          </body>
        </body>
      </body>
    </body>

    <!-- Objetos sueltos sobre la mesa (free joint = pueden moverse/caer) -->
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
      <geom type="sphere" size="0.03" rgba="1 0.9 0 1" mass="0.05"/>
    </body>

    <!-- Cámara fija, mirando hacia la mesa -->
    <body name="foco" pos="0.5 0 0.5"/>
    <camera name="cam_mesa" pos="1.4 0 1.1" mode="targetbody" target="foco"/>
  </worldbody>

  <!-- Un actuador de posición por articulación: ctrl = ángulo objetivo (rad) -->
  <actuator>
    <position name="a_yaw"    joint="hombro_yaw"   kp="40" kv="4"/>
    <position name="a_pitch"  joint="hombro_pitch" kp="40" kv="4"/>
    <position name="a_codo"   joint="codo"         kp="30" kv="3"/>
    <position name="a_muneca" joint="muneca"       kp="10" kv="1"/>
  </actuator>
</mujoco>
"""

ANCHO, ALTO = 640, 480


def main():
    # -----------------------------------------------------------------------
    # 2) Cargar modelo (model = estructura fija, data = estado que cambia)
    # -----------------------------------------------------------------------
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    # Pose objetivo del brazo: [yaw, pitch, codo, muñeca] en radianes
    data.ctrl[:] = [0.3, 0.7, 1.2, 0.5]

    # -----------------------------------------------------------------------
    # 3) Simular 2 segundos para que el brazo llegue a la pose y todo se asiente
    # -----------------------------------------------------------------------
    pasos = int(2.0 / model.opt.timestep)
    for _ in range(pasos):
        mujoco.mj_step(model, data)

    pos_efector = data.site("efector").xpos
    print(f"Posición del efector (x, y, z) en metros: {np.round(pos_efector, 3)}")
    for nombre in ("cubo_rojo", "cubo_verde", "esfera_amarilla"):
        print(f"{nombre:16s}: {np.round(data.body(nombre).xpos, 3)}")

    # -----------------------------------------------------------------------
    # 4) Renderizar la cámara: RGB y profundidad
    # -----------------------------------------------------------------------
    renderer = mujoco.Renderer(model, height=ALTO, width=ANCHO)

    renderer.update_scene(data, camera="cam_mesa")
    rgb = renderer.render()  # (ALTO, ANCHO, 3) uint8, orden RGB

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="cam_mesa")
    depth = renderer.render()  # (ALTO, ANCHO) float32, en metros
    renderer.disable_depth_rendering()
    renderer.close()

    print(f"RGB shape: {rgb.shape}, depth rango: {depth.min():.2f}–{depth.max():.2f} m")

    # -----------------------------------------------------------------------
    # 5) Guardar imágenes (OpenCV usa BGR, por eso convertimos)
    # -----------------------------------------------------------------------
    cv2.imwrite("src/rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # Para visualizar: recortamos a 3 m (el fondo/cielo da ~75 m y aplasta el contraste)
    d = np.clip(depth, 0, 3.0)
    d8 = (255 * d / 3.0).astype(np.uint8)
    cv2.imwrite("src/depth.png", d8)
    np.save("src/depth.npy", depth)  # valores reales en metros, para usar después
    print("Guardados: rgb.png, depth.png, depth.npy")

    # -----------------------------------------------------------------------
    # 6) Visor interactivo opcional
    # -----------------------------------------------------------------------
    if "--viewer" in sys.argv:
        from mujoco import viewer

        print("Abriendo visor... (cerrá la ventana para terminar)")
        viewer.launch(model, data)


if __name__ == "__main__":
    main()