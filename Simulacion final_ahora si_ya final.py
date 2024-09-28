import numpy as np
import matplotlib.pyplot as plt

# Parámetros
num_robots = 10
target_position = np.array([12.5, 15.0])  # Posición del objetivo como float
obstacles = np.array([[5.0, 5.0], [8.0, 8.0]])  # Posiciones de los obstáculos como float
k_att = 600.0  # Coeficiente de potencial atractivo más fuerte
k_rep = 500.0  # Potencial repulsivo fuerte para obstáculos
k_LJ = 0.1  # Coeficiente de potencial Lennard-Jones mucho más débil para robots
r_min = 1.0  # Distancia mínima para el potencial Lennard-Jones
r_safe = 3.0  # Distancia segura para el potencial repulsivo de los obstáculos
target_threshold = 5.0  # Distancia a la que los robots comienzan a desacelerar cerca del objetivo
min_distance_to_target = 0.5  # Distancia mínima para dejar de moverse hacia el objetivo
close_to_target_threshold = 4.0  # Distancia donde los robots solo sienten la atracción hacia el objetivo
damping_factor = 0.3  # Damping más fuerte para desacelerar a los robots cerca del objetivo
max_speed = 3.0  # Velocidad máxima a la que pueden viajar los robots

dt = 0.1  # Paso de tiempo de la simulación

# Inicializar posiciones de los robots aleatoriamente con orientación como float
robot_positions = np.hstack((np.random.rand(num_robots, 2) * 10, np.zeros((num_robots, 1))))  # [x, y, orientación]
robot_positions = robot_positions.astype(float)  # Asegurar que todas las posiciones son float

# Bucle de simulación
for t in range(200):
    # Inicializar velocidades
    robot_velocities = np.zeros((num_robots, 2), dtype=float)
    
    # Calcular distancias entre robots
    robot_distances = np.linalg.norm(robot_positions[:, np.newaxis, :2] - robot_positions[np.newaxis, :, :2], axis=2)
    
    # Actualizar velocidades de los robots usando campos potenciales
    for i in range(num_robots):
        # Distancia al objetivo
        distance_to_target = np.linalg.norm(target_position - robot_positions[i, :2])

        # La fuerza atractiva se hace más fuerte a medida que los robots se acercan al objetivo
        if distance_to_target > min_distance_to_target:
            # Desaceleración gradual cerca del objetivo
            att_gradient = k_att * (target_position - robot_positions[i, :2]) / max(distance_to_target, target_threshold)
        else:
            # Sin atracción si está dentro de la distancia de parada al objetivo
            att_gradient = np.array([0.0, 0.0], dtype=float)

        # Solo aplicar fuerzas repulsivas si el robot no está demasiado cerca del objetivo
        if distance_to_target > close_to_target_threshold:
            # Inicializar potencial repulsivo y gradiente
            rep_gradient_obstacles = np.array([0.0, 0.0], dtype=float)
            LJ_gradient = np.array([0.0, 0.0], dtype=float)

            # Calcular potencial repulsivo de los obstáculos
            for j in range(len(obstacles)):
                obstacle_position = obstacles[j]
                obstacle_distance = np.linalg.norm(robot_positions[i, :2] - obstacle_position)

                if obstacle_distance < r_safe:
                    # Repulsión más fuerte cuando los robots están demasiado cerca de los obstáculos
                    rep_gradient_obstacles += k_rep * (1 / obstacle_distance - 1 / r_safe)**2 * \
                                              (robot_positions[i, :2] - obstacle_position) / obstacle_distance**3

            # Calcular potencial repulsivo y gradiente de otros robots
            for j in range(num_robots):
                if j != i:
                    robot_distance = np.linalg.norm(robot_positions[i, :2] - robot_positions[j, :2])
                    if robot_distance < r_min:
                        # Aplicar repulsión Lennard-Jones más débil entre robots cerca del objetivo
                        LJ_gradient += k_LJ * (12 * (r_min / robot_distance)**13 - 12 * (r_min / robot_distance)**7) * \
                                       (robot_positions[i, :2] - robot_positions[j, :2]) / robot_distance**2
        else:
            # Sin fuerzas repulsivas cuando están cerca del objetivo
            rep_gradient_obstacles = np.array([0.0, 0.0], dtype=float)
            LJ_gradient = np.array([0.0, 0.0], dtype=float)

        # Calcular gradiente total (combinar fuerzas atractivas y repulsivas)
        total_gradient = att_gradient + rep_gradient_obstacles + LJ_gradient

        # Aplicar un damping más fuerte a medida que se acercan al objetivo para reducir el exceso
        total_gradient *= damping_factor if distance_to_target < target_threshold else 1.0

        # Limitar la velocidad máxima para evitar que los robots se muevan demasiado rápido
        total_gradient_magnitude = np.linalg.norm(total_gradient)
        if total_gradient_magnitude > 0:
            total_gradient = total_gradient / total_gradient_magnitude * min(total_gradient_magnitude, max_speed)
        
        # Calcular velocidad lineal (v) y velocidad angular (omega)
        v = np.linalg.norm(total_gradient)  # Velocidad lineal proporcional a la magnitud del gradiente
        theta = np.arctan2(total_gradient[1], total_gradient[0])  # Dirección del gradiente
        omega = (theta - robot_positions[i, 2])  # Velocidad angular basada en la diferencia en orientación

        # Actualizar posición y orientación del robot
        robot_positions[i, 0] += v * np.cos(robot_positions[i, 2]) * dt  # Actualizar coordenada x
        robot_positions[i, 1] += v * np.sin(robot_positions[i, 2]) * dt  # Actualizar coordenada y
        robot_positions[i, 2] += omega * dt  # Actualizar orientación

   # Visualización (grafica de las posiciones de los robots, objetivo y obstáculos)
    plt.figure(1)
    plt.clf()
    plt.quiver(robot_positions[:, 0], robot_positions[:, 1], 
               np.cos(robot_positions[:, 2]), np.sin(robot_positions[:, 2]), 
               color='b', scale=20, label='Robots')  # Usar flechas para los robots
    plt.scatter(robot_positions[:, 0], robot_positions[:, 1], c='b', label='Robots', s=50)
    plt.scatter(target_position[0], target_position[1], c='g', label='Objetivo', s=50)
    plt.scatter(obstacles[:, 0], obstacles[:, 1], c='r', label='Obstáculos', s=50)
    plt.xlim([0, 20])
    plt.ylim([0, 20])
    plt.title(f'Paso de Tiempo: {t + 1}')
    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')
    plt.legend()
    plt.pause(0.01)

plt.show()

