import streamlit as st
import math
import heapq
import time
from typing import Dict, List, Tuple, Optional
from collections import deque
import folium
from streamlit_folium import folium_static
import pandas as pd

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Rutas Óptimas - Cuenca",
    page_icon="🗺️",
    layout="wide"
)

# Datos de los nodos de Cuenca (16 puntos de interés)
CUENCA_NODES: Dict[str, Dict[str, float]] = {
    "Catedral Nueva": {"lat": -2.8975, "lon": -79.005, "descripcion": "Centro histórico de Cuenca"},
    "Parque Calderón": {"lat": -2.89741, "lon": -79.00438, "descripcion": "Corazón de Cuenca"},
    "Puente Roto": {"lat": -2.90423, "lon": -79.00142, "descripcion": "Monumento histórico"},
    "Museo Pumapungo": {"lat": -2.90607, "lon": -78.99681, "descripcion": "Museo de antropología"},
    "Terminal Terrestre": {"lat": -2.89222, "lon": -78.99277, "descripcion": "Terminal de autobuses"},
    "Mirador de Turi": {"lat": -2.92583, "lon": -79.0040, "descripcion": "Mirador con vista panorámica"},
    # 10 PUNTOS ADICIONALES
    "Universidad de Cuenca": {"lat": -2.90138, "lon": -79.00583, "descripcion": "Universidad principal de la ciudad"},
    "Plaza de las Flores": {"lat": -2.89694, "lon": -79.00472, "descripcion": "Mercado de flores tradicional"},
    "Barranco": {"lat": -2.90111, "lon": -79.00083, "descripcion": "Paseo junto al río Tomebamba"},
    "Mercado 10 de Agosto": {"lat": -2.89389, "lon": -79.00611, "descripcion": "Mercado tradicional de Cuenca"},
    "Parque de la Madre": {"lat": -2.88833, "lon": -79.00250, "descripcion": "Parque recreativo familiar"},
    "Todos Santos": {"lat": -2.90972, "lon": -79.00583, "descripcion": "Iglesia histórica y barrio tradicional"},
    "Mall del Río": {"lat": -2.91083, "lon": -78.99472, "descripcion": "Centro comercial moderno"},
    "Estadio Alejandro Serrano": {"lat": -2.88944, "lon": -78.99889, "descripcion": "Estadio de fútbol principal"},
    "Parque El Paraíso": {"lat": -2.92222, "lon": -78.99722, "descripcion": "Parque recreativo grande"},
    "Aeropuerto Mariscal Lamar": {"lat": -2.88944, "lon": -78.98472, "descripcion": "Aeropuerto de Cuenca"},
}

# Conexiones del grafo (ampliado con los nuevos nodos)
GRAPH_EDGES = {
    "Catedral Nueva": ["Parque Calderón", "Puente Roto", "Museo Pumapungo", "Plaza de las Flores", "Universidad de Cuenca"],
    "Parque Calderón": ["Catedral Nueva", "Terminal Terrestre", "Puente Roto", "Plaza de las Flores", "Mercado 10 de Agosto"],
    "Puente Roto": ["Catedral Nueva", "Parque Calderón", "Museo Pumapungo", "Mirador de Turi", "Barranco", "Universidad de Cuenca"],
    "Museo Pumapungo": ["Catedral Nueva", "Puente Roto", "Terminal Terrestre", "Barranco", "Mall del Río"],
    "Terminal Terrestre": ["Parque Calderón", "Museo Pumapungo", "Mirador de Turi", "Estadio Alejandro Serrano", "Aeropuerto Mariscal Lamar"],
    "Mirador de Turi": ["Puente Roto", "Terminal Terrestre", "Todos Santos", "Parque El Paraíso"],
    "Universidad de Cuenca": ["Catedral Nueva", "Puente Roto", "Barranco", "Todos Santos"],
    "Plaza de las Flores": ["Catedral Nueva", "Parque Calderón", "Mercado 10 de Agosto"],
    "Barranco": ["Puente Roto", "Museo Pumapungo", "Universidad de Cuenca", "Mall del Río"],
    "Mercado 10 de Agosto": ["Parque Calderón", "Plaza de las Flores", "Parque de la Madre"],
    "Parque de la Madre": ["Mercado 10 de Agosto", "Estadio Alejandro Serrano", "Aeropuerto Mariscal Lamar"],
    "Todos Santos": ["Universidad de Cuenca", "Mirador de Turi", "Mall del Río"],
    "Mall del Río": ["Museo Pumapungo", "Barranco", "Todos Santos", "Parque El Paraíso"],
    "Estadio Alejandro Serrano": ["Terminal Terrestre", "Parque de la Madre", "Aeropuerto Mariscal Lamar"],
    "Parque El Paraíso": ["Mirador de Turi", "Mall del Río", "Aeropuerto Mariscal Lamar"],
    "Aeropuerto Mariscal Lamar": ["Terminal Terrestre", "Parque de la Madre", "Estadio Alejandro Serrano", "Parque El Paraíso"],
}

# Funciones de distancia
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia Haversine en km."""
    R = 6371.0
    lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def euclidean_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia euclidiana aproximada en km (111 km ≈ 1°)."""
    return math.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) * 111.0

# Clase para el algoritmo A*
class AStarPathFinder:
    def __init__(self, nodes: Dict, edges: Dict):
        self.nodes = nodes
        self.edges = edges
        self.explored: List[str] = []
        self.frontier: List[Tuple[float, int, str, List[str], float]] = []

    def heuristic(self, node: str, goal: str) -> float:
        n, g = self.nodes[node], self.nodes[goal]
        return euclidean_distance(n["lat"], n["lon"], g["lat"], g["lon"])

    def get_distance(self, node1: str, node2: str) -> float:
        n1, n2 = self.nodes[node1], self.nodes[node2]
        return haversine_distance(n1["lat"], n1["lon"], n2["lat"], n2["lon"])

    def find_path(self, start: str, goal: str) -> Tuple[Optional[List[str]], float, int, float]:
        start_time = time.time()
        self.explored = []
        self.frontier = []
        counter = 0
        heapq.heappush(self.frontier, (0.0, counter, start, [start], 0.0))
        visited = set()

        while self.frontier:
            f_score, _, current, path, g_score = heapq.heappop(self.frontier)
            
            if current in visited:
                continue
            
            visited.add(current)
            self.explored.append(current)
            
            if current == goal:
                execution_time = time.time() - start_time
                return path, g_score, len(self.explored), execution_time
            
            for neighbor in self.edges.get(current, []):
                if neighbor in visited:
                    continue
                
                edge_cost = self.get_distance(current, neighbor)
                new_g = g_score + edge_cost
                h = self.heuristic(neighbor, goal)
                counter += 1
                heapq.heappush(self.frontier, (new_g + h, counter, neighbor, path + [neighbor], new_g))
        
        execution_time = time.time() - start_time
        return None, float("inf"), len(self.explored), execution_time

# Algoritmo BFS (Búsqueda en Amplitud)
class BFSPathFinder:
    def __init__(self, nodes: Dict, edges: Dict):
        self.nodes = nodes
        self.edges = edges
        self.explored: List[str] = []

    def get_distance(self, node1: str, node2: str) -> float:
        n1, n2 = self.nodes[node1], self.nodes[node2]
        return haversine_distance(n1["lat"], n1["lon"], n2["lat"], n2["lon"])

    def find_path(self, start: str, goal: str) -> Tuple[Optional[List[str]], float, int, float]:
        start_time = time.time()
        self.explored = []
        queue = deque([[start]])
        visited = {start}

        while queue:
            path = queue.popleft()
            current = path[-1]
            self.explored.append(current)
            
            if current == goal:
                # Calcular distancia total
                total_distance = 0
                for i in range(len(path) - 1):
                    total_distance += self.get_distance(path[i], path[i + 1])
                execution_time = time.time() - start_time
                return path, total_distance, len(self.explored), execution_time
            
            for neighbor in self.edges.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        
        execution_time = time.time() - start_time
        return None, float("inf"), len(self.explored), execution_time

# Algoritmo DFS (Búsqueda en Profundidad)
class DFSPathFinder:
    def __init__(self, nodes: Dict, edges: Dict):
        self.nodes = nodes
        self.edges = edges
        self.explored: List[str] = []

    def get_distance(self, node1: str, node2: str) -> float:
        n1, n2 = self.nodes[node1], self.nodes[node2]
        return haversine_distance(n1["lat"], n1["lon"], n2["lat"], n2["lon"])

    def find_path(self, start: str, goal: str) -> Tuple[Optional[List[str]], float, int, float]:
        start_time = time.time()
        self.explored = []
        stack = [[start]]
        visited = {start}

        while stack:
            path = stack.pop()
            current = path[-1]
            self.explored.append(current)
            
            if current == goal:
                # Calcular distancia total
                total_distance = 0
                for i in range(len(path) - 1):
                    total_distance += self.get_distance(path[i], path[i + 1])
                execution_time = time.time() - start_time
                return path, total_distance, len(self.explored), execution_time
            
            for neighbor in reversed(self.edges.get(current, [])):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(path + [neighbor])
        
        execution_time = time.time() - start_time
        return None, float("inf"), len(self.explored), execution_time

# Función para crear el mapa
def create_map(start_node: str, goal_node: str, path: Optional[List[str]] = None):
    # Calcular el centro del mapa
    start_coords = CUENCA_NODES[start_node]
    goal_coords = CUENCA_NODES[goal_node]
    center_lat = (start_coords["lat"] + goal_coords["lat"]) / 2
    center_lon = (start_coords["lon"] + goal_coords["lon"]) / 2
    
    # Crear el mapa
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    
    # Dibujar todas las aristas (en gris)
    for node, neighbors in GRAPH_EDGES.items():
        node_coords = CUENCA_NODES[node]
        for neighbor in neighbors:
            neighbor_coords = CUENCA_NODES[neighbor]
            folium.PolyLine(
                locations=[
                    [node_coords["lat"], node_coords["lon"]],
                    [neighbor_coords["lat"], neighbor_coords["lon"]]
                ],
                color="gray",
                weight=2,
                opacity=0.4
            ).add_to(m)
    
    # Dibujar la ruta óptima si existe
    if path:
        for i in range(len(path) - 1):
            node1 = CUENCA_NODES[path[i]]
            node2 = CUENCA_NODES[path[i + 1]]
            folium.PolyLine(
                locations=[
                    [node1["lat"], node1["lon"]],
                    [node2["lat"], node2["lon"]]
                ],
                color="blue",
                weight=5,
                opacity=0.8
            ).add_to(m)
    
    # Agregar marcadores para todos los nodos
    for name, coords in CUENCA_NODES.items():
        if name == start_node:
            color = "green"
            icon = "play"
        elif name == goal_node:
            color = "red"
            icon = "stop"
        elif path and name in path:
            color = "blue"
            icon = "info-sign"
        else:
            color = "gray"
            icon = "map-marker"
        
        folium.Marker(
            location=[coords["lat"], coords["lon"]],
            popup=f"<b>{name}</b><br>{coords['descripcion']}",
            tooltip=name,
            icon=folium.Icon(color=color, icon=icon)
        ).add_to(m)
    
    return m

# Interfaz de Streamlit
def main():
    st.title("🗺️ Sistema de Rutas Óptimas - Cuenca")
    st.markdown("**Comparación de Algoritmos de Búsqueda**: A*, BFS y DFS")
    
    st.markdown("---")
    
    # Sidebar para controles
    with st.sidebar:
        st.header("🎯 Selección de Ruta")
        
        node_names = list(CUENCA_NODES.keys())
        
        start_node = st.selectbox(
            "📍 Punto de Inicio",
            node_names,
            index=0
        )
        
        goal_node = st.selectbox(
            "🏁 Punto de Destino",
            node_names,
            index=5
        )
        
        st.markdown("---")
        st.header("🔬 Algoritmo a Usar")
        algorithm = st.radio(
            "Selecciona el algoritmo:",
            ["A* (Búsqueda Informada)", "BFS (Amplitud)", "DFS (Profundidad)", "Comparar Todos"],
            index=0
        )
        
        search_button = st.button("🔍 Buscar Ruta", type="primary")
        
        st.markdown("---")
        st.markdown("### ℹ️ Información")
        st.info(f"**Total de nodos:** {len(CUENCA_NODES)}\n\n**Algoritmos disponibles:**\n- A*: Búsqueda informada con heurística\n- BFS: Búsqueda en amplitud\n- DFS: Búsqueda en profundidad")
    
    # Validación
    if start_node == goal_node:
        st.error("⚠️ El punto de inicio y destino deben ser diferentes")
        return
    
    # Ejecutar búsqueda
    if search_button or True:
        
        if algorithm == "Comparar Todos":
            st.header("📊 Comparación de Algoritmos")
            
            # Ejecutar los tres algoritmos
            astar = AStarPathFinder(CUENCA_NODES, GRAPH_EDGES)
            bfs = BFSPathFinder(CUENCA_NODES, GRAPH_EDGES)
            dfs = DFSPathFinder(CUENCA_NODES, GRAPH_EDGES)
            
            with st.spinner("Ejecutando los tres algoritmos..."):
                astar_path, astar_dist, astar_nodes, astar_time = astar.find_path(start_node, goal_node)
                bfs_path, bfs_dist, bfs_nodes, bfs_time = bfs.find_path(start_node, goal_node)
                dfs_path, dfs_dist, dfs_nodes, dfs_time = dfs.find_path(start_node, goal_node)
            
            # Crear tabla comparativa
            comparison_data = {
                "Algoritmo": ["A* (Informada)", "BFS (Amplitud)", "DFS (Profundidad)"],
                "Distancia (km)": [
                    f"{astar_dist:.3f}" if astar_path else "N/A",
                    f"{bfs_dist:.3f}" if bfs_path else "N/A",
                    f"{dfs_dist:.3f}" if dfs_path else "N/A"
                ],
                "Nodos Explorados": [astar_nodes, bfs_nodes, dfs_nodes],
                "Tiempo (ms)": [
                    f"{astar_time*1000:.2f}",
                    f"{bfs_time*1000:.2f}",
                    f"{dfs_time*1000:.2f}"
                ],
                "Paradas": [
                    len(astar_path) if astar_path else 0,
                    len(bfs_path) if bfs_path else 0,
                    len(dfs_path) if dfs_path else 0
                ]
            }
            
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Análisis
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🏆 Ruta Más Corta", "A*" if astar_dist <= min(bfs_dist, dfs_dist) else ("BFS" if bfs_dist <= dfs_dist else "DFS"))
            
            with col2:
                st.metric("⚡ Más Rápido", 
                         "A*" if astar_time <= min(bfs_time, dfs_time) else ("BFS" if bfs_time <= dfs_time else "DFS"))
            
            with col3:
                st.metric("🎯 Menos Nodos Explorados", 
                         "A*" if astar_nodes <= min(bfs_nodes, dfs_nodes) else ("BFS" if bfs_nodes <= dfs_nodes else "DFS"))
            
            # Mostrar detalles de A* (el mejor)
            st.markdown("---")
            st.header("🗺️ Visualización - Ruta A* (Óptima)")
            
            col_left, col_right = st.columns([1, 2])
            
            with col_left:
                if astar_path:
                    st.markdown("### 🛣️ Ruta Detallada (A*)")
                    for i, node in enumerate(astar_path, 1):
                        if i == 1:
                            emoji = "🟢"
                        elif i == len(astar_path):
                            emoji = "🔴"
                        else:
                            emoji = "🔵"
                        st.markdown(f"{emoji} **{i}.** {node}")
                        if i < len(astar_path):
                            next_node = astar_path[i]
                            segment_distance = astar.get_distance(node, next_node)
                            st.caption(f"   ↓ {segment_distance:.2f} km")
            
            with col_right:
                m = create_map(start_node, goal_node, astar_path)
                folium_static(m, width=700, height=500)
        
        else:
            # Ejecutar algoritmo individual
            if "A*" in algorithm:
                finder = AStarPathFinder(CUENCA_NODES, GRAPH_EDGES)
                algo_name = "A*"
            elif "BFS" in algorithm:
                finder = BFSPathFinder(CUENCA_NODES, GRAPH_EDGES)
                algo_name = "BFS"
            else:
                finder = DFSPathFinder(CUENCA_NODES, GRAPH_EDGES)
                algo_name = "DFS"
            
            with st.spinner(f"Buscando con {algo_name}..."):
                path, distance, nodes_explored, execution_time = finder.find_path(start_node, goal_node)
            
            # Crear columnas para resultados y mapa
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.header(f"📊 Resultados - {algo_name}")
                
                if path:
                    st.success("✅ Ruta encontrada exitosamente")
                    
                    # Métricas
                    st.metric("📏 Distancia Total", f"{distance:.2f} km")
                    st.metric("🔢 Nodos Explorados", nodes_explored)
                    st.metric("⏱️ Tiempo de Ejecución", f"{execution_time*1000:.2f} ms")
                    st.metric("📍 Paradas en la Ruta", len(path))
                    
                    # Mostrar la ruta
                    st.markdown("### 🛣️ Ruta Detallada")
                    for i, node in enumerate(path, 1):
                        if i == 1:
                            emoji = "🟢"
                        elif i == len(path):
                            emoji = "🔴"
                        else:
                            emoji = "🔵"
                        st.markdown(f"{emoji} **{i}.** {node}")
                        if i < len(path):
                            next_node = path[i]
                            segment_distance = finder.get_distance(node, next_node)
                            st.caption(f"   ↓ {segment_distance:.2f} km")
                else:
                    st.error("❌ No se encontró una ruta entre los puntos seleccionados")
            
            with col2:
                st.header("🗺️ Visualización del Mapa")
                m = create_map(start_node, goal_node, path)
                folium_static(m, width=700, height=600)
    
    # Sección de información de puntos de interés
    st.markdown("---")
    st.header("📍 Puntos de Interés en Cuenca (16 Lugares)")
    
    cols = st.columns(4)
    for idx, (name, data) in enumerate(CUENCA_NODES.items()):
        with cols[idx % 4]:
            with st.container():
                st.markdown(f"**{name}**")
                st.caption(data["descripcion"])
                st.caption(f"📍 {data['lat']:.5f}, {data['lon']:.5f}")
                st.markdown("---")

if __name__ == "__main__":
    main()