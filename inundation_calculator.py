#inundation_calculator.py
import pandas as pd
import geopandas as gpd
import numpy as np
from collections import deque
from utils.visualization import visualize_flood_area
from utils.file_utils import ensure_directory_exists
from config import OUTPUT_FOLDER


def load_shapefile_and_initialize_grid(shapefile_path, y_pred_sample):
    """
    Shapefile을 로드하고 Junction 데이터와 예측값을 통합합니다.
    """
    gdf = gpd.read_file(shapefile_path)
    grid_data = gdf[['row_index', 'col_index', 'Elevation', 'Junction']].copy()
    
    grid_data['Junction'] = grid_data['Junction'].apply(
        lambda x: f"J{int(x)}" if pd.notnull(x) and not str(x).startswith('J') else x
    )
    
    # 모든 시간 단계 초기화 (flooding_value_1 ~ flooding_value_4)
    for t in range(4):
        grid_data.loc[:, f'flooding_value_{t+1}'] = np.nan

    max_row, max_col = y_pred_sample.shape[1:3]

    for idx, row in grid_data.iterrows():
        row_index = int(row['row_index'])
        col_index = int(row['col_index'])

        if row_index >= max_row or col_index >= max_col:
            print(f"경고: row_index={row_index}, col_index={col_index}가 y_pred_sample의 범위를 초과했습니다.")
            continue

        for t in range(4):
            grid_data.at[idx, f'flooding_value_{t+1}'] = y_pred_sample[t, row_index, col_index, 0]
    
    return grid_data


def find_inundation_low_points(x, y, grid_array):
    """침수 최저점 찾기."""
    queue = deque([(x, y)])
    visited = set()
    visited.add((x, y))
    lowest_points = [(x, y)]
    lowest_elevation = grid_array[y, x]['elevation']
    while queue:
        current_x, current_y = queue.popleft()
        current_elevation = grid_array[current_y, current_x]['elevation']
        neighbors = [(current_x + dx, current_y + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if (dx, dy) != (0, 0)]
        for nx, ny in neighbors:
            if 0 <= nx < 64 and 0 <= ny < 64 and (nx, ny) not in visited:
                visited.add((nx, ny))
                neighbor_elevation = grid_array[ny, nx]['elevation']
                if neighbor_elevation < lowest_elevation:
                    lowest_points = [(nx, ny)]
                    lowest_elevation = neighbor_elevation
                elif neighbor_elevation == lowest_elevation:
                    lowest_points.append((nx, ny))
                if neighbor_elevation <= current_elevation:
                    queue.append((nx, ny))
    return lowest_points, lowest_elevation


def find_connected_same_elevation_cells(x, y, elevation, grid_array):
    """같은 고도의 셀 연결."""
    queue = deque([(x, y)])
    visited = set()
    visited.add((x, y))
    connected_cells = [(x, y)]
    while queue:
        current_x, current_y = queue.popleft()
        neighbors = [(current_x + dx, current_y + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if (dx, dy) != (0, 0)]
        for nx, ny in neighbors:
            if 0 <= nx < 64 and 0 <= ny < 64 and (nx, ny) not in visited:
                if grid_array[ny, nx]['elevation'] == elevation:
                    queue.append((nx, ny))
                    visited.add((nx, ny))
                    connected_cells.append((nx, ny))
    return connected_cells


def compute_total_flooding(H, elevation_groups, cell_area):
    """총 침수량 계산."""
    total_flooding_computed = 0
    for elevation, cells in elevation_groups.items():
        flooded_cells_count = len(cells)
        total_flooding_computed += (H - elevation) * cell_area * flooded_cells_count
    return total_flooding_computed


def find_optimal_H(total_flooding, elevation_groups, cell_area, H_min, H_max, tolerance=1e-5):
    """최적 H 값 찾기."""
    while H_max - H_min > tolerance:
        H_mid = (H_min + H_max) / 2
        total_flooding_computed = compute_total_flooding(H_mid, elevation_groups, cell_area)

        if total_flooding_computed < total_flooding:
            H_min = H_mid
        else:
            H_max = H_mid

    return (H_min + H_max) / 2


def initialize_grid_array(grid_data):
    """
    grid_data를 기반으로 grid_array 생성.
    """
    grid_array = np.zeros((64, 64), dtype=[('elevation', 'f8'), ('flooding_value', 'f8')])
    for _, row in grid_data.iterrows():
        x, y = int(row['col_index']), int(row['row_index'])
        flooding_value_t_plus_10 = row['flooding_value_4']
        grid_array[y, x] = (row['Elevation'], flooding_value_t_plus_10)
    return grid_array


def calculate_inundation(y_pred_sample, shapefile_path, sample_idx=0):
    """
    침수 계산 및 시각화 수행.
    """
    # Step 1: Grid 데이터 초기화
    grid_data = load_shapefile_and_initialize_grid(shapefile_path, y_pred_sample)
    grid_array = initialize_grid_array(grid_data)

    # Step 2: Junction 기반 침수 셀 초기화
    flooded_cells = set()
    for _, row in grid_data.iterrows():
        junction_name = row['Junction']
        if pd.notnull(junction_name):  # Junction이 존재하는 경우만 처리
            row_index = int(row['row_index'])
            col_index = int(row['col_index'])

            # 침수 최저점 찾기
            low_points, elevation = find_inundation_low_points(col_index, row_index, grid_array)
            
            # 동일 고도 연결 셀 탐색
            for low_x, low_y in low_points:
                flooded_cells.update(find_connected_same_elevation_cells(low_x, low_y, elevation, grid_array))

    # Step 3: 초기 H 계산
    lowest_elevation = min(grid_array[ly, lx]['elevation'] for lx, ly in flooded_cells if grid_array[ly, lx]['elevation'] != 999)
    cell_area = 244.1406  # 각 셀의 면적
    total_flooding = np.sum(grid_array['flooding_value']) * 600  # t+10의 총유출량 계산

    def calculate_initial_H(flooded_cells, lowest_elevation, total_flooding, cell_area):
        """
        초기 H 계산.
        """
        flooded_cells_count = len(flooded_cells)
        if flooded_cells_count == 0:
            return 0  # flooded_cells가 없으면 H는 0
        H = (total_flooding / (cell_area * flooded_cells_count)) + lowest_elevation
        return H

    # 초기 H 계산
    initial_H = calculate_initial_H(flooded_cells, lowest_elevation, total_flooding, cell_area)

    # Step 4: 침수 범위 확장 및 H 업데이트
    while True:
        new_flooded_cells = set(flooded_cells)
        for x, y in flooded_cells:
            neighbors = [(x + dx, y + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if (dx, dy) != (0, 0)]
            for nx, ny in neighbors:
                if 0 <= nx < 64 and 0 <= ny < 64 and (nx, ny) not in flooded_cells:
                    adjacent_elevation = grid_array[ny, nx]['elevation']
                    if adjacent_elevation <= initial_H:
                        new_flooded_cells.update(find_connected_same_elevation_cells(nx, ny, adjacent_elevation, grid_array))

        if new_flooded_cells == flooded_cells:
            break

        flooded_cells = new_flooded_cells

        # H 값 갱신 및 침수된 고도 그룹 업데이트
        elevation_groups = {}
        for x, y in flooded_cells:
            elevation = grid_array[y, x]['elevation']
            if elevation <= initial_H:  # H보다 높은 고도는 제외
                if elevation not in elevation_groups:
                    elevation_groups[elevation] = []
                elevation_groups[elevation].append((x, y))

        initial_H = find_optimal_H(total_flooding, elevation_groups, cell_area, lowest_elevation, initial_H)

    # Step 5: 모든 침수 깊이 계산
    all_inundation_depths = []
    cell_depths = {}
    for elevation, cells in elevation_groups.items():
        for cell in cells:
            inundation_depth = initial_H - elevation
            all_inundation_depths.append(inundation_depth)
            cell_depths[cell] = inundation_depth

    min_depth = min(all_inundation_depths)
    max_depth = max(all_inundation_depths)

    # Step 6: 정규화 및 그룹별 출력
    print("\n고도 그룹별 침수 깊이 (정규화 전 및 후):")
    normalized_inundation = {}
    for elevation, cells in elevation_groups.items():
        # 침수된 셀이 포함된 고도 그룹만 처리
        if not cells:  # 셀이 없는 경우 무시
            continue

        # 침수 깊이 계산 (모든 셀이 동일한 깊이를 가짐)
        inundation_depth = initial_H - elevation

        # 정규화 수행
        if 0 <= inundation_depth <= 0.2:
            normalized_depth = inundation_depth  # 정규화 생략
        else:
            normalized_depth = (inundation_depth - min_depth) / (max_depth - min_depth) if max_depth > min_depth else 0

        # 출력
        print(f"Elevation: {elevation:.2f}m")
        print(f"  Depth: {inundation_depth:.2f}m")
        print(f"  Normalized Depth: {normalized_depth:.2f}")

        # 셀별 정규화 값 저장
        for cell in cells:
            normalized_inundation[cell] = normalized_depth

    # Step 7: 시각화 호출
    ensure_directory_exists(OUTPUT_FOLDER)
    visualize_flood_area(grid_array, flooded_cells, OUTPUT_FOLDER, sample_idx, initial_H, normalized_inundation)

    print(f"최종 H 값: {initial_H}")

