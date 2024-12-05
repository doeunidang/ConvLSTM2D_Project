#visualization.py
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib as mpl
import numpy as np
import os

def visualize_flood_area(grid_array, flooded_cells, output_path, sample_idx, final_H, normalized_inundation):
    """
    침수 영역 시각화 (고도별 색상 배경 포함).
    """
    # 저장 경로 생성
    os.makedirs(output_path, exist_ok=True)

    # 시각화 시작
    plt.figure(figsize=(10, 10))
    
    # 고도 배열 생성
    elevation_array = grid_array['elevation'].copy()
    elevation_array[elevation_array == 999] = np.nan  # 고도 999를 NaN으로 변환

    # 'terrain' 컬러맵 설정
    cmap = mpl.cm.get_cmap("terrain").copy()
    cmap.set_bad(color='black')  # NaN 값을 까만색으로 표시
    norm = plt.Normalize(vmin=-1, vmax=np.nanmax(elevation_array))  # NaN 제외한 최대값 계산

    # 고도를 배경으로 표시
    plt.imshow(elevation_array, cmap=cmap, norm=norm, origin='lower')
    plt.colorbar(label="Elevation (m)")

    # 침수된 셀 표시
    for (x, y), norm_inundation in normalized_inundation.items():
        if 0 <= norm_inundation <= 0.2:
            color = 'cornflowerblue'
        elif 0.2 < norm_inundation <= 0.3:
            color = 'royalblue'
        elif 0.3 < norm_inundation <= 0.5:
            color = 'mediumblue'
        elif norm_inundation > 0.5:
            color = 'darkblue'
        else:
            continue  # 비정상 값 무시
        plt.plot(x, y, 's', markersize=5, color=color)

    # 침수심 범례 추가
    legend_elements = [
        Patch(color='cornflowerblue', label='0 ~ 0.2m'),
        Patch(color='royalblue', label='0.2 ~ 0.3m'),
        Patch(color='mediumblue', label='0.3 ~ 0.5m'),
        Patch(color='darkblue', label='> 0.5m')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # y축 반전
    plt.gca().invert_yaxis()

    # 그래프 제목
    plt.title(f"Flooded Areas - Sample {sample_idx}\nFlood Depth (H): {final_H:.2f}m")

    # 그래프 저장
    output_file = os.path.join(output_path, f"flooded_area_sample_{sample_idx}.png")
    plt.savefig(output_file, dpi=300)
    plt.close()

    # 저장 완료 메시지
    print(f"Saved visualization at {output_file}")
