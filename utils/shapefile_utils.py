# shapefile_utils.py
import geopandas as gpd

def load_shapefile(shapefile_path):
    """
    주어진 shapefile 경로에서 데이터를 로드합니다.

    Args:
        shapefile_path (str): shapefile 경로.

    Returns:
        geopandas.GeoDataFrame: 로드된 shapefile 데이터.
    """
    try:
        gdf = gpd.read_file(shapefile_path)
        return gdf
    except Exception as e:
        print(f"Shapefile 로드 중 오류 발생: {e}")
        return None
