import math
import requests
import os
from shutil import copyfile

# ------------------- Helper Functions -------------------
def lat_lon_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    n = 2.0 ** zoom
    x = int((lon_rad + math.pi) / (2 * math.pi) * n)
    y = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n)
    return (x, y)

def tile_coords_to_lat_lon(x, y, zoom):
    n = 2.0 ** zoom
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def lat_lon_to_pixel_in_tile(target_lat, target_lon, x, y, zoom, tile_size):
    nw_lat, nw_lon = tile_coords_to_lat_lon(x, y, zoom)
    se_lat, se_lon = tile_coords_to_lat_lon(x + 1, y + 1, zoom)
    west_lon, east_lon = nw_lon, se_lon
    north_lat, south_lat = nw_lat, se_lat

    def lat_to_mercator(lat):
        lat_rad = math.radians(lat)
        return math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad))

    pixel_x = ((target_lon - west_lon) / (east_lon - west_lon)) * tile_size
    top_mercator = lat_to_mercator(north_lat)
    bottom_mercator = lat_to_mercator(south_lat)
    target_mercator = lat_to_mercator(target_lat)
    pixel_y = ((target_mercator - top_mercator) / (bottom_mercator - top_mercator)) * tile_size

    return int(pixel_x), int(pixel_y)

# ------------------- Directional Box Generation -------------------
def generate_directional_boxes(base_x, base_y, box_size=64, shift=32, tile_size=512):
    directions = {
        'center': (0, 0),
        'left': (-shift, 0),
        'right': (shift, 0),
        'up': (0, -shift),
        'down': (0, shift)
    }

    half_size = box_size // 2
    annotations = []

    for name, (dx, dy) in directions.items():
        new_x = max(half_size, min(tile_size - half_size, base_x + dx))
        new_y = max(half_size, min(tile_size - half_size, base_y + dy))

        x_center = new_x / tile_size
        y_center = new_y / tile_size
        width = height = box_size / tile_size

        annotations.append((name, x_center, y_center, width, height))

    return annotations

# ------------------- Satellite Tile Retrieval -------------------
def get_satellite_tile(lat, lon, zoom, tile_format, api_key, tile_size=512):
    x, y = lat_lon_to_tile(lat, lon, zoom)
    url = f'https://maps.hereapi.com/v3/base/mc/{zoom}/{x}/{y}/{tile_format}?style=satellite.day&size={tile_size}&apiKey={api_key}'
    response = requests.get(url)
    return response.content if response.status_code == 200 else None

# ------------------- Execution -------------------
api_key = '<API-KEY>'  # Replace with your HERE API key
zoom_level = 18
tile_size = 512
tile_format = 'png'
class_id = 0
box_size = 64
shift = 32

# Create output directories
os.makedirs('images', exist_ok=True)
os.makedirs('annotations', exist_ok=True)

locations = [
    #  (49.14067, 8.16832),
    # (49.13895, 8.16271),
    (49.1567, 8.15384),
    (49.25072, 6.82029),
    (49.34079, 6.74129),
    (49.35256, 6.77284),
    (49.24494, 6.86146),
    (49.23333, 6.96245),
    ( 49.22119, 7.0162)
]

for latitude, longitude in locations:
    # Get tile coordinates and pixel position
    x_tile, y_tile = lat_lon_to_tile(latitude, longitude, zoom_level)
    pixel_x, pixel_y = lat_lon_to_pixel_in_tile(latitude, longitude,
                                               x_tile, y_tile, zoom_level, tile_size)

    # Retrieve satellite tile once
    image_data = get_satellite_tile(latitude, longitude, zoom_level, tile_format, api_key)
    
    if image_data:
        # Generate directional annotations
        annotations = generate_directional_boxes(pixel_x, pixel_y, box_size, shift)
        
        # Base filename without extension
        base_filename = f'satellite_tile_{latitude}_{longitude}'

        # Create individual files for each direction
        for direction, x_center, y_center, width, height in annotations:
            # Create unique image copy
            image_filename = f'{base_filename}_{direction}.{tile_format}'
            with open(f'images/{image_filename}', 'wb') as img_file:
                img_file.write(image_data)

            # Create corresponding annotation file
            annotation_filename = f'{base_filename}_{direction}.txt'
            with open(f'annotations/{annotation_filename}', 'w') as ann_file:
                ann_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        print(f"Processed {latitude}, {longitude} with {len(annotations)} directions")
    else:
        print(f"Failed to retrieve tile for {latitude}, {longitude}")