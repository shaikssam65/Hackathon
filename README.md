**A machine learning-powered solution for validating motorway sign associations against geospatial topology rules**


## Project Overview

This repository contains an innovative solution for validating motorway sign associations using satellite imagery and geospatial analysis. The system automatically detects three common data quality issues in road infrastructure databases:

1. **Ghost Signs** 🚫📌 - Signs that exist in databases but not in reality  
2. **Misattributed Signs** 🗺️🔀 - Signs associated with wrong road segments  
3. **Incorrect Road Attributes** 🛣️❌ - Properly located signs with wrong access characteristics  

## Key Features

- 🛰️ **Satellite Image Analysis** - Automated sign verification using CNN models
- 📍 **Precision Geospatial Matching** - Spatial joins with 20cm accuracy
- 🗄️ **Data Versioning** - Automatic tracking of topology corrections
- 📊 **Validation Dashboard** - Interactive Folium visualization of results

## Solution Architecture

### Pipeline

```mermaid
graph TD
  A[Sign Location] --> B{Tile Fetching}
  B --> C[Data Annotation]
  B --> D[CNN Detection]
  C --> E[Update Sign]
  C/D --> F[20m Radius Check]
  F --> G[Topology Updates]
```

<p align="center">
  <img src="data_annotation.jpg" alt="data_annotation" width="300" height="300"/>
  <img src="Corrected topology.png" alt="Corrected topology" width="300" height="300"/>
  <img src="detection.jpg" alt="motorsign_detection" width="300" height="300"/>
</p>


## Requirements

```text
geopandas==0.14.2
tensorflow==2.16.1
torch==2.3.0
torchvision==0.18.0
scikit-learn==1.4.2
pyproj==3.6.1
folium==0.15.1
requests==2.32.3
matplotlib==3.8.4
shapely==2.0.4
```





