# OptiTrack

OptiTrack is a web application designed to help optimize shipment data by assigning vehicles and calculating efficient routes. Users can upload a CSV file with shipment details, process it, and download the optimized data in CSV format. The app uses Flask for the backend and provides a user-friendly frontend for file uploading and downloading. 

**Upcoming Feature**: Route display using the Folium library to visualize optimized routes on a map.

## Features

- Upload a CSV file containing shipment data.
- Process the shipment data to assign vehicles and optimize routes.
- Download the optimized shipment data as a CSV file.
- Visual feedback through a progress bar while the file is being processed.
- **(Upcoming Feature)** Route visualization using the **Folium** library to display the optimized routes on a map.

## Technologies Used

- **Backend**: Flask, Python
- **Frontend**: HTML, CSS, JavaScript
- **Clustering Algorithms**: DBSCAN, KMeans (for optimizing vehicle assignments)
- **Geospatial Analysis**: Geopy for distance calculations
- **Graph Algorithms**: NetworkX for Minimum Spanning Tree calculations
- **Mapping**: Folium (for route display, coming soon)

## Setup

### Prerequisites

To run this project locally, make sure you have the following installed:

- Python 3.x
- Flask
- Pandas
- NumPy
- Scikit-learn
- Geopy
- NetworkX
- Folium (for future route display feature)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/csv-optimizer.git
