# OptiTrack


This web application allows users to upload a CSV file containing shipment data, processes it, and provides an optimized output in the form of another CSV file. It is built using Flask for the backend and includes a user-friendly frontend for file uploading and download.

## Features

- Upload CSV file with shipment data.
- Process the data to assign vehicles and optimize routes.
- Download the processed data as a CSV file.
- Visual feedback using a progress bar while the file is being processed.

## Technologies Used

- **Backend**: Flask, Python
- **Frontend**: HTML, CSS, JavaScript
- **Clustering Algorithms**: DBSCAN, KMeans (for optimizing vehicle assignments)
- **Geospatial Analysis**: Geopy for distance calculation
- **Graph Algorithms**: NetworkX for Minimum Spanning Tree calculation

## Setup

### Prerequisites

- Python 3.x
- Flask
- Pandas
- NumPy
- Scikit-learn
- Geopy
- NetworkX

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/csv-optimizer.git
