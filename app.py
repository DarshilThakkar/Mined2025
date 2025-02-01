import os
import pandas as pd
import numpy as np
import networkx as nx
from flask import Flask, request, render_template, send_file, jsonify
from sklearn.cluster import DBSCAN, KMeans
from geopy.distance import geodesic

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 📌 Convert Excel to CSV
def convert_excel_to_csv(excel_path):
    xls = pd.ExcelFile(excel_path)
    csv_filename = f"{UPLOAD_FOLDER}/converted.csv"
    df = pd.read_excel(xls, sheet_name=0)  # Assuming first sheet
    df.to_csv(csv_filename, index=False)
    return csv_filename

# 📌 Read Parcel Data
# 📌 Read Parcel Data
def read_parcel_data(csv_file):
    df = pd.read_csv(csv_file)
    df.sort_values(by='Shipment ID', inplace=True)  # Sort by Shipment ID
    return df


# 📌 Compute Minimum Spanning Tree Distance
def compute_mst_distance(cluster_points, warehouse):
    G = nx.Graph()
    points = [warehouse] + cluster_points
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = geodesic(points[i], points[j]).km
            G.add_edge(i, j, weight=distance)
    mst = nx.minimum_spanning_tree(G)
    return sum(weight for _, _, weight in mst.edges(data='weight')) * 2

# 📌 Split Cluster Based on Distance
def split_cluster(cluster_data, max_distance, warehouse_coords):
    cluster_points = [x[1] for x in cluster_data]
    if compute_mst_distance(cluster_points, warehouse_coords) <= max_distance:
        return [cluster_data]
    k = max(2, len(cluster_data) // 2)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(cluster_points)
    sub_clusters = {}
    for i, label in enumerate(kmeans.labels_):
        sub_clusters.setdefault(label, []).append(cluster_data[i])
    return list(sub_clusters.values())

# 📌 Create Clusters
def create_clusters(df, warehouse_coords):
    vehicle_types = {
        "3W": {"eps": 1.5 / 111, "min": 3, "max": 5, "max_distance": 15},
        "4W-EV": {"eps": 2.0 / 111, "min": 5, "max": 8, "max_distance": 20},
        "4W": {"eps": 3.5 / 111, "min": 13, "max": 25, "max_distance": float("inf")}
    }
    cluster_assignments = []
    cluster_id_counter = 0
    for time_slot, group in df.groupby("Delivery Timeslot"):
        remaining_shipments = list(zip(group['Shipment ID'], group['Latitude'], group['Longitude']))
        for vehicle, params in vehicle_types.items():
            coordinates = [(lat, lon) for _, lat, lon in remaining_shipments]
            shipment_ids = [sid for sid, _, _ in remaining_shipments]
            if len(coordinates) < params['min']:
                continue
            dbscan = DBSCAN(eps=params['eps'], min_samples=params['min'], metric="euclidean")
            labels = dbscan.fit_predict(np.array(coordinates))
            clusters = {}
            for i, label in enumerate(labels):
                if label != -1:
                    clusters.setdefault(label, []).append((shipment_ids[i], coordinates[i]))
            final_clusters = {}
            for key, cluster_data in clusters.items():
                sub_clusters = split_cluster(cluster_data, params['max_distance'], warehouse_coords)
                for sub_cluster in sub_clusters:
                    if len(sub_cluster) >= params['min']:
                        final_clusters[cluster_id_counter] = (sub_cluster, vehicle)
                        cluster_id_counter += 1
            for cluster_id, (cluster_data, vehicle) in final_clusters.items():
                for shipment_id, coords in cluster_data:
                    cluster_assignments.append({
                        "TRIP ID": cluster_id, "Shipment ID": shipment_id, 
                        "Latitude": coords[0], "Longitude": coords[1], 
                        "TIME SLOT": time_slot, "Vehicle_Type": vehicle
                    })
                remaining_shipments = [(sid, lat, lon) for sid, lat, lon in remaining_shipments if sid not in [x[0] for x in cluster_data]]
    return pd.DataFrame(cluster_assignments)

def assign_vehicles(cluster_df, warehouse_coords):
    """Assigns vehicles based on optimized clustering data."""
    assigned_vehicles = []

    for time_slot, group in cluster_df.groupby("TIME SLOT"):
        vehicle_counters = {"3W": 1, "4W-EV": 1, "4W": 1}
        cluster_counts = group.groupby("TRIP ID").size().reset_index(name="Shipments")

        for _, row in cluster_counts.iterrows():
            cluster_id = row["TRIP ID"]
            shipment_count = row["Shipments"]
            vehicle_type = group[group["TRIP ID"] == cluster_id]["Vehicle_Type"].values[0]
            cluster_points = [(row["Latitude"], row["Longitude"]) for _, row in group[group["TRIP ID"] == cluster_id].iterrows()]
            total_distance = compute_mst_distance(cluster_points, warehouse_coords)
            
            trip_time = total_distance * 5 + shipment_count * 10  # Estimated trip time
            capacity_utilization = shipment_count / (5 if vehicle_type == "3W" else 8 if vehicle_type == "4W-EV" else 25)
            time_utilization = trip_time / 150
            distance_utilization = total_distance / (15 if vehicle_type == "3W" else 20 if vehicle_type == "4W-EV" else float("inf"))

            assigned_vehicles.append({
                "TRIP ID": cluster_id, "TIME SLOT": time_slot, 
                "Shipments": shipment_count, "MST_DIST": total_distance, 
                "TRIP_TIME": trip_time, "Vehicle_Type": vehicle_type, 
                "CAPACITY_UTI": capacity_utilization, "TIME_UTI": time_utilization, 
                "COV_UTI": distance_utilization
            })

    vehicle_df = pd.DataFrame(assigned_vehicles)

    # Instead of overwriting, merge directly with the original cluster_df
    merged_df = cluster_df.merge(vehicle_df, on=["TRIP ID", "TIME SLOT"], how="left").drop_duplicates()

    return merged_df

# 📌 Flask Routes
@app.route("/")
def upload_form():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    """ Handles file upload, processing, and returning the optimized shipment file. """
    file = request.files.get("file")
    
    if not file:
        return jsonify({'error': 'No file uploaded!'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file format. Please upload a CSV file.'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Process the CSV file
    df = read_parcel_data(file_path)
    warehouse_coords = [19.075887, 72.877911]
    cluster_df = create_clusters(df, warehouse_coords)
    final_df = assign_vehicles(cluster_df, warehouse_coords)  # Assign vehicles and get final_df

    output_file = os.path.join(UPLOAD_FOLDER, "optimized_shipments.csv")
    final_df.to_csv(output_file, index=False)  # Only save final_df

    return send_file(output_file, as_attachment=True, download_name="optimized_shipments.csv")


if __name__ == "__main__":
    app.run(debug=True)
