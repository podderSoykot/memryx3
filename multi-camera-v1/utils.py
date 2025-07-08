from datetime import datetime
import pytz
import numpy as np
import sqlite3
import pickle
import cv2
# import backup  # Disabled for local-only operation


def save_new_face(db_path, embedding, uuid, model, detector, timestamp, apparent_age, dominant_gender, normalization):
    """

    :param db_path: (str) database path
    :param embedding: (vector) face embbeding
    :param uuid: (uuid) face uuid
    :param model: (str) model for identification
    :param detector: (str) detector model of faces
    :param timestamp: (isotime) timestamp of creation
    :param apparent_age: (int) apparent age of
    :param dominant_gender: (str) apparent gender
    :param normalization: (str) normalization of face embedding data
    :return: True when saved
    """
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Extract the embedding from the first detected face (assuming only one face is detected)
    actual_embedding = np.array(embedding[0]['embedding'])

    serialized_embedding = pickle.dumps(actual_embedding)

    # Define the SQL query to insert the face embedding
    insert_query = "INSERT INTO face_embeddings (embedding_data, identity, model, detector, timestamp, apparent_age," \
                   " dominant_gender, normalization) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"

    # Execute the SQL query
    cursor.execute(insert_query, (serialized_embedding, uuid, model, detector, timestamp, apparent_age,
                                  dominant_gender, normalization))

    # Commit the changes to the database
    conn.commit()
    # Close the cursor and database connection
    cursor.close()
    conn.close()

    return True


def save_record(db_path, uuid, model_name, detector_backend, timestamp):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Define the SQL query to insert the face embedding
    insert_query = "INSERT INTO records (identity, model, detector, timestamp) VALUES (?, ?, ?, ?)"
    # Execute the SQL query
    cursor.execute(insert_query, (uuid, model_name, detector_backend, timestamp))
    # Commit the changes to the database
    conn.commit()
    # Close the cursor and database connection
    cursor.close()
    conn.close()

    return True

def save_image(file_name, timestamp):
    # Connect to the SQLite database
    conn = sqlite3.connect('faces.db')  # Replace with your database file
    cursor = conn.cursor()

    # Insert the image data into the pictures table
    try:
        cursor.execute("INSERT INTO pictures (name, timestamp) VALUES (?, ?)",
                       (file_name, timestamp))
        conn.commit()
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return False
    finally:
        conn.close()

    return True


def create_epochtime():
    #create epoch time in ms
    epoch_time = int(datetime.now().timestamp())

    return epoch_time


def create_isotime():
    # Get the local timezone
    local_timezone = datetime.now(pytz.utc).astimezone().tzinfo

    # Get the current local time
    local_time = datetime.now(local_timezone)

    # Convert to ISO format
    iso_string = local_time.isoformat()

    print(iso_string)
    return iso_string


def distance_box_centers(box1, box2):
    x1_center = box1[0] + box1[2] / 2.0
    y1_center = box1[1] + box1[3] / 2.0
    x2_center = box2[0] + box2[2] / 2.0
    y2_center = box2[1] + box2[3] / 2.0
    return np.sqrt((x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2)


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = boxAArea + boxBArea - interArea

    return interArea / float(unionArea)


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Both bounding boxes should be in the format (x, y, width, height).
    """

    # Coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[0] + bb1[2], bb2[0] + bb2[2])
    y_bottom = min(bb1[1] + bb1[3], bb2[1] + bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = bb1[2] * bb1[3]
    bb2_area = bb2[2] * bb2[3]

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    print("iou: ", iou)

    return iou


def create_tracker(tracker_type):
    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    print("Tracker Created")
    return tracker