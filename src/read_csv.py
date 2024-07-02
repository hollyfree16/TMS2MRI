#!/usr/bin/python3


import csv


def process_csv(csv_filename):
    results = []
    with open(csv_filename, mode='r', newline='') as file:
        reader = csv.reader(file)

        next(reader)

        for row in reader:
            try:
                subject_id = row[0]
                x = float(row[1])
                y = float(row[2])
                z = float(row[3])
                label = row[4]
                sourcedata = row[5]
                results.append((subject_id, x, y, z, label, sourcedata))

                # Print or process the values as needed
                #print(f"Subject ID: {subject_id}, Coordinates: ({x}, {y}, {z}), Label: {label}")

            except ValueError as e:
                print(f"Error processing row: {row}. {e}")

    return results
