from LS_Post_data_reader import Model
import numpy as np

folder = r"C:\Users\nir\Projects\FInal-Project\Analysis\single_element_mode_1_two_ways"
m = Model(folder, "simgle_element_mode_1.k")

eid = 14501
elem = m.get_element(eid)
print("Element type:", type(elem).__name__)
print("node_ids:", elem.node_ids)
print("initial_node_coords:")
for nid, c in elem.initial_node_coords.items():
    print(f"  {nid}: ({c['x']:.4f}, {c['y']:.4f}, {c['z']:.4f})")

# Faces
faces = elem.get_faces()
print("Faces:", faces)
bottom = faces[0]
top = faces[1]

# Face normal
normal = elem.get_face_normal_direction(bottom)
print(f"Bottom face normal: {normal}")

# Area
try:
    area = elem.calculate_face_area(bottom)
    print(f"Bottom face area: {area}")
except Exception as e:
    print(f"Area error: {e}")

print(f"elem.area (property): {elem.area}")

# Stress data
if elem.stress_data is not None:
    print("Stress columns:", list(elem.stress_data.columns))
    print("Stress shape:", elem.stress_data.shape)
    print("Stress first 5 rows:")
    print(elem.stress_data.head(5))
    print("Stress last 5 rows:")
    print(elem.stress_data.tail(5))
else:
    print("NO stress data")

# Node data
if elem.node_data is not None:
    print("Node data shape:", elem.node_data.shape)
    avail = elem.node_data.index.get_level_values("id").unique().tolist()
    print("Available node IDs:", avail)
else:
    print("NO node data")

# Cohesive separation
try:
    sep = elem.get_cohesive_separation()
    print("\nCohesive separation shape:", sep.shape)
    print("Separation first 5:")
    print(sep.head(5))
    print("Max magnitude:", sep["magnitude"].max())
except Exception as e:
    print(f"Separation error: {e}")

# Gc calculation
try:
    result, gc = elem.calculate_Gc_by_integration(use_cohesive_separation=True, mode="I")
    print(f"\nGc (Mode I): {gc}")
    print("Result columns:", list(result.columns))
    print("Result first 5:")
    print(result.head(5))
    print("Result last 5:")
    print(result.tail(5))
    print("Max separation:", result["separation"].max())
    print("Max traction:", result["traction"].max())
    print("Max G_cumulative:", result["G_cumulative"].max())
except Exception as e:
    print(f"Gc error: {e}")

# Internal energy
try:
    ie = elem.calculate_internal_energy(use_cohesive_separation=True)
    print(f"\nInternal energy (last value): {ie.iloc[-1]}")
    print(f"Internal energy max: {ie.max()}")
except Exception as e:
    print(f"IE error: {e}")

print(f"\nExpected GIC from material card: 0.256 J/m^2")
print(f"Element area should be ~0.25 mm^2 (0.5 x 0.5)")
