"""
Parser for LS-DYNA keyword files (.k files)
Extracts element-to-node mapping and node coordinates.
"""

def parse_lsdyna_keyfile(filepath):
    """
    Parse an LS-DYNA keyword file and extract:
    - Element to node mapping from *ELEMENT_SOLID
    - Node coordinates from *NODE
    
    Args:
        filepath: Path to the .k file
        
    Returns:
        elements: dict mapping element_id -> {'pid': part_id, 'nodes': [n1, n2, ..., n8]}
        nodes: dict mapping node_id -> {'x': x, 'y': y, 'z': z}
    """
    elements = {}
    nodes = {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    current_keyword = None
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('$'):
            continue
        
        # Check for keyword (starts with *)
        if line.startswith('*'):
            current_keyword = line.split('_')[0] if '_' in line else line
            # Handle full keyword for specific sections
            if line.startswith('*ELEMENT_SOLID'):
                current_keyword = '*ELEMENT_SOLID'
            elif line.startswith('*NODE'):
                current_keyword = '*NODE'
            continue
        
        # Parse NODE data
        if current_keyword == '*NODE':
            parts = line.split()
            if len(parts) >= 4:
                try:
                    nid = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    nodes[nid] = {'x': x, 'y': y, 'z': z}
                except ValueError:
                    continue
        
        # Parse ELEMENT_SOLID data
        elif current_keyword == '*ELEMENT_SOLID':
            parts = line.split()
            if len(parts) >= 10:  # eid, pid, n1-n8
                try:
                    eid = int(parts[0])
                    pid = int(parts[1])
                    node_ids = [int(parts[i]) for i in range(2, 10)]
                    elements[eid] = {'pid': pid, 'nodes': node_ids}
                except ValueError:
                    continue
    
    return elements, nodes


def print_summary(elements, nodes):
    """Print a summary of the parsed data."""
    print("=" * 50)
    print("LS-DYNA Keyword File Parser Results")
    print("=" * 50)
    
    print(f"\nTotal Nodes: {len(nodes)}")
    print("\nNode Coordinates:")
    print("-" * 50)
    for nid, coords in nodes.items():
        print(f"  Node {nid}: x={coords['x']:.4f}, y={coords['y']:.4f}, z={coords['z']:.4f}")
    
    print(f"\nTotal Solid Elements: {len(elements)}")
    print("\nElement to Node Mapping:")
    print("-" * 50)
    for eid, data in elements.items():
        print(f"  Element {eid} (Part {data['pid']}): Nodes {data['nodes']}")


if __name__ == "__main__":
    # Path to the keyword file
    keyfile_path = r"c:\Users\nir\Desktop\Final_Project\analysis\single_element_mode_1_two_ways\simgle_element_mode_1.k"
    
    # Parse the file
    elements, nodes = parse_lsdyna_keyfile(keyfile_path)
    
    # Print summary
    print_summary(elements, nodes)
    
    # Example: Access specific data
    print("\n" + "=" * 50)
    print("Example Data Access:")
    print("=" * 50)
    
    # Get nodes for a specific element
    if elements:
        first_eid = list(elements.keys())[0]
        print(f"\nNodes of element {first_eid}: {elements[first_eid]['nodes']}")
        
        # Get coordinates of those nodes
        print(f"\nCoordinates of nodes in element {first_eid}:")
        for nid in elements[first_eid]['nodes']:
            if nid in nodes:
                coords = nodes[nid]
                print(f"  Node {nid}: ({coords['x']}, {coords['y']}, {coords['z']})")
