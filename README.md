# LS-DYNA Post-Processor UI

Interactive web-based visualization tool for LS-DYNA simulation results using Streamlit.

## Features

- 🎯 **Element Selection**: Choose any element from your simulation to analyze
- 📊 **Multiple Plots**:
    - Stress components over time
    - Displacement components over time
    - Velocity components over time
    - Internal energy over time
    - Energy release rate (G_c) over time
    - Cohesive separation (relative displacement)
    - Traction-separation curves (stress vs. separation)
- 📋 **Element Summary**: View node coordinates and metadata
- 📈 **Interactive Visualizations**: Hover, zoom, and pan on all plots

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run LS_Post_UI.py
```

The application will open automatically in your default browser (typically at `http://localhost:8501`).

## Usage

1. **Configure Model**:
    - In the sidebar, enter the path to your LS-DYNA analysis folder
    - Specify the keyword file name (.k file)
    - Click "Load Model"

2. **Select Element**:
    - Use the dropdown to choose which element to analyze
    - Click "Show Summary" to view node coordinates

3. **Explore Data**:
    - Use the tabs to switch between different plot types
    - Hover over plotting areas for detailed values
    - Use Plotly tools to zoom, pan, and download plots

## Example Path

```
C:\Users\nir\Desktop\Final_Project\analysis\single_element_mode_1_two_ways
```

Keyword file: `simgle_element_mode_1.k`

## Supported Output Files

The model loader automatically detects and loads:

- `nodout`: Nodal time-history data (displacements, velocities, accelerations)
- `elout`: Element stress time-history data
- `matsum`: Part internal energy time-history
- `.k` file: Element connectivity and initial node coordinates

## Data Calculated

- **Cohesive Separation**: Relative displacement between top and bottom faces of cohesive elements
    - Formula: δ = u_top - u_bottom
- **G_c (Energy Release Rate)**: Critical energy release rate
    - Method 1: G_c = Max Internal Energy / Face Area
    - Method 2: G_c = ∫ σ(δ) dδ (Area under traction-separation curve)

## Tips

- The app caches model data for faster switching between elements
- All plots are interactive - use the Plotly toolbar for zoom/pan/export
- Stress and displacement plots show individual node data points
- The traction-separation curve shows the total G_c value in the title

## Troubleshooting

**"No data available" warnings**:

- Ensure the output files (nodout, elout, matsum) exist in the analysis folder
- Check that the .k file path is correct

**Slow loading**:

- Large simulations take time to parse; be patient on first load
- Subsequent element selections are cached and instant

**Empty plots**:

- Some elements may not have certain data types
- Check the console output for detailed error messages
