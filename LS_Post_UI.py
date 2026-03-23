"""
LS-DYNA Post-Processing Interactive UI using Streamlit
Allows selection of elements and visualization of simulation results
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Optional, List, Dict
from LS_Post_data_reader import Model

# Page configuration
st.set_page_config(
    page_title="LS-DYNA Post Processor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 LS-DYNA Post-Processor")
st.markdown("Interactive visualization of LS-DYNA simulation results")


def load_model_cached(folder_path: str, keyfile: str) -> Optional[Model]:
    """Load model with caching to avoid reloading."""
    try:
        with st.spinner(f"Loading model from {folder_path}..."):
            model = Model(
                folder=folder_path,
                keyfile=keyfile,
                load_nodout=True,
                load_elout=True,
                load_matsum=True,
            )
        st.success("✓ Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def plot_stress_over_time(element_ids: List[int], model: Model, components: List[str]) -> Optional[go.Figure]:
    """Plot stress components over time for one or multiple elements."""
    try:
        if not element_ids or not components:
            st.warning("Please select at least one element and one stress component")
            return None
        
        fig = go.Figure()
        
        for element_id in element_ids:
            element = model.get_element(element_id)
            if element.stress_data is None:
                st.warning(f"No stress data available for element {element_id}")
                continue
            
            stress_data = element.stress_data.copy()
            
            # Add selected stress components
            for col in components:
                if col in stress_data.columns:
                    fig.add_trace(go.Scatter(
                        y=stress_data[col],
                        x=stress_data.index,
                        mode='lines',
                        name=f"Elem {element_id} - {col}",
                        hovertemplate='<b>%{fullData.name}</b><br>Time: %{x:.4f}<br>Stress: %{y:.4E}<extra></extra>'
                    ))
        
        fig.update_layout(
            title="Stress Components Over Time",
            xaxis_title="Time (s)",
            yaxis_title="Stress (Pa)",
            hovermode='x unified',
            template='plotly_white',
            height=500,
            legend=dict(x=0.01, y=0.99)
        )
        
        return fig
    except Exception as e:
        st.error(f"Error plotting stress: {str(e)}")
        return None


def calculate_average_element_displacement(element, directions: List[str]) -> Optional[Dict[str, pd.DataFrame]]:
    """Calculate average displacement for top and bottom faces separately and their difference.
    
    Returns:
        Dictionary with keys 'top', 'bottom', 'diff' containing average displacement DataFrames
    """
    try:
        if element.node_data is None:
            return None
        
        # Get top and bottom faces
        faces = element.get_faces()
        bottom_face = faces[0]  # (n1, n2, n3, n4)
        top_face = faces[1]     # (n5, n6, n7, n8)
        
        # Collect displacements for bottom face nodes
        bottom_displacements = []
        for node_id in bottom_face:
            displacement = element.get_node_displacement(node_id)
            if displacement is not None and not displacement.empty:
                bottom_displacements.append(displacement)
        
        # Collect displacements for top face nodes
        top_displacements = []
        for node_id in top_face:
            displacement = element.get_node_displacement(node_id)
            if displacement is not None and not displacement.empty:
                top_displacements.append(displacement)
        
        if not bottom_displacements or not top_displacements:
            return None
        
        # Calculate average across bottom face nodes
        avg_bottom = pd.concat(bottom_displacements).groupby(level=0).mean()
        avg_bottom = avg_bottom[directions] if directions else avg_bottom
        
        # Calculate average across top face nodes
        avg_top = pd.concat(top_displacements).groupby(level=0).mean()
        avg_top = avg_top[directions] if directions else avg_top
        
        # Calculate difference (top - bottom)
        avg_diff = avg_top - avg_bottom
        
        return {
            'top': avg_top,
            'bottom': avg_bottom,
            'diff': avg_diff
        }
    except Exception:
        return None


def plot_displacement_over_time(element_ids: List[int], model: Model, directions: List[str], use_average: bool = False) -> Optional[go.Figure]:
    """Plot displacement components over time for one or multiple elements.
    
    Args:
        element_ids: List of element IDs to plot
        model: The model containing element data
        directions: List of displacement directions to plot
        use_average: If True, plot average displacement per element instead of individual nodes
    """
    try:
        if not element_ids or not directions:
            st.warning("Please select at least one element and one displacement direction")
            return None
        
        fig = go.Figure()
        
        for element_id in element_ids:
            element = model.get_element(element_id)
            if element.node_data is None:
                st.warning(f"No nodal data available for element {element_id}")
                continue
            
            if use_average:
                # Plot average displacement for top/bottom faces and difference
                avg_displacements = calculate_average_element_displacement(element, directions)
                if avg_displacements is not None:
                    for col in directions:
                        # Plot top face average
                        if col in avg_displacements['top'].columns:
                            fig.add_trace(go.Scatter(
                                y=avg_displacements['top'][col],
                                x=avg_displacements['top'].index,
                                mode='lines',
                                name=f"Elem {element_id} - Top {col}",
                                hovertemplate='<b>%{fullData.name}</b><br>Time: %{x:.4f}<br>Disp: %{y:.4E}<extra></extra>',
                                line=dict(width=2, dash='dot')
                            ))
                        
                        # Plot bottom face average
                        if col in avg_displacements['bottom'].columns:
                            fig.add_trace(go.Scatter(
                                y=avg_displacements['bottom'][col],
                                x=avg_displacements['bottom'].index,
                                mode='lines',
                                name=f"Elem {element_id} - Bottom {col}",
                                hovertemplate='<b>%{fullData.name}</b><br>Time: %{x:.4f}<br>Disp: %{y:.4E}<extra></extra>',
                                line=dict(width=2, dash='dash')
                            ))
                        
                        # Plot difference (top - bottom)
                        if col in avg_displacements['diff'].columns:
                            fig.add_trace(go.Scatter(
                                y=avg_displacements['diff'][col],
                                x=avg_displacements['diff'].index,
                                mode='lines',
                                name=f"Elem {element_id} - Diff {col}",
                                hovertemplate='<b>%{fullData.name}</b><br>Time: %{x:.4f}<br>Disp: %{y:.4E}<extra></extra>',
                                line=dict(width=2.5)
                            ))
            else:
                # Plot displacement for each node in the element
                for node_id in element.node_ids:
                    try:
                        displacement = element.get_node_displacement(node_id)
                        if displacement is not None and not displacement.empty:
                            for col in directions:
                                if col in displacement.columns:
                                    fig.add_trace(go.Scatter(
                                        y=displacement[col],
                                        x=displacement.index,
                                        mode='lines',
                                        name=f"Elem {element_id} - Node {node_id} - {col}",
                                        hovertemplate='<b>%{fullData.name}</b><br>Time: %{x:.4f}<br>Disp: %{y:.4E}<extra></extra>'
                                    ))
                    except Exception:
                        continue
        
        title_suffix = " (Top/Bottom/Difference)" if use_average else ""
        fig.update_layout(
            title=f"Displacements Over Time{title_suffix}",
            xaxis_title="Time (s)",
            yaxis_title="Displacement (m)",
            hovermode='x unified',
            template='plotly_white',
            height=500,
            legend=dict(x=0.01, y=0.99, maxheight=150)
        )
        
        return fig
    except Exception as e:
        st.error(f"Error plotting displacement: {str(e)}")
        return None


def _prepare_matsum_series(series: pd.Series, model_times: np.ndarray) -> pd.Series:
    # Keep only matsum timestamps and preserve them. Add a zero at model end if element is deleted early.
    if series is None or series.empty:
        return series

    out = series.copy().sort_index()
    if model_times.size > 0:
        final_time = float(np.max(model_times))
        last_time = float(out.index.max())
        if final_time > last_time:
            out.loc[final_time] = 0.0
            out = out.sort_index()
    return out


def _prepare_calculated_series(series: pd.Series, model_times: np.ndarray) -> pd.Series:
    # Keep calculated timestamps, and append zeros after last known time for all remaining model times.
    if series is None or series.empty:
        return series

    out = series.copy().sort_index()
    if model_times.size > 0:
        final_time = float(np.max(model_times))
        last_time = float(out.index.max())

        # Add any missing model times beyond the last known calculation time with zeros
        if final_time > last_time:
            times_to_add = model_times[model_times > last_time]
            if times_to_add.size > 0:
                add_index = pd.Index(times_to_add)
                out = out.reindex(out.index.union(add_index))
                out.loc[out.index > last_time] = 0.0

    return out

def plot_internal_energy_over_time(
    element_ids: List[int],
    model: Model,
    show_matsum: bool = True,
    show_calculated: bool = False,
    show_combined: bool = False,
    divide_by_area: bool = False,
) -> Optional[go.Figure]:
    """Plot internal energy over time for one or multiple elements.

    Args:
        show_matsum: If True, show part matsum energy
        show_calculated: If True, show calculated cumulative energy
        show_combined: If True, plot both on same graph
        divide_by_area: If True, divide calculated energy by element area
    """
    try:
        if not element_ids:
            st.warning("Please select at least one element")
            return None

        fig = go.Figure()

        for element_id in element_ids:
            element = model.get_element(element_id)
            part = model.get_part(element.pid)

            if show_matsum and part.internal_energy is not None and not part.internal_energy.empty:
                energy_series = part.internal_energy.copy()
                source_tag = "Matsum"
                if divide_by_area:
                    energy_series = energy_series / element.area
                    source_tag += " / Area"

                energy_series = _prepare_matsum_series(energy_series, model.times)

                fig.add_trace(go.Scatter(
                    y=energy_series,
                    x=energy_series.index,
                    mode='lines',
                    name=f"Elem {element_id} (Part {element.pid}, {source_tag})",
                    hovertemplate='<b>%{fullData.name}</b><br>Time: %{x:.4f}<br>Energy: %{y:.4E}<extra></extra>'
                ))

            if show_calculated:
                try:
                    cumulative_energy = element.calculate_internal_energy(use_cohesive_separation=True)
                    energy_series = cumulative_energy
                    if divide_by_area:
                        energy_series = energy_series / element.area
                    source_tag = "Calculated" + (" / Area" if divide_by_area else "")

                    energy_series = _prepare_calculated_series(energy_series, model.times)

                    fig.add_trace(go.Scatter(
                        y=energy_series,
                        x=energy_series.index,
                        mode='lines',
                        name=f"Elem {element_id} ({source_tag})",
                        hovertemplate='<b>%{fullData.name}</b><br>Time: %{x:.4f}<br>Energy: %{y:.4E}<extra></extra>'
                    ))
                except Exception as e:
                    st.warning(f"Could not compute internal energy for element {element_id}: {str(e)}")
                    continue

        title_suffix = []
        if show_matsum:
            suffix = "Matsum"
            if divide_by_area:
                suffix += " / Area"
            title_suffix.append(suffix)
        if show_calculated:
            suffix = "Calculated"
            if divide_by_area:
                suffix += " / Area"
            title_suffix.append(suffix)
        title_suffix = " & ".join(title_suffix) if title_suffix else "Energy"

        yaxis_label = "Internal Energy"
        if divide_by_area and (show_matsum or show_calculated):
            yaxis_label += " (J/m²)"
        else:
            yaxis_label += " (J)"

        fig.update_layout(
            title=f"Internal Energy Over Time ({title_suffix})",
            xaxis_title="Time (s)",
            yaxis_title=yaxis_label,
            hovermode='x unified',
            template='plotly_white',
            height=500,
        )
        
        return fig
    except Exception as e:
        st.error(f"Error plotting internal energy: {str(e)}")
        return None


def plot_gc_over_time(
    element_ids: List[int],
    model: Model,
    mode: str = "I",
) -> Optional[go.Figure]:
    """Plot G_c (energy release rate) over time for one or multiple elements.

    Args:
        mode: "I" for Mode I (G_IC), "II" for Mode II (G_IIC), "C" for mixed (G_C).
    """
    try:
        if not element_ids:
            st.warning("Please select at least one element")
            return None
        
        fig = go.Figure()
        
        for element_id in element_ids:
            element = model.get_element(element_id)
            
            try:
                result_df, final_Gc = element.calculate_Gc_by_integration(
                    use_cohesive_separation=True,
                    mode=mode,
                )

                if result_df is None or result_df.empty:
                    st.warning(f"Could not calculate G_c for element {element_id}")
                    continue
                
                # Plot cumulative G_c over time (already per unit area)
                g_values = result_df['G_cumulative']
                g_values = _prepare_calculated_series(g_values, model.times)

                fig.add_trace(go.Scatter(
                    y=g_values,
                    x=g_values.index,
                    mode='lines',
                    name=f"Elem {element_id} (Final G: {final_Gc:.4E})",
                    hovertemplate='<b>%{fullData.name}</b><br>Time: %{x:.4f}<br>G: %{y:.4E}<extra></extra>'
                ))
            except Exception as e:
                st.warning(f"Error calculating G_c for element {element_id}: {str(e)}")
                continue
        
        title_map = {
            "I": "G_IC (Mode I)",
            "II": "G_IIC (Mode II)",
            "C": "G_C (Mixed Mode)",
        }
        fig.update_layout(
            title=f"Critical Energy Release Rate Over Time ({title_map.get(mode, mode)})",
            xaxis_title="Time (s)",
            yaxis_title="Energy (J/m²)",
            hovermode='x unified',
            template='plotly_white',
            height=500,
        )
        
        return fig
    except Exception as e:
        st.error(f"Error plotting G_c: {str(e)}")
        return None



def plot_traction_separation_curve(
    element_ids: List[int],
    model: Model,
    mode: str = "I",
) -> Optional[go.Figure]:
    """Plot traction-separation curve (stress vs. separation) for one or multiple elements."""
    try:
        if not element_ids:
            st.warning("Please select at least one element")
            return None
        
        fig = go.Figure()
        
        for element_id in element_ids:
            element = model.get_element(element_id)
            
            traction_sep_df = element.get_traction_separation_data(
                use_cohesive_separation=True,
                mode=mode,
            )
            
            if traction_sep_df is None or traction_sep_df.empty:
                st.warning(f"No traction-separation data available for element {element_id}")
                continue
            
            if mode == "C":
                # Plot mixed-mode (effective) curve + individual Mode I/II curves
                if 'separation_mixed' in traction_sep_df and 'traction_mixed' in traction_sep_df:
                    fig.add_trace(go.Scatter(
                        x=traction_sep_df['separation_mixed'],
                        y=traction_sep_df['traction_mixed'],
                        mode='lines',
                        name=f"Elem {element_id} (Mixed Mode)",
                        line=dict(width=3),
                        legendgroup=f"elem_{element_id}"
                    ))

                if 'separationI' in traction_sep_df and 'tractionI' in traction_sep_df:
                    fig.add_trace(go.Scatter(
                        x=traction_sep_df['separationI'],
                        y=traction_sep_df['tractionI'],
                        mode='lines',
                        name=f"Elem {element_id} (Mode I)",
                        line=dict(width=2, dash='dot'),
                        legendgroup=f"elem_{element_id}"
                    ))
                if 'separationII' in traction_sep_df and 'tractionII' in traction_sep_df:
                    fig.add_trace(go.Scatter(
                        x=traction_sep_df['separationII'],
                        y=traction_sep_df['tractionII'],
                        mode='lines',
                        name=f"Elem {element_id} (Mode II)",
                        line=dict(width=2, dash='dash'),
                        legendgroup=f"elem_{element_id}"
                    ))
            else:
                fig.add_trace(go.Scatter(
                    x=traction_sep_df['separation'],
                    y=traction_sep_df['traction'],
                    mode='lines',
                    name=f"Elem {element_id} (Mode {mode})",
                    hovertemplate='Separation: %{x:.4E}<br>Traction: %{y:.4E}<extra></extra>',
                    line=dict(width=2),
                    legendgroup=f"elem_{element_id}"
                ))
                
                # Fill area under curve
                fig.add_trace(go.Scatter(
                    x=traction_sep_df['separation'],
                    y=traction_sep_df['traction'],
                    fill='tozeroy',
                    mode='none',
                    name=f'Elem {element_id} - Area',
                    hoverinfo='skip',
                    fillcolor=f'rgba({hash(element_id) % 255}, {(hash(element_id) * 2) % 255}, {(hash(element_id) * 3) % 255}, 0.2)',
                    showlegend=False,
                    legendgroup=f"elem_{element_id}"
                ))
        
        title_map = {
            "I": "Mode I (normal)",
            "II": "Mode II (shear)",
            "C": "Mixed Mode (I + II)",
        }
        fig.update_layout(
            title=f"Traction-Separation Curve ({title_map.get(mode, mode)})",
            xaxis_title="Separation (m)",
            yaxis_title="Traction (Pa)",
            hovermode='closest',
            template='plotly_white',
            height=500,
        )
        
        return fig
    except Exception as e:
        st.error(f"Error plotting traction-separation curve: {str(e)}")
        return None


def display_element_summary(element_id: int, model: Model):
    """Display element information summary."""
    try:
        element = model.get_element(element_id)
        part = model.get_part(element.pid)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Element ID", element_id)
        with col2:
            st.metric("Part ID", element.pid)
        with col3:
            st.metric("Number of Nodes", len(element.node_ids))
        with col4:
            if part.internal_energy is not None:
                st.metric("Max Energy (J)", f"{part.get_max_internal_energy():.4E}")
        
        # Element node information
        st.subheader("Node Information")
        node_data = []
        for nid in element.node_ids:
            coords = element.initial_node_coords.get(nid, {})
            node_data.append({
                'Node ID': nid,
                'X (m)': coords.get('x', 0),
                'Y (m)': coords.get('y', 0),
                'Z (m)': coords.get('z', 0),
            })
        
        st.dataframe(pd.DataFrame(node_data), use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying element summary: {str(e)}")


# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Get query parameters (persists across refreshes)
    try:
        # Try new API first (Streamlit >= 1.30)
        query_params = st.query_params
        folder_from_url = query_params.get("folder", None)
        keyfile_from_url = query_params.get("keyfile", None)
    except AttributeError:
        # Fall back to old API
        query_params = st.experimental_get_query_params()
        folder_from_url = query_params.get("folder", [None])[0]
        keyfile_from_url = query_params.get("keyfile", [None])[0]
    
    # Initialize with query params or defaults
    default_folder = folder_from_url if folder_from_url else r"C:\Users\nir\Desktop\Final_Project\analysis\single_element_mode_1_two_ways"
    default_keyfile = keyfile_from_url if keyfile_from_url else "simgle_element_mode_1.k"
    
    # Folder and keyfile selection
    folder_path = st.text_input(
        "Analysis Folder Path",
        value=default_folder,
        help="Path to folder containing LS-DYNA output files",
        key="folder_input"
    )
    
    keyfile_name = st.text_input(
        "Keyword File Name",
        value=default_keyfile,
        help="Name of the .k file in the analysis folder",
        key="keyfile_input"
    )
    
    if st.button("🔄 Load Model", use_container_width=True):
        # Save to query parameters (persists across refreshes)
        try:
            # Try new API first (Streamlit >= 1.30)
            st.query_params["folder"] = folder_path
            st.query_params["keyfile"] = keyfile_name
        except (AttributeError, TypeError):
            # Fall back to old API
            st.experimental_set_query_params(folder=folder_path, keyfile=keyfile_name)
        
        # Load the model
        st.session_state.model = load_model_cached(folder_path, keyfile_name)
        st.session_state.model_loaded = True


# Main content
if 'model' not in st.session_state:
    st.info("👈 Configure the analysis folder and keyword file in the sidebar, then click 'Load Model' to begin.")
else:
    model = st.session_state.model
    
    if model is None:
        st.error("Failed to load model. Check the folder path and keyfile name.")
    else:
        # Reset widget states when a new model is loaded
        if st.session_state.get('_last_model_id') != id(model):
            st.session_state._last_model_id = id(model)
            # Clear stress component selection to use default
            if 'stress_components' in st.session_state:
                del st.session_state.stress_components
        
        # Initialize saved selections in session state
        if 'saved_selections' not in st.session_state:
            st.session_state.saved_selections = {}
        
        # Element selection
        st.header("🔍 Element Selection")
        
        # Saved selection management
        with st.expander("📁 Manage Saved Selections"):
            # Show predefined sets from keyword file
            if model.solid_sets:
                st.write("**Predefined Element Sets (from keyword file):**")
                
                for set_name, element_ids in model.solid_sets.items():
                    # Filter to only include elements that exist in the model
                    valid_elements = [eid for eid in element_ids if eid in model.element_ids]
                    
                    if valid_elements:
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"🔷 {set_name}: {len(valid_elements)} element(s)")
                        
                        with col2:
                            if st.button("Load", key=f"load_preset_{set_name}", use_container_width=True):
                                st.session_state.element_selector = valid_elements.copy()
                                st.rerun()
                
                st.divider()
            
            # User-defined saved selections
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selection_name = st.text_input(
                    "Selection Name",
                    placeholder="Enter name for current selection",
                    key="selection_name_input"
                )
            
            with col2:
                st.write("")
                st.write("")
                if st.button("💾 Save Current Selection", use_container_width=True):
                    if selection_name and st.session_state.get('element_selector'):
                        st.session_state.saved_selections[selection_name] = st.session_state.element_selector.copy()
                        st.success(f"✓ Saved selection '{selection_name}'")
                    elif not selection_name:
                        st.warning("Please enter a name for the selection")
                    else:
                        st.warning("No elements selected to save")
            
            # Display and load user-saved selections
            if st.session_state.saved_selections:
                st.write("**Your Saved Selections:**")
                
                for name, elements in list(st.session_state.saved_selections.items()):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"📌 {name}: {len(elements)} element(s)")
                    
                    with col2:
                        if st.button("Load", key=f"load_{name}", use_container_width=True):
                            st.session_state.element_selector = elements.copy()
                            st.rerun()
                    
                    with col3:
                        if st.button("🗑️", key=f"delete_{name}", use_container_width=True):
                            del st.session_state.saved_selections[name]
                            st.rerun()
            else:
                st.info("No saved selections yet. Select elements below and save them.")
        
        # Element multiselect
        selected_elements = st.multiselect(
            "Select Elements:",
            options=model.element_ids,
            format_func=lambda x: f"Element {x}",
            default=[model.element_ids[0]] if model.element_ids else [],
            key="element_selector"
        )
        
        if selected_elements:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📋 Show Summary", use_container_width=True):
                    st.session_state.show_summary = not st.session_state.get('show_summary', False)
            
            with col2:
                st.write("")  # Spacing
            
            if st.session_state.get('show_summary', False):
                # Show summary for first selected element
                display_element_summary(selected_elements[0], model)
            
            # Plot selection tabs
            st.header("📈 Visualization")
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "Stress",
                "Displacement",
                "Internal Energy",
                "G_c"
            ])
            
            with tab1:
                st.subheader("Stress Components Over Time")
                # Get available stress components from first element
                element = model.get_element(selected_elements[0])
                available_stress = []
                if element.stress_data is not None:
                    available_stress = [col for col in element.stress_data.columns]
                
                # Determine default: sig_zz if available, otherwise first 3 components
                default_stress = ['sig_zz'] if 'sig_zz' in available_stress else (available_stress[:3] if len(available_stress) > 0 else [])
                
                stress_components = st.multiselect(
                    "Select Stress Components:",
                    options=available_stress,
                    default=default_stress,
                    key="stress_components"
                )
                
                if stress_components:
                    fig = plot_stress_over_time(selected_elements, model, stress_components)
                    if fig:
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            key=f"stress_{'_'.join(map(str, selected_elements))}_{'_'.join(stress_components)}",
                        )
            
            with tab2:
                st.subheader("Displacement Components Over Time")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    displacement_directions = st.multiselect(
                        "Select Displacement Directions:",
                        options=['x_disp', 'y_disp', 'z_disp'],
                        default=['z_disp'],
                        key="displacement_directions"
                    )
                
                with col2:
                    use_average = st.checkbox(
                        "Show Face Averages",
                        value=False,
                        key="use_average_displacement",
                        help="Show average displacement for top face, bottom face, and their difference (top - bottom)"
                    )
                
                if displacement_directions:
                    fig = plot_displacement_over_time(selected_elements, model, displacement_directions, use_average)
                    if fig:
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            key=f"disp_{'_'.join(map(str, selected_elements))}_{'_'.join(displacement_directions)}_{use_average}",
                        )
            
            with tab3:
                st.subheader("Internal Energy Over Time")

                col1, col2 = st.columns(2)
                with col1:
                    show_matsum = st.checkbox("Show Matsum", value=True, key="show_matsum")
                with col2:
                    show_calculated = st.checkbox("Show Calculated", value=False, key="show_calculated")
                
                divide_by_area = st.checkbox("Divide Calculated by Area", value=False, key="divide_energy_by_area")
                show_combined = st.checkbox("Show Both on One Graph", value=False, key="show_combined")
                
                # Handle non-area case: side by side when both selected and not combined
                if not divide_by_area and show_matsum and show_calculated and not show_combined:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Matsum Internal Energy**")
                        fig_matsum = plot_internal_energy_over_time(selected_elements, model, show_matsum=True, show_calculated=False, show_combined=False, divide_by_area=False)
                        if fig_matsum:
                            fig_matsum.update_layout(title="Matsum Internal Energy Over Time")
                            st.plotly_chart(
                                fig_matsum,
                                use_container_width=True,
                                key=f"matsum_energy_{'_'.join(map(str, selected_elements))}_no_area",
                            )
                    with col2:
                        st.markdown("**Calculated Internal Energy**")
                        fig_calc = plot_internal_energy_over_time(selected_elements, model, show_matsum=False, show_calculated=True, show_combined=False, divide_by_area=False)
                        if fig_calc:
                            fig_calc.update_layout(title="Calculated Internal Energy Over Time")
                            st.plotly_chart(
                                fig_calc,
                                use_container_width=True,
                                key=f"calculated_energy_{'_'.join(map(str, selected_elements))}_no_area",
                            )
                
                if divide_by_area:
                    if show_combined:
                        # Show both matsum and calculated on the same graph, divided by area
                        fig_energy = plot_internal_energy_over_time(selected_elements, model, show_matsum=True, show_calculated=True, show_combined=True, divide_by_area=True)
                        if fig_energy:
                            fig_energy.update_layout(title="Internal Energy / Area Over Time")
                            st.plotly_chart(
                                fig_energy,
                                use_container_width=True,
                                key=f"energy_both_{'_'.join(map(str, selected_elements))}_area",
                            )
                    else:
                        # Show side by side: matsum/area and G_C mixed
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Matsum Energy / Area**")
                            fig_matsum = plot_internal_energy_over_time(selected_elements, model, show_matsum=True, show_calculated=False, show_combined=False, divide_by_area=True)
                            if fig_matsum:
                                fig_matsum.update_layout(title="Matsum Energy / Area Over Time")
                                st.plotly_chart(
                                    fig_matsum,
                                    use_container_width=True,
                                    key=f"matsum_area_{'_'.join(map(str, selected_elements))}",
                                )
                        with col2:
                            st.markdown("**G_C (Mixed)**")
                            fig_gc = plot_gc_over_time(selected_elements, model, mode="C")
                            if fig_gc:
                                fig_gc.update_layout(title="Critical Energy Release Rate (G_C) Over Time")
                                st.plotly_chart(
                                    fig_gc,
                                    use_container_width=True,
                                    key=f"gc_mixed_side_{'_'.join(map(str, selected_elements))}",
                                )
                elif not (not divide_by_area and show_matsum and show_calculated and not show_combined) and (show_matsum or show_calculated):
                    fig = plot_internal_energy_over_time(selected_elements, model, show_matsum=show_matsum, show_calculated=show_calculated, show_combined=show_combined, divide_by_area=divide_by_area)
                    if fig:
                        if divide_by_area:
                            fig.update_layout(title="Internal Energy / Area Over Time")
                        else:
                            fig.update_layout(title="Internal Energy Over Time")
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            key=f"energy_{'_'.join(map(str, selected_elements))}_{show_matsum}_{show_calculated}_{show_combined}_{divide_by_area}",
                        )
            
            with tab4:
                st.subheader("Energy Release Rate (G_c) & Traction-Separation")

                mode = st.selectbox(
                    "Select fracture mode:",
                    options=["I", "II", "C"],
                    format_func=lambda x: {"I": "Mode I (G_IC)", "II": "Mode II (G_IIC)", "C": "Mixed (G_C)"}[x],
                    help="Choose which fracture mode to calculate and plot. Mixed mode sums Mode I + Mode II energy.",
                )

                st.markdown("---")

                st.subheader("Energy Release Rate Over Time")
                fig = plot_gc_over_time(selected_elements, model, mode=mode)
                if fig:
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key=f"gc_over_time_{'_'.join(map(str, selected_elements))}_{mode}",
                    )

                st.markdown("---")

                st.subheader("Traction-Separation Curve")
                fig = plot_traction_separation_curve(selected_elements, model, mode=mode)
                if fig:
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key=f"traction_separation_{'_'.join(map(str, selected_elements))}_{mode}",
                    )
        else:
            st.info("👆 Select at least one element to view plots")
        
        # Model statistics footer
        st.divider()
        st.subheader("📊 Model Statistics")
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            st.metric("Total Elements", len(model.element_ids))
        with stat_col2:
            st.metric("Total Parts", len(model.part_ids))
        with stat_col3:
            st.metric("Total Nodes", len(model.node_ids))
        with stat_col4:
            st.metric("Timesteps", len(model.times))
