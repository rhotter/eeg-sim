# %%
import numpy as np
import mne
import matplotlib.pyplot as plt
from pathlib import Path

# %% [markdown]
# # EEG Lead Field (Jacobian) Visualization
# This notebook demonstrates how to visualize the EEG lead field distribution for specific electrodes using MNE-Python.

# %%
# Set up the directory for storing MNE sample data
subjects_dir = mne.datasets.sample.data_path() / "subjects"

# %% [markdown]
# ## Create sample EEG data with standard 10-20 montage

# %%
# Create sample data with 10-20 electrode positions
ch_names = "Fz Cz Pz Oz Fp1 Fp2 F3 F4 F7 F8 C3 C4 T7 T8 P3 P4 P7 P8 O1 O2".split()
data = np.random.RandomState(0).randn(len(ch_names), 1000)
info = mne.create_info(ch_names, 1000.0, "eeg")
raw = mne.io.RawArray(data, info)

# %% [markdown]
# ## Set up the head model and source space

# %%
# Download fsaverage files
fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
subjects_dir = Path(fs_dir).parent

# Set up the source space
src = mne.setup_source_space(
    "fsaverage", spacing="oct6", subjects_dir=subjects_dir, add_dist=False
)

# Get the BEM solution
conductivity = (0.3, 0.006, 0.3)  # for three layers
model = mne.make_bem_model(
    "fsaverage", ico=4, conductivity=conductivity, subjects_dir=subjects_dir
)
bem = mne.make_bem_solution(model)

# %% [markdown]
# ## Set up the montage and compute the forward solution

# %%
# Setup the montage
montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage)

# Compute transformation matrix
fiducials = "estimated"  # get fiducials from the standard montage
trans = "fsaverage"  # use fsaverage transformation

# Compute forward solution
fwd = mne.make_forward_solution(
    raw.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0
)
# %%
# Convert to fixed orientation
fwd_fixed = mne.convert_forward_solution(fwd, force_fixed=True, surf_ori=True)

# %% [markdown]
# ## Visualize the lead field for a specific electrode

# %%
# Get the lead field matrix
leadfield = fwd_fixed["sol"]["data"]

# Select a specific electrode (e.g., Cz which is often used as reference)
electrode_idx = ch_names.index("Cz")

# Get the lead field for this electrode
electrode_leadfield = leadfield[electrode_idx]

# Split the leadfield into left and right hemispheres
n_sources_lh = len(fwd_fixed["src"][0]["vertno"])
n_sources_rh = len(fwd_fixed["src"][1]["vertno"])

leadfield_lh = electrode_leadfield[:n_sources_lh]
leadfield_rh = electrode_leadfield[n_sources_lh:]

# %% [markdown]
# ## Plot the lead field distribution on the brain

# %%
# Create source estimate object for visualization
vertices = [fwd_fixed["src"][0]["vertno"], fwd_fixed["src"][1]["vertno"]]

# %% [markdown]
# The visualization above shows how the electrical potential measured at the Cz electrode
# is influenced by different source locations in the brain. Brighter colors indicate
# regions where neural activity has a stronger influence on the measurement at Cz.

# %%
import mne.viz

mne.viz.set_3d_backend("pyvistaqt")

# %%
# Create an stc with one time point, so data has shape=(n_vertices_total, n_times)
data_stc = np.concatenate([leadfield_lh, leadfield_rh])
data_stc = data_stc[:, np.newaxis]  # shape => (n_sources, 1)

stc = mne.SourceEstimate(
    data_stc,
    vertices=vertices,  # [lh_vertno, rh_vertno]
    tmin=0.0,
    tstep=1.0,  # dummy
    subject="fsaverage",  # make sure to match your subject
)

# Now plot with MNE's built-in 3D viewer
brain = stc.plot(
    subject="fsaverage",
    surface="pial",
    subjects_dir=subjects_dir,
    hemi="both",
    time_viewer=False,
    views=["lat"],
    size=(800, 800),
    colormap="plasma",
    clim=dict(kind="value", lims=[0, 0.5 * data_stc.max(), data_stc.max()]),
    smoothing_steps=5,  # Smooth the data for better visualization
    transparent=False,  # Make the surface fully opaque
)
