import parselmouth
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file
sound = parselmouth.Sound("03a01Fa.wav")

# Extract pitch
pitch = sound.to_pitch()
pitch_values = pitch.selected_array['frequency']

# Extract formants (this will give us formants across all time frames)
formant = sound.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5000)

# Get time frames
time_frames = formant.xs()

# Store formants across all time frames (for each time point)
formants = []

for t in time_frames:
    frame_formants = []
    for i in range(1, 3):  # Extracting formants F1 to F5
        formant_value = formant.get_value_at_time(i, t)
        frame_formants.append(formant_value)
    formants.append(frame_formants)

# Convert to numpy array for easier handling
formants = np.array(formants)

# Print all formants for each time frame (as a list)
print("Formants for each time frame:")
for i, frame_formants in enumerate(formants):
    print(f"Time {time_frames[i]:.3f} s: {frame_formants}")

# Plot pitch contour
plt.figure()
plt.plot(pitch.xs(), pitch_values, label="Pitch")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Pitch Contour")
plt.legend()

# Plot formants (we will plot only the first two formants for simplicity)
plt.figure()
plt.plot(time_frames, formants[:, 0], label="Formant 1 (F1)")
plt.plot(time_frames, formants[:, 1], label="Formant 2 (F2)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Formant Contours")
plt.legend()
plt.show()
