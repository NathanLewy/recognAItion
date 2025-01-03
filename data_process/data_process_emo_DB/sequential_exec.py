from Filteredaudio import AudioFilter
from csv_organizer import CsvBatchTransformer
from csv_to_train_test import CsvDataOrganizer
from shuffle import CSVProcessor
from wav_to_csv_opensmile import WavToCsvExtractor

# List of all possible columns from eGeMAPSv02 feature set
features = [
'file', 'start', 'end', 'Loudness_sma3', 'alphaRatio_sma3',
       'hammarbergIndex_sma3', 'slope0-500_sma3', 'slope500-1500_sma3',
       'spectralFlux_sma3', 'mfcc1_sma3', 'mfcc2_sma3', 'mfcc3_sma3',
       'mfcc4_sma3', 'F0semitoneFrom27.5Hz_sma3nz', 'jitterLocal_sma3nz',
       'shimmerLocaldB_sma3nz', 'HNRdBACF_sma3nz', 'logRelF0-H1-H2_sma3nz',
       'logRelF0-H1-A3_sma3nz', 'F1frequency_sma3nz', 'F1bandwidth_sma3nz',
       'F1amplitudeLogRelF0_sma3nz', 'F2frequency_sma3nz',
       'F2bandwidth_sma3nz', 'F2amplitudeLogRelF0_sma3nz',
       'F3frequency_sma3nz', 'F3bandwidth_sma3nz',
       'F3amplitudeLogRelF0_sma3nz'
]
columns_to_keep = [
    'jitterLocal_sma3nz',
    'F0semitoneFrom27.5Hz_sma3nz',
    'hammarbergIndex_sma3',
    'F1amplitudeLogRelF0_sma3nz',

]
num_segments = 20
#emotions_to_pick_from = ["F","N","W","T","A","E"] il faut changer d'autres parties du code pour que ça marche ça 
#W = Anger, E = disgust, A = fear, F = happineess, T = sadness, N = neutral
#W = 0, E = 1, A = 2,F = 3,T = 4, N = 5
emotions = ['T','F','N','E','A','W']

filter_ang_hap = AudioFilter(emotions)
extractor = WavToCsvExtractor(num_segments,columns_to_keep = columns_to_keep)
transformer = CsvBatchTransformer()
organizer = CsvDataOrganizer()




filter_ang_hap.filter_and_copy_files()
extractor.process_all_files()
transformer.process_all_files()
organizer.create_samples_and_labels()

processor = CSVProcessor()
processor.shuffle_and_extract(
    output_sample_file="/home/pierres/Projet_S7/recognAItion/data_emodb/sample_eval.csv",
    output_label_file="/home/pierres/Projet_S7/recognAItion/data_emodb/label_eval.csv",
    fraction=0.2  # Extract 20% of the data
)

