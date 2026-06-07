import pandas as pd
import os

# Count images in each folder
base = r'c:\Users\amogh\EarlyLiverFailurePrediction\Ultrasonic_dataset\Dataset\Dataset'
for stage in ['F0','F1','F2','F3','F4']:
    path = os.path.join(base, stage)
    imgs = [f for f in os.listdir(path) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
    print(f'{stage}: {len(imgs)} images')

cohort = pd.read_csv(r'c:\Users\amogh\EarlyLiverFailurePrediction\cirrhosis_cohort.csv')
labels = pd.read_csv(r'c:\Users\amogh\EarlyLiverFailurePrediction\decompensation_labels.csv')
labs   = pd.read_csv(r'c:\Users\amogh\EarlyLiverFailurePrediction\labs_cirrhosis.csv')

print(f'\nCohort patients: {len(cohort)}')
print(f'Label patients:  {len(labels)}')
print(f'Decompensation 90d positive rate: {labels["decompensation_90day"].mean():.1%}')
print(f'Mortality 30d positive rate:      {labels["mortality_30day"].mean():.1%}')
print(f'Lab rows:        {len(labs):,}')
print(f'Unique patients in labs: {labs["subject_id"].nunique()}')
print()
print('Lab test distribution:')
print(labs['lab_test_name'].value_counts())
