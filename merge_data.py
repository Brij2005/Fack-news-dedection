import pandas as pd
import os

fake_path = os.path.join('data', 'Fake.csv')
true_path = os.path.join('data', 'True.csv')

if os.path.exists(fake_path) and os.path.exists(true_path):
    print("Files mil gayin! Dono ko merge (jod) rahe hain...")
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    fake_df['label'] = 1
    true_df['label'] = 0
    combined_df = pd.concat([fake_df, true_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)
    combined_df.to_csv(os.path.join('data', 'train.csv'), index=False)
    print("Mubarak ho! data/train.csv file ban gayi hai.")
else:
    print("Error: Kripya data folder me 'Fake.csv' aur 'True.csv' dono files rakhein.")
