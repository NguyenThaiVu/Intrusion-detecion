import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import pandas as pd


def Read_Dataset(cols=None):

    df1 = pd.read_csv(r"data/MachineLearningCSV/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    df2 = pd.read_csv(r"data/MachineLearningCSV/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
    df3 = pd.read_csv(r"data/MachineLearningCSV/Friday-WorkingHours-Morning.pcap_ISCX.csv")
    # df4 = pd.read_csv(r"data/MachineLearningCSV/Monday-WorkingHours.pcap_ISCX.csv")
    df5 = pd.read_csv(r"data/MachineLearningCSV/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
    df6 = pd.read_csv(r"data/MachineLearningCSV/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
    df7 = pd.read_csv(r"data/MachineLearningCSV/Tuesday-WorkingHours.pcap_ISCX.csv")
    df8 = pd.read_csv(r"data/MachineLearningCSV/Wednesday-workingHours.pcap_ISCX.csv")

    df = pd.concat([df1, df2, df3, df5, df6, df7, df8])
    return df


def main():

    df = Read_Dataset()
    df = df.drop_duplicates()

    print("Shape of Dataframe: {}".format(df.shape))
    df.to_csv(r'data/intrusion_detection_dataset.csv', index=False)

    print("Num of label: {}".format(len(df[' Label'].unique())))


main()