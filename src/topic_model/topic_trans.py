import pandas as pd
import os
import numpy as np


def train_index(train_dir, dev_dir, test_dir):

    df_train = pd.read_csv(train_dir)
    print(df_train.shape)
    df_dev = pd.read_csv(dev_dir)
    print(df_dev.shape)
    df = pd.concat([df_train, df_dev])
    print(df.shape)
    df = df[["Participant_ID", "Gender"]].copy()

    print(df.shape)

    return df


def topic_selection(transcript):
    interest = [
        "recently that you really enjoy",
        "traveling",
        "travel alot",
        "family",
        "fun",
        "best friend",
        "weekend",
    ]
    sleep = ["good night's sleep", "don't sleep well"]
    feeling_depressed = [
        "really happy",
        "behavior",
        " disturbing thought",
        "feel_lately",
    ]
    failure = ["regret", "guilty", "proud", "being_parent", "best_quality"]
    personality = ["introvert", "shyoutgoing"]
    dignose = ["ptsd", "depression", "therapy is useful"]
    parent = [
        "hard_parent",
        "best_parent",
        "easy_parent",
        "your_kid",
        "differnet_parent",
    ]

    ques = [interest, sleep, feeling_depressed, failure, personality, dignose, parent]
    topic_name = []
    for topic_count, topic in enumerate(ques):
        for sub_topic_count, sub_topic in enumerate(topic):
            # remove nan
            if type(transcript) == float:
                print(transcript)
                return "problem"
            if type(transcript) != float:
                if sub_topic in transcript:

                    topic_name.append([topic_count, sub_topic_count])

    return topic_name


def data_retrieve(working_dir, train_id):

    participants = train_id
    transcripts = pd.DataFrame()

    for index_p, row in participants.iterrows():
        #                print(int(row.Participant_ID))
        filename = str(int(row.Participant_ID)) + "_TRANSCRIPT.csv"

        location = os.path.join(working_dir, filename)
        temp = pd.read_csv(location, sep="\t")
        # remove nan
        temp = temp.dropna(subset=["value"])
        temp["topic"] = np.nan
        temp["topic_value"] = np.nan
        temp["sub_topic"] = np.nan
        temp["participant"] = row.Participant_ID
        #               temp['Gender']=row.Gender

        for index_t, row_t in temp.iterrows():
            if row_t.speaker == "Ellie":
                topic = topic_selection(row_t.value)

                if topic != [] and len(topic) > 1:
                    print(filename)
                    df_try = row_t
                    temp.append([df_try] * len(topic), ignore_index=True)
                    print(temp)

                for words in topic:
                    check = False
                    if temp["speaker"][index_t + 1] == "Participant":
                        temp["topic"][index_t + 1] = words[0]
                        temp["sub_topic"][index_t + 1] = words[1]
                        temp["topic_value"][index_t + 1] = row_t.value

                    if temp["speaker"].iloc[index_t + 2] == "Participant":
                        check = True
                        temp["topic"][index_t + 2] = words[0]
                        temp["sub_topic"][index_t + 2] = words[1]
                        temp["topic_value"][index_t + 2] = row_t.value

                    if temp["speaker"][index_t + 3] == "Participant" and check:
                        temp["topic"][index_t + 3] = words[0]
                        temp["sub_topic"][index_t + 3] = words[1]
                        temp["topic_value"][index_t + 2] = row_t.value

        temp.dropna(inplace=True)
        transcripts = pd.concat([transcripts, temp], axis=0)
        print(transcripts.shape)

    print(transcripts.shape)
    return transcripts


if __name__ == "__main__":

    train_dir = "../data/depression_data/train_split_Depression_AVEC2017.csv"
    dev_dir = "../data/depression_data/dev_split_Depression_AVEC2017.csv"
    test_dir = "../data/depression_data/test_split_Depression_AVEC2017.csv"
    train_id = train_index(train_dir, dev_dir, test_dir)

    working_dir = "../data/raw_data/transcripts/"
    transcripts = data_retrieve(working_dir, train_id)

    transcripts.to_csv("transcripts_check.csv", index=False, sep="\t", encoding="utf-8")
