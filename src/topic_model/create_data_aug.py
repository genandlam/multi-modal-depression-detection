import pandas as pd
import os
from pydub import AudioSegment
import os
import numpy as np
import pandas as pd
import os
import random

np.random.seed(15)  # for reproducibility
import os
import matplotlib

matplotlib.use("Agg")  # No pictures displayed
import pylab
import librosa
import librosa.display
import numpy as np
from itertools import combinations


topic_to_subtopic_to_category = {
    0: {
        0: "did_recently",
        1: "enjoy_travelling",
        3: "family_relationship",
        4: "do_for_fun",
        5: "best_friend",
        6: "ideal_weekend",
    },
    1: {0: "easy_sleep", 1: "sleep_badly"},
    2: {
        0: "happy_last_time",
        1: "behaviour_changes",
        2: "disturbing_thoughts",
        3: "feel_lately",
    },
    3: {
        0: "any_regret",
        1: "feel_guilty",
        2: "most_proud",
    },
    4: {
        0: "introvert",
        1: "shy_outgoing",
    },
    5: {
        0: "ptsd_diagnosed",
        1: "depression_diagnosed",
        2: "therapy_useful",
    },
    6: {2: "easy_parent"},
}


def preprocess(text):
    return text


# need to edit return function
def segment_all():

    transcripts = pd.read_csv("transcripts_check.csv", sep="\t")

    participant_audio_text = {}
    prev_topic = ""
    prev_subtopic = ""
    prev_participant = -1

    for idx, row in transcripts.iterrows():
        participant = row.participant
        topic = int(row.topic)
        subtopic = int(row.sub_topic)
        text = row.value
        startt = row.start_time
        stopt = row.stop_time

        if participant not in participant_audio_text:
            # Create blank entry for new participant
            participant_audio_text[participant] = ["", [], []]

        if (
            participant == prev_participant
            and topic == prev_topic
            and subtopic == prev_subtopic
        ):
            # If previous participant and topic+subtopic, don't pad special tokens
            proc_text = preprocess(text) + " "
            participant_audio_text[participant][0] += proc_text
            participant_audio_text[participant][1][-1] += proc_text
            participant_audio_text[participant][2][-1] += (
                "," + str(startt) + "," + str(stopt)
            )
        else:
            # If different topic+subtopic, pad special token infront of text before appending to full text and topic-wise text
            proc_text = (
                topic_to_subtopic_to_category[topic][subtopic]
                + " "
                + preprocess(text)
                + " "
            )
            participant_audio_text[participant][0] += proc_text
            participant_audio_text[participant][1].append(proc_text)
            participant_audio_text[participant][2].append(
                str(startt) + "," + str(stopt)
            )

        prev_participant = participant
        prev_topic = topic
        prev_subtopic = subtopic

    return participant_audio_text


def segment_want(df, participant_id, counter):

    combined = AudioSegment.empty()
    path = "../data/raw_data/audio/"
    out_dir = "../data/raw_data/data_aug_audio/"

    for index3 in df:

        index2 = index3.split(",")
        for count, item in enumerate(index2):
            if count % 2 == 0:
                t1 = item
                print(count)
                len(index2)
                t2 = index2[count + 1]

                # TIMESTAMP IS IN SECONDS DF
                t1 = float(t1) * 1000  # Works in milliseconds
                t2 = float(t2) * 1000

                # Obtaining audio file of full audio.wav
                file_name = str(participant_id) + "_AUDIO.wav"
                newAudio = AudioSegment.from_wav(os.path.join(path, file_name))
                newAudio1 = newAudio[t1:t2]
                combined += newAudio1
                file_new_name = str(participant_id) + str(counter) + "_AUDIO.wav"
                print(file_new_name)

    combined.export(os.path.join(out_dir, file_new_name), format="wav")


# fft size / sampling rate = __hz (resolution of each spectral line)
def mel_spec(working_dir):

    y, sr = librosa.load(working_dir)
    S = librosa.feature.melspectrogram(y=y, sr=sr)

    # Convert to log scale (dB). We'll use the peak power as reference.
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    t_mel = np.transpose(log_S)

    return t_mel


def build_class_dictionaries(dir_name, partic_id, counter):

    normal_audio_dict = dict()
    mat = mel_spec(dir_name + str(participant_id) + str(counter) + "_AUDIO.wav")

    return mat


def creating_aug():

    participant_audio_text = segment_all()
    min_len = {
        0: 10,
        1: 5,
    }  # Minimum length of a transcript above which we can do augmentation
    aug_count = {0: 3, 1: 8}  # Number of augmented transcripts to be created
    data_train = {"Text": [], "Targets": []}

    # data_test = {"Text": [], "Targets": []}
    df_topic = pd.read_csv("transcripts_check.csv", sep="\t")
    true_value = []

    scores_train = pd.read_csv(
        "../data/depression_data/train_split_Depression_AVEC2017.csv"
    )
    # participants id
    participants = scores_train.Participant_ID.unique()

    scores = scores_train.set_index("Participant_ID")
    scores = scores["PHQ8_Binary"]
    audio_train = []

    for participant_id in participants:

        participant_df = df_topic[(df_topic["participant"] == participant_id)]
        #   print('first {}'.format(len(participant_audio_text[participant_id][1])))

        if (
            len(participant_audio_text[participant_id][1])
            > min_len[scores[participant_id]]
        ):
            counter = 0
            t_lens = np.random.randint(
                low=min_len[scores[participant_id]],
                high=len(participant_audio_text[participant_id][1]),
                size=aug_count[scores[participant_id]],
            )
            print("t_lens{}".format(t_lens))

            for t_len in t_lens:
                # Generate list of all combinations of topic texts of t_len
                high = len(participant_audio_text[participant_id][1])

                combs = list(combinations(list(range(0, high)), t_len))
                # Select a random combination
                print("length of array {}".format(high))

                t_comb = list(combs[np.random.randint(len(combs))])
                # Shuffle the topic texts in selected combination
                np.random.shuffle(t_comb)
                # arrangement of topics
                print(t_comb)
                # Add augmented transcript
                temp = []
                temp_audio = []
                for item in t_comb:
                    print(participant_id)
                    print(item)
                    print(participant_audio_text[participant_id][2][item])
                    temp_audio.append(participant_audio_text[participant_id][2][item])
                    temp.append(participant_audio_text[participant_id][1][item])

                segment_want(temp_audio, participant_id, counter)
                data_train["Text"].append(" ".join(temp))
                data_train["Targets"].append(scores[participant_id])

                mat = build_class_dictionaries(
                    "../data/raw_data/data_aug_audio/", participant_id, counter
                )
                audio_train.append(mat)

                counter += 1

        return audio_train, data_train


if __name__ == "__main__":

    audio_train, data_train = creating_aug()
    print("Saving npz file locally...")
    np.savez("../data/processed_data_aug/train_samples.npz", audio_train)
    np.savez("../data/processed_data_aug/train_labels.npz", data_train["Targets"])

    audio_train, data_train = creating_aug()
    np.savez("../data/processed_data_aug/train_samples3.npz", audio_train)
    np.savez("../data/processed_data_aug/train_labels3.npz", data_train["Targets"])
