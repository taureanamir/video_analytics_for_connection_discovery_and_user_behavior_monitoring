import os
import random
import shutil
import math

class SeparateTrainTest:

    def __init__(self, source_path, destination_path, test_set_percent):
        self.source_path = source_path
        self.destination_path = destination_path
        self.test_set_percent = test_set_percent


    def separate(self):
        self._separate_train_test()


    def _separate_train_test(self):
        train = "train/"
        test = "test/"

        final_destination_train = self.destination_path + train
        if not os.path.exists(final_destination_train):
            os.makedirs(final_destination_train)

        for name in os.listdir(self.source_path):
            if name == ".DS_Store":
                continue

            a = []
            for file in os.listdir(self.source_path + name):
                if file == ".DS_Store":
                    continue
                a.append(file)

            final_destination_test = self.destination_path + test + name
            if not os.path.exists(final_destination_test):
                os.makedirs(final_destination_test)

            # print("-----------AAAA------------", a)
            num_of_files = math.floor(len(a) * (test_set_percent/100))
            print(a)
            print("Num of files: ", num_of_files)
            for i in range(num_of_files):
                # print ("-----------------------", a)
                temp = random.choice(a)
                # print("# TEMP: ", temp)
                # print("source ", self.source_path + name)
                # print("destination ", self.destination_path + name)
                shutil.move(self.source_path + name + "/" + temp, final_destination_test)
                del a[a.index(temp)]  # deletes the previously selected file.

            shutil.move(self.source_path + name, final_destination_train)


if __name__ == '__main__':
    source_path = "/mnt/drive/Amir/Thesis/dataset/face-recognition/Final-face-dataset-no-train-test-separation/"
    destination_path = "/mnt/drive/Amir/Thesis/dataset/face-recognition/Final-face-dataset/"
    test_set_percent=20
    # img_ext = ".png"
    separate = SeparateTrainTest(source_path, destination_path,test_set_percent)
    separate.separate()
