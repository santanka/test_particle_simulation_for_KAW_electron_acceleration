import os

# 入力ファイル名
input_filename = "/home/satanka/Documents/test_particle_simulation_for_KAW_electron_acceleration/distribution_function/double_wave_packet/initial_distribution/initial_condition.dat"

# 出力ディレクトリ名
output_dirname = "/home/satanka/Documents/test_particle_simulation_for_KAW_electron_acceleration/distribution_function/double_wave_packet/initial_distribution/initial_condition"

# 出力ファイル名のフォーマット
output_filename_format = "initial_condition{:03d}.dat"

file_threads = 952

# 入力ファイルからデータを読み込む
with open(input_filename, 'r') as input_file:
    data = input_file.readlines()
    N = len(data)

# 1ファイルあたりのデータ数
data_per_file = N // file_threads + 1

# 出力ディレクトリを作成する
os.makedirs(output_dirname, exist_ok=True)

# データをファイルに書き出す
for i in range(file_threads):
    # 出力ファイル名を作成する
    output_filename = os.path.join(output_dirname, output_filename_format.format(i))

    # 出力ファイルを開く
    with open(output_filename, 'w') as output_file:
        # データを書き出す
        for j in range(i * data_per_file, min((i + 1) * data_per_file, N)):
            output_file.write(data[j])