import glob
import pandas as pd
import os
import numpy as np

# time_stepの指定
time_step = 200000

elementary_charge = 1.60217662E-19
electron_mass = 9.10938356E-31
moment = 7.75E22
l_shell = 9E0
planet_radius = 6.371E6
B0_eq = (1E-7 * moment) / (l_shell * planet_radius)**3E0 # [T]
omega_ce = elementary_charge * B0_eq / electron_mass # [rad/s]

time_step_omega_ce = time_step / omega_ce
formatted_number = format(time_step_omega_ce, ".6E")  # 指数表記に変換

print(formatted_number) # 確認用


# ディレクトリresults_particleの中のmyrank*ディレクトリを検索
dir_name = '/mnt/j/KAW_simulation_data/distribution_function/double_wave_packet/simulation_set_600V_gradient_1_threshold_5_wavephase_270_Epara/results_particle/'

myrank_dirs = glob.glob(f'{dir_name}myrank*')

# 結果を格納するための空のデータフレームを作成
result_df = pd.DataFrame()

for myrank_dir in myrank_dirs:
    files = glob.glob(os.path.join(myrank_dir, 'count_at_equator*.dat'))
    
    for file in files:
        if os.stat(file).st_size > 0:
            data = pd.read_csv(file, delimiter='\s+', header=None)

            # 1列目の値と指定した値が指数表記で小数点6位まで一致する行を抽出
            selected_data = data[np.isclose(data[0].astype(float), float(formatted_number), rtol=1e-6, atol=1e-6)]

            result_df = pd.concat([result_df, selected_data], ignore_index=True)
        else:
            print(f"Empty file: {file}")

# 結果をカンマ区切りでファイルに保存
result_df.to_csv(f'{dir_name}result_{str(formatted_number)}.dat', sep=',', header=False, index=False)