import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count


wave_scalar_potential = 2000E0 # [V]
initial_wave_phase = 0E0 # [deg]
gradient_parameter = 2E0 # []
wave_threshold = 5E0 # [deg]

wavekind_list = [r'EparaBpara', r'Epara']

filename_base_1 = f'/mnt/j/KAW_simulation_data/single_test_particle/double_wave_packet/0.15Hz/results_particle_{str(int(wave_scalar_potential))}V' \
        + f'_gradient_{int(gradient_parameter)}_threshold_{int(wave_threshold)}_wavephase_{int(initial_wave_phase)}_{wavekind_list[0]}'

filename_base_2 = f'/mnt/j/KAW_simulation_data/single_test_particle/double_wave_packet/0.15Hz/results_particle_{str(int(wave_scalar_potential))}V' \
        + f'_gradient_{int(gradient_parameter)}_threshold_{int(wave_threshold)}_wavephase_{int(initial_wave_phase)}_{wavekind_list[1]}'

Data_energy_1 = {
    "initial pitch angle": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85], # [deg]
    "100": np.zeros(17),
    "200": np.zeros(17),
    "300": np.zeros(17),
    "400": np.zeros(17),
    "500": np.zeros(17),
    "600": np.zeros(17),
    "700": np.zeros(17),
    "800": np.zeros(17),
    "900": np.zeros(17),
    "1000": np.zeros(17),
}

Data_energy_1 = pd.DataFrame(Data_energy_1)

Data_equatorial_pitch_angle_1 = {
    "initial pitch angle": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85], # [deg]
    "100": np.zeros(17),
    "200": np.zeros(17),
    "300": np.zeros(17),
    "400": np.zeros(17),
    "500": np.zeros(17),
    "600": np.zeros(17),
    "700": np.zeros(17),
    "800": np.zeros(17),
    "900": np.zeros(17),
    "1000": np.zeros(17),
}

Data_equatorial_pitch_angle_1 = pd.DataFrame(Data_equatorial_pitch_angle_1)

Data_energy_2 = {
    "initial pitch angle": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85], # [deg]
    "100": np.zeros(17),
    "200": np.zeros(17),
    "300": np.zeros(17),
    "400": np.zeros(17),
    "500": np.zeros(17),
    "600": np.zeros(17),
    "700": np.zeros(17),
    "800": np.zeros(17),
    "900": np.zeros(17),
    "1000": np.zeros(17),
}

Data_energy_2 = pd.DataFrame(Data_energy_2)

Data_equatorial_pitch_angle_2 = {
    "initial pitch angle": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85], # [deg]
    "100": np.zeros(17),
    "200": np.zeros(17),
    "300": np.zeros(17),
    "400": np.zeros(17),
    "500": np.zeros(17),
    "600": np.zeros(17),
    "700": np.zeros(17),
    "800": np.zeros(17),
    "900": np.zeros(17),
    "1000": np.zeros(17),
}

Data_equatorial_pitch_angle_2 = pd.DataFrame(Data_equatorial_pitch_angle_2)

Data_energy_dif = {
    "initial pitch angle": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85], # [deg]
    "100": np.zeros(17),
    "200": np.zeros(17),
    "300": np.zeros(17),
    "400": np.zeros(17),
    "500": np.zeros(17),
    "600": np.zeros(17),
    "700": np.zeros(17),
    "800": np.zeros(17),
    "900": np.zeros(17),
    "1000": np.zeros(17),
}

Data_energy_dif = pd.DataFrame(Data_energy_dif)

Data_equatorial_pitch_angle_dif = {
    "initial pitch angle": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85], # [deg]
    "100": np.zeros(17),
    "200": np.zeros(17),
    "300": np.zeros(17),
    "400": np.zeros(17),
    "500": np.zeros(17),
    "600": np.zeros(17),
    "700": np.zeros(17),
    "800": np.zeros(17),
    "900": np.zeros(17),
    "1000": np.zeros(17),
}

Data_equatorial_pitch_angle_dif = pd.DataFrame(Data_equatorial_pitch_angle_dif)


def main(count_energy, count_pitch_angle):
    print(count_energy, count_pitch_angle)
    count_all = count_energy * 17 + count_pitch_angle + 1
    count_front = (count_all-1) // 5
    filename_1 = f'{filename_base_1}/myrank000/particle_trajectory{str(count_front).zfill(2)}-{str(count_all).zfill(3)}.dat'
    filename_2 = f'{filename_base_2}/myrank000/particle_trajectory{str(count_front).zfill(2)}-{str(count_all).zfill(3)}.dat'
    data_1 = np.genfromtxt(filename_1, dtype=np.float64)
    data_2 = np.genfromtxt(filename_2, dtype=np.float64)
    energy_1 = data_1[-1, 6]
    equatorial_pitch_angle_1 = data_1[-1, 7]
    energy_2 = data_2[-1, 6]
    equatorial_pitch_angle_2 = data_2[-1, 7]
    return count_energy, count_pitch_angle, energy_1, equatorial_pitch_angle_1, energy_2, equatorial_pitch_angle_2

if __name__ == '__main__':
    num_processes = cpu_count()
    
    with Pool(num_processes) as p:
        results = []
        for count_energy in range(10):
            for count_pitch_angle in range(17):
                result = p.apply_async(main, args=(count_energy, count_pitch_angle))
                results.append(result)
        for result in results:
            count_energy, count_pitch_angle, energy_1, equatorial_pitch_angle_1, energy_2, equatorial_pitch_angle_2 = result.get()
            Data_energy_1.at[count_pitch_angle, str(count_energy * 100 + 100)] = energy_1
            Data_equatorial_pitch_angle_1.at[count_pitch_angle, str(count_energy * 100 + 100)] = equatorial_pitch_angle_1
            Data_energy_2.at[count_pitch_angle, str(count_energy * 100 + 100)] = energy_2
            Data_equatorial_pitch_angle_2.at[count_pitch_angle, str(count_energy * 100 + 100)] = equatorial_pitch_angle_2
            Data_energy_dif.at[count_pitch_angle, str(count_energy * 100 + 100)] = energy_1 - energy_2
            Data_equatorial_pitch_angle_dif.at[count_pitch_angle, str(count_energy * 100 + 100)] = equatorial_pitch_angle_1 - equatorial_pitch_angle_2
            print(count_energy, count_pitch_angle, Data_energy_1.at[count_pitch_angle, str(count_energy * 100 + 100)], Data_equatorial_pitch_angle_1.at[count_pitch_angle, str(count_energy * 100 + 100)], \
                Data_energy_2.at[count_pitch_angle, str(count_energy * 100 + 100)], Data_equatorial_pitch_angle_2.at[count_pitch_angle, str(count_energy * 100 + 100)], \
                Data_energy_dif.at[count_pitch_angle, str(count_energy * 100 + 100)], Data_equatorial_pitch_angle_dif.at[count_pitch_angle, str(count_energy * 100 + 100)])


    Data_energy_dif.to_csv(f'{filename_base_1}/difference_energy_{wavekind_list[0]}_{wavekind_list[1]}.csv', index=False)
    Data_equatorial_pitch_angle_dif.to_csv(f'{filename_base_1}/difference_equatorial_pitch_angle_{wavekind_list[0]}_{wavekind_list[1]}.csv', index=False)
    Data_energy_1.to_csv(f'{filename_base_1}/energy_{wavekind_list[0]}.csv', index=False)
    Data_equatorial_pitch_angle_1.to_csv(f'{filename_base_1}/equatorial_pitch_angle_{wavekind_list[0]}.csv', index=False)
    Data_energy_2.to_csv(f'{filename_base_2}/energy_{wavekind_list[1]}.csv', index=False)
    Data_equatorial_pitch_angle_2.to_csv(f'{filename_base_2}/equatorial_pitch_angle_{wavekind_list[1]}.csv', index=False)
    print(r'finish')
    quit()