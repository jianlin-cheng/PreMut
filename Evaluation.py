import os
import subprocess
import csv
from collections import defaultdict
import re
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='Evaluation of Performance of PreMut')
parser.add_argument('dataset', help='name of the dataset, either MutData2022 or MutData2023')
args = parser.parse_args()

if args.dataset == 'MutData2022':
    test_data__cluster_lst = [['3LL2_A'], ['6JZZ_A'], ['4Y2C_A', '3KMQ_A', '3KMS_A', '4WZQ_A', '3KNA_A', '3KOA_A', '4ZP9_A', '4ZPA_A', '4ZPB_A', '4ZPC_A', '4ZPD_A', '4WFX_A', '4WFY_A'], ['1A40_A', '1A54_A', '1QUK_A', '1A55_A', '1QUL_A', '1IXG_A', '1IXI_A'], ['2PVQ_A'], ['1B41_B'], ['2EIB_A', '2EIC_A', '2EID_A'], ['1K0I_A', '1K0J_A', '1K0L_A'], ['4TTO_A', '4TTN_A'], ['2B5W_A'], ['1VEN_A', '1VEO_A', '1ITC_A'], ['1KVA_A', '1KVB_A', '1KVC_A', '3AA3_A'], ['1WTX_A', '1WTR_A', '1WTW_A'], ['1J47_A'], ['3F45_A'], ['1G2U_A', '1GC9_A'], ['3ZI8_A'], ['2HMJ_A', '2HML_A', '2HMN_A'], ['1KVW_A', '1KVX_A', '1KVY_A', '2ZP4_A', '2ZP5_A', '1N29_A'], ['6Z28_A', '7ARU_A', '6F6Q_A', '6F75_A', '6F79_A', '3V38_A', '6Q4L_A'], ['3NYT_A', '3NYS_A'], ['2GB2_A', '2GBA_A', '3IE9_A', '3IEA_A', '4P5R_A', '1SF3_A', '2IDQ_A', '4P5S_A', '2IDS_A']]
elif args.dataset == 'MutData2023':
    test_data__cluster_lst = [['8EET_A'], ['1G1H_A', '4QBE_A', '1G1G_A', '7S4F_A'], ['1L07_A', '1L33_A', '1L24_A', '1L11_A', '1L53_A', '1L23_A', '1L47_A', '1L03_A', '1L15_A', '1L37_A', '1L31_A', '1L09_A', '1L45_A', '1L17_A', '1L16_A', '1L38_A', '1L20_A', '1L13_A', '1L04_A', '1L25_A', '1L05_A', '1L30_A', '1L43_A', '1L28_A', '1L42_A', '1L48_A', '1L14_A', '1L29_A', '1L44_A', '1L34_A', '1L27_A', '1L32_A', '1L52_A', '1L18_A', '1L26_A', '1L46_A', '1L19_A', '1L08_A', '1L06_A', '1L22_A', '1L02_A', '1L12_A'], ['8EJ9_A', '8DFE_A', '8EHL_A', '7TVS_A', '8DI3_A', '8DKJ_A', '7MGR_A'], ['7ZDR_C', '7ZDW_C'], ['7XC9_A', '7XCQ_A'], ['8CX4_A', '3FT3_A', '3FT2_A', '7N2O_A'], ['2JQX_A'], ['8J0U_A', '8I81_A', '8I77_A'], ['4A84_A'], ['1LOU_A'], ['7U2F_A', '3FQY_A'], ['5AF5_A', '8E7O_D', '7S6O_A']]
print('Evaluating on {0}'.format(args.dataset))
def parse_gdt_ha_score(text):
    gdt_ha_score = None
    pattern = r'GDT-HA-score= ([0-9.]+)'

    match = re.search(pattern, text)
    if match:
        gdt_ha_score = float(match.group(1))
    
    return gdt_ha_score
def get_metrics(prediction_path,ground_truth_path):
    command = './TMalign {0} {1} -outfmt 2'.format(prediction_path,ground_truth_path)
    result = subprocess.check_output(command, shell=True)
    result = result.decode()
    try:
        tmscore = float(result.split('\n')[1].split('\t')[3])
        
    except:
        tmscore = float(result.split('\n')[2].split('\t')[3])
    command = './TMscore {0} {1}'.format(prediction_path,ground_truth_path)
    result = subprocess.check_output(command, shell=True)
    result = result.decode()

    gdt_ha_score = parse_gdt_ha_score(result)
    return [tmscore, gdt_ha_score]
def parse_specs_score(text):
    lines = text.split('\n')
    for line in lines:
        if 'SPECS-score' in line:
            specs_score = line.split()[2]
            return float(specs_score)
def use_specs(prediction_path,ground_truth_path):
    command = './SPECS -m {0} -n {1}'.format(prediction_path,ground_truth_path)
    result = subprocess.check_output(command, shell=True)
    result = result.decode()
    return parse_specs_score(result)
                                             
                                             


def get_per_cluster_average(dict_):
    lst = []
    for key,items in dict_.items():
        lst.append(sum(items)/len(items))
    return sum(lst)/len(lst)
def get_per_target_average(dict_):
    lst = []
    for key,items in dict_.items():
        for item in items:
            lst.append(item)
    return sum(lst)/len(lst)
def sort_keys(myDict):
    myKeys = list(myDict.keys())
    myKeys.sort()
    sorted_dict = {i: myDict[i] for i in myKeys}
    return sorted_dict


def display_all_scores(score_name,premut_cluster_results_gdthascore,premut_cluster_results_gdthascore_refined,alphafold_cluster_results_gdthascore):
    print('Per Cluster Average {0}'.format(score_name))
    print('PreMut')
    print(get_per_cluster_average(premut_cluster_results_gdthascore))
   
    print('PreMut Refined')
    print(get_per_cluster_average(premut_cluster_results_gdthascore_refined))
   
    print('AlphaFold')
    print(get_per_cluster_average(alphafold_cluster_results_gdthascore))
    
    print('Per Target Average {0}'.format(score_name))
    print('PreMut')

    print(get_per_target_average(premut_cluster_results_gdthascore))
   
    print('PreMut Refined')
    print(get_per_target_average(premut_cluster_results_gdthascore_refined))
    
    print('AlphaFold')
    print(get_per_target_average(alphafold_cluster_results_gdthascore))

def save_dict_to_csv(my_dict, column_names, csv_file):
   
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the header row with specified column names
        writer.writerow(column_names)

        # Write the dictionary values to the CSV file
        for key, values in my_dict.items():
            writer.writerow([key] + values)
def get_results(AlphaFold_dir = 'AlphaFold_Predictions_on_MutData2022_test',PreMut_dir='PreMut_Predictions_on_MutData2022_test',ground_truth_dir='ground_truth_MutData2022_test',PreMut_Refined_dir='PreMut_Refined_Predictions_on_MutData2022_test',dataset='MutData2022'):
    

    alphafold_cluster_results = defaultdict(list)
    alphafold_cluster_results_rmsd = defaultdict(list)
    alphafold_cluster_results_gdthascore = defaultdict(list)
    alphafold_cluster_results_specs = defaultdict(list)


    premut_cluster_results = defaultdict(list)
    premut_cluster_results_rmsd = defaultdict(list)
    premut_cluster_results_gdthascore = defaultdict(list)
    premut_cluster_results_specs = defaultdict(list)

    

    premut_cluster_results_refined = defaultdict(list)
    premut_cluster_results_rmsd_refined = defaultdict(list)
    premut_cluster_results_gdthascore_refined = defaultdict(list)
    premut_cluster_results_specs_refined = defaultdict(list)
    tmscore_dict = defaultdict(list)
    gdtha_score_dict = defaultdict(list)
    specs_score_dict = defaultdict(list)
    for files in os.listdir(AlphaFold_dir):
        if files.endswith('.pdb'):
            # print(files)
            alphafold_prediction_path = os.path.join(AlphaFold_dir,files)
            tmp = files.split('_')
            name = tmp[0] +'_'+ tmp[1] +'_'+ tmp[2] +'_'+ tmp[3] +'_'+ tmp[4] +'_'+ tmp[5] +'_'+ tmp[6]
            ground_truth_path = os.path.join(ground_truth_dir,name+'_groundtruth.pdb')
            mutant_name = tmp[5]
            # print(name)
            premut_prediction_path = os.path.join(PreMut_dir,name+'_predicted.pdb')
            premut_prediction_path_refined = os.path.join(PreMut_Refined_dir,name+'_predicted.pdb')

            ground_truth_path = os.path.join(ground_truth_dir,name+'_groundtruth.pdb')
            for i,items in enumerate(test_data__cluster_lst):
                
                if tmp[5]+'_'+tmp[6] in items:
                    # print(i+1)
                    cluster_num = i+1
            



                
            alphafold_tmscore,  alphafold_gdtha = get_metrics(prediction_path=alphafold_prediction_path,ground_truth_path=ground_truth_path)
            premut_tmscore,  premut_gdtha = get_metrics(prediction_path=premut_prediction_path,ground_truth_path=ground_truth_path)
            premut_tmscore_refined,  premut_gdtha_refined = get_metrics(prediction_path=premut_prediction_path_refined,ground_truth_path=ground_truth_path)

            alphafold_specs = use_specs(prediction_path=alphafold_prediction_path,ground_truth_path=ground_truth_path)
            premut_specs = use_specs(prediction_path=premut_prediction_path,ground_truth_path=ground_truth_path)
            premut_specs_refined = use_specs(prediction_path=premut_prediction_path_refined,ground_truth_path=ground_truth_path)
            tmscore_dict[name] = [alphafold_tmscore,premut_tmscore,premut_tmscore_refined]
            gdtha_score_dict[name] = [alphafold_gdtha,premut_gdtha,premut_gdtha_refined]
            specs_score_dict[name] = [alphafold_specs,premut_specs,premut_specs_refined]

            alphafold_cluster_results[cluster_num].append(alphafold_tmscore)
            alphafold_cluster_results_gdthascore[cluster_num].append(alphafold_gdtha)
            alphafold_cluster_results_specs[cluster_num].append(alphafold_specs)

            premut_cluster_results[cluster_num].append(premut_tmscore)
            premut_cluster_results_gdthascore[cluster_num].append(premut_gdtha)
            premut_cluster_results_specs[cluster_num].append(premut_specs)


            premut_cluster_results_refined[cluster_num].append(premut_tmscore_refined)
            premut_cluster_results_gdthascore_refined[cluster_num].append(premut_gdtha_refined)
            premut_cluster_results_specs_refined[cluster_num].append(premut_specs_refined)


    save_dict_to_csv(my_dict=tmscore_dict,column_names=['Data', 'AlphaFold', 'PreMut', 'PreMut_Refined'],csv_file='TMscores_{0}.csv'.format(dataset))
    save_dict_to_csv(my_dict=gdtha_score_dict,column_names=['Data', 'AlphaFold', 'PreMut', 'PreMut_Refined'],csv_file='GDTHAscores_{0}.csv'.format(dataset))
    save_dict_to_csv(my_dict=specs_score_dict,column_names=['Data', 'AlphaFold', 'PreMut', 'PreMut_Refined'],csv_file='SPECS_{0}.csv'.format(dataset))

    display_all_scores(score_name='GDT-HA Score',
    premut_cluster_results_gdthascore=premut_cluster_results_gdthascore,
    premut_cluster_results_gdthascore_refined=premut_cluster_results_gdthascore_refined,
    alphafold_cluster_results_gdthascore=alphafold_cluster_results_gdthascore)

    display_all_scores(score_name='TM Score',
    premut_cluster_results_gdthascore=premut_cluster_results,
    premut_cluster_results_gdthascore_refined=premut_cluster_results_refined,
    alphafold_cluster_results_gdthascore=alphafold_cluster_results)

    display_all_scores(score_name='SPECS',
    premut_cluster_results_gdthascore=premut_cluster_results_specs,
    premut_cluster_results_gdthascore_refined=premut_cluster_results_specs_refined,
    alphafold_cluster_results_gdthascore=alphafold_cluster_results_specs)

    
    
    


            

if args.dataset == 'MutData2022':
    get_results()
elif args.dataset == 'MutData2023':
    get_results(AlphaFold_dir='AlphaFold_Predictions_on_MutData2023_test',PreMut_dir='PreMut_Predictions_on_MutData2023_test',ground_truth_dir='ground_truth_MutData2023_test',PreMut_Refined_dir='PreMut_Refined_Predictions_on_MutData2023_test',dataset=args.dataset)

