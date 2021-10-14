import yaml
import io

with open("example.yaml", 'r') as stream:
    base_yaml = yaml.safe_load(stream)

behav_measures = [
    'Gender',
    'Flanker_Unadj',
    'WM_Task_Acc',
    'PMAT24_A_CR',
    'MMSE_Score',
    'PSQI_AmtSleep',
    'PicSeq_Unadj',
    'ReadEng_Unadj',
    'PicVocab_Unadj',
    'DDisc_AUC_200',
    'DDisc_AUC_40K',
    'VSPLOT_CRTE',
    'SCPT_SEN',
    'SCPT_SPEC',
    'IWRD_TOT',
    'IWRD_RTC',
    'ER40ANG',
    'ER40HAP',
    'ER40FEAR',
    'ER40SAD',
    'ER40NOE',
    'AngAffect_Unadj',
    'AngAggr_Unadj',
    'AngHostil_Unadj',
    'FearAffect_Unadj',
    'FearSomat_Unadj',
    'Sadness_Unadj',
    'LifeSatisf_Unadj',
    'PosAffect_Unadj',
    'Friendship_Unadj',
    'Loneliness_Unadj',
    'PercHostil_Unadj',
    'PercReject_Unadj',
    'PercStress_Unadj',
    'EmotSupp_Unadj',
    'InstruSupp_Unadj',
    'SelfEff_Unadj',
    'Emotion_Task_Acc',
    'Language_Task_Acc',
    'Relational_Task_Acc',
    'Social_Task_Perc_NLR',
    'Social_Task_Perc_Random',
    'Social_Task_Perc_TOM',
    'Social_Task_Perc_Unsure',
    'NEOFAC_A',
    'NEOFAC_C',
    'NEOFAC_E',
    'NEOFAC_N',
    'NEOFAC_O']


for measure in behav_measures:
    base_yaml['exp_dir'] = 'exp/Liangwei/' + measure
    base_yaml['dataset']['name'] = measure
    base_yaml['device'] = 'cuda:0'
    base_yaml['test']['test_model_name'] = 'model_snapshot_' + measure + '.pth'
    base_yaml['dataset']['data_folder_path'] = '/home/yousefabadi/amir/prepared_data'
    base_yaml['dataset']['idx2names_path_train'] = '/home/yousefabadi/amir/data_idx2name_train.pkl'
    base_yaml['dataset']['idx2names_path_test'] = '/home/yousefabadi/amir/data_idx2name_test.pkl'
    base_yaml['train']['batch_size'] = 40
    base_yaml['train']['display_iter'] = 1
    with io.open('liangwei_' + measure + '.yaml', 'w', encoding='utf8') as outfile:
        yaml.dump(base_yaml, outfile, default_flow_style=False, allow_unicode=True)
