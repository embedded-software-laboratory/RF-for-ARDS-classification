import numpy as np
mapping_extraction_lab_rev = {
'Respiratory Rate' : 0,
'Spontaneous Rate' : 1,
'Peak Airway/Pressure' : 2,
'paCO2_(ohne_Temp-Korrektur)' : 3,
'lactate' : 4,
'WBC x 1000' : 5,
'-lymphs' : 6,
'platelets x 1000' : 7,
'CRP' : 8,
'BUN' : 9,
'creatinine' : 10,
'BNP' : 11,
'total bilirubin' : 12,
'Hgb' : 13,
'Hct' : 14,
'albumin' : 15,
'AST (SGOT)' : 16,
'ALT (SGPT)' : 17,
'troponin - T' : 18,
'CPK' : 19,
'CPK-MB' : 20,
'LDH' : 21,
'amylase' : 22,
'PT - INR' : 23,
'pTT' : 24,
'lipase' : 25
}
mapping_extraction_lab = {
0 : 'Respiratory Rate',
1 : 'Spontaneous Rate',
2 : 'Peak Airway/Pressure',
3 : 'paCO2_(ohne_Temp-Korrektur)',
4 : 'lactate',
5 : 'WBC x 1000',
6 : '-lymphs',
7 : 'platelets x 1000',
8 : 'CRP',
9 : 'BUN',
10 : 'creatinine',
11 : 'BNP',
12 : 'total bilirubin',
13 : 'Hgb',
14 : 'Hct',
15 : 'albumin',
16 : 'AST (SGOT)',
17 : 'ALT (SGPT)',
18 : 'troponin - T',
19 : 'CPK',
20 : 'CPK-MB',
21 : 'LDH',
22 : 'amylase',
23 : 'PT - INR',
24 : 'pTT',
25 : 'lipase'
}
mapping_extraction_respcharting_rev = {
'TOTAL RR' : 0,
'RR Spont' : 1,
'RR (patient)' : 2,
'Spontaneous Respiratory Rate' : 3,
'Tidal Volume Observed (VT)' : 4,
'Exhaled TV (machine)' : 5,
'Exhaled Vt' : 6,
'Spont TV' : 7,
'Exhaled TV (patient)' : 8,
'Vt Spontaneous (mL)' : 9,
'PD I:E RATIO' : 10,
'TV/kg IBW' : 11,
'Plateau Pressure' : 12,
'Peak Insp. Pressure' : 13,
'Compliance' : 14,
'Static Compliance' : 15,
'EtCO2' : 16,
'SaO2' : 17
}
mapping_extraction_respcharting = {
0 : 'TOTAL RR',
1 : 'RR Spont',
2 : 'RR (patient)',
3 : 'Spontaneous Respiratory Rate',
4 : 'Tidal Volume Observed (VT)',
5 : 'Exhaled TV (machine)',
6 : 'Exhaled Vt',
7 : 'Spont TV',
8 : 'Exhaled TV (patient)',
9 : 'Vt Spontaneous (mL)',
10 : 'PD I:E RATIO',
11 : 'TV/kg IBW',
12 : 'Plateau Pressure',
13 : 'Peak Insp. Pressure',
14 : 'Compliance',
15 : 'Static Compliance',
16 : 'EtCO2',
17 : 'SaO2'
}
mapping_extraction_infusion_drug_rev = {
'Dobutamine (mcg/min)' : 0,
'dobutrex (mcg/kg/min)' : 0,
'dobutrex (mg/kg/min)' : 0,
'DOBUTamine MAX 1000 mg Dextrose 5% 250 ml  Premix (mcg/kg/min)' : 0,
'DOBUTamine STD 500 mg Dextrose 5% 250 ml  Premix (mcg/kg/min)' : 0,
'Epinepherine (mcg/min)' : 1,
'Epinephrine (mcg/hr)' : 1,
'Epinephrine (mcg/kg/min)' : 1,
'Epinephrine (mcg/min)' : 1,
'Epinephrine (mg/hr)' : 1,
'Epinephrine (mg/kg/min)' : 1,
'EPINEPHrine(Adrenalin)MAX 30 mg Sodium Chloride 0.9% 250 ml (mcg/min)' : 1,
'EPINEPHrine(Adrenalin)STD 4 mg Sodium Chloride 0.9% 250 ml (mcg/min)' : 1,
'EPINEPHrine(Adrenalin)STD 4 mg Sodium Chloride 0.9% 500 ml (mcg/min)' : 1,
'EPINEPHrine(Adrenalin)STD 7 mg Sodium Chloride 0.9% 250 ml (mcg/min)' : 1,
'Norepinephrine (mcg/min)' : 2,
'Norepinephrine (mcg/hr)' : 2,
'Norepinephrine (mcg/kg/min)' : 2,
'Norepinephrine (mg/hr)' : 2,
'Norepinephrine (mcg/kg/hr)' : 2,
'Norepinephrine (mg/kg/min)' : 2,
'Norepinephrine (mg/min)' : 2,
'Norepinephrine MAX 32 mg Dextrose 5% 250 ml (mcg/min)' : 2,
'Norepinephrine MAX 32 mg Dextrose 5% 500 ml (mcg/min)' : 2,
'Norepinephrine STD 32 mg Dextrose 5% 282 ml (mcg/min)' : 2,
'Norepinephrine STD 32 mg Dextrose 5% 500 ml (mcg/min)' : 2,
'Norepinephrine STD 4 mg Dextrose 5% 250 ml (mcg/min)' : 2,
'Norepinephrine STD 4 mg Dextrose 5% 500 ml (mcg/min)' : 2,
'Norepinephrine STD 8 mg Dextrose 5% 250 ml (mcg/min)' : 2,
'Norepinephrine STD 8 mg Dextrose 5% 500 ml (mcg/min)' : 2,
'Vasopressin 20 Units Sodium Chloride 0.9% 100 ml (units/hr)' : 3,
'Vasopressin 20 Units Sodium Chloride 0.9% 250 ml (units/hr)' : 3,
'Vasopressin 40 Units Sodium Chloride 0.9% 100 ml (units/hr)' : 3,
'Vasopressin 40 Units Sodium Chloride 0.9% 100 ml (units/kg/hr)' : 3,
'Vasopressin 40 Units Sodium Chloride 0.9% 100 ml (units/min)' : 3,
'Vasopressin 40 Units Sodium Chloride 0.9% 200 ml (units/min)' : 3,
'Vasopressin (units/kg/min)' : 3,
'Vasopressin (units/hr)' : 3,
'Vasopressin (units/min)' : 3,
'Milrinone (mcg/kg/min)' : 4,
'Milrinone (mcg/kg/hr)' : 4,
'Milrinone (Primacor) 40 mg Dextrose 5% 200 ml (mcg/kg/min)' : 4,
'Propofol (Diprivan) 1000 mg  100 ml  Premix (mcg/kg/min)' : 5,
'Propofol (Diprivan) 1000 mg  130 ml  Premix (mcg/kg/min)' : 5,
'Propofol (Diprivan) 1000 mg Sterile Water (SWFI) 100 ml  Premix (mcg/kg/min)' : 5,
'Propofol (mg/kg/hr)' : 5,
'Propofol (mcg/hr)' : 5,
'Propofol (mcg/kg/hr)' : 5,
'Propofol (mcg/kg/min)' : 5,
'Propofol (mcg/min)' : 5,
'Propofol (mg/hr)' : 5,
'Propofol (mg/kg/min)' : 5,
'Propofol (mg/min)' : 5,
'Midazolam (mg/hr)' : 6,
'Midazolam (mcg/hr)' : 6,
'Midazolam (mcg/kg/min)' : 6,
'Midazolam (mcg/kg/hr)' : 6,
'Midazolam (mg/kg/hr)' : 6,
'Midazolam (Versed) 100 mg Sodium Chloride 0.9% 100 ml (mg/hr)' : 6,
'Midazolam (Versed) 100 mg Sodium Chloride 0.9% 350 ml (mg/hr)' : 6,
'Versed (mg/hr)' : 6,
'Dexmedetomidine (mcg/hr)' : 7,
'Dexmedetomidine (mcg/kg/hr)' : 7,
'Dexmedetomidine (mcg/kg/hr) ()' : 7,
'Dexmedetomidine (mcg/kg/hr) (mcg/kg/hr)' : 7,
'Dexmedetomidine (mcg/kg/min)' : 7,
'Dexmedetomidine (mcg/min)' : 7,
'Dexmedetomidine (mg/hr)' : 7,
'Dexmedetomidine (mg/kg/hr)' : 7,
'Dexmedetomidine (mg/kg/min)' : 7,
'Dexmedetomidine Inj 400 Mcg in Dextrose 5% 100 ml (mcg/kg/hr)' : 7,
'Dexmedetomidine Inj 400 Mcg in Dextrose 5% 100 ml (mcg/min)' : 7,
'Dexmedetomidine Inj 400 Mcg in Sodium Chloride 0.9% 100 ml (mcg/kg/hr)' : 7,
'Dexmedetomidine Inj 400 Mcg in Sodium Chloride 0.9% 100 ml (mcg/kg/min)' : 7,
'Dexmedetomidine Inj 400 Mcg in Sodium Chloride 0.9% 100 ml (mcg/min)' : 7,
'Ketamine  (mg/hr)' : 8,
'ketamine (mcg/kg/min)' : 8,
'Ketamine (mg/hr)' : 8,
'Ketamine (mg/kg/hr)' : 8,
'Ketamine (mg/kg/min)' : 8,
'Ketamine (mg/min)' : 8,
'Ketamine500mg/250ccNS (mg/hr)' : 8,
'Other Med 500 mg Sodium Chloride 0.9% 250 ml Ketamine HCl (mcg/kg/min)' : 8,
'Fentanyl (mcg/hr)' : 9,
'Fentanyl (mcg/kg/min)' : 9,
'fentanyl  (mcg/hr)' : 9,
'Fentanyl (mcg/kg/hr)' : 9,
'Fentanyl (mcg/min)' : 9,
'Fentanyl (mg/hr)' : 9,
'FentaNYL (Sublimaze) 2500 mcg Sodium Chloride 0.9% 250 ml  Premix (mcg/hr)' : 9,
'FentaNYL (Sublimaze) 2500 mcg Sodium Chloride 0.9% 500 ml  Premix (mcg/hr)' : 9,
'Morphine (mg/hr)' : 10,
'morph (mg/hr)' : 10,
'Morphine 250 mg Sodium Chloride 0.9% 250 ml  Premix (mg/hr)' : 10,
'Morphine 250 mg Sodium Chloride 0.9% 500 ml  Premix (mg/hr)' : 10,
'Furosemide (Lasix) MAX 500 mg Sodium Chloride 0.9% 100 ml (mg/hr)' : 11,
'Furosemide (Lasix) STD 250 mg Sodium Chloride 0.9% 250 ml (mg/hr)' : 11,
'Furosemide (mg/hr)' : 11
}
mapping_extraction_infusion_drug = {
0 : ['Dobutamine (mcg/min)', 'dobutrex (mcg/kg/min)', 'dobutrex (mg/kg/min)', 'DOBUTamine MAX 1000 mg Dextrose 5% 250 ml  Premix (mcg/kg/min)', 'DOBUTamine STD 500 mg Dextrose 5% 250 ml  Premix (mcg/kg/min)'],
1 : ['Epinepherine (mcg/min)', 'Epinephrine (mcg/hr)', 'Epinephrine (mcg/kg/min)', 'Epinephrine (mcg/min)', 'Epinephrine (mg/hr)', 'Epinephrine (mg/kg/min)', 'EPINEPHrine(Adrenalin)MAX 30 mg Sodium Chloride 0.9% 250 ml (mcg/min)', 'EPINEPHrine(Adrenalin)STD 4 mg Sodium Chloride 0.9% 250 ml (mcg/min)', 'EPINEPHrine(Adrenalin)STD 4 mg Sodium Chloride 0.9% 500 ml (mcg/min)', 'EPINEPHrine(Adrenalin)STD 7 mg Sodium Chloride 0.9% 250 ml (mcg/min)'],
2 : ['Norepinephrine (mcg/min)', 'Norepinephrine (mcg/hr)', 'Norepinephrine (mcg/kg/min)', 'Norepinephrine (mg/hr)', 'Norepinephrine (mcg/kg/hr)', 'Norepinephrine (mg/kg/min)', 'Norepinephrine (mg/min)', 'Norepinephrine MAX 32 mg Dextrose 5% 250 ml (mcg/min)', 'Norepinephrine MAX 32 mg Dextrose 5% 500 ml (mcg/min)', 'Norepinephrine STD 32 mg Dextrose 5% 282 ml (mcg/min)', 'Norepinephrine STD 32 mg Dextrose 5% 500 ml (mcg/min)', 'Norepinephrine STD 4 mg Dextrose 5% 250 ml (mcg/min)', 'Norepinephrine STD 4 mg Dextrose 5% 500 ml (mcg/min)', 'Norepinephrine STD 8 mg Dextrose 5% 250 ml (mcg/min)', 'Norepinephrine STD 8 mg Dextrose 5% 500 ml (mcg/min)'],
3 : ['Vasopressin 20 Units Sodium Chloride 0.9% 100 ml (units/hr)', 'Vasopressin 20 Units Sodium Chloride 0.9% 250 ml (units/hr)', 'Vasopressin 40 Units Sodium Chloride 0.9% 100 ml (units/hr)', 'Vasopressin 40 Units Sodium Chloride 0.9% 100 ml (units/kg/hr)', 'Vasopressin 40 Units Sodium Chloride 0.9% 100 ml (units/min)', 'Vasopressin 40 Units Sodium Chloride 0.9% 200 ml (units/min)', 'Vasopressin (units/kg/min)', 'Vasopressin (units/hr)', 'Vasopressin (units/min)'],
4 : ['Milrinone (mcg/kg/min)', 'Milrinone (mcg/kg/hr)', 'Milrinone (Primacor) 40 mg Dextrose 5% 200 ml (mcg/kg/min)'],
5 : ['Propofol (Diprivan) 1000 mg  100 ml  Premix (mcg/kg/min)', 'Propofol (Diprivan) 1000 mg  130 ml  Premix (mcg/kg/min)', 'Propofol (Diprivan) 1000 mg Sterile Water (SWFI) 100 ml  Premix (mcg/kg/min)', 'Propofol (mg/kg/hr)', 'Propofol (mcg/hr)', 'Propofol (mcg/kg/hr)', 'Propofol (mcg/kg/min)', 'Propofol (mcg/min)', 'Propofol (mg/hr)', 'Propofol (mg/kg/min)', 'Propofol (mg/min)'],
6 : ['Midazolam (mg/hr)', 'Midazolam (mcg/hr)', 'Midazolam (mcg/kg/min)', 'Midazolam (mcg/kg/hr)', 'Midazolam (mg/kg/hr)', 'Midazolam (Versed) 100 mg Sodium Chloride 0.9% 100 ml (mg/hr)', 'Midazolam (Versed) 100 mg Sodium Chloride 0.9% 350 ml (mg/hr)', 'Versed (mg/hr)'],
7 : ['Dexmedetomidine (mcg/hr)', 'Dexmedetomidine (mcg/kg/hr)', 'Dexmedetomidine (mcg/kg/hr) ()', 'Dexmedetomidine (mcg/kg/hr) (mcg/kg/hr)', 'Dexmedetomidine (mcg/kg/min)', 'Dexmedetomidine (mcg/min)', 'Dexmedetomidine (mg/hr)', 'Dexmedetomidine (mg/kg/hr)', 'Dexmedetomidine (mg/kg/min)', 'Dexmedetomidine Inj 400 Mcg in Dextrose 5% 100 ml (mcg/kg/hr)', 'Dexmedetomidine Inj 400 Mcg in Dextrose 5% 100 ml (mcg/min)', 'Dexmedetomidine Inj 400 Mcg in Sodium Chloride 0.9% 100 ml (mcg/kg/hr)', 'Dexmedetomidine Inj 400 Mcg in Sodium Chloride 0.9% 100 ml (mcg/kg/min)', 'Dexmedetomidine Inj 400 Mcg in Sodium Chloride 0.9% 100 ml (mcg/min)'],
8 : ['Ketamine  (mg/hr)', 'ketamine (mcg/kg/min)', 'Ketamine (mg/hr)', 'Ketamine (mg/kg/hr)', 'Ketamine (mg/kg/min)', 'Ketamine (mg/min)', 'Ketamine500mg/250ccNS (mg/hr)', 'Other Med 500 mg Sodium Chloride 0.9% 250 ml Ketamine HCl (mcg/kg/min)'],
9 : ['Fentanyl (mcg/hr)', 'Fentanyl (mcg/kg/min)', 'fentanyl  (mcg/hr)', 'Fentanyl (mcg/kg/hr)', 'Fentanyl (mcg/min)', 'Fentanyl (mg/hr)', 'FentaNYL (Sublimaze) 2500 mcg Sodium Chloride 0.9% 250 ml  Premix (mcg/hr)', 'FentaNYL (Sublimaze) 2500 mcg Sodium Chloride 0.9% 500 ml  Premix (mcg/hr)'],
10 : ['Morphine (mg/hr)', 'morph (mg/hr)', 'Morphine 250 mg Sodium Chloride 0.9% 250 ml  Premix (mg/hr)', 'Morphine 250 mg Sodium Chloride 0.9% 500 ml  Premix (mg/hr)'],
11 : ['Furosemide (Lasix) MAX 500 mg Sodium Chloride 0.9% 100 ml (mg/hr)', 'Furosemide (Lasix) STD 250 mg Sodium Chloride 0.9% 250 ml (mg/hr)', 'Furosemide (mg/hr)']
}
mapping_preprocessing_variabel_index={
'AF' : 0,
'AF_spontan' : 1,
'Albumin' : 2,
'Bilirubin_ges.' : 3,
'CK' : 4,
'Compliance' : 5,
'DAP' : 6,
'GOT' : 7,
'GPT' : 8,
'HF' : 9,
'Haematokrit' : 10,
'Haemoglobin' : 11,
'Harnstoff' : 12,
'Horowitz-Quotient_(ohne_Temp-Korrektur)' : 13,
'INR' : 14,
'Koerperkerntemperatur' : 15,
'Kreatinin' : 16,
'Laktat_arteriell' : 17,
'Leukozyten' : 18,
'Lipase_MIMIC' : 19,
'MAP' : 20,
'PEEP' : 21,
'P_EI' : 22,
'SAP' : 23,
'SaO2' : 24,
'SpO2' : 25,
'Thrombozyten' : 26,
'ZVD' : 27,
'individuelles_Tidalvolumen_pro_kg_idealem_Koerpergewicht' : 28,
'pTT' : 29,
'paCO2_(ohne_Temp-Korrektur)' : 30,
'paO2_(ohne_Temp-Korrektur)' : 31,
'SVRI' : 32,
'etCO2' : 33,
'CRP' : 34,
'HZV_(kontinuierlich)' : 35,
'MPAP' : 36,
'Lymphozyten_absolut' : 37,
'BNP_MIMIC' : 38,
'Lymphozyten_prozentual' : 39,
'DPAP' : 40,
'Troponin' : 41,
'I:E' : 42,
'PVRI' : 43,
'FiO2' : 44,
'Amylase' : 45,
'deltaP' : 46,
'SPAP' : 47,
'PCWP' : 48,
'Furosemid_intravenoes_kontinuierlich' : 49,
'Norepinephrin_intravenoes_kontinuierlich' : 50,
'Propofol_intravenoes_kontinuierlich' : 51,
'Vasopressin_intravenoes_kontinuierlich' : 52,
'Midazolam_intravenoes_kontinuierlich' : 53,
'Morphin_intravenoes_kontinuierlich' : 54,
'Milrinon_intravenoes_kontinuierlich' : 55,
'Fentanyl_intravenoes_kontinuierlich' : 56,
'Dobutamin_intravenoes_kontinuierlich' : 57,
'Ketanest_intravenoes_kontinuierlich' : 58,
'Dexmedetomidin_intravenoes_kontinuierlich' : 59,
'Tidalvolumen' : 60,
'Tidalvolumen_spont' : 61,
'CK-MB_MIMIC' : 62,
'LDH_MIMIC' : 63,
'Epinephrin_intravenoes_kontinuierlich' : 64
}
mapping_preprocessing_variabel_index_rev={
 0 : 'AF',
 1 : 'AF_spontan',
 2 : 'Albumin',
 3 : 'Bilirubin_ges.',
 4 : 'CK',
 5 : 'Compliance',
 6 : 'DAP',
 7 : 'GOT',
 8 : 'GPT',
 9 : 'HF',
 10 : 'Haematokrit',
 11 : 'Haemoglobin',
 12 : 'Harnstoff',
 13 : 'Horowitz-Quotient_(ohne_Temp-Korrektur)',
 14 : 'INR',
 15 : 'Koerperkerntemperatur',
 16 : 'Kreatinin',
 17 : 'Laktat_arteriell',
 18 : 'Leukozyten',
 19 : 'Lipase_MIMIC',
 20 : 'MAP',
 21 : 'PEEP',
 22 : 'P_EI',
 23 : 'SAP',
 24 : 'SaO2',
 25 : 'SpO2',
 26 : 'Thrombozyten',
 27 : 'ZVD',
 28 : 'individuelles_Tidalvolumen_pro_kg_idealem_Koerpergewicht',
 29 : 'pTT',
 30 : 'paCO2_(ohne_Temp-Korrektur)',
 31 : 'paO2_(ohne_Temp-Korrektur)',
 32 : 'SVRI',
 33 : 'etCO2',
 34 : 'CRP',
 35 : 'HZV_(kontinuierlich)',
 36 : 'MPAP',
 37 : 'Lymphozyten_absolut',
 38 : 'BNP_MIMIC',
 39 : 'Lymphozyten_prozentual',
 40 : 'DPAP',
 41 : 'Troponin',
 42 : 'I:E',
 43 : 'PVRI',
 44 : 'FiO2',
 45 : 'Amylase',
 46 : 'deltaP',
 47 : 'SPAP',
 48 : 'PCWP',
 49 : 'Furosemid_intravenoes_kontinuierlich',
 50 : 'Norepinephrin_intravenoes_kontinuierlich',
 51 : 'Propofol_intravenoes_kontinuierlich',
 52 : 'Vasopressin_intravenoes_kontinuierlich',
 53 : 'Midazolam_intravenoes_kontinuierlich',
 54 : 'Morphin_intravenoes_kontinuierlich',
 55 : 'Milrinon_intravenoes_kontinuierlich',
 56 : 'Fentanyl_intravenoes_kontinuierlich',
 57 : 'Dobutamin_intravenoes_kontinuierlich',
 58 : 'Ketanest_intravenoes_kontinuierlich',
 59 : 'Dexmedetomidin_intravenoes_kontinuierlich',
 60 : 'Tidalvolumen',
 61 : 'Tidalvolumen_spont',
 62 : 'CK-MB_MIMIC',
 63 : 'LDH_MIMIC',
 64 : 'Epinephrin_intravenoes_kontinuierlich' 
}
mapping_extraction_preprocessing_lab={
'paCO2_(ohne_Temp-Korrektur)' : 30,
'lactate' : 17,
'WBC x 1000' : 18,
'-lymphs' : 39,
'platelets x 1000' : 26,
'CRP' : 34,
'BUN' : 12,
'creatinine' : 16,
'BNP_MIMIC' : 38,
'total bilirubin' : 3,
'Hgb' : 11,
'Hct' : 10,
'albumin' : 2,
'AST (SGOT)' : 7,
'ALT (SGPT)' : 8,
'troponin - T' : 41,
'CPK' : 4,
'CPK-MB' : 62,
'LDH' : 63,
'amylase' : 45,
'PT - INR' : 14,
'pTT' : 29,
'lipase' : 19
}
mapping_extraction_preprocessing_respcharting={
'TOTAL RR' : np.nan,
'RR Spont' : np.nan,
'RR (patient)' : np.nan,
'Spontaneous Respiratory Rate' : np.nan,
'Tidal Volume Observed (VT)' : np.nan,
'Exhaled TV (machine)' : np.nan,
'Exhaled Vt' : np.nan,
'Spont TV' : np.nan,
'Exhaled TV (patient)' : np.nan,
'Vt Spontaneous (mL)' : np.nan,
'PD I:E RATIO' : 42,
'TV/kg IBW' : 28,
'Plateau Pressure' : np.nan,
'Peak Insp. Pressure' : np.nan,
'Compliance' : np.nan,
'Static Compliance' : np.nan,
'EtCO2' : np.nan,
'SaO2' : 24
}
mapping_extraction_preprocessing_drugs={
'Dobutamine (mcg/min)' : 57,
'dobutrex (mcg/kg/min)' : 57,
'dobutrex (mg/kg/min)' : 57,
'DOBUTamine MAX 1000 mg Dextrose 5% 250 ml  Premix (mcg/kg/min)' : 57,
'DOBUTamine STD 500 mg Dextrose 5% 250 ml  Premix (mcg/kg/min)' : 57,
'Epinepherine (mcg/min)' : 64,
'Epinephrine (mcg/hr)' : 64,
'Epinephrine (mcg/kg/min)' : 64,
'Epinephrine (mcg/min)' : 64,
'Epinephrine (mg/hr)' : 64,
'Epinephrine (mg/kg/min)' : 64,
'EPINEPHrine(Adrenalin)MAX 30 mg Sodium Chloride 0.9% 250 ml (mcg/min)' : 64,
'EPINEPHrine(Adrenalin)STD 4 mg Sodium Chloride 0.9% 250 ml (mcg/min)' : 64,
'EPINEPHrine(Adrenalin)STD 4 mg Sodium Chloride 0.9% 500 ml (mcg/min)' : 64,
'EPINEPHrine(Adrenalin)STD 7 mg Sodium Chloride 0.9% 250 ml (mcg/min)' : 64,
'Norepinephrine (mcg/min)' : 50,
'Norepinephrine (mcg/hr)' : 50,
'Norepinephrine (mcg/kg/min)' : 50,
'Norepinephrine (mg/hr)' : 50,
'Norepinephrine (mcg/kg/hr)' : 50,
'Norepinephrine (mg/kg/min)' : 50,
'Norepinephrine (mg/min)' : 50,
'Norepinephrine MAX 32 mg Dextrose 5% 250 ml (mcg/min)' : 50,
'Norepinephrine MAX 32 mg Dextrose 5% 500 ml (mcg/min)' : 50,
'Norepinephrine STD 32 mg Dextrose 5% 282 ml (mcg/min)' : 50,
'Norepinephrine STD 32 mg Dextrose 5% 500 ml (mcg/min)' : 50,
'Norepinephrine STD 4 mg Dextrose 5% 250 ml (mcg/min)' : 50,
'Norepinephrine STD 4 mg Dextrose 5% 500 ml (mcg/min)' : 50,
'Norepinephrine STD 8 mg Dextrose 5% 250 ml (mcg/min)' : 50,
'Norepinephrine STD 8 mg Dextrose 5% 500 ml (mcg/min)' : 50,
'Vasopressin 20 Units Sodium Chloride 0.9% 100 ml (units/hr)' : 52,
'Vasopressin 20 Units Sodium Chloride 0.9% 250 ml (units/hr)' : 52,
'Vasopressin 40 Units Sodium Chloride 0.9% 100 ml (units/hr)' : 52,
'Vasopressin 40 Units Sodium Chloride 0.9% 100 ml (units/kg/hr)' : 52,
'Vasopressin 40 Units Sodium Chloride 0.9% 100 ml (units/min)' : 52,
'Vasopressin 40 Units Sodium Chloride 0.9% 200 ml (units/min)' : 52,
'Vasopressin (units/kg/min)' : 52,
'Vasopressin (units/hr)' : 52,
'Vasopressin (units/min)' : 52,
'Milrinone (mcg/kg/min)' : 55,
'Milrinone (mcg/kg/hr)' : 55,
'Milrinone (Primacor) 40 mg Dextrose 5% 200 ml (mcg/kg/min)' : 55,
'Propofol (Diprivan) 1000 mg  100 ml  Premix (mcg/kg/min)' : 51,
'Propofol (Diprivan) 1000 mg  130 ml  Premix (mcg/kg/min)' : 51,
'Propofol (Diprivan) 1000 mg Sterile Water (SWFI) 100 ml  Premix (mcg/kg/min)' : 51,
'Propofol (mg/kg/hr)' : 51,
'Propofol (mcg/hr)' : 51,
'Propofol (mcg/kg/hr)' : 51,
'Propofol (mcg/kg/min)' : 51,
'Propofol (mcg/min)' : 51,
'Propofol (mg/hr)' : 51,
'Propofol (mg/kg/min)' : 51,
'Propofol (mg/min)' : 51,
'Midazolam (mg/hr)' : 53,
'Midazolam (mcg/hr)' : 53,
'Midazolam (mcg/kg/min)' : 53,
'Midazolam (mcg/kg/hr)' : 53,
'Midazolam (mg/kg/hr)' : 53,
'Midazolam (Versed) 100 mg Sodium Chloride 0.9% 100 ml (mg/hr)' : 53,
'Midazolam (Versed) 100 mg Sodium Chloride 0.9% 350 ml (mg/hr)' : 53,
'Versed (mg/hr)' : 53,
'Dexmedetomidine (mcg/hr)' : 59,
'Dexmedetomidine (mcg/kg/hr)' : 59,
'Dexmedetomidine (mcg/kg/hr) ()' : 59,
'Dexmedetomidine (mcg/kg/hr) (mcg/kg/hr)' : 59,
'Dexmedetomidine (mcg/kg/min)' : 59,
'Dexmedetomidine (mcg/min)' : 59,
'Dexmedetomidine (mg/hr)' : 59,
'Dexmedetomidine (mg/kg/hr)' : 59,
'Dexmedetomidine (mg/kg/min)' : 59,
'Dexmedetomidine Inj 400 Mcg in Dextrose 5% 100 ml (mcg/kg/hr)' : 59,
'Dexmedetomidine Inj 400 Mcg in Dextrose 5% 100 ml (mcg/min)' : 59,
'Dexmedetomidine Inj 400 Mcg in Sodium Chloride 0.9% 100 ml (mcg/kg/hr)' : 59,
'Dexmedetomidine Inj 400 Mcg in Sodium Chloride 0.9% 100 ml (mcg/kg/min)' : 59,
'Dexmedetomidine Inj 400 Mcg in Sodium Chloride 0.9% 100 ml (mcg/min)' : 59,
'Ketamine  (mg/hr)' : 58,
'ketamine (mcg/kg/min)' : 58,
'Ketamine (mg/hr)' : 58,
'Ketamine (mg/kg/hr)' : 58,
'Ketamine (mg/kg/min)' : 58,
'Ketamine (mg/min)' : 58,
'Ketamine500mg/250ccNS (mg/hr)' : 58,
'Other Med 500 mg Sodium Chloride 0.9% 250 ml Ketamine HCl (mcg/kg/min)' : 58,
'Fentanyl (mcg/hr)' : 56,
'Fentanyl (mcg/kg/min)' : 56,
'fentanyl  (mcg/hr)' : 56,
'Fentanyl (mcg/kg/hr)' : 56,
'Fentanyl (mcg/min)' : 56,
'Fentanyl (mg/hr)' : 56,
'FentaNYL (Sublimaze) 2500 mcg Sodium Chloride 0.9% 250 ml  Premix (mcg/hr)' : 56,
'FentaNYL (Sublimaze) 2500 mcg Sodium Chloride 0.9% 500 ml  Premix (mcg/hr)' : 56,
'Morphine (mg/hr)' : 54,
'morph (mg/hr)' : 54,
'Morphine 250 mg Sodium Chloride 0.9% 250 ml  Premix (mg/hr)' : 54,
'Morphine 250 mg Sodium Chloride 0.9% 500 ml  Premix (mg/hr)' : 54,
'Furosemide (Lasix) MAX 500 mg Sodium Chloride 0.9% 100 ml (mg/hr)' : 49,
'Furosemide (Lasix) STD 250 mg Sodium Chloride 0.9% 250 ml (mg/hr)' : 49,
'Furosemide (mg/hr)' : 49
}
mapping_drug_index_needed_unit={
 49 : ['(mg/hr)'],
 54 : ['(mg/hr)'],
 50 : ['(mcg/kg/min)'],
 51 : ['(mg/hr)'],
 52 : ['(units/min)'],
 53 : ['(mg/hr)'],
 55 : ['(mcg/kg/min)'],
 56 : ['(mg/hr)'],
 57 : ['(mcg/kg/min)'],
 58 : ['(mg/hr)'],
 59 : ['(mcg/kg/min)'],
 64 : ['(mcg/kg/min)'] 
}
mapping_drug_index_used_unit={
 49 : ['(mg/hr)'],
 50 : ['(mcg/min)', '(mcg/hr)', '(mcg/kg/min)', '(mg/hr)', '(mcg/kg/hr)', '(mg/kg/min)', '(mg/min)'],
 51 : ['(mcg/kg/min)', '(mg/kg/hr)', '(mcg/hr)', '(mcg/kg/hr)', '(mcg/min)', '(mg/hr)', '(mg/kg/min)', '(mg/min)'],
 52 : ['(units/hr)', '(units/kg/hr)', '(units/min)', '(units/kg/min)'],
 53 : ['(mg/hr)', '(mcg/hr)', '(mcg/kg/min)', '(mcg/kg/hr)', '(mg/kg/hr)'],
 54 : ['(mg/hr)'],
 55 : ['(mg/hr)', '(mcg/hr)', '(mcg/kg/min)', '(mcg/kg/hr)', '(mg/kg/hr)'],
 56 : ['(mcg/hr)', '(mcg/kg/min)', '(mcg/kg/hr)', '(mcg/min)', '(mg/hr)'],
 57 : ['(mcg/min)', '(mcg/kg/min)', '(mg/kg/min)'],
 58 : ['(mg/hr)', '(mcg/kg/min)', '(mg/kg/hr)', '(mg/kg/min)', '(mg/min)'],
 59 : ['(mcg/hr)', '(mcg/kg/hr)', '(mcg/kg/min)', '(mcg/min)', '(mg/hr)', '(mg/kg/hr)', '(mg/kg/min)'],
 64 : ['(mcg/min)', '(mcg/hr)', '(mcg/kg/min)', '(mg/hr)', '(mg/kg/min)'] 
}
